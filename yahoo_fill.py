"""
Yahoo Finance fill for tickers missing from OHLC.parquet  (run daily by GitHub Actions, or by hand)

The Terminal (index.html) charts weekly bars from OHLC.parquet. Some tickers in
Results.parquet have no rows there (mostly NSE SME listings). This script fetches
their weekly OHLCV from Yahoo Finance into Yahoo_OHLC.parquet, which index.html
reads and uses ONLY for tickers that OHLC.parquet does not have.

GitHub Actions (.github/workflows/yahoo_fill.yml) runs "all" Mon–Sat ~5:40 PM IST (backups ~8:15 PM, ~10:50 PM).
Commands:

    python yahoo_fill.py            same as "status"
    python yahoo_fill.py status     counts + lists, no network
    python yahoo_fill.py new        full history for missing tickers not fetched yet
    python yahoo_fill.py update     refresh Yahoo tickers whose data is out of date
    python yahoo_fill.py clean      drop Yahoo tickers that OHLC.parquet now has
    python yahoo_fill.py all        clean + update + new

Options:
    --tickers A,B,C     only these tickers (with new / update)
    --limit N           at most N tickers this run
    --retry-notfound    "new" also retries tickers Yahoo did not have last time

Files written (same folder):
    Yahoo_OHLC.parquet       Date, Ticker, Open, High, Low, Close, Volume  (same schema as OHLC.parquet)
    Yahoo_OHLC_meta.json     per ticker: Yahoo symbol, name, fetch time, last bar, or "notfound"

Yahoo symbol: TICKER.NS first, then TICKER.BO. Yahoo labels many NSE SME stocks
"MUTUALFUND"; their bars are real, so the name is saved in the meta file for checking.
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OHLC = os.path.join(HERE, 'OHLC.parquet')
RESULTS = os.path.join(HERE, 'Results.parquet')
YF_PARQUET = os.path.join(HERE, 'Yahoo_OHLC.parquet')
YF_META = os.path.join(HERE, 'Yahoo_OHLC_meta.json')

HISTORY_START = datetime(2019, 7, 1, tzinfo=timezone.utc)   # OHLC.parquet starts 2019-07
UPDATE_OVERLAP_DAYS = 28                                     # re-fetch the last 4 weeks on update
REQUEST_GAP_S = 0.4
IST = timezone(timedelta(hours=5, minutes=30))
COLS = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']

session = requests.Session()
session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'


# ── week / staleness rules (index.html uses the same ones) ──────────────────
def last_market_close_utc(now):
    """Most recent trading-day close (Mon–Fri 15:30 IST) at or before `now`, as UTC."""
    d = now.astimezone(IST)
    close = d.replace(hour=15, minute=30, second=0, microsecond=0)
    if close > d:
        close -= timedelta(days=1)
    while close.weekday() >= 5:                         # Saturday / Sunday
        close -= timedelta(days=1)
    return close.astimezone(timezone.utc)


def is_stale(meta_rec, latest_parquet_week, now):
    """Out of date when the last bar is older than OHLC.parquet's newest week, or a
    trading day has closed since the last fetch (so the running-week bar is
    refreshed every day, and a new week's bar is picked up on its first day)."""
    last = meta_rec.get('last')
    if not last:
        return True
    if last < latest_parquet_week:
        return True
    fetched = datetime.fromisoformat(meta_rec['fetched'])
    return fetched < last_market_close_utc(now)


# ── Yahoo ────────────────────────────────────────────────────────────────────
def fetch_chart(symbol, start):
    url = 'https://query1.finance.yahoo.com/v8/finance/chart/' + urllib.parse.quote(symbol)
    params = {'interval': '1wk', 'period1': int(start.timestamp()),
              'period2': int(time.time()) + 86400, 'events': 'history'}
    for attempt in range(4):
        r = session.get(url, params=params, timeout=20)
        if r.status_code == 429:
            wait = 5 * (2 ** attempt)
            print(f'    rate-limited, waiting {wait}s')
            time.sleep(wait)
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        res = (r.json().get('chart') or {}).get('result')
        return res[0] if res else None
    raise RuntimeError('still rate-limited after retries')


def bars_from_chart(ticker, res):
    ts = res.get('timestamp') or []
    q = ((res.get('indicators') or {}).get('quote') or [{}])[0]
    by_week = {}
    for i, t in enumerate(ts):
        c = (q.get('close') or [None] * len(ts))[i]
        if not c or c <= 0:
            continue
        o = (q.get('open') or [None] * len(ts))[i] or c
        h = (q.get('high') or [None] * len(ts))[i] or max(o, c)
        lo = (q.get('low') or [None] * len(ts))[i] or min(o, c)
        v = (q.get('volume') or [None] * len(ts))[i] or 0
        h, lo = max(h, o, c), min(lo, o, c)
        if lo <= 0:
            continue
        day = datetime.fromtimestamp(t, IST).date()
        monday = (day - timedelta(days=day.weekday())).isoformat()
        # Yahoo can repeat the running week (weekly + latest daily stamp): the later row wins
        by_week[monday] = (monday, ticker, round(o, 2), round(h, 2), round(lo, 2), round(c, 2), int(v))
    return [by_week[k] for k in sorted(by_week)]


def fetch_ticker(ticker, start, known_symbol=None):
    """Returns (symbol, name, rows) or None when Yahoo has no data for .NS or .BO."""
    for sym in ([known_symbol] if known_symbol else [ticker + '.NS', ticker + '.BO']):
        res = fetch_chart(sym, start)
        time.sleep(REQUEST_GAP_S)
        if not res:
            continue
        rows = bars_from_chart(ticker, res)
        if rows:
            meta = res.get('meta') or {}
            return sym, meta.get('longName') or meta.get('shortName') or '', rows
    return None


# ── files ────────────────────────────────────────────────────────────────────
def load_state():
    ohlc = pd.read_parquet(OHLC, columns=['Date', 'Ticker'])
    parquet_tickers = set(ohlc['Ticker'].astype(str).str.strip().str.upper())
    latest_week = str(ohlc['Date'].max())[:10]
    res = pd.read_parquet(RESULTS, columns=['Ticker'])
    universe = {t for t in res['Ticker'].dropna().astype(str).str.strip().str.upper()
                if t and '_' not in t}                # skip index rows like Nifty_TOTAL
    yf = pd.read_parquet(YF_PARQUET) if os.path.exists(YF_PARQUET) else pd.DataFrame(columns=COLS)
    meta = {}
    if os.path.exists(YF_META):
        with open(YF_META, encoding='utf-8') as f:
            meta = json.load(f).get('tickers', {})
    return parquet_tickers, latest_week, universe, yf, meta


def save(yf, meta):
    yf = yf.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    yf['Volume'] = yf['Volume'].astype('int64')
    tmp = YF_PARQUET + '.tmp'
    yf[COLS].to_parquet(tmp, index=False)
    os.replace(tmp, YF_PARQUET)
    with open(YF_META + '.tmp', 'w', encoding='utf-8') as f:
        json.dump({'updated': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                   'tickers': meta}, f, indent=1, sort_keys=True)
    os.replace(YF_META + '.tmp', YF_META)


def classify(parquet_tickers, latest_week, universe, yf, meta):
    now = datetime.now(timezone.utc)
    have = set(yf['Ticker'].unique()) if len(yf) else set()
    missing = universe - parquet_tickers
    notfound = {t for t in missing if meta.get(t, {}).get('status') == 'notfound' and t not in have}
    return {
        'missing': missing,
        'have': have & missing,
        'stale': {t for t in have & missing if is_stale(meta.get(t, {}), latest_week, now)},
        'new': missing - have - notfound,
        'notfound': notfound,
        'removable': have & parquet_tickers,
    }


def print_status(c, latest_week):
    print(f"OHLC.parquet newest week: {latest_week}")
    print(f"Tickers in Results.parquet with no OHLC.parquet data: {len(c['missing'])}")
    print(f"  Yahoo data present      : {len(c['have'])}  ({len(c['stale'])} need update)")
    print(f"  Need full download (new): {len(c['new'])}")
    print(f"  Not found on Yahoo      : {len(c['notfound'])}")
    print(f"Yahoo tickers now in OHLC.parquet (removable): {len(c['removable'])}")
    for key, label in (('stale', 'Need update'), ('new', 'Need full download'), ('removable', 'Removable')):
        if c[key]:
            items = sorted(c[key])
            print(f"\n{label}: " + ', '.join(items[:60]) + (f' … (+{len(items) - 60})' if len(items) > 60 else ''))


# ── commands ─────────────────────────────────────────────────────────────────
def pick(targets, args):
    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers.split(',') if t.strip()}
        targets = [t for t in targets if t in wanted]
    targets = sorted(targets)
    return targets[:args.limit] if args.limit else targets


def run_fetch(targets, yf, meta, full):
    if not targets:
        print('Nothing to fetch.')
        return yf
    print(f"Fetching {len(targets)} ticker(s) from Yahoo ({'full history' if full else 'update'}) …")
    fetched = {}                                        # ticker -> DataFrame of fetched weeks
    had = set(yf['Ticker'].unique()) if len(yf) else set()
    done = notfound = failed = 0
    try:
        for i, t in enumerate(targets, 1):
            rec = meta.get(t, {})
            if full or not rec.get('last'):
                start = HISTORY_START
            else:
                start = datetime.strptime(rec['last'], '%Y-%m-%d').replace(tzinfo=timezone.utc) - timedelta(days=UPDATE_OVERLAP_DAYS)
            try:
                got = fetch_ticker(t, start, None if full else rec.get('symbol'))
            except Exception as e:                      # network / HTTP error: keep old data
                failed += 1
                print(f'  [{i}/{len(targets)}] {t}: failed ({e})')
                continue
            now = datetime.now(timezone.utc).isoformat(timespec='seconds')
            if not got:
                notfound += 1
                if t not in had:
                    meta[t] = {'status': 'notfound', 'fetched': now}
                print(f'  [{i}/{len(targets)}] {t}: not on Yahoo')
                continue
            sym, name, rows = got
            fetched[t] = pd.DataFrame(rows, columns=COLS)
            last = rows[-1][0]
            if not full and rec.get('last') and rec['last'] > last:
                last = rec['last']
            meta[t] = {'status': 'ok', 'symbol': sym, 'name': name, 'fetched': now, 'last': last}
            done += 1
            print(f'  [{i}/{len(targets)}] {t}: {len(rows)} weeks via {sym}  {name}')
    except KeyboardInterrupt:
        print('\nStopped — saving what was fetched so far.')

    if fetched:
        keep = yf
        if len(keep):
            if full:                                    # full history replaces the ticker
                keep = keep[~keep['Ticker'].isin(fetched.keys())]
            else:                                       # update replaces only re-fetched weeks
                first = {t: df['Date'].min() for t, df in fetched.items()}
                cut = keep['Ticker'].map(first)
                keep = keep[cut.isna() | (keep['Date'] < cut.fillna(''))]
        yf = pd.concat([keep] + list(fetched.values()), ignore_index=True)
    print(f'Done: {done} fetched, {notfound} not on Yahoo, {failed} failed.')
    return yf


def main():
    ap = argparse.ArgumentParser(description='Manual Yahoo Finance fill for tickers missing from OHLC.parquet')
    ap.add_argument('command', nargs='?', default='status', choices=['status', 'new', 'update', 'clean', 'all'])
    ap.add_argument('--tickers')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--retry-notfound', action='store_true')
    args = ap.parse_args()

    parquet_tickers, latest_week, universe, yf, meta = load_state()
    c = classify(parquet_tickers, latest_week, universe, yf, meta)
    if args.command == 'status':
        print_status(c, latest_week)
        return

    changed = False
    if args.command in ('clean', 'all') and c['removable']:
        yf = yf[~yf['Ticker'].isin(c['removable'])]
        for t in c['removable']:
            meta.pop(t, None)
        print(f"Removed {len(c['removable'])} ticker(s) now in OHLC.parquet: {', '.join(sorted(c['removable']))}")
        changed = True
    if args.command in ('update', 'all'):
        yf = run_fetch(pick(c['stale'], args), yf, meta, False)
        changed = True
    if args.command in ('new', 'all'):
        targets = c['new'] | (c['notfound'] if args.retry_notfound else set())
        yf = run_fetch(pick(targets, args), yf, meta, True)
        changed = True
    if changed:
        save(yf, meta)
        print(f'Saved {YF_PARQUET} ({yf["Ticker"].nunique()} tickers, {len(yf)} rows).')
    print()
    print_status(classify(parquet_tickers, latest_week, universe, yf, meta), latest_week)


if __name__ == '__main__':
    sys.exit(main())
