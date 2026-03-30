"""
collect_data.py — Collect historical + live option data for all Indian index options.

Assets: NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY (NSE) + SENSEX, BANKEX (BSE)

Usage:
    python collect_data.py              # historical bulk (run once)
    python collect_data.py --live 60    # live snapshots every 60s
    python collect_data.py --both 60    # historical once + live loop
"""

import argparse
import sqlite3
import time
import pandas as pd
from datetime import datetime, timedelta, timezone

import pytz
from dotenv import load_dotenv
from nubra_python_sdk.start_sdk import InitNubraSdk, NubraEnv
from nubra_python_sdk.marketdata.market_data import MarketData
from nubra_python_sdk.marketdata.validation import ExchangeEnum

load_dotenv()
DB_PATH = "data.db"
IST = pytz.timezone("Asia/Kolkata")


def market_status() -> str:
    """Returns: 'open', 'pre_open', 'post_market', 'closed'"""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return "closed"
    t = now.strftime("%H:%M")
    if "09:00" <= t < "09:15":
        return "pre_open"
    if "09:15" <= t <= "15:30":
        return "open"
    if "15:30" < t <= "18:00":
        return "post_market"
    return "closed"


def next_open() -> str:
    now = datetime.now(IST)
    days_ahead = 1
    if now.weekday() == 4:
        days_ahead = 3
    elif now.weekday() == 5:
        days_ahead = 2
    return (now + timedelta(days=days_ahead)).strftime("%a %d %b 09:15 IST")


ASSETS = [
    ("NIFTY", ExchangeEnum.NSE, "NSE", "INDEX"),
    ("BANKNIFTY", ExchangeEnum.NSE, "NSE", "INDEX"),
    ("FINNIFTY", ExchangeEnum.NSE, "NSE", "INDEX"),
    ("MIDCPNIFTY", ExchangeEnum.NSE, "NSE", "INDEX"),
    ("SENSEX", ExchangeEnum.BSE, "BSE", "INDEX"),
    ("BANKEX", ExchangeEnum.BSE, "BSE", "INDEX"),
]

MONTH_MAP = {
    1: "JAN",
    2: "FEB",
    3: "MAR",
    4: "APR",
    5: "MAY",
    6: "JUN",
    7: "JUL",
    8: "AUG",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}

OPT_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "iv_mid",
    "delta",
    "gamma",
    "theta",
    "vega",
    "cumulative_oi",
]


# ── DB ────────────────────────────────────────────────────────────────────────


def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS option_chain_snapshot (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at INTEGER NOT NULL,
            asset        TEXT NOT NULL,
            exchange     TEXT NOT NULL,
            expiry       TEXT NOT NULL,
            option_type  TEXT NOT NULL,
            inst_id      INTEGER,
            strike       REAL,
            ltp          REAL,
            iv           REAL,
            delta        REAL,
            gamma        REAL,
            theta        REAL,
            vega         REAL,
            oi           INTEGER,
            volume       INTEGER
        );
        CREATE TABLE IF NOT EXISTS historical_candle (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol   TEXT NOT NULL,
            exchange TEXT NOT NULL,
            interval TEXT NOT NULL,
            ts       INTEGER NOT NULL,
            open     REAL, high REAL, low REAL, close REAL, volume INTEGER,
            UNIQUE(symbol, exchange, interval, ts)
        );
        CREATE TABLE IF NOT EXISTS historical_option (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol   TEXT NOT NULL,
            asset    TEXT NOT NULL,
            exchange TEXT NOT NULL,
            interval TEXT NOT NULL,
            ts       INTEGER NOT NULL,
            open     REAL, high REAL, low REAL, close REAL, volume INTEGER,
            iv       REAL, delta REAL, gamma REAL, theta REAL, vega REAL,
            oi       INTEGER,
            UNIQUE(symbol, exchange, interval, ts)
        );
        CREATE INDEX IF NOT EXISTS idx_chain_asset
            ON option_chain_snapshot(asset, expiry, collected_at);
        CREATE INDEX IF NOT EXISTS idx_candle_sym
            ON historical_candle(symbol, exchange, interval, ts);
        CREATE INDEX IF NOT EXISTS idx_hist_opt
            ON historical_option(asset, exchange, interval, ts);
    """
    )
    conn.commit()
    return conn


# ── Helpers ───────────────────────────────────────────────────────────────────


def nse_symbol(asset: str, expiry: str, strike: int, opt_type: str) -> str:
    """NIFTY26MAR23300CE"""
    y = expiry[2:4]
    mon = MONTH_MAP[int(expiry[4:6])]
    return f"{asset}{y}{mon}{strike}{opt_type}"


def bse_symbol(asset: str, expiry: str, strike: int, opt_type: str) -> str:
    """SENSEX26MAR75300CE"""
    y = expiry[2:4]
    mon = MONTH_MAP[int(expiry[4:6])]
    return f"{asset}{y}{mon}{strike}{opt_type}"


def strikes_near_atm(chain, n: int = 12) -> list:
    atm = chain.at_the_money_strike
    all_s = sorted({x.strike_price for x in (chain.ce or [])})
    if not all_s:
        return []
    idx = min(range(len(all_s)), key=lambda i: abs(all_s[i] - atm))
    lo, hi = max(0, idx - n), min(len(all_s), idx + n + 1)
    return [s // 100 for s in all_s[lo:hi]]


def _pts(stock, field: str) -> dict:
    return {p.timestamp: p.value for p in (getattr(stock, field, None) or [])}


# ── Save functions ────────────────────────────────────────────────────────────


def save_index_candles(conn, hist_resp, symbol: str, exchange: str, interval: str):
    try:
        v0 = hist_resp.result[0].values[0]
        stock = v0.get(symbol)
        if not stock or not stock.close:
            return 0
        closes = _pts(stock, "close")
        opens = _pts(stock, "open")
        highs = _pts(stock, "high")
        lows = _pts(stock, "low")
        volumes = _pts(stock, "tick_volume")
        rows = [
            (
                symbol,
                exchange,
                interval,
                ts,
                opens.get(ts, 0) / 100,
                highs.get(ts, 0) / 100,
                lows.get(ts, 0) / 100,
                v / 100,
                volumes.get(ts, 0),
            )
            for ts, v in closes.items()
        ]
    except (AttributeError, IndexError, KeyError, TypeError) as e:
        print(f"    [db] Index parse error {symbol}: {e}")
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO historical_candle
            (symbol,exchange,interval,ts,open,high,low,close,volume)
        VALUES (?,?,?,?,?,?,?,?,?)
    """,
        rows,
    )
    conn.commit()
    return len(rows)


def save_option_candles(
    conn, hist_resp, symbol: str, asset: str, exchange: str, interval: str
):
    try:
        v0 = hist_resp.result[0].values[0]
        stock = list(v0.values())[0]
        closes = _pts(stock, "close")
        if not closes:
            return 0
        opens = _pts(stock, "open")
        highs = _pts(stock, "high")
        lows = _pts(stock, "low")
        volumes = _pts(stock, "tick_volume")
        ivs = _pts(stock, "iv_mid")
        deltas = _pts(stock, "delta")
        gammas = _pts(stock, "gamma")
        thetas = _pts(stock, "theta")
        vegas = _pts(stock, "vega")
        ois = _pts(stock, "cumulative_oi")
        rows = [
            (
                symbol,
                asset,
                exchange,
                interval,
                ts,
                opens.get(ts, 0) / 100,
                highs.get(ts, 0) / 100,
                lows.get(ts, 0) / 100,
                v / 100,
                volumes.get(ts, 0),
                ivs.get(ts),
                deltas.get(ts),
                gammas.get(ts),
                thetas.get(ts),
                vegas.get(ts),
                ois.get(ts),
            )
            for ts, v in closes.items()
        ]
    except (AttributeError, IndexError, KeyError, TypeError):
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO historical_option
            (symbol,asset,exchange,interval,ts,open,high,low,close,volume,
             iv,delta,gamma,theta,vega,oi)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,
        rows,
    )
    conn.commit()
    return len(rows)


def save_live_snapshot(conn, wrapper, asset: str, exchange: str):
    """Insert snapshot only if data changed since last run (dedup by ATM ltp)."""
    now   = int(time.time())
    chain = wrapper.chain
    expiry = chain.expiry or ""
    rows = []
    for opt_type, items in [("CE", chain.ce or []), ("PE", chain.pe or [])]:
        for item in items:
            sp  = item.strike_price
            ltp = item.last_traded_price
            rows.append((
                now, asset, exchange, expiry, opt_type,
                item.ref_id,
                sp / 100 if sp is not None else None,
                ltp / 100 if ltp is not None else None,
                item.iv, item.delta, item.gamma,
                item.theta, item.vega,
                item.open_interest, item.volume,
            ))

    if not rows:
        return 0

    # Dedup: skip if ATM ltp unchanged since last snapshot
    mid = len(rows) // 2
    new_ltp = rows[mid][7]
    if new_ltp is not None:
        last = conn.execute(
            "SELECT ltp FROM option_chain_snapshot "
            "WHERE asset=? AND expiry=? ORDER BY collected_at DESC LIMIT 1",
            (asset, expiry)
        ).fetchone()
        if last and last[0] is not None and abs(last[0] - new_ltp) < 0.01:
            return 0  # unchanged — skip

    conn.executemany(
        """
        INSERT INTO option_chain_snapshot
            (collected_at,asset,exchange,expiry,option_type,
             inst_id,strike,ltp,iv,delta,gamma,theta,vega,oi,volume)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,
        rows,
    )
    conn.commit()
    return len(rows)


def collect_historical(md: MarketData, conn: sqlite3.Connection):
    print("\n[collect] === HISTORICAL BULK FETCH ===")
    end_dt = datetime.now(timezone.utc)

    # Smart start date: only fetch data we don't already have
    def _last_candle_date(asset_sym, exc_str):
        row = conn.execute(
            "SELECT MAX(ts) FROM historical_candle WHERE symbol=? AND exchange=? AND interval='1d'",
            (asset_sym, exc_str)
        ).fetchone()
        if row and row[0]:
            last_ts = pd.Timestamp(row[0], unit='ns', tz='UTC')
            # Start from day after last candle
            return last_ts + timedelta(days=1)
        return end_dt - timedelta(days=90)  # default: 90 days back

    end_s = end_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    def _hist(payload, retries=2):
        """Call historical_data with retry on 403."""
        for attempt in range(retries):
            try:
                return md.historical_data(payload)
            except Exception as e:
                if "403" in str(e) and attempt < retries - 1:
                    print("    [auth] 403 — refreshing session...")
                    try:
                        md.auth_flow()  # re-authenticate
                    except Exception:
                        pass
                    time.sleep(2)
                else:
                    raise
        return None

    for asset, exc, exc_str, itype in ASSETS:
        print(f"\n  [{asset}]")

        # 1. Index OHLCV — only fetch dates not already in DB
        try:
            asset_start = _last_candle_date(asset, exc_str)
            if asset_start.date() >= end_dt.date():
                print(f"    index candles: already up to date")
            else:
                hist = _hist({
                    "exchange": exc_str,
                    "type": itype,
                    "values": [asset],
                    "fields": ["open", "high", "low", "close", "tick_volume"],
                    "startDate": asset_start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "endDate": end_s,
                    "interval": "1d",
                    "intraDay": False,
                    "realTime": False,
                })
                n = save_index_candles(conn, hist, asset, exc_str, "1d")
                print(f"    index candles: {n} new")
        except Exception as e:  # noqa: BLE001
            print(f"    index error: {e}")
        # 2. Option chain → strikes
        try:
            wrapper = md.option_chain(instrument=asset, exchange=exc)
            chain = wrapper.chain
            expiries = (chain.all_expiries or [chain.expiry])[:2]
            strikes = strikes_near_atm(chain, n=12)
            sym_fn = bse_symbol if exc_str == "BSE" else nse_symbol
        except Exception as e:  # noqa: BLE001
            print(f"    chain error: {e}")
            continue

        # 3. Historical option candles
        total = 0
        for expiry in expiries:
            for strike in strikes:
                for opt_type in ("CE", "PE"):
                    sym = sym_fn(asset, expiry, strike, opt_type)
                    try:
                        hist = _hist(
                            {
                                "exchange": exc_str,
                                "type": "OPT",
                                "values": [sym],
                                "fields": OPT_FIELDS,
                                "startDate": start_s,
                                "endDate": end_s,
                                "interval": "1d",
                                "intraDay": False,
                                "realTime": False,
                            }
                        )
                        total += save_option_candles(
                            conn, hist, sym, asset, exc_str, "1d"
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    time.sleep(0.12)

        print(f"    option candles: {total}")


def collect_live(md: MarketData, conn: sqlite3.Connection):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[collect] {ts} — live snapshots")
    total = 0
    for asset, exc, exc_str, _ in ASSETS:
        try:
            wrapper = md.option_chain(instrument=asset, exchange=exc)
            n = save_live_snapshot(conn, wrapper, asset, exc_str)
            total += n
            # Next expiry
            expiries = wrapper.chain.all_expiries or []
            if len(expiries) > 1:
                try:
                    w2 = md.option_chain(
                        instrument=asset, expiry=expiries[1], exchange=exc
                    )
                    total += save_live_snapshot(conn, w2, asset, exc_str)
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001
            print(f"  [{asset}] error: {e}")
    print(f"  saved {total} rows total")


def row_counts(conn):
    c1 = conn.execute("SELECT COUNT(*) FROM option_chain_snapshot").fetchone()[0]
    c2 = conn.execute("SELECT COUNT(*) FROM historical_candle").fetchone()[0]
    c3 = conn.execute("SELECT COUNT(*) FROM historical_option").fetchone()[0]
    print(f"[db] live={c1}  index_candles={c2}  opt_candles={c3}")


# ── Entry ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", type=int, default=0, help="Live snapshots every N seconds"
    )
    parser.add_argument(
        "--both", type=int, default=0, help="Historical once + live loop"
    )
    parser.add_argument(
        "--historical", action="store_true", help="Force historical bulk fetch"
    )
    parser.add_argument(
        "--eod", action="store_true", help="End-of-day snapshot (after 15:30)"
    )
    args = parser.parse_args()

    status = market_status()
    now_ist = datetime.now(IST).strftime("%H:%M IST")
    print(f"[collect] Market status: {status.upper()} ({now_ist})")

    print("[collect] Authenticating (Production)...")
    nubra = InitNubraSdk(NubraEnv.PROD, env_creds=True)
    md = MarketData(nubra)
    conn = init_db(DB_PATH)
    print("[collect] Ready")

    # ── EOD snapshot (run after 15:30, before 18:00) ──────────────────────────
    if args.eod or status == "post_market":
        print(
            "[collect] Post-market: fetching EOD data (today's completed candle + final chain)"
        )
        collect_historical(md, conn)  # picks up today's completed daily candle
        collect_live(md, conn)  # final option chain snapshot with EOD prices
        row_counts(conn)
        print("[collect] EOD collection done.")
        return

    # ── Explicit historical ───────────────────────────────────────────────────
    if args.historical:
        collect_historical(md, conn)
        row_counts(conn)
        print("\n[collect] Done. Now run:  python build_model.py")
        return

    # ── Live loop (market hours) ──────────────────────────────────────────────
    if args.live > 0:
        if status not in ("open", "pre_open"):
            print(f"[collect] Market is {status}. Live data not available.")
            print(f"[collect] Next open: {next_open()}")
            print("[collect] Running historical fetch instead...")
            collect_historical(md, conn)
            row_counts(conn)
            return
        print(f"[collect] Live loop every {args.live}s. Ctrl+C to stop.")
        try:
            while True:
                if market_status() not in ("open", "pre_open"):
                    print("[collect] Market closed. Switching to EOD collection...")
                    collect_historical(md, conn)
                    collect_live(md, conn)
                    row_counts(conn)
                    print(f"[collect] Done. Next open: {next_open()}")
                    break
                collect_live(md, conn)
                row_counts(conn)
                time.sleep(args.live)
        except KeyboardInterrupt:
            print("\n[collect] Stopped.")
        return

    # ── Both: historical + live ───────────────────────────────────────────────
    if args.both > 0:
        collect_historical(md, conn)
        row_counts(conn)
        if status not in ("open", "pre_open"):
            print(f"[collect] Market is {status}. Skipping live loop.")
            print(f"[collect] Next open: {next_open()}")
            return
        print(f"\n[collect] Live loop every {args.both}s. Ctrl+C to stop.")
        try:
            while True:
                if market_status() not in ("open", "pre_open"):
                    print("[collect] Market closed. Final EOD snapshot...")
                    collect_live(md, conn)
                    row_counts(conn)
                    break
                collect_live(md, conn)
                row_counts(conn)
                time.sleep(args.both)
        except KeyboardInterrupt:
            print("\n[collect] Stopped.")
        return

    # ── Default: auto-detect ──────────────────────────────────────────────────
    if status == "open":
        print("[collect] Market is OPEN — running live snapshot + historical")
        collect_historical(md, conn)
        collect_live(md, conn)
    elif status == "post_market":
        print("[collect] Post-market — fetching EOD data")
        collect_historical(md, conn)
        collect_live(md, conn)
    else:
        print(f"[collect] Market is {status} — fetching historical only")
        print(f"[collect] Next open: {next_open()}")
        collect_historical(md, conn)

    row_counts(conn)
    print("\n[collect] Done. Now run:  python build_model.py")


if __name__ == "__main__":
    main()
