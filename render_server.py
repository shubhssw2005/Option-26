"""
render_server.py — Production FastAPI server for Render deployment.
Uses direct HTTP calls to Nubra API (no SDK auth issues).
"""

import os
import asyncio
import sqlite3
import threading
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytz
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
IS_RENDER  = os.getenv("RENDER", "") == "true"
DB_PATH    = "/tmp/data.db"        if IS_RENDER else "data.db"
MODELS_DIR = "/tmp/trained_models" if IS_RENDER else "trained_models"
IST        = pytz.timezone("Asia/Kolkata")

os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["DB_PATH"]    = DB_PATH
os.environ["MODELS_DIR"] = MODELS_DIR

# Lazy imports — heavy ML libs loaded on first use to avoid slow startup
_lazy = {}

def _vol():
    if "vol" not in _lazy:
        from models.vol_models import fit_garch, realized_vol
        _lazy["vol"] = (fit_garch, realized_vol)
    return _lazy["vol"]

def _strategy():
    if "strategy" not in _lazy:
        from models.strategy_engine import score_strategies, classify_regime, STRATEGY_CATALOG, liquidity_score
        _lazy["strategy"] = (score_strategies, classify_regime, STRATEGY_CATALOG, liquidity_score)
    return _lazy["strategy"]

def _signals():
    if "signals" not in _lazy:
        from realtime_model import generate_signals, generate_all_signals
        _lazy["signals"] = (generate_signals, generate_all_signals)
    return _lazy["signals"]

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "client": None,  # NubraDirectClient
    "index_cache": {},
    "greeks_cache": {},
    "ws_connected": False,
}
_lock = threading.Lock()


def query_df(sql: str, params=()):
    with sqlite3.connect(DB_PATH) as c:
        return pd.read_sql_query(sql, c, params=params)


def market_status() -> str:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return "closed"
    t = now.strftime("%H:%M")
    if "09:15" <= t <= "15:30":
        return "open"
    if "15:30" < t <= "18:00":
        return "post_market"
    return "closed"


# ── Keep-alive ────────────────────────────────────────────────────────────────
def _keep_alive():
    import urllib.request

    time.sleep(60)
    while True:
        try:
            port = os.getenv("PORT", "10000")
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
        except Exception:
            pass
        time.sleep(840)


# ── Background scheduler ──────────────────────────────────────────────────────
def _scheduler():
    eod_done = None
    while True:
        try:
            mkt = market_status()
            today = datetime.now(IST).date()
            client = state["client"]

            if mkt == "open" and client:
                _collect_live(client)
                time.sleep(60)
            elif mkt == "post_market" and eod_done != today and client:
                _collect_historical(client)
                eod_done = today
                time.sleep(3600)
            else:
                time.sleep(300)
        except Exception as e:
            logger.error(f"[scheduler] {e}")
            time.sleep(60)


def _collect_live(client):
    try:
        from collect_data import init_db, save_live_snapshot
        from nubra_python_sdk.marketdata.validation import ExchangeEnum

        # Use direct client for option chain
        conn = init_db(DB_PATH)
        for asset, exc_str in [
            ("NIFTY", "NSE"),
            ("BANKNIFTY", "NSE"),
            ("FINNIFTY", "NSE"),
            ("MIDCPNIFTY", "NSE"),
        ]:
            try:
                data = client.option_chain(asset, exc_str)
                # Parse and save
                chain = data.get("chain", {})
                now = int(time.time())
                rows = []
                for opt_type, key in [("CE", "ce"), ("PE", "pe")]:
                    for item in chain.get(key, []):
                        sp = item.get("sp") or item.get("strike_price")
                        ltp = item.get("ltp") or item.get("last_traded_price")
                        rows.append(
                            (
                                now,
                                asset,
                                exc_str,
                                chain.get("expiry", ""),
                                opt_type,
                                item.get("ref_id"),
                                sp / 100 if sp else None,
                                ltp / 100 if ltp else None,
                                item.get("iv"),
                                item.get("delta"),
                                item.get("gamma"),
                                item.get("theta"),
                                item.get("vega"),
                                item.get("oi") or item.get("open_interest"),
                                item.get("volume"),
                            )
                        )
                if rows:
                    conn.executemany(
                        """INSERT INTO option_chain_snapshot
                        (collected_at,asset,exchange,expiry,option_type,inst_id,strike,ltp,iv,delta,gamma,theta,vega,oi,volume)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        rows,
                    )
                    conn.commit()
            except Exception as e:
                logger.warning(f"[live] {asset}: {e}")
        conn.close()
    except Exception as e:
        logger.error(f"[live] {e}")


def _collect_historical(client):
    try:
        from collect_data import init_db, save_index_candles

        conn = init_db(DB_PATH)
        for asset, exc in [
            ("NIFTY", "NSE"),
            ("BANKNIFTY", "NSE"),
            ("FINNIFTY", "NSE"),
            ("MIDCPNIFTY", "NSE"),
            ("SENSEX", "BSE"),
            ("BANKEX", "BSE"),
        ]:
            try:
                closes = client.get_historical_closes(asset, exc, "1d", 90)
                if closes:
                    rows = [
                        (
                            asset,
                            exc,
                            "1d",
                            p.get("ts", p.get("timestamp")),
                            0,
                            0,
                            0,
                            (p.get("v", p.get("value", 0))) / 100,
                            0,
                        )
                        for p in closes
                    ]
                    conn.executemany(
                        """INSERT OR REPLACE INTO historical_candle
                        (symbol,exchange,interval,ts,open,high,low,close,volume)
                        VALUES (?,?,?,?,?,?,?,?,?)""",
                        rows,
                    )
                    conn.commit()
                    logger.info(f"[hist] {asset}: {len(rows)} candles")
            except Exception as e:
                logger.warning(f"[hist] {asset}: {e}")
        conn.close()
    except Exception as e:
        logger.error(f"[hist] {e}")


# ── App lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    from nubra_client import get_client
    import glob, shutil

    # Copy models from repo to /tmp
    for src in glob.glob("trained_models/*.pkl"):
        dst = os.path.join(MODELS_DIR, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            logger.info(f"[startup] Copied {os.path.basename(src)}")

    # Init DB
    from collect_data import init_db

    init_db(DB_PATH)

    # Auth
    client = get_client()
    if client:
        state["client"] = client
        logger.info("[server] Authenticated ✓")

        # Fetch historical data in background
        def _load():
            time.sleep(3)
            count = query_df("SELECT COUNT(*) as n FROM historical_candle").iloc[0]["n"]
            if count < 10:
                logger.info("[startup] Fetching historical data...")
                _collect_historical(client)

        threading.Thread(target=_load, daemon=True).start()
        threading.Thread(target=_scheduler, daemon=True).start()
        threading.Thread(target=_keep_alive, daemon=True).start()
    else:
        logger.warning(
            "[server] No auth — set NUBRA_SESSION_TOKEN + NUBRA_DEVICE_ID on Render"
        )

    logger.info("[server] Ready")
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Options Intelligence API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def root():
    idx = os.path.join(FRONTEND_DIR, "index.html")
    return HTMLResponse(open(idx).read()) if os.path.exists(idx) else {"status": "ok"}


@app.get("/health")
def health():
    import glob

    models = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    return {
        "status": "ok",
        "authenticated": state["client"] is not None,
        "ws_connected": state["ws_connected"],
        "market_status": market_status(),
        "ist_time": datetime.now(IST).strftime("%H:%M:%S IST"),
        "models_loaded": [os.path.basename(m) for m in models],
        "db_path": DB_PATH,
    }


@app.get("/vol-forecast")
def vol_forecast(
    symbol: str = Query("NIFTY"),
    exchange: str = Query("NSE"),
    interval: str = Query("1d"),
):
    # Try DB first
    df = query_df(
        "SELECT ts, close FROM historical_candle WHERE symbol=? AND exchange=? AND interval=? ORDER BY ts",
        (symbol, exchange, interval),
    )

    # Fetch live if DB empty
    if len(df) < 30 and state["client"]:
        try:
            closes = state["client"].get_historical_closes(
                symbol, exchange, interval, 90
            )
            if closes:
                df = pd.DataFrame(
                    [
                        {
                            "ts": p.get("ts", p.get("timestamp")),
                            "close": (p.get("v", p.get("value", 0))) / 100,
                        }
                        for p in closes
                    ]
                )
                # Save to DB
                try:
                    with sqlite3.connect(DB_PATH) as conn:
                        rows = [
                            (
                                symbol,
                                exchange,
                                interval,
                                r["ts"],
                                0,
                                0,
                                0,
                                r["close"],
                                0,
                            )
                            for _, r in df.iterrows()
                        ]
                        conn.executemany(
                            "INSERT OR REPLACE INTO historical_candle (symbol,exchange,interval,ts,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?,?,?)",
                            rows,
                        )
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[vol] live fetch: {e}")

    if len(df) < 30:
        raise HTTPException(404, f"No data for {symbol} ({len(df)} candles)")

    try:
        prices = df["close"]
        # Run only GARCH(1,1) for speed — skip EGARCH/GJR on Render
        g = fit_garch(prices)
        rv20 = realized_vol(prices, 20)
        rv5 = realized_vol(prices, 5)
        try:
            sar = {"forecast": [], "model": "disabled"}  # too slow on Render
        except Exception:
            sar = {"forecast": [], "model": ""}

        # Use GARCH only for ensemble on Render (EGARCH/GJR too slow)
        models = {}
        if "error" not in g:
            models["garch"] = g

        ens_1d = g.get("vol_1d", 1.5) if "error" not in g else 1.5
        ens_ann = g.get("vol_ann", 24.0) if "error" not in g else 24.0

        return {
            "asset": symbol,
            "ensemble_vol_1d": round(ens_1d, 4),
            "ensemble_vol_ann": round(ens_ann, 4),
            "realized_vol_20d": rv20,
            "realized_vol_5d": rv5,
            "current_iv_pct": None,
            "iv_percentile": None,
            "vol_risk_premium": None,
            "pcr": None,
            "sarima_forecast": sar.get("forecast", []),
            "sarima_model": sar.get("model", ""),
            "vol_regime": "fair",
            "models": models,
        }
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.get("/historical")
def historical(
    symbol: str = Query("NIFTY"),
    exchange: str = Query("NSE"),
    itype: str = Query("INDEX"),
    interval: str = Query("1d"),
    start: str = Query("2025-01-01T03:45:00.000Z"),
    end: str = Query(None),
    fields: str = Query("open,high,low,close,tick_volume"),
):
    client = state["client"]
    if not client:
        raise HTTPException(
            503, "SDK not ready — set NUBRA_SESSION_TOKEN + NUBRA_DEVICE_ID"
        )
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
    try:
        data = client.historical_data(
            {
                "exchange": exchange,
                "type": itype,
                "values": [symbol],
                "fields": fields.split(","),
                "startDate": start,
                "endDate": end,
                "interval": interval,
                "intraDay": False,
                "realTime": False,
            }
        )
        # Convert to frontend-compatible format
        result = []
        for item in data.get("result", []):
            values_list = []
            for sym_dict in item.get("values", []):
                sym_entry = {}
                for sym_name, sym_data in sym_dict.items():
                    sym_entry[sym_name] = {}
                    for field in fields.split(","):
                        pts = (
                            sym_data.get(field, [])
                            if isinstance(sym_data, dict)
                            else []
                        )
                        sym_entry[sym_name][field] = [
                            {
                                "timestamp": p.get("ts", p.get("timestamp")),
                                "value": p.get("v", p.get("value")),
                            }
                            for p in pts
                        ]
                values_list.append(sym_entry)
            result.append(
                {
                    "exchange": item.get("exchange"),
                    "type": item.get("type"),
                    "values": values_list,
                }
            )
        return {
            "market_time": data.get("market_time"),
            "message": data.get("message"),
            "result": result,
        }
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.get("/option-chain")
def option_chain(
    instrument: str = Query("NIFTY"),
    exchange: str = Query("NSE"),
    expiry: str = Query(...),
):
    client = state["client"]
    if not client:
        raise HTTPException(503, "Not authenticated")
    try:
        return client.option_chain(instrument, exchange, expiry)
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.get("/signals")
def signals(asset: str = Query("NIFTY")):
    try:
        generate_signals, _ = _signals()
        return generate_signals(asset)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.get("/signals/all")
def signals_all():
    _, generate_all_signals = _signals()
    return generate_all_signals()


@app.get("/strategy-recommend")
@app.get("/strategy/recommend")
def strategy_recommend(
    asset: str = Query("NIFTY"),
    dte: int = Query(7),
    spot_change: float = Query(None),
    iv_level: float = Query(None),
    iv_rank: float = Query(None),
    risk_tolerance: str = Query("medium"),
):
    try:
        df_spot = query_df(
            "SELECT close FROM historical_candle WHERE symbol=? AND exchange=? AND interval='1d' ORDER BY ts DESC LIMIT 25",
            (asset, "BSE" if asset in ("SENSEX", "BANKEX") else "NSE"),
        )
        spot_val = float(df_spot["close"].iloc[0]) if not df_spot.empty else 23300.0
        if spot_change is None:
            spot_change = (
                float(
                    (df_spot["close"].iloc[0] - df_spot["close"].iloc[1])
                    / df_spot["close"].iloc[1]
                    * 100
                )
                if len(df_spot) > 1
                else 0.0
            )
        if iv_level is None:
            iv_level = 25.0
        if iv_rank is None:
            iv_rank = 50.0

        try:
            generate_signals, _ = _signals()
            sigs = generate_signals(asset)
            ce_score = (
                float(sigs.get("CE", [{}])[0].get("signal_score", 0.5))
                if sigs.get("CE")
                else 0.5
            )
            pe_score = (
                float(sigs.get("PE", [{}])[0].get("signal_score", 0.5))
                if sigs.get("PE")
                else 0.5
            )
        except Exception:
            ce_score, pe_score = 0.5, 0.5

        spot_ret5 = (
            float(
                (df_spot["close"].iloc[0] - df_spot["close"].iloc[4])
                / df_spot["close"].iloc[4]
            )
            if len(df_spot) >= 5
            else 0.0
        )
        spot_ret20 = (
            float(
                (df_spot["close"].iloc[0] - df_spot["close"].iloc[-1])
                / df_spot["close"].iloc[-1]
            )
            if len(df_spot) >= 20
            else 0.0
        )

        score_strategies, classify_regime, STRATEGY_CATALOG, liquidity_score = _strategy()
        regime = classify_regime(spot_ret5, spot_ret20, 25.0, iv_rank, 1.0, 1.5)

        df_chain = query_df(
            "SELECT option_type, strike/100.0 as strike, ltp/100.0 as ltp, iv, delta, gamma, theta, vega, oi, volume FROM option_chain_snapshot WHERE asset=? ORDER BY collected_at DESC, strike LIMIT 400",
            (asset,),
        )
        liq = liquidity_score(df_chain) if not df_chain.empty else 0.5

        ranked = score_strategies(
            regime, ce_score, pe_score, iv_level, iv_rank, 1.0, dte, liq
        )

        trend = (
            "Strong Bearish"
            if spot_ret5 < -0.03
            else (
                "Bearish"
                if spot_ret5 < 0
                else "Bullish" if spot_ret5 > 0.03 else "Neutral"
            )
        )
        vol_env = (
            "High Volatility"
            if iv_level > 30
            else "Low Volatility" if iv_level < 15 else "Normal Volatility"
        )

        top = [
            {
                "strategy_key": s["key"],
                "name": s["name"],
                "category": s["category"],
                "description": s["best_when"],
                "best_for": s["best_when"],
                "risk_profile": s["max_loss"],
                "score": s["score"] * 100,
                "trend_fit": s["direction"],
                "rank": i + 1,
            }
            for i, s in enumerate(ranked[:10])
        ]

        return {
            "asset": asset,
            "spot": round(spot_val, 2),
            "iv_level": iv_level,
            "iv_rank": iv_rank,
            "dte": dte,
            "spot_change_pct": round(spot_change, 2),
            "risk_tolerance": risk_tolerance,
            "regime": regime["regime"],
            "regime_description": regime["regime"] + " — " + vol_env,
            "condition_analysis": {
                "trend": trend,
                "volatility_environment": vol_env,
                "iv_status": (
                    "IV Rank High"
                    if iv_rank > 70
                    else "IV Rank Low" if iv_rank < 30 else "IV Rank Neutral"
                ),
                "skew_analysis": "Normal Skew",
                "overall_outlook": trend
                + " market. "
                + ("Sell premium" if iv_rank > 60 else "Buy options")
                + " favoured.",
            },
            "top_recommendations": top,
        }
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.get("/strategy-details/{strategy_key}")
def strategy_details(strategy_key: str, asset: str = Query("NIFTY")):
    _, _, STRATEGY_CATALOG, _ = _strategy()
    s = STRATEGY_CATALOG.get(strategy_key)
    if not s:
        raise HTTPException(404, f"Unknown: {strategy_key}")
    return {
        "strategy_key": strategy_key,
        "name": s.name,
        "category": s.category,
        "description": s.best_when,
        "best_for": s.best_when,
        "risk_profile": s.max_loss,
        "max_profit": s.max_profit,
        "breakeven": s.breakeven,
    }


@app.get("/iv-surface")
def iv_surface(asset: str = Query("NIFTY")):
    df = query_df(
        "SELECT strike, iv, option_type, ltp, expiry FROM option_chain_snapshot WHERE asset=? AND iv IS NOT NULL ORDER BY collected_at DESC, strike",
        (asset,),
    )
    if df.empty:
        return []
    df = df.drop_duplicates(subset=["strike", "option_type"], keep="first")
    return df.replace({float("nan"): None}).to_dict(orient="records")


@app.get("/live/index")
def live_index():
    with _lock:
        return list(state["index_cache"].values())


@app.get("/live/greeks")
def live_greeks():
    with _lock:
        return list(state["greeks_cache"].values())


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with _lock:
                payload = {
                    "indexes": list(state["index_cache"].values()),
                    "greeks": list(state["greeks_cache"].values()),
                }
            await websocket.send_json(payload)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
