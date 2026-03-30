"""
render_server.py — Production entry point for Render deployment.

Differences from vol_server.py:
  - No interactive OTP (uses TOTP or cached token)
  - SQLite DB loaded from persistent disk (/data/data.db on Render)
  - Background scheduler: collects data + retrains daily
  - CORS configured for Netlify frontend URL
"""

import os
import asyncio
import threading
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime

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
# Always use /tmp on Render (free tier has no persistent disk)
DB_PATH    = os.getenv("DB_PATH",    "/tmp/data.db"        if IS_RENDER else "data.db")
MODELS_DIR = os.getenv("MODELS_DIR", "/tmp/trained_models" if IS_RENDER else "trained_models")
NETLIFY_URL = os.getenv("NETLIFY_URL", "*")
IST = pytz.timezone("Asia/Kolkata")

os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["DB_PATH"] = DB_PATH
os.environ["MODELS_DIR"] = MODELS_DIR

# ── Import app modules ────────────────────────────────────────────────────────
import sqlite3
from models.vol_models import vol_ensemble, fit_sarima, iv_surface
from models.strategy_engine import (
    score_strategies,
    classify_regime,
    STRATEGY_CATALOG,
    liquidity_score,
)
from realtime_vol import forecast_asset, forecast_all
from realtime_model import generate_signals, generate_all_signals

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "nubra": None,
    "market_data": None,
    "socket": None,
    "greeks_cache": {},
    "index_cache": {},
    "ws_connected": False,
}
_lock = threading.Lock()


def query_df(sql: str, params=()):
    with sqlite3.connect(DB_PATH) as c:
        return pd.read_sql_query(sql, c, params=params)


# ── Market hours ──────────────────────────────────────────────────────────────
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


# ── Background scheduler ──────────────────────────────────────────────────────
def _background_scheduler():
    """
    Runs in a background thread on Render.
    - Market open: collect live snapshots every 60s
    - Post-market: EOD collection + model retrain
    - Closed: sleep
    """
    import importlib

    logger.info("[scheduler] Started")
    eod_done_date = None

    while True:
        try:
            mkt = market_status()
            today = datetime.now(IST).date()

            if mkt == "open" and state["market_data"]:
                # Live snapshot
                from collect_data import init_db, collect_live

                conn = init_db(DB_PATH)
                collect_live(state["market_data"], conn)
                conn.close()
                time.sleep(60)

            elif (
                mkt == "post_market" and eod_done_date != today and state["market_data"]
            ):
                logger.info("[scheduler] Post-market: EOD collection")
                from collect_data import init_db, collect_historical, collect_live

                conn = init_db(DB_PATH)
                collect_historical(state["market_data"], conn)
                collect_live(state["market_data"], conn)
                conn.close()
                logger.info("[scheduler] Retraining models...")
                try:
                    import subprocess, sys

                    subprocess.run(
                        [
                            sys.executable,
                            "build_model.py",
                            "NIFTY",
                            "BANKNIFTY",
                            "FINNIFTY",
                            "MIDCPNIFTY",
                        ],
                        check=True,
                        timeout=600,
                    )
                    # Reload models
                    import realtime_model

                    realtime_model._model_cache.clear()
                    importlib.reload(realtime_model)
                    logger.info("[scheduler] Models retrained and reloaded")
                except Exception as e:
                    logger.error(f"[scheduler] Retrain failed: {e}")
                eod_done_date = today
                time.sleep(3600)

            else:
                time.sleep(300)  # 5 min

        except Exception as e:
            logger.error(f"[scheduler] Error: {e}")
            time.sleep(60)


# ── WS callbacks ──────────────────────────────────────────────────────────────
def _on_index(msg):
    items = msg if isinstance(msg, list) else [msg]
    with _lock:
        for item in items:
            name = item.get("indexname") or item.get("index_name")
            if name:
                state["index_cache"][name] = item


def _on_greeks(msg):
    items = msg if isinstance(msg, list) else [msg]
    with _lock:
        for item in items:
            iid = item.get("inst_id")
            if iid:
                state["greeks_cache"][iid] = item


def _on_connect(msg):
    state["ws_connected"] = True
    logger.info(f"[ws] Connected: {msg}")


def _on_close(reason):
    state["ws_connected"] = False
    logger.warning(f"[ws] Closed: {reason}")


def _on_error(err):
    logger.error(f"[ws] Error: {err}")


# ── App lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        from nubra_python_sdk.start_sdk import InitNubraSdk, NubraEnv
        from nubra_python_sdk.marketdata.market_data import MarketData
        from nubra_python_sdk.ticker.websocketdata import NubraDataSocket
        from auto_auth import get_authenticated_client

        logger.info("[server] Authenticating...")
        nubra = get_authenticated_client()
        state["nubra"] = nubra
        state["market_data"] = MarketData(nubra)

        # WebSocket
        socket = NubraDataSocket(
            client=nubra,
            on_index_data=_on_index,
            on_greeks_data=_on_greeks,
            on_connect=_on_connect,
            on_close=_on_close,
            on_error=_on_error,
            reconnect=True,
            persist_subscriptions=True,
        )
        state["socket"] = socket

        def _run_ws():
            socket.connect()
            time.sleep(2)
            socket.subscribe(
                ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"],
                data_type="index",
                exchange="NSE",
            )
            socket.keep_running()

        threading.Thread(target=_run_ws, daemon=True).start()

        # Background scheduler (data collection + daily retrain)
        threading.Thread(target=_background_scheduler, daemon=True).start()

        # On free tier: fetch historical data on startup if DB is empty
        def _initial_load():
            time.sleep(5)  # let server fully start first
            try:
                from collect_data import init_db, collect_historical

                conn = init_db(DB_PATH)
                count = conn.execute(
                    "SELECT COUNT(*) FROM historical_candle"
                ).fetchone()[0]
                if count < 10:
                    logger.info("[startup] DB empty — fetching historical data...")
                    collect_historical(state["market_data"], conn)
                    logger.info("[startup] Historical data loaded")
                else:
                    logger.info(
                        f"[startup] DB has {count} candles — skipping initial load"
                    )
                conn.close()
            except Exception as e:
                logger.error(f"[startup] Initial load error: {e}")

        threading.Thread(target=_initial_load, daemon=True).start()

        logger.info("[server] Ready")
    except Exception as e:
        logger.error(f"[server] Startup error: {e}")
    yield
    if state.get("socket"):
        try:
            state["socket"].disconnect()
        except Exception:
            pass


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Options Intelligence API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[NETLIFY_URL, "http://localhost:3000", "http://localhost:8000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ── Copy all endpoints from vol_server.py ────────────────────────────────────
# Import and re-export all routes
import importlib, sys

sys.path.insert(0, os.path.dirname(__file__))

# Re-implement key endpoints inline (avoids circular import)


@app.get("/")
async def root():
    idx = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(idx):
        return HTMLResponse(open(idx).read())
    return {"message": "Options Intelligence API", "docs": "/docs"}


@app.get("/health")
def health():
    now = datetime.now(IST)
    t = now.strftime("%H:%M")
    mkt = market_status()
    return {
        "status": "ok",
        "ws_connected": state["ws_connected"],
        "authenticated": state["nubra"] is not None,
        "market_status": mkt,
        "ist_time": now.strftime("%H:%M:%S IST"),
        "note": (
            "Market open — live streaming"
            if mkt == "open"
            else "Post-market/closed — using last data"
        ),
    }


@app.get("/vol-forecast")
def vol_forecast(
    symbol: str = Query("NIFTY"),
    exchange: str = Query("NSE"),
    interval: str = Query("1d"),
):
    df = query_df(
        "SELECT ts, close FROM historical_candle WHERE symbol=? AND exchange=? AND interval=? ORDER BY ts",
        (symbol, exchange, interval),
    )
    if len(df) < 30:
        raise HTTPException(404, "Not enough data")
    try:
        result = forecast_asset(symbol)
        return result
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.get("/vol-forecast-all")
def vol_forecast_all():
    return forecast_all()


@app.get("/strategy-recommend")
def strategy_recommend(
    asset: str = Query("NIFTY"),
    dte: int = Query(7),
    spot_change: float = Query(None),
    iv_level: float = Query(None),
    iv_rank: float = Query(None),
    risk_tolerance: str = Query("medium"),
):
    try:
        from vol_server import strategy_recommend_endpoint

        return strategy_recommend_endpoint(
            asset=asset,
            dte=dte,
            spot_change=spot_change,
            iv_level=iv_level,
            iv_rank=iv_rank,
            risk_tolerance=risk_tolerance,
        )
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.get("/strategy/recommend")
def strategy_recommend_v2(asset: str = Query("NIFTY"), dte: int = Query(7)):
    try:
        from vol_server import strategy_recommend

        return strategy_recommend(asset=asset, dte=dte)
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.get("/strategy-details/{strategy_key}")
def strategy_details(strategy_key: str, asset: str = Query("NIFTY")):
    try:
        from vol_server import strategy_details_endpoint

        return strategy_details_endpoint(strategy_key=strategy_key, asset=asset)
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.get("/signals")
def signals(asset: str = Query("NIFTY")):
    try:
        return generate_signals(asset)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@app.get("/signals/all")
def signals_all():
    return generate_all_signals()


@app.get("/option-chain")
def option_chain(
    instrument: str = Query("NIFTY"),
    exchange: str = Query("NSE"),
    expiry: str = Query(...),
):
    md = state["market_data"]
    if not md:
        raise HTTPException(503, "SDK not ready")
    from nubra_python_sdk.marketdata.validation import ExchangeEnum

    exc = ExchangeEnum.NSE if exchange.upper() == "NSE" else ExchangeEnum.BSE
    return md.option_chain(
        instrument=instrument, expiry=expiry, exchange=exc
    ).model_dump()


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
    md = state["market_data"]
    if not md:
        raise HTTPException(503, "SDK not ready")
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return md.historical_data(
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
    ).model_dump()


@app.get("/iv-surface")
def iv_surface_endpoint(asset: str = Query("NIFTY")):
    df = query_df(
        "SELECT strike, iv, option_type, ltp, expiry FROM option_chain_snapshot "
        "WHERE asset=? AND iv IS NOT NULL ORDER BY collected_at DESC, strike",
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
