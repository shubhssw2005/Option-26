"""
vol_server.py — FastAPI backend using the official Nubra Python SDK.
Serves option chain, historical data, Greeks, vol forecasts, and signals.
WebSocket collector uses NubraDataSocket (handles auth tokens internally).
"""

import os
import sqlite3
import asyncio
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from nubra_python_sdk.start_sdk import InitNubraSdk, NubraEnv
from nubra_python_sdk.marketdata.market_data import MarketData
from nubra_python_sdk.ticker.websocketdata import NubraDataSocket
from nubra_python_sdk.ticker.validation import IntervalEnum
from nubra_python_sdk.trading.trading_enum import ExchangeEnum

from models.vol_models import vol_ensemble, fit_sarima, iv_surface
from models.strategy_engine import (
    score_strategies,
    calculate_strikes,
    liquidity_score,
    classify_regime,
    STRATEGY_CATALOG,
)
from realtime_vol import forecast_asset, forecast_all
from realtime_model import generate_signals, generate_all_signals

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "data.db")
USE_UAT = os.getenv("NUBRA_ENV", "uat").lower() != "production"

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "nubra": None,  # InitNubraSdk instance
    "market_data": None,  # MarketData instance
    "socket": None,  # NubraDataSocket instance
    "greeks_cache": {},  # inst_id -> tick dict
    "index_cache": {},  # indexname -> tick dict
    "option_cache": {},  # asset -> option chain dict
    "ws_connected": False,
}
_lock = threading.Lock()


# ── DB helpers ────────────────────────────────────────────────────────────────


def query_df(sql: str, params=()) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn, params=params)


# ── SDK WebSocket callbacks ───────────────────────────────────────────────────


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


def _on_option(msg):
    with _lock:
        asset = msg.get("asset")
        if asset:
            state["option_cache"][asset] = msg


def _on_connect(msg):
    state["ws_connected"] = True
    print(f"[ws] Connected: {msg}")


def _on_close(reason):
    state["ws_connected"] = False
    print(f"[ws] Closed: {reason}")


def _on_error(err):
    print(f"[ws] Error: {err}")


# ── WebSocket collector (background thread) ───────────────────────────────────


def start_ws_collector(nubra: InitNubraSdk):
    """Start NubraDataSocket in a background thread."""
    socket = NubraDataSocket(
        client=nubra,
        on_index_data=_on_index,
        on_greeks_data=_on_greeks,
        on_option_data=_on_option,
        on_connect=_on_connect,
        on_close=_on_close,
        on_error=_on_error,
        reconnect=True,
        persist_subscriptions=True,
    )
    state["socket"] = socket

    def _run():
        socket.connect()
        # Give WS time to establish before subscribing
        time.sleep(2)
        # Subscribe NIFTY + BANKNIFTY index stream
        # Pass exchange as string "NSE", not ExchangeEnum (SDK formats it directly)
        socket.subscribe(
            ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"],
            data_type="index",
            exchange="NSE",
        )
        socket.keep_running()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return socket


# ── App lifecycle ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI):
    env = NubraEnv.UAT if USE_UAT else NubraEnv.PROD
    print(f"[server] Initialising SDK ({env.value})...")
    nubra = InitNubraSdk(env, env_creds=True)
    state["nubra"] = nubra
    state["market_data"] = MarketData(nubra)
    start_ws_collector(nubra)
    print("[server] Ready")
    yield
    if state["socket"]:
        try:
            state["socket"].disconnect()
        except Exception:  # noqa: BLE001
            pass


app = FastAPI(title="Nubra Options Dashboard", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST endpoints ────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    import pytz
    from datetime import datetime

    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    t = now.strftime("%H:%M")
    wd = now.weekday()
    if wd >= 5:
        mkt = "closed"
    elif "09:15" <= t <= "15:30":
        mkt = "open"
    elif "15:30" < t <= "18:00":
        mkt = "post_market"
    else:
        mkt = "closed"

    return {
        "status": "ok",
        "ws_connected": state["ws_connected"],
        "authenticated": state["nubra"] is not None,
        "market_status": mkt,
        "ist_time": now.strftime("%H:%M:%S IST"),
        "note": (
            "Live data available 09:15–15:30 IST. Post-market uses EOD snapshot."
            if mkt != "open"
            else "Market open — live data streaming"
        ),
    }


@app.get("/option-chain")
def option_chain(
    instrument: str = Query("NIFTY"),
    exchange: str = Query("NSE"),
    expiry: str = Query(...),
):
    md: MarketData = state["market_data"]
    if not md:
        raise HTTPException(status_code=503, detail="SDK not ready")
    exc = ExchangeEnum.NSE if exchange.upper() == "NSE" else ExchangeEnum.BSE
    result = md.option_chain(instrument=instrument, expiry=expiry, exchange=exc)
    # SDK returns OptionChainWrapper (Pydantic model) — serialise to dict
    return result.model_dump()


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
    md: MarketData = state["market_data"]
    if not md:
        raise HTTPException(status_code=503, detail="SDK not ready")
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
    request = {
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
    return md.historical_data(request).model_dump()


@app.get("/subscribe/option")
def subscribe_option(
    asset: str = Query("NIFTY"),
    expiry: str = Query(...),
    exchange: str = Query("NSE"),
):
    """Subscribe to live option chain stream for a given asset+expiry."""
    socket: NubraDataSocket = state["socket"]
    if not socket:
        raise HTTPException(status_code=503, detail="WebSocket not ready")
    exc = "NSE" if exchange.upper() == "NSE" else "BSE"
    symbol = f"{asset}:{expiry}"
    socket.subscribe([symbol], data_type="option", exchange=exc)
    return {"subscribed": symbol}


@app.get("/subscribe/greeks")
def subscribe_greeks(symbols: str = Query(...)):
    """Subscribe to Greeks stream. symbols = comma-separated instrument names."""
    socket: NubraDataSocket = state["socket"]
    if not socket:
        raise HTTPException(status_code=503, detail="WebSocket not ready")
    syms = [s.strip() for s in symbols.split(",")]
    socket.subscribe(syms, data_type="greeks", exchange=ExchangeEnum.NSE)
    return {"subscribed": syms}


@app.get("/vol-forecast")
def vol_forecast(
    symbol: str = Query("NIFTY"),
    exchange: str = Query("NSE"),
    interval: str = Query("1d"),
):
    df = query_df(
        """SELECT ts, close FROM historical_candle
           WHERE symbol=? AND exchange=? AND interval=? ORDER BY ts""",
        (symbol, exchange, interval),
    )
    if len(df) < 30:
        raise HTTPException(status_code=404, detail="Not enough historical data in DB")
    prices = df["close"]
    result = vol_ensemble(prices)
    result["sarima"] = fit_sarima(prices)
    return result


@app.get("/signals")
def signals(asset: str = Query("NIFTY")):
    try:
        return generate_signals(asset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/signals/all")
def signals_all():
    return generate_all_signals()


@app.get("/live/greeks")
def live_greeks():
    with _lock:
        return list(state["greeks_cache"].values())


@app.get("/live/index")
def live_index():
    """Get live index data from database - polls every 5 seconds on frontend"""
    assets = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
    result = []
    for asset in assets:
        df = query_df(
            """SELECT symbol, MAX(ts) as latest_ts, close, open, high, low, volume
               FROM historical_candle WHERE symbol=? GROUP BY symbol""",
            (asset,),
        )
        if not df.empty:
            row = df.iloc[0]
            prev_df = query_df(
                """SELECT close FROM historical_candle 
                   WHERE symbol=? AND ts < ? ORDER BY ts DESC LIMIT 1""",
                (asset, int(row["latest_ts"])),
            )
            prev_close = (
                float(prev_df.iloc[0]["close"])
                if not prev_df.empty
                else float(row["close"])
            )
            change = float(row["close"]) - prev_close
            change_pct = (change / prev_close * 100) if prev_close > 0 else 0

            result.append(
                {
                    "indexname": row["symbol"],
                    "index_value": float(row["close"]) * 100,
                    "close": float(row["close"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "volume": int(row["volume"]) if pd.notna(row["volume"]) else 0,
                    "change": change,
                    "changeprecent": change_pct,
                    "timestamp": int(row["latest_ts"]),
                    "source": "nubra_api",
                }
            )
    return result


@app.get("/live/option")
def live_option(asset: str = Query("NIFTY")):
    with _lock:
        data = state["option_cache"].get(asset)
    if not data:
        raise HTTPException(status_code=404, detail="No live option data yet")
    return data


@app.get("/iv-surface")
def iv_surface_endpoint(asset: str = Query("NIFTY"), expiry: str = Query(None)):
    """Get option chain data with IV for pricing engine"""
    if expiry:
        df = query_df(
            """SELECT strike, iv, option_type, ltp, expiry FROM option_chain_snapshot
               WHERE asset=? AND expiry=? AND iv IS NOT NULL
               ORDER BY strike""",
            (asset, expiry),
        )
    else:
        df = query_df(
            """SELECT strike, iv, option_type, ltp, expiry FROM option_chain_snapshot
               WHERE asset=? AND iv IS NOT NULL
               ORDER BY collected_at DESC, strike""",
            (asset,),
        )
    if df.empty:
        return []
    df = df.drop_duplicates(subset=["strike", "option_type"], keep="first")
    df = df.replace({pd.NA: None, float("nan"): None})
    return df.to_dict(orient="records")


@app.get("/iv-3d-surface")
def iv_3d_surface_endpoint(asset: str = Query("NIFTY")):
    """Get IV data for 3D surface with strike and expiry dimensions"""
    df = query_df(
        """SELECT strike, expiry, iv, option_type FROM option_chain_snapshot
           WHERE asset=? AND iv IS NOT NULL
           ORDER BY expiry, strike""",
        (asset,),
    )
    if df.empty:
        return {"error": "No data for this asset"}

    from datetime import datetime

    expiries = sorted(df["expiry"].unique())
    strikes = sorted(df["strike"].unique())

    surface_data = []
    for exp in expiries:
        exp_df = df[df["expiry"] == exp]

        dte = 0
        try:
            exp_date = datetime.strptime(exp, "%Y%m%d")
            dte = max(1, (exp_date - datetime.now()).days)
        except:
            dte = 30

        for strike in strikes:
            strike_row = exp_df[exp_df["strike"] == strike]
            ce_iv = strike_row[strike_row["option_type"] == "CE"]["iv"].values
            pe_iv = strike_row[strike_row["option_type"] == "PE"]["iv"].values

            surface_data.append(
                {
                    "expiry": exp,
                    "strike": float(strike) / 100,
                    "dte": dte,
                    "ce_iv": (
                        float(ce_iv[0]) * 100
                        if len(ce_iv) > 0 and not pd.isna(ce_iv[0])
                        else None
                    ),
                    "pe_iv": (
                        float(pe_iv[0]) * 100
                        if len(pe_iv) > 0 and not pd.isna(pe_iv[0])
                        else None
                    ),
                }
            )

    return {
        "asset": asset,
        "expiries": expiries,
        "dtes": [
            max(1, (datetime.strptime(e, "%Y%m%d") - datetime.now()).days)
            for e in expiries
        ],
        "strikes": [float(s) / 100 for s in strikes],
        "data": surface_data,
    }


# ── Black-Scholes & Monte Carlo Options Pricing ───────────────────────────────

from scipy.stats import norm
import numpy as np


@app.get("/bs-pricing")
def bs_pricing_endpoint(
    asset: str = Query("NIFTY"),
    strike: float = Query(None),
    expiry: str = Query(None),
    option_type: str = Query("CE"),
    custom_iv: float = Query(None),
    custom_tte: float = Query(None),
    custom_spot: float = Query(None),
    rate: float = Query(0.065),
):
    """Black-Scholes options pricing with real Nubra data"""
    spot = custom_spot
    if spot is None:
        df = query_df(
            "SELECT close FROM historical_candle WHERE symbol=? ORDER BY ts DESC LIMIT 1",
            (asset,),
        )
        spot = float(df.iloc[0]["close"]) if not df.empty else 23300.0

    T = custom_tte
    if T is None and expiry:
        try:
            exp_date = datetime.strptime(expiry, "%Y%m%d")
            T = max(1 / 365, (exp_date - datetime.now()).days / 365)
        except:
            T = 7 / 365
    elif T is None:
        T = 7 / 365

    iv = custom_iv
    if iv is None and strike and expiry:
        df = query_df(
            """SELECT iv FROM option_chain_snapshot 
               WHERE asset=? AND strike=? AND expiry=? AND option_type=? AND iv IS NOT NULL
               ORDER BY collected_at DESC LIMIT 1""",
            (asset, strike * 100, expiry, option_type),
        )
        iv = float(df.iloc[0]["iv"]) if not df.empty else 0.20
    elif iv is None:
        iv = 0.20

    K = strike if strike else spot
    r = rate
    sigma = iv / 100

    d1 = (np.log(spot / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.upper() == "CE":
        price = spot * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (spot * sigma * np.sqrt(T))
        theta = (
            -(spot * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        ) / 365
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (spot * sigma * np.sqrt(T))
        theta = (
            -(spot * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        ) / 365

    vega = spot * norm.pdf(d1) * np.sqrt(T) / 100
    rho = (
        K
        * T
        * np.exp(-r * T)
        * (norm.cdf(d2) if option_type.upper() == "CE" else -norm.cdf(-d2))
        / 100
    )

    return {
        "asset": asset,
        "spot": round(spot, 2),
        "strike": K,
        "expiry": expiry,
        "dte": round(T * 365, 1),
        "tte": round(T, 6),
        "iv": round(iv, 2),
        "rate": round(r * 100, 2),
        "option_type": option_type.upper(),
        "d1": round(d1, 4),
        "d2": round(d2, 4),
        "price": round(price, 2),
        "greeks": {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta": round(theta, 4),
            "vega": round(vega, 4),
            "rho": round(rho, 4),
        },
    }


@app.get("/mc-pricing")
def mc_pricing_endpoint(
    asset: str = Query("NIFTY"),
    strike: float = Query(None),
    expiry: str = Query(None),
    option_type: str = Query("CE"),
    custom_iv: float = Query(None),
    custom_tte: float = Query(None),
    custom_spot: float = Query(None),
    rate: float = Query(0.065),
    simulations: int = Query(100000),
    steps: int = Query(252),
):
    """Monte Carlo option pricing simulation with GBM paths"""
    np.random.seed(42)

    spot = custom_spot
    if spot is None:
        df = query_df(
            "SELECT close FROM historical_candle WHERE symbol=? ORDER BY ts DESC LIMIT 1",
            (asset,),
        )
        spot = float(df.iloc[0]["close"]) if not df.empty else 23300.0

    T = custom_tte
    if T is None and expiry:
        try:
            exp_date = datetime.strptime(expiry, "%Y%m%d")
            T = max(1 / 365, (exp_date - datetime.now()).days / 365)
        except:
            T = 7 / 365
    elif T is None:
        T = 7 / 365

    iv = custom_iv
    if iv is None and strike and expiry:
        df = query_df(
            """SELECT iv FROM option_chain_snapshot 
               WHERE asset=? AND strike=? AND expiry=? AND option_type=? AND iv IS NOT NULL
               ORDER BY collected_at DESC LIMIT 1""",
            (asset, strike * 100, expiry, option_type),
        )
        iv = float(df.iloc[0]["iv"]) if not df.empty else 0.20
    elif iv is None:
        iv = 0.20

    K = strike if strike else spot
    r = rate
    sigma = iv / 100
    dt = T / steps

    S = np.full(simulations, spot)
    paths = [S.copy()]

    for _ in range(steps):
        Z = np.random.standard_normal(simulations)
        S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        if len(paths) <= 10 or _ % (steps // 10) == 0:
            paths.append(S.copy())

    if option_type.upper() == "CE":
        payoff = np.maximum(S - K, 0)
    else:
        payoff = np.maximum(K - S, 0)

    mc_price = np.exp(-r * T) * np.mean(payoff)
    mc_std = np.exp(-r * T) * np.std(payoff) / np.sqrt(simulations)
    mc_std_full = np.std(payoff) / np.sqrt(simulations)

    percentile_5 = np.percentile(S, 5)
    percentile_50 = np.percentile(S, 50)
    percentile_95 = np.percentile(S, 95)

    sample_paths = paths[:: max(1, len(paths) // 20)][:20]

    return {
        "asset": asset,
        "spot": round(spot, 2),
        "strike": K,
        "expiry": expiry,
        "dte": round(T * 365, 1),
        "iv": round(iv, 2),
        "rate": round(r * 100, 2),
        "option_type": option_type.upper(),
        "simulations": simulations,
        "steps": steps,
        "mc_price": round(mc_price, 2),
        "mc_std": round(mc_std, 4),
        "mc_std_full": round(mc_std_full, 2),
        "ci_95_lower": round(mc_price - 1.96 * mc_std, 2),
        "ci_95_upper": round(mc_price + 1.96 * mc_std, 2),
        "terminal_prices": {
            "p5": round(percentile_5, 2),
            "p50": round(percentile_50, 2),
            "p95": round(percentile_95, 2),
        },
        "sample_paths": [[round(x, 2) for x in p] for p in sample_paths],
    }


@app.get("/mc-paths-chart")
def mc_paths_chart_endpoint(
    asset: str = Query("NIFTY"),
    strike: float = Query(None),
    expiry: str = Query(None),
    option_type: str = Query("CE"),
    custom_iv: float = Query(None),
    custom_tte: float = Query(None),
    custom_spot: float = Query(None),
    rate: float = Query(0.065),
    paths_to_show: int = Query(100),
    simulations: int = Query(100000),
    steps: int = Query(252),
):
    """Get Monte Carlo paths for visualization"""
    np.random.seed(42)

    spot = custom_spot
    if spot is None:
        df = query_df(
            "SELECT close FROM historical_candle WHERE symbol=? ORDER BY ts DESC LIMIT 1",
            (asset,),
        )
        spot = float(df.iloc[0]["close"]) if not df.empty else 23300.0

    T = custom_tte
    if T is None and expiry:
        try:
            exp_date = datetime.strptime(expiry, "%Y%m%d")
            T = max(1 / 365, (exp_date - datetime.now()).days / 365)
        except:
            T = 7 / 365
    elif T is None:
        T = 7 / 365

    iv = custom_iv
    if iv is None:
        iv = 0.20

    K = strike if strike else spot
    r = rate
    sigma = iv / 100
    dt = T / steps

    all_paths = []
    time_points = np.linspace(0, T, steps + 1)

    for _ in range(paths_to_show):
        Z = np.random.standard_normal(steps)
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        log_returns = np.cumsum(drift + diffusion)
        path = spot * np.exp(log_returns)
        path = np.insert(path, 0, spot)
        all_paths.append(path.tolist())

    if option_type.upper() == "CE":
        terminal_payoffs = np.maximum(np.array([p[-1] for p in all_paths]) - K, 0)
    else:
        terminal_payoffs = np.maximum(K - np.array([p[-1] for p in all_paths]), 0)

    mean_path = np.mean(all_paths, axis=0).tolist()
    p5_path = np.percentile(all_paths, 5, axis=0).tolist()
    p95_path = np.percentile(all_paths, 95, axis=0).tolist()

    return {
        "time_points": [round(t, 4) for t in time_points],
        "paths": all_paths,
        "mean_path": mean_path,
        "p5_path": p5_path,
        "p95_path": p95_path,
        "spot": spot,
        "strike": K,
        "option_type": option_type.upper(),
        "terminal_payoffs": terminal_payoffs.tolist(),
    }


# ── Strategy Recommendation Engine ─────────────────────────────────────────────

import sys

sys.path.insert(0, os.path.dirname(__file__))
from models.strategy_recommender import (
    StrategyRecommender,
    StrategyLibrary,
    MarketRegime,
)
from models.deep_strategy_recommender import DeepStrategyRecommender

_strategy_lib = StrategyLibrary()
_deep_recommender = DeepStrategyRecommender()
_rule_based_recommender = StrategyRecommender()


@app.get("/strategy-recommend")
def strategy_recommend_endpoint(
    asset: str = Query("NIFTY"),
    spot_change: float = Query(None),
    iv_level: float = Query(None),
    iv_rank: float = Query(None),
    skew: float = Query(0.0),
    dte: int = Query(7),
    realized_vol: float = Query(None),
    risk_tolerance: str = Query("medium"),
    top_n: int = Query(10),
):
    """ML-powered strategy recommendation based on market conditions"""

    # Get current spot and calculate spot_change from live data
    spot_df = query_df(
        "SELECT close FROM historical_candle WHERE symbol=? ORDER BY ts DESC LIMIT 20",
        (asset,),
    )
    spot_val = float(spot_df.iloc[0]["close"]) if not spot_df.empty else 23300.0

    if spot_change is None:
        if len(spot_df) > 1:
            prev_close = float(spot_df.iloc[1]["close"])
            spot_change = ((spot_val - prev_close) / prev_close) * 100
        else:
            spot_change = 0.0

    # Get ATM IV from option chain data
    if iv_level is None:
        atm_strike = spot_val * 100
        df = query_df(
            """SELECT iv FROM option_chain_snapshot 
               WHERE asset=? AND iv IS NOT NULL 
               ORDER BY ABS(strike - ?) ASC LIMIT 1""",
            (asset, atm_strike),
        )
        iv_level = float(df.iloc[0]["iv"] * 100) if not df.empty else 25.0

    # Calculate IV Rank from historical volatility
    if iv_rank is None:
        if realized_vol is None:
            realized_vol = iv_level * 0.8
        if iv_level > 0 and realized_vol > 0:
            iv_rank = min(100, max(0, (iv_level / realized_vol) * 50))
        else:
            iv_rank = 50.0

    deep_recs = _deep_recommender.recommend(
        spot_change=spot_change,
        iv_level=iv_level,
        iv_rank=iv_rank,
        skew=skew,
        dte=dte,
        realized_vol=realized_vol,
        risk_tolerance=risk_tolerance,
    )

    condition_analysis = _deep_recommender.get_condition_analysis(
        spot_change=spot_change, iv_level=iv_level, iv_rank=iv_rank, skew=skew
    )

    spot = query_df(
        "SELECT close FROM historical_candle WHERE symbol=? ORDER BY ts DESC LIMIT 1",
        (asset,),
    )
    spot_val = float(spot.iloc[0]["close"]) if not spot.empty else 23300.0

    iv_df = query_df(
        """SELECT strike, iv FROM option_chain_snapshot 
           WHERE asset=? AND iv IS NOT NULL AND expiry=(SELECT expiry FROM option_chain_snapshot WHERE asset=? AND iv IS NOT NULL ORDER BY collected_at DESC LIMIT 1)
           ORDER BY strike""",
        (asset, asset),
    )

    strikes = iv_df["strike"].tolist() if not iv_df.empty else [spot_val * 100]
    ivs = (iv_df["iv"] * 100).tolist() if not iv_df.empty else [iv_level]

    trend = "neutral"
    if spot_change > 1:
        trend = "bullish"
    elif spot_change < -1:
        trend = "bearish"

    rule_rec = _rule_based_recommender.recommend(
        asset=asset,
        spot=spot_val,
        strikes=strikes,
        ivs=ivs,
        atm_iv=iv_level,
        iv_hist=realized_vol,
        trend=trend,
        iv_rank=iv_rank,
        dte=dte,
        risk_tolerance=risk_tolerance,
    )

    merged_recs = {}
    for rec in deep_recs[: top_n * 2]:
        key = rec["strategy_key"]
        strat = _strategy_lib.get_strategy(key)

        if strat:
            if "bull" in trend:
                trend_fit = (
                    "bullish"
                    if (
                        strat.bullish or strat.category in ["Vertical Spread", "Income"]
                    )
                    else "neutral"
                )
            elif "bear" in trend:
                trend_fit = (
                    "bearish"
                    if (
                        strat.bearish
                        or strat.category in ["Vertical Spread", "Hedging"]
                    )
                    else "neutral"
                )
            else:
                trend_fit = "neutral"
        else:
            trend_fit = "neutral"

        if key not in merged_recs:
            merged_recs[key] = {
                "strategy_key": key,
                "name": strat.name if strat else key.replace("_", " ").title(),
                "category": strat.category if strat else "Unknown",
                "description": strat.description if strat else "",
                "best_for": strat.best_for if strat else "",
                "risk_profile": strat.risk_profile if strat else "",
                "score": rec["score"],
                "deep_score": rec["score"],
                "rule_score": 0,
                "trend_fit": trend_fit,
            }

    for rec in rule_rec.get("all_strategies", [])[: top_n * 2]:
        key = rec["strategy_key"]
        if key in merged_recs:
            merged_recs[key]["rule_score"] = rec["score"]
            merged_recs[key]["score"] = (
                merged_recs[key]["deep_score"] + rec["score"]
            ) / 2

    final_recs = sorted(merged_recs.values(), key=lambda x: x["score"], reverse=True)[
        :top_n
    ]

    for i, rec in enumerate(final_recs):
        rec["rank"] = i + 1
        rec["score"] = round(rec["score"], 1)
        rec["deep_score"] = round(rec["deep_score"], 1)
        rec["rule_score"] = round(rec["rule_score"], 1)

    return {
        "asset": asset,
        "spot": round(spot_val, 2),
        "iv_level": round(iv_level, 2),
        "iv_rank": round(iv_rank, 1),
        "skew": round(skew, 2),
        "dte": dte,
        "spot_change_pct": round(spot_change, 2),
        "risk_tolerance": risk_tolerance,
        "condition_analysis": condition_analysis,
        "regime": rule_rec.get("regime", "neutral"),
        "regime_description": rule_rec.get("regime_description", ""),
        "top_recommendations": final_recs[:5],
        "all_strategies": final_recs,
    }


@app.get("/strategy-details/{strategy_key}")
def strategy_details_endpoint(strategy_key: str):
    """Get detailed information about a specific strategy"""
    strat = _strategy_lib.get_strategy(strategy_key)
    if not strat:
        return {"error": "Strategy not found"}

    return {
        "strategy_key": strategy_key,
        "name": strat.name,
        "category": strat.category,
        "description": strat.description,
        "best_for": strat.best_for,
        "risk_profile": strat.risk_profile,
        "characteristics": {
            "bullish": strat.bullish,
            "bearish": strat.bearish,
            "neutral": strat.neutral,
            "volatility_trade": strat.volatility_trade,
        },
    }


@app.get("/market-regime")
def market_regime_endpoint(asset: str = Query("NIFTY")):
    """Analyze current market regime using real data"""

    spot_df = query_df(
        "SELECT close FROM historical_candle WHERE symbol=? ORDER BY ts DESC LIMIT 100",
        (asset,),
    )

    if spot_df.empty:
        return {"error": "No data available"}

    closes = spot_df["close"].values
    spot = closes[0]

    returns = np.diff(np.log(closes)) if len(closes) > 1 else np.array([0])
    realized_vol = (
        float(np.std(returns) * np.sqrt(252) * 100) if len(returns) > 1 else 20.0
    )

    spot_20d_ago = closes[min(19, len(closes) - 1)]
    spot_change = (
        ((spot - spot_20d_ago) / spot_20d_ago) * 100 if spot_20d_ago > 0 else 0
    )

    iv_df = query_df(
        """SELECT strike, iv FROM option_chain_snapshot 
           WHERE asset=? AND iv IS NOT NULL ORDER BY collected_at DESC LIMIT 200""",
        (asset,),
    )

    atm_iv = 25.0
    skew = 0.0

    if not iv_df.empty:
        atm_iv = float(iv_df.iloc[len(iv_df) // 2]["iv"] * 100)

        otm_ivs = iv_df[iv_df["strike"] > spot * 100]["iv"] * 100
        itm_ivs = iv_df[iv_df["strike"] < spot * 100]["iv"] * 100
        if len(otm_ivs) > 0 and len(itm_ivs) > 0:
            skew = float(np.mean(otm_ivs) - np.mean(itm_ivs))

    if spot_change > 1:
        trend = "strong_bullish" if spot_change > 2 else "mild_bullish"
    elif spot_change < -1:
        trend = "strong_bearish" if spot_change < -2 else "mild_bearish"
    else:
        trend = "neutral"

    vol_regime = "high" if atm_iv > 30 else "low"

    if "bull" in trend and vol_regime == "low":
        regime = "BULL_LOW_VOL"
        regime_desc = "Bullish trend with low volatility - ideal for debit spreads and covered calls"
    elif "bull" in trend and vol_regime == "high":
        regime = "BULL_HIGH_VOL"
        regime_desc = "Bullish trend with high volatility - consider hedging strategies"
    elif "bear" in trend and vol_regime == "low":
        regime = "BEAR_LOW_VOL"
        regime_desc = "Bearish trend with low volatility - credit spreads preferred"
    elif "bear" in trend and vol_regime == "high":
        regime = "BEAR_HIGH_VOL"
        regime_desc = "Bearish trend with high volatility - protective strategies"
    elif trend == "neutral" and vol_regime == "low":
        regime = "NEUTRAL_LOW_VOL"
        regime_desc = (
            "Range-bound market with low volatility - wait or calendar spreads"
        )
    else:
        regime = "NEUTRAL_HIGH_VOL"
        regime_desc = (
            "Range-bound with elevated volatility - premium selling opportunities"
        )

    iv_rank = 50.0
    if realized_vol > 0:
        iv_rank = min(100, max(0, (atm_iv / realized_vol) * 50))

    if skew > 3:
        regime = "SKEW_BEARISH"
        regime_desc = (
            "Elevated put skew indicates bearish sentiment - protective puts favored"
        )
    elif skew < -3:
        regime = "SKEW_BULLISH"
        regime_desc = (
            "Elevated call skew indicates bullish sentiment - call spreads favored"
        )

    return {
        "asset": asset,
        "spot": round(spot, 2),
        "spot_change_20d": round(spot_change, 2),
        "realized_vol": round(realized_vol, 2),
        "atm_iv": round(atm_iv, 2),
        "iv_rank": round(iv_rank, 1),
        "iv_skew": round(skew, 2),
        "trend": trend,
        "vol_regime": vol_regime,
        "regime": regime,
        "regime_description": regime_desc,
        "recommendations": {
            "if_bullish": [
                "Bull Call Spread",
                "Bull Put Spread",
                "Covered Call",
                "Diagonal Call",
            ],
            "if_bearish": [
                "Bear Put Spread",
                "Bear Call Spread",
                "Protective Put",
                "Put Ratio Backspread",
            ],
            "if_neutral": [
                "Short Straddle",
                "Iron Condor",
                "Short Strangle",
                "Calendar Spread",
            ],
            "if_high_iv": [
                "Short Straddle",
                "Short Strangle",
                "Iron Condor",
                "Bear Call Spread",
            ],
            "if_low_iv": [
                "Long Straddle",
                "Long Strangle",
                "Protective Put",
                "Call Ratio Backspread",
            ],
        },
    }


# ── Frontend WebSocket (push live ticks to browser) ───────────────────────────


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


# ── Serve Frontend ─────────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

# Serve all static files (js, css, images) from frontend/
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def root():
    with open(os.path.join(FRONTEND_DIR, "index.html"), "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import Response

    return Response(content=b"", media_type="image/x-icon")


# ── Vol Forecast (new clean endpoint) ────────────────────────────────────────


@app.get("/vol-forecast-v2")
def vol_forecast_v2(asset: str = Query("NIFTY")):
    """Full vol forecast: GARCH ensemble + SARIMA + IV percentile + PCR."""
    try:
        return forecast_asset(asset)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/vol-forecast-all")
def vol_forecast_all_endpoint():
    """Vol forecast for all assets."""
    try:
        return forecast_all()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e


# ── Strategy Engine (new clean endpoints) ────────────────────────────────────


@app.get("/strategy/recommend")
def strategy_recommend(
    asset: str = Query("NIFTY"),
    dte: int = Query(7),
):
    """
    ML-driven strategy recommendation using:
    - GARCH vol forecast vs IV percentile
    - Signal scores from ensemble model
    - PCR, regime classification, liquidity
    """
    import sqlite3 as _sq

    # 1. Vol forecast
    try:
        vf = forecast_asset(asset)
    except Exception:
        vf = {}

    # 2. Signal scores
    try:
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

    # 3. Spot returns for regime
    exchange = "BSE" if asset in ("SENSEX", "BANKEX") else "NSE"
    df_spot = query_df(
        "SELECT close FROM historical_candle WHERE symbol=? AND exchange=? AND interval='1d' ORDER BY ts DESC LIMIT 25",
        (asset, exchange),
    )
    spot_ret_5 = 0.0
    spot_ret_20 = 0.0
    if len(df_spot) >= 5:
        spot_ret_5 = float(
            (df_spot["close"].iloc[0] - df_spot["close"].iloc[4])
            / df_spot["close"].iloc[4]
        )
    if len(df_spot) >= 20:
        spot_ret_20 = float(
            (df_spot["close"].iloc[0] - df_spot["close"].iloc[19])
            / df_spot["close"].iloc[19]
        )

    # 4. Live chain for liquidity
    df_chain = query_df(
        """SELECT option_type, strike/100.0 as strike, ltp/100.0 as ltp,
                  iv, delta, gamma, theta, vega, oi, volume
           FROM option_chain_snapshot WHERE asset=?
           ORDER BY collected_at DESC, strike LIMIT 400""",
        (asset,),
    )
    liq = liquidity_score(df_chain) if not df_chain.empty else 0.5

    # 5. Regime
    regime = classify_regime(
        spot_ret_5=spot_ret_5,
        spot_ret_20=spot_ret_20,
        realized_vol=vf.get("realized_vol_20d") or 20.0,
        iv_percentile=vf.get("iv_percentile") or 50.0,
        pcr=vf.get("pcr") or 1.0,
        garch_vol_1d=vf.get("garch_vol_1d") or 1.5,
    )

    # 6. Score strategies
    ranked = score_strategies(
        regime=regime,
        signal_score_ce=ce_score,
        signal_score_pe=pe_score,
        atm_iv=vf.get("current_iv_pct") or 20.0,
        iv_percentile=vf.get("iv_percentile") or 50.0,
        pcr=vf.get("pcr") or 1.0,
        dte=dte,
        liquidity_score=liq,
    )

    return {
        "asset": asset,
        "regime": regime,
        "vol_forecast": vf,
        "signal_ce": round(ce_score, 4),
        "signal_pe": round(pe_score, 4),
        "liquidity": liq,
        "strategies": ranked[:10],
    }


@app.get("/strategy/strikes")
def strategy_strikes(
    asset: str = Query("NIFTY"),
    strategy: str = Query("long_straddle"),
    lot_size: int = Query(75),
):
    """Calculate exact strikes, premiums, P&L for a given strategy."""
    df_chain = query_df(
        """SELECT option_type, strike/100.0 as strike, ltp/100.0 as ltp,
                  iv, delta, gamma, theta, vega, oi, volume
           FROM option_chain_snapshot WHERE asset=?
           ORDER BY collected_at DESC, strike LIMIT 400""",
        (asset,),
    )
    if df_chain.empty:
        raise HTTPException(status_code=404, detail="No chain data")

    df_chain = df_chain.drop_duplicates(subset=["option_type", "strike"], keep="first")

    df_spot = query_df(
        "SELECT close FROM historical_candle WHERE symbol=? ORDER BY ts DESC LIMIT 1",
        (asset,),
    )
    spot = float(df_spot["close"].iloc[0]) if not df_spot.empty else 23300.0

    result = calculate_strikes(strategy, spot, df_chain, lot_size)
    return result


@app.get("/strategy/list")
def strategy_list():
    """List all available strategies with metadata."""
    return [
        {
            "key": k,
            "name": v.name,
            "category": v.category,
            "direction": v.direction,
            "vol_view": v.vol_view,
            "risk_score": v.risk_score,
            "best_when": v.best_when,
            "margin_required": v.margin_required,
        }
        for k, v in STRATEGY_CATALOG.items()
    ]
