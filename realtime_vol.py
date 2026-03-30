"""
realtime_vol.py — Real-time volatility forecasting for all assets.
Uses GARCH ensemble + SARIMA + realized vol + IV percentile.
"""

import os
import sqlite3
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from models.vol_models import (
    fit_garch,
    fit_egarch,
    fit_gjr_garch,
    fit_sarima,
    realized_vol,
    vol_ensemble,
)

load_dotenv()
DB_PATH = os.getenv("DB_PATH", "data.db")
EXCHANGE_MAP = {"SENSEX": "BSE", "BANKEX": "BSE"}
ASSETS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX"]


def load_spot(asset: str) -> pd.Series:
    exchange = EXCHANGE_MAP.get(asset, "NSE")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ts, close FROM historical_candle "
        "WHERE symbol=? AND exchange=? AND interval='1d' ORDER BY ts",
        conn,
        params=(asset, exchange),
    )
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"], unit="ns")
    return df.set_index("ts")["close"]


def iv_percentile(asset: str, window: int = 252) -> float:
    """Rolling IV percentile from historical_option data."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ts, AVG(iv) as avg_iv FROM historical_option "
        "WHERE asset=? AND iv IS NOT NULL AND iv > 0 "
        "GROUP BY ts ORDER BY ts",
        conn,
        params=(asset,),
    )
    conn.close()
    if len(df) < 5:
        return 50.0
    df["ts"] = pd.to_datetime(df["ts"], unit="ns")
    current_iv = df["avg_iv"].iloc[-1]
    hist_iv = df["avg_iv"].tail(window)
    pct = float((hist_iv <= current_iv).mean() * 100)
    return round(pct, 1)


def pcr(asset: str) -> float:
    """Put-Call Ratio from latest option chain snapshot."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT option_type, SUM(oi) as total_oi
        FROM option_chain_snapshot
        WHERE asset=? AND collected_at = (
            SELECT MAX(collected_at) FROM option_chain_snapshot WHERE asset=?
        )
        GROUP BY option_type
    """,
        conn,
        params=(asset, asset),
    )
    conn.close()
    if df.empty:
        return 1.0
    oi_map = df.set_index("option_type")["total_oi"].to_dict()
    ce_oi = oi_map.get("CE", 1)
    pe_oi = oi_map.get("PE", 1)
    return round(pe_oi / max(ce_oi, 1), 3)


def forecast_asset(asset: str) -> dict:
    """Full vol forecast for one asset."""
    spot = load_spot(asset)
    if len(spot) < 30:
        return {"asset": asset, "error": "insufficient data"}

    ens = vol_ensemble(spot)
    sarima = fit_sarima(spot)
    rv20 = realized_vol(spot, window=20)
    rv5 = realized_vol(spot, window=5)
    ivp = iv_percentile(asset)
    p = pcr(asset)

    # Vol risk premium = IV - Realized vol
    current_iv = None
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT AVG(iv)*100 FROM historical_option WHERE asset=? AND iv>0 "
        "ORDER BY ts DESC LIMIT 50",
        (asset,),
    ).fetchone()
    conn.close()
    if row and row[0]:
        current_iv = round(float(row[0]), 2)

    vrp = round(current_iv - rv20, 2) if current_iv and rv20 else None

    return {
        "asset": asset,
        "garch_vol_1d": ens.get("ensemble_vol_1d"),
        "garch_vol_ann": ens.get("ensemble_vol_ann"),
        "realized_vol_20d": rv20,
        "realized_vol_5d": rv5,
        "current_iv_pct": current_iv,
        "iv_percentile": ivp,
        "vol_risk_premium": vrp,
        "pcr": p,
        "sarima_forecast": sarima.get("forecast", []),
        "sarima_model": sarima.get("model", ""),
        "models": ens.get("models", {}),
        "vol_regime": ("cheap" if ivp < 30 else "expensive" if ivp > 70 else "fair"),
    }


def forecast_all() -> dict:
    """Forecast vol for all assets."""
    return {asset: forecast_asset(asset) for asset in ASSETS}


def run_loop(interval_sec: int = 60):
    print(f"[vol] Starting vol forecast loop every {interval_sec}s")
    while True:
        for asset in ASSETS:
            try:
                r = forecast_asset(asset)
                print(
                    f"[vol] {asset}: "
                    f"GARCH_1d={r.get('garch_vol_1d')}%  "
                    f"RV20={r.get('realized_vol_20d')}%  "
                    f"IV={r.get('current_iv_pct')}%  "
                    f"IVP={r.get('iv_percentile')}%  "
                    f"VRP={r.get('vol_risk_premium')}  "
                    f"PCR={r.get('pcr')}  "
                    f"regime={r.get('vol_regime')}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"[vol] {asset} error: {e}")
        time.sleep(interval_sec)


if __name__ == "__main__":
    import sys

    secs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    run_loop(secs)
