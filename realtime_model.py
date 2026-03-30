"""
realtime_model.py — Real-time signal generation using the trained ensemble.
Feature engineering mirrors build_model.py exactly to avoid train/serve skew.
"""

import os
import re
import sqlite3
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from models.ensemble import OptionEnsemble

load_dotenv()
def _db(): return os.getenv("DB_PATH", "data.db")
def _models_dir(): return os.getenv("MODELS_DIR", "trained_models")

ASSETS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX"]
EXCHANGE_MAP = {"SENSEX": "BSE", "BANKEX": "BSE"}

MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

_model_cache: dict[str, OptionEnsemble] = {}


def load_model(asset: str) -> OptionEnsemble:
    if asset not in _model_cache:
        path = os.path.join(_models_dir(), f"{asset.lower()}_ensemble.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No model for {asset}. Run: python build_model.py {asset}"
            )
        _model_cache[asset] = OptionEnsemble.load(path)
    return _model_cache[asset]


def _load_spot(asset: str) -> pd.Series:
    exchange = EXCHANGE_MAP.get(asset, "NSE")
    conn = sqlite3.connect(_db())
    df = pd.read_sql_query(
        "SELECT ts, close FROM historical_candle "
        "WHERE symbol=? AND exchange=? AND interval='1d' ORDER BY ts",
        conn,
        params=(asset, exchange),
    )
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"], unit="ns")
    return df.set_index("ts")["close"]


def _parse_expiry(symbol: str) -> pd.Timestamp | None:
    m = re.search(r"(\d{2})([A-Z]{3})(\d{4,5})(CE|PE)$", symbol)
    if not m:
        return None
    yy, mon, _, _ = m.groups()
    month = MONTH_MAP.get(mon)
    if not month:
        return None
    import calendar

    year = 2000 + int(yy)
    last_day = calendar.monthrange(year, month)[1]
    dt = pd.Timestamp(year=year, month=month, day=last_day)
    while dt.weekday() != 3:
        dt -= pd.Timedelta(days=1)
    return dt


def load_latest_chain(asset: str, option_type: str = "CE") -> pd.DataFrame:
    conn = sqlite3.connect(_db())
    df = pd.read_sql_query(
        """
        SELECT * FROM option_chain_snapshot
        WHERE asset=? AND option_type=?
        ORDER BY collected_at DESC
        LIMIT 500
    """,
        conn,
        params=(asset, option_type),
    )
    conn.close()
    return df


def build_features(df: pd.DataFrame, asset: str, opt_type_enc: int) -> tuple:
    """
    Build features matching build_model.py exactly.
    Returns (X array, feature_names list).
    """
    df = df.copy()

    # Coerce all numeric columns
    for col in [
        "ltp",
        "strike",
        "iv",
        "delta",
        "gamma",
        "theta",
        "vega",
        "oi",
        "volume",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["close"] = df["ltp"].fillna(0)
    df["ts"] = pd.to_datetime(df["collected_at"], unit="s")
    df = df.sort_values("ts").reset_index(drop=True)

    # ── Spot price ────────────────────────────────────────────────────────────
    spot = _load_spot(asset)
    latest_spot = float(spot.iloc[-1]) if not spot.empty else 1.0
    spot_ret1 = float(spot.pct_change(1).iloc[-1]) if len(spot) > 1 else 0.0
    spot_ret5 = float(spot.pct_change(5).iloc[-1]) if len(spot) > 5 else 0.0

    df["spot"] = latest_spot
    df["spot_ret_1"] = spot_ret1
    df["spot_ret_5"] = spot_ret5

    # ── True moneyness ────────────────────────────────────────────────────────
    df["true_moneyness"] = df["strike"] / df["spot"].replace(0, np.nan)

    # ── Days to expiry ────────────────────────────────────────────────────────
    now = pd.Timestamp.now()
    df["expiry_dt"] = df["strike"].apply(lambda _: None)  # placeholder
    # Use expiry from snapshot if available
    if "expiry" in df.columns:

        def parse_dte(exp_str):
            try:
                exp = pd.to_datetime(str(exp_str), format="%Y%m%d")
                return max(0, (exp - now).days)
            except Exception:
                return 30

        df["dte"] = df["expiry"].apply(parse_dte)
    else:
        df["dte"] = 30

    # ── Per-symbol momentum (use collected_at ordering) ───────────────────────
    # Group by strike as proxy for symbol in live data
    g = df.groupby("strike")
    df["close_ret_1"] = g["close"].pct_change(1).fillna(0)
    df["close_ret_5"] = g["close"].pct_change(5).fillna(0)
    df["oi_change"] = g["oi"].diff().fillna(0)
    df["vol_20"] = (
        g["close"]
        .transform(
            lambda x: x.pct_change().rolling(20, min_periods=3).std() * np.sqrt(252)
        )
        .fillna(0)
    )
    df["iv_rank"] = (
        g["iv"]
        .transform(lambda x: x.rolling(20, min_periods=3).rank(pct=True))
        .fillna(0.5)
    )

    # ── Greeks derived ────────────────────────────────────────────────────────
    df["delta_abs"] = df["delta"].abs()
    df["opt_type_enc"] = opt_type_enc

    from models.ensemble import FEATURES as MODEL_FEATURES

    avail = [f for f in MODEL_FEATURES if f in df.columns]
    X = df[avail].fillna(0).values.astype(np.float32)
    return X, avail, df


def generate_signals(asset: str = "NIFTY") -> dict:
    model = load_model(asset)
    signals = {}

    for opt_type, enc in [("CE", 1), ("PE", 0)]:
        df = load_latest_chain(asset, opt_type)
        if df.empty:
            signals[opt_type] = []
            continue

        try:
            X, avail, df_feat = build_features(df, asset, enc)
        except Exception as e:  # noqa: BLE001
            signals[opt_type] = []
            continue

        if len(X) < 2:
            signals[opt_type] = []
            continue

        try:
            probas = model.predict_proba(X)[:, 1]
        except Exception:  # noqa: BLE001
            probas = np.full(len(X), 0.5)

        df_feat = df_feat.copy()
        df_feat["signal_score"] = probas

        # Return latest snapshot per strike with highest score
        latest = df_feat.sort_values("ts").groupby("strike").last().reset_index()
        latest["signal_score"] = df_feat.groupby("strike")["signal_score"].last().values

        cols = [
            "strike",
            "ltp",
            "iv",
            "delta",
            "gamma",
            "theta",
            "vega",
            "oi",
            "volume",
            "true_moneyness",
            "dte",
            "signal_score",
        ]
        top = latest.nlargest(10, "signal_score")[
            [c for c in cols if c in latest.columns]
        ]
        signals[opt_type] = top.round(4).to_dict(orient="records")

    return signals


def generate_all_signals() -> dict:
    results = {}
    for asset in ASSETS:
        try:
            results[asset] = generate_signals(asset)
        except FileNotFoundError:
            results[asset] = {"error": f"No model for {asset}"}
        except Exception as e:  # noqa: BLE001
            results[asset] = {"error": str(e)}
    return results
