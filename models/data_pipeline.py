"""
data_pipeline.py — Model-specific EDA + feature engineering.

Each model family has different requirements:

  GARCH/EGARCH/GJR-GARCH:
    - Input: univariate log-returns of underlying spot price
    - Needs: stationarity (ADF), ARCH effects (Ljung-Box), no scaling
    - Features: just the return series itself

  SARIMA:
    - Input: univariate log-returns (differenced to achieve stationarity)
    - Needs: ADF test, ACF/PACF analysis for order selection
    - Features: just the return series itself

  CatBoost / XGBoost / LightGBM (tree models):
    - Input: tabular cross-sectional features per option per day
    - Needs: no scaling (trees are scale-invariant), handle nulls, no leakage
    - Features: Greeks, momentum, IV rank, moneyness, OI change, spot returns

  LSTM / Transformer (deep sequential models):
    - Input: time-ordered sequences per symbol, shape (N, seq_len, features)
    - Needs: StandardScaler per feature, fixed sequence length, enough history
    - Features: same as tree models but ordered temporally per symbol
"""

import re
import calendar
import sqlite3
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

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

# ── Feature sets per model family ─────────────────────────────────────────────

# Tree models: tabular, scale-invariant, handles nulls
TREE_FEATURES = [
    "close",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "oi",
    "volume",
    "true_moneyness",
    "dte",
    "close_ret_1",
    "close_ret_5",
    "spot_ret_1",
    "spot_ret_5",
    "oi_change",
    "iv_rank",
    "delta_abs",
    "vol_20",
    "opt_type_enc",
]

# Deep models: same features but need scaling + temporal ordering
DEEP_FEATURES = [
    "close",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "oi",
    "volume",
    "true_moneyness",
    "dte",
    "close_ret_1",
    "close_ret_5",
    "spot_ret_1",
    "spot_ret_5",
    "oi_change",
    "iv_rank",
    "delta_abs",
    "vol_20",
    "opt_type_enc",
]

# All features (union)
ALL_FEATURES = list(dict.fromkeys(TREE_FEATURES + DEEP_FEATURES))


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_spot(db_path: str, asset: str) -> pd.Series:
    exchange = EXCHANGE_MAP.get(asset, "NSE")
    conn = sqlite3.connect(db_path)
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
    year = 2000 + int(yy)
    last_day = calendar.monthrange(year, month)[1]
    dt = pd.Timestamp(year=year, month=month, day=last_day)
    while dt.weekday() != 3:  # last Thursday
        dt -= pd.Timedelta(days=1)
    return dt


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GARCH / EGARCH / GJR-GARCH DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


class GarchPipeline:
    """
    EDA + data prep for GARCH-family models.
    Input: spot price series.
    Output: log-return series (stationary, exhibits ARCH effects).
    """

    def __init__(self, db_path: str, asset: str):
        self.db_path = db_path
        self.asset = asset
        self.spot = _load_spot(db_path, asset)
        self.returns = None
        self.eda_results = {}

    def run_eda(self) -> dict:
        """Run all EDA checks required before fitting GARCH."""
        spot = self.spot
        if len(spot) < 30:
            return {"error": "insufficient data"}

        log_ret = np.log(spot / spot.shift(1)).dropna() * 100
        self.returns = log_ret

        # 1. Stationarity of returns (must be stationary for GARCH)
        adf = adfuller(log_ret)
        self.eda_results["adf_stat"] = round(float(adf[0]), 4)
        self.eda_results["adf_pvalue"] = round(float(adf[1]), 6)
        self.eda_results["stationary"] = bool(adf[1] < 0.05)

        # 2. ARCH effects (Ljung-Box on squared returns)
        lb = acorr_ljungbox(log_ret**2, lags=[5, 10], return_df=True)
        self.eda_results["arch_effects"] = bool((lb["lb_pvalue"] < 0.05).any())
        self.eda_results["lb_pvalue_5"] = round(float(lb["lb_pvalue"].iloc[0]), 6)

        # 3. Descriptive stats
        self.eda_results["n"] = len(log_ret)
        self.eda_results["mean"] = round(float(log_ret.mean()), 4)
        self.eda_results["std"] = round(float(log_ret.std()), 4)
        self.eda_results["skew"] = round(float(log_ret.skew()), 4)
        self.eda_results["kurtosis"] = round(float(log_ret.kurtosis()), 4)

        # 4. Best GARCH order by AIC
        from arch import arch_model

        best_aic, best_order = np.inf, (1, 1)
        for p in [1, 2]:
            for q in [1, 2]:
                try:
                    res = arch_model(log_ret, vol="Garch", p=p, q=q).fit(
                        disp="off", show_warning=False
                    )
                    if res.aic < best_aic:
                        best_aic, best_order = res.aic, (p, q)
                except Exception:
                    pass
        self.eda_results["best_garch_order"] = best_order
        self.eda_results["best_garch_aic"] = round(float(best_aic), 2)

        return self.eda_results

    def get_returns(self) -> pd.Series:
        """Return the log-return series ready for GARCH fitting."""
        if self.returns is None:
            self.run_eda()
        return self.returns


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SARIMA DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


class SarimaPipeline:
    """
    EDA + data prep for SARIMA.
    Input: spot price series.
    Output: stationary log-return series + suggested ARIMA order.
    """

    def __init__(self, db_path: str, asset: str):
        self.db_path = db_path
        self.asset = asset
        self.spot = _load_spot(db_path, asset)
        self.returns = None
        self.eda_results = {}

    def run_eda(self) -> dict:
        spot = self.spot
        if len(spot) < 50:
            return {"error": "insufficient data (need 50+)"}

        log_ret = np.log(spot / spot.shift(1)).dropna() * 100
        self.returns = log_ret

        # 1. ADF on log prices (should be non-stationary → needs differencing)
        adf_price = adfuller(np.log(spot))
        self.eda_results["price_stationary"] = bool(adf_price[1] < 0.05)

        # 2. ADF on returns (should be stationary → d=1 confirmed)
        adf_ret = adfuller(log_ret)
        self.eda_results["returns_stationary"] = bool(adf_ret[1] < 0.05)
        self.eda_results["suggested_d"] = 0 if adf_ret[1] < 0.05 else 1

        # 3. ACF/PACF for order hints
        conf = 1.96 / np.sqrt(len(log_ret))
        acf_vals = acf(log_ret, nlags=20, fft=True)
        pacf_vals = pacf(log_ret, nlags=20)
        sig_acf = [i for i in range(1, 21) if abs(acf_vals[i]) > conf]
        sig_pacf = [i for i in range(1, 21) if abs(pacf_vals[i]) > conf]
        self.eda_results["sig_acf_lags"] = sig_acf
        self.eda_results["sig_pacf_lags"] = sig_pacf

        # Suggest p from PACF, q from ACF
        self.eda_results["suggested_p"] = min(sig_pacf[0], 3) if sig_pacf else 1
        self.eda_results["suggested_q"] = min(sig_acf[0], 3) if sig_acf else 1

        self.eda_results["n"] = len(log_ret)
        return self.eda_results

    def get_returns(self) -> pd.Series:
        if self.returns is None:
            self.run_eda()
        return self.returns


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TREE MODEL DATA PIPELINE (CatBoost / XGBoost / LightGBM)
# ═══════════════════════════════════════════════════════════════════════════════


class TreePipeline:
    """
    EDA + feature engineering for tree-based models.
    - No scaling needed (trees are scale-invariant)
    - Handles nulls via fillna
    - Temporal train/test split (no shuffle)
    - Filters deep OTM options
    """

    def __init__(self, db_path: str, asset: str):
        self.db_path = db_path
        self.asset = asset
        self.spot = _load_spot(db_path, asset)
        self.df = None
        self.eda_results = {}

    def run_eda(self, df: pd.DataFrame) -> dict:
        """EDA checks specific to tree models."""
        # 1. Class balance
        self.eda_results["positive_rate"] = round(float(df["target"].mean()), 4)

        # 2. Feature-target correlations
        feat_cols = [f for f in TREE_FEATURES if f in df.columns]
        corr = df[feat_cols + ["target"]].corr()["target"].drop("target")
        self.eda_results["top_corr_features"] = (
            corr.abs().sort_values(ascending=False).head(5).round(4).to_dict()
        )

        # 3. Null rates
        null_rates = df[feat_cols].isnull().mean()
        self.eda_results["null_rates"] = null_rates[null_rates > 0].round(4).to_dict()

        # 4. Outlier counts (>3 std)
        outliers = {}
        for col in ["close", "iv", "delta", "gamma", "theta", "vega"]:
            if col in df.columns:
                s = df[col].dropna()
                z = (s - s.mean()) / (s.std() + 1e-9)
                outliers[col] = int((z.abs() > 3).sum())
        self.eda_results["outliers_3std"] = outliers

        # 5. Moneyness distribution
        if "true_moneyness" in df.columns:
            m = df["true_moneyness"]
            self.eda_results["moneyness"] = {
                "mean": round(float(m.mean()), 3),
                "std": round(float(m.std()), 3),
                "itm": int((m < 0.97).sum()),
                "atm": int(((m >= 0.97) & (m <= 1.03)).sum()),
                "otm": int((m > 1.03).sum()),
            }

        return self.eda_results

    def build(self) -> pd.DataFrame:
        """Full feature engineering pipeline for tree models."""
        exchange = EXCHANGE_MAP.get(self.asset, "NSE")
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            """
            SELECT symbol, asset, ts, close, iv, delta, gamma, theta, vega, oi, volume
            FROM historical_option
            WHERE asset=? AND close IS NOT NULL AND close > 0
            ORDER BY ts, symbol
        """,
            conn,
            params=(self.asset,),
        )
        conn.close()

        if df.empty:
            return df

        df["ts"] = pd.to_datetime(df["ts"], unit="ns")
        df["opt_type"] = df["symbol"].str[-2:]
        df["opt_type_enc"] = (df["opt_type"] == "CE").astype(int)
        df["strike"] = df["symbol"].str.extract(r"(\d{4,6})(?:CE|PE)$").astype(float)

        # Join spot
        spot = self.spot
        df = df.set_index("ts").join(spot.rename("spot"), how="left").reset_index()
        df["spot"] = df["spot"].ffill().bfill()

        # True moneyness
        df["true_moneyness"] = df["strike"] / df["spot"].replace(0, np.nan)

        # Filter near-ATM only (0.85–1.15)
        df = df[(df["true_moneyness"] >= 0.85) & (df["true_moneyness"] <= 1.15)].copy()
        if df.empty:
            return df

        # Days to expiry
        df["expiry_dt"] = df["symbol"].apply(_parse_expiry)
        df["dte"] = (df["expiry_dt"] - df["ts"]).dt.days.clip(lower=0).fillna(30)

        # Sort for temporal features
        df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
        g = df.groupby("symbol")

        # Per-symbol momentum
        df["close_ret_1"] = g["close"].pct_change(1).fillna(0)
        df["close_ret_5"] = g["close"].pct_change(5).fillna(0)
        df["oi_change"] = g["oi"].diff().fillna(0)
        df["vol_20"] = (
            g["close"]
            .transform(
                lambda x: x.pct_change().rolling(20, min_periods=5).std() * np.sqrt(252)
            )
            .fillna(0)
        )
        df["iv_rank"] = (
            g["iv"]
            .transform(lambda x: x.rolling(20, min_periods=3).rank(pct=True))
            .fillna(0.5)
        )

        # Spot returns (underlying momentum)
        spot_r1 = spot.pct_change(1).rename("spot_ret_1")
        spot_r5 = spot.pct_change(5).rename("spot_ret_5")
        df = df.set_index("ts").join(spot_r1).join(spot_r5).reset_index()
        df["spot_ret_1"] = df["spot_ret_1"].fillna(0)
        df["spot_ret_5"] = df["spot_ret_5"].fillna(0)

        # Greeks derived
        df["delta_abs"] = df["delta"].abs()

        # Clip outliers at 3-sigma per feature (tree models still benefit)
        for col in ["close", "oi", "volume", "oi_change"]:
            if col in df.columns:
                mu, sd = df[col].mean(), df[col].std()
                df[col] = df[col].clip(mu - 3 * sd, mu + 3 * sd)

        # Fill nulls
        for col in ["iv", "delta", "gamma", "theta", "vega", "oi", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Target: next-day close up?
        df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
        df["close_next"] = df.groupby("symbol")["close"].shift(-1)
        df["target"] = (df["close_next"] > df["close"]).astype(float)
        df = df.dropna(subset=["close_next"])

        self.df = df
        self.run_eda(df)
        return df

    def get_Xy(self) -> tuple:
        """Return (X_train, y_train, X_test, y_test, feature_names) with temporal split."""
        if self.df is None:
            self.build()
        df = self.df
        avail = [f for f in TREE_FEATURES if f in df.columns]

        # Temporal split: sort by ts
        df_s = df.sort_values("ts").reset_index(drop=True)
        X = df_s[avail].values.astype(np.float32)
        y = df_s["target"].values.astype(np.float32)

        split = int(len(X) * 0.8)
        return X[:split], y[:split], X[split:], y[split:], avail


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DEEP MODEL DATA PIPELINE (LSTM / Transformer)
# ═══════════════════════════════════════════════════════════════════════════════


class DeepPipeline:
    """
    EDA + feature engineering for LSTM and Transformer.
    - Requires StandardScaler (deep models are NOT scale-invariant)
    - Sequences built per-symbol (temporal order within each option)
    - Only symbols with >= seq_len observations are used
    - Train/test split is temporal (no future data in training)
    """

    def __init__(self, db_path: str, asset: str, seq_len: int = 10):
        self.db_path = db_path
        self.asset = asset
        self.seq_len = seq_len
        self.scaler = StandardScaler()
        self.spot = _load_spot(db_path, asset)
        self.eda_results = {}

    def run_eda(self, df: pd.DataFrame) -> dict:
        """EDA checks specific to sequential deep models."""
        seq_counts = df.groupby("symbol")["ts"].count()
        self.eda_results["total_symbols"] = int(seq_counts.shape[0])
        self.eda_results["symbols_ge_seq_len"] = int((seq_counts >= self.seq_len).sum())
        self.eda_results["symbols_ge_20"] = int((seq_counts >= 20).sum())
        self.eda_results["mean_seq_len"] = round(float(seq_counts.mean()), 1)
        self.eda_results["min_seq_len"] = int(seq_counts.min())
        self.eda_results["max_seq_len"] = int(seq_counts.max())

        # Check for temporal gaps within symbols
        gaps = []
        for sym, grp in df.groupby("symbol"):
            ts_sorted = grp["ts"].sort_values()
            diffs = ts_sorted.diff().dropna().dt.days
            if (diffs > 5).any():  # gap > 5 trading days
                gaps.append(sym)
        self.eda_results["symbols_with_gaps"] = len(gaps)

        return self.eda_results

    def build(self) -> pd.DataFrame:
        """Build feature-engineered DataFrame for deep models."""
        # Reuse TreePipeline for feature engineering (same features)
        tree = TreePipeline(self.db_path, self.asset)
        df = tree.build()
        if df.empty:
            return df

        self.run_eda(df)
        return df

    def make_sequences(self, df: pd.DataFrame) -> tuple:
        """
        Build (X_train, y_train, X_test, y_test) as 3D arrays.
        Sequences are built per-symbol, split temporally.
        Scaler is fit on training data only.
        """
        avail = [f for f in DEEP_FEATURES if f in df.columns]
        cutoff_date = df["ts"].quantile(0.8)  # 80th percentile date

        train_seqs_X, train_seqs_y = [], []
        test_seqs_X, test_seqs_y = [], []

        for sym, grp in df.groupby("symbol"):
            grp = grp.sort_values("ts").reset_index(drop=True)
            if len(grp) < self.seq_len + 1:
                continue

            X_sym = grp[avail].values.astype(np.float32)
            y_sym = grp["target"].values.astype(np.float32)

            for i in range(self.seq_len, len(grp)):
                seq_X = X_sym[i - self.seq_len : i]
                seq_y = y_sym[i]
                ts_i = grp["ts"].iloc[i]

                if ts_i <= cutoff_date:
                    train_seqs_X.append(seq_X)
                    train_seqs_y.append(seq_y)
                else:
                    test_seqs_X.append(seq_X)
                    test_seqs_y.append(seq_y)

        if not train_seqs_X:
            return None, None, None, None

        X_tr = np.array(train_seqs_X, dtype=np.float32)
        y_tr = np.array(train_seqs_y, dtype=np.float32)

        # Fit scaler on training data only (flatten, scale, reshape)
        n_tr, sl, nf = X_tr.shape
        X_tr_flat = X_tr.reshape(-1, nf)
        # Replace any NaN/inf before fitting scaler
        X_tr_flat = np.nan_to_num(X_tr_flat, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler.fit(X_tr_flat)
        X_tr = np.nan_to_num(
            self.scaler.transform(X_tr_flat), nan=0.0, posinf=0.0, neginf=0.0
        ).reshape(n_tr, sl, nf)

        if test_seqs_X:
            X_te = np.array(test_seqs_X, dtype=np.float32)
            y_te = np.array(test_seqs_y, dtype=np.float32)
            n_te = X_te.shape[0]
            X_te = np.nan_to_num(
                self.scaler.transform(X_te.reshape(-1, nf)),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).reshape(n_te, sl, nf)
        else:
            X_te = np.empty((0, sl, nf), dtype=np.float32)
            y_te = np.empty(0, dtype=np.float32)

        return X_tr, y_tr, X_te, y_te

    def transform_new(self, X: np.ndarray) -> np.ndarray:
        """Scale new data using the fitted scaler."""
        n, sl, nf = X.shape
        return self.scaler.transform(X.reshape(-1, nf)).reshape(n, sl, nf)
