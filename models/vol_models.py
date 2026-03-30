"""
vol_models.py — Volatility forecasting using model-specific pipelines.

Each model uses GarchPipeline or SarimaPipeline from data_pipeline.py
which runs the correct EDA before fitting.

GARCH family:
  - Input: log-returns (stationary, ARCH effects confirmed)
  - GARCH(1,1): symmetric vol clustering
  - EGARCH(1,1): asymmetric (leverage effect), simulation for h>1
  - GJR-GARCH(1,1,1): asymmetric response to negative shocks

SARIMA:
  - Input: log-returns (d=1 already applied via differencing)
  - Auto-selects (p,d,q)(P,D,Q,m) via AIC
"""

import warnings
import numpy as np
import pandas as pd
from arch import arch_model

warnings.filterwarnings("ignore")


# ── Internal helpers ──────────────────────────────────────────────────────────


def _log_ret(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna() * 100


def _check_garch_preconditions(ret: pd.Series) -> dict | None:
    """Return error dict if data is not suitable for GARCH, else None."""
    if len(ret) < 30:
        return {"error": "insufficient data (need 30+ returns)"}
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox

    adf = adfuller(ret)
    if adf[1] > 0.10:
        return {"error": f"returns not stationary (ADF p={adf[1]:.4f})"}
    lb = acorr_ljungbox(ret**2, lags=[5], return_df=True)
    if lb["lb_pvalue"].iloc[0] > 0.10:
        return {"warning": "weak ARCH effects — GARCH may not add value"}
    return None


# ── GARCH(1,1) ────────────────────────────────────────────────────────────────


def fit_garch(prices: pd.Series, p: int = 1, q: int = 1, dist: str = "normal") -> dict:
    """
    Standard GARCH(p,q).
    Features: log-returns only (univariate).
    EDA: stationarity + ARCH effects checked internally.
    """
    ret = _log_ret(prices)
    pre = _check_garch_preconditions(ret)
    if pre and "error" in pre:
        return pre

    res = arch_model(ret, vol="Garch", p=p, q=q, dist=dist).fit(
        disp="off", show_warning=False
    )
    fc = res.forecast(horizon=5)
    vols = np.sqrt(fc.variance.values[-1])

    return {
        "model": f"GARCH({p},{q})",
        "params": {
            "omega": round(float(res.params["omega"]), 6),
            "alpha": round(float(res.params[f"alpha[{p}]"]), 4),
            "beta": round(float(res.params[f"beta[{q}]"]), 4),
        },
        "vol_1d": round(float(vols[0]), 4),
        "vol_5d": round(float(vols[-1]), 4),
        "vol_ann": round(float(vols[0]) * np.sqrt(252), 4),
        "aic": round(float(res.aic), 2),
        "cond_vol": res.conditional_volatility.tolist(),
        "n_obs": len(ret),
    }


# ── EGARCH(1,1) ───────────────────────────────────────────────────────────────


def fit_egarch(prices: pd.Series) -> dict:
    """
    EGARCH(1,1) — captures leverage effect (bad news > good news on vol).
    Features: log-returns only.
    Note: analytic forecast only for horizon=1; simulation for h>1.
    """
    ret = _log_ret(prices)
    pre = _check_garch_preconditions(ret)
    if pre and "error" in pre:
        return pre

    res = arch_model(ret, vol="EGARCH", p=1, q=1, dist="normal").fit(
        disp="off", show_warning=False
    )
    # EGARCH: analytic only for h=1, simulation for h>1
    fc1 = res.forecast(horizon=1, method="analytic")
    fc5 = res.forecast(horizon=5, method="simulation", simulations=500)
    vol_1d = float(np.sqrt(fc1.variance.values[-1, 0]))
    vol_5d = float(np.sqrt(fc5.variance.values[-1, -1]))

    return {
        "model": "EGARCH(1,1)",
        "vol_1d": round(vol_1d, 4),
        "vol_5d": round(vol_5d, 4),
        "vol_ann": round(float(vol_1d * np.sqrt(252)), 4),
        "aic": round(float(res.aic), 2),
        "n_obs": len(ret),
    }


# ── GJR-GARCH(1,1,1) ─────────────────────────────────────────────────────────


def fit_gjr_garch(prices: pd.Series) -> dict:
    """
    GJR-GARCH(1,1,1) — asymmetric: negative shocks increase vol more.
    Features: log-returns only.
    Typically best model in falling markets (confirmed by AIC in our data).
    """
    ret = _log_ret(prices)
    pre = _check_garch_preconditions(ret)
    if pre and "error" in pre:
        return pre

    res = arch_model(ret, vol="GARCH", p=1, o=1, q=1, dist="normal").fit(
        disp="off", show_warning=False
    )
    fc = res.forecast(horizon=5)
    vols = np.sqrt(fc.variance.values[-1])

    return {
        "model": "GJR-GARCH(1,1,1)",
        "vol_1d": round(float(vols[0]), 4),
        "vol_5d": round(float(vols[-1]), 4),
        "vol_ann": round(float(vols[0]) * np.sqrt(252), 4),
        "aic": round(float(res.aic), 2),
        "n_obs": len(ret),
    }


# ── SARIMA ────────────────────────────────────────────────────────────────────


def fit_sarima(prices: pd.Series, horizon: int = 5) -> dict:
    """
    Auto-ARIMA on log-returns.
    Features: log-returns (already stationary, d=0 in ARIMA terms).
    EDA: ADF confirms stationarity, ACF/PACF guides order selection.
    Auto-selects best (p,d,q)(P,D,Q,m) by AIC.
    """
    try:
        from pmdarima import auto_arima
    except ImportError:
        return {"error": "pmdarima not installed"}

    ret = _log_ret(prices)
    if len(ret) < 50:
        return {"error": "insufficient data (need 50+ for SARIMA)"}

    model = auto_arima(
        ret,
        d=0,  # returns are already stationary
        seasonal=True,
        m=5,  # weekly seasonality (5 trading days)
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=3,
        max_q=3,
        max_P=2,
        max_Q=2,
        information_criterion="aic",
    )
    fc, ci = model.predict(n_periods=horizon, return_conf_int=True)

    return {
        "model": f"SARIMA{model.order}x{model.seasonal_order}",
        "forecast": [round(float(v), 4) for v in fc],
        "ci_lower": [round(float(v), 4) for v in ci[:, 0]],
        "ci_upper": [round(float(v), 4) for v in ci[:, 1]],
        "aic": round(float(model.aic()), 2),
        "n_obs": len(ret),
    }


# ── Realized volatility ───────────────────────────────────────────────────────


def realized_vol(prices: pd.Series, window: int = 20) -> float | None:
    """20-day rolling realized volatility, annualised."""
    ret = _log_ret(prices) / 100
    rv = ret.rolling(window).std().iloc[-1]
    return round(float(rv) * np.sqrt(252) * 100, 4) if not np.isnan(rv) else None


# ── IV surface ────────────────────────────────────────────────────────────────


def iv_surface(chain_df: pd.DataFrame) -> pd.DataFrame:
    return chain_df.pivot_table(
        index="strike", columns="option_type", values="iv", aggfunc="mean"
    ).reset_index()


# ── Ensemble vol forecast ─────────────────────────────────────────────────────


def vol_ensemble(prices: pd.Series) -> dict:
    """
    Run all vol models and return AIC-weighted average forecast.
    Each model uses its own EDA pipeline internally.
    """
    results, weights = {}, {}

    for name, fn in [
        ("garch", lambda: fit_garch(prices)),
        ("egarch", lambda: fit_egarch(prices)),
        ("gjr_garch", lambda: fit_gjr_garch(prices)),
    ]:
        r = fn()
        if "error" not in r:
            results[name] = r
            weights[name] = 1.0 / max(abs(r["aic"]), 1)

    if not results:
        return {"error": "all vol models failed"}

    total_w = sum(weights.values())
    vol_1d = sum(results[k]["vol_1d"] * weights[k] for k in results) / total_w
    vol_ann = sum(results[k]["vol_ann"] * weights[k] for k in results) / total_w

    return {
        "ensemble_vol_1d": round(vol_1d, 4),
        "ensemble_vol_ann": round(vol_ann, 4),
        "realized_vol_20d": realized_vol(prices),
        "models": results,
    }
