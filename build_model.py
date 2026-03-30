"""
build_model.py — Train per-asset ensembles using model-specific pipelines.

Pipeline per model family:
  GARCH/EGARCH/GJR/SARIMA → GarchPipeline / SarimaPipeline
    Input: spot log-returns (univariate)
    EDA: ADF stationarity, Ljung-Box ARCH effects, ACF/PACF

  CatBoost / XGBoost / LightGBM → TreePipeline
    Input: tabular option features (no scaling)
    EDA: class balance, feature-target correlation, outlier check

  LSTM / Transformer → DeepPipeline
    Input: per-symbol scaled sequences shape (N, seq_len, n_features)
    EDA: sequence length distribution, temporal gap check
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from dotenv import load_dotenv

from models.data_pipeline import (
    GarchPipeline,
    SarimaPipeline,
    TreePipeline,
    DeepPipeline,
    TREE_FEATURES,
)
from models.vol_models import fit_garch, fit_egarch, fit_gjr_garch, fit_sarima
from models.ensemble import OptionEnsemble

load_dotenv()
DB_PATH = os.getenv("DB_PATH", "data.db")
MODELS_DIR = os.getenv("MODELS_DIR", "trained_models")
os.makedirs(MODELS_DIR, exist_ok=True)

ASSETS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX"]


def run_vol_eda(asset: str):
    """Run and print EDA for GARCH + SARIMA pipelines."""
    print(f"\n  [Vol EDA — {asset}]")
    gp = GarchPipeline(DB_PATH, asset)
    eda = gp.run_eda()
    if "error" in eda:
        print(f"    GARCH EDA: {eda['error']}")
        return

    print(f"    Returns: n={eda['n']}  mean={eda['mean']}%  std={eda['std']}%")
    print(f"    Skew={eda['skew']}  Kurtosis={eda['kurtosis']}")
    print(f"    ADF p={eda['adf_pvalue']} → stationary={eda['stationary']}")
    print(f"    ARCH effects: {eda['arch_effects']} (LB p={eda['lb_pvalue_5']})")
    print(
        f"    Best GARCH order: {eda['best_garch_order']} AIC={eda['best_garch_aic']}"
    )

    sp = SarimaPipeline(DB_PATH, asset)
    seda = sp.run_eda()
    if "error" not in seda:
        print(
            f"    SARIMA: d={seda['suggested_d']}  "
            f"sig_ACF={seda['sig_acf_lags']}  "
            f"sig_PACF={seda['sig_pacf_lags']}"
        )


def run_tree_eda(asset: str, tree: TreePipeline):
    """Print EDA results from TreePipeline."""
    eda = tree.eda_results
    if not eda:
        return
    print(f"\n  [Tree EDA — {asset}]")
    print(f"    Class balance: {eda.get('positive_rate', '?'):.2%} positive")
    print(f"    Top correlated features: {eda.get('top_corr_features', {})}")
    print(f"    Null rates: {eda.get('null_rates', {})}")
    print(f"    Outliers (>3σ): {eda.get('outliers_3std', {})}")
    m = eda.get("moneyness", {})
    if m:
        print(f"    Moneyness: ITM={m['itm']} ATM={m['atm']} OTM={m['otm']}")


def run_deep_eda(asset: str, deep: DeepPipeline):
    """Print EDA results from DeepPipeline."""
    eda = deep.eda_results
    if not eda:
        return
    print(f"\n  [Deep EDA — {asset}]")
    print(f"    Total symbols: {eda.get('total_symbols')}")
    print(f"    Symbols ≥ seq_len({deep.seq_len}): {eda.get('symbols_ge_seq_len')}")
    print(f"    Symbols ≥ 20 obs: {eda.get('symbols_ge_20')}")
    print(
        f"    Seq len: min={eda.get('min_seq_len')} "
        f"mean={eda.get('mean_seq_len')} max={eda.get('max_seq_len')}"
    )
    print(f"    Symbols with temporal gaps: {eda.get('symbols_with_gaps')}")


def train_asset(asset: str):
    print(f"\n{'='*60}")
    print(f"  ASSET: {asset}")
    print(f"{'='*60}")

    # ── 1. Vol model EDA (GARCH / SARIMA) ────────────────────────────────────
    run_vol_eda(asset)

    # ── 2. Tree pipeline ──────────────────────────────────────────────────────
    tree = TreePipeline(DB_PATH, asset)
    df = tree.build()

    if df.empty:
        print(f"\n[build] No data for {asset}. Run collect_data.py first.")
        return

    run_tree_eda(asset, tree)

    X_tr, y_tr, X_te, y_te, feat_names = tree.get_Xy()
    print(
        f"\n  [Tree split] train={len(X_tr)} test={len(X_te)} features={len(feat_names)}"
    )

    if len(X_tr) < 80:
        print(f"[build] Not enough training data ({len(X_tr)} rows). Need 80+.")
        return

    # ── 3. Deep pipeline ──────────────────────────────────────────────────────
    deep = DeepPipeline(DB_PATH, asset, seq_len=10)
    df_deep = deep.build()
    run_deep_eda(asset, deep)

    X_tr_seq, y_tr_seq, X_te_seq, y_te_seq = None, None, None, None
    if not df_deep.empty:
        X_tr_seq, y_tr_seq, X_te_seq, y_te_seq = deep.make_sequences(df_deep)
        if X_tr_seq is not None:
            print(
                f"  [Deep split] train_seqs={len(X_tr_seq)} "
                f"test_seqs={len(X_te_seq) if X_te_seq is not None else 0} "
                f"shape={X_tr_seq.shape}"
            )

    # ── 4. Train ensemble ─────────────────────────────────────────────────────
    ensemble = OptionEnsemble()
    ensemble.fit(
        X_tr,
        y_tr,
        X_te,
        y_te,
        feature_names=feat_names,
        X_tr_seq=X_tr_seq,
        y_tr_seq=y_tr_seq,
        X_val_seq=X_te_seq,
        y_val_seq=y_te_seq,
    )

    # ── 5. Final evaluation ───────────────────────────────────────────────────
    preds = ensemble.predict(X_te)
    probas = ensemble.predict_proba(X_te)[:, 1]
    print(f"\n[build] {asset} Test Report:")
    print(classification_report(y_te, preds))
    print(f"ROC-AUC: {roc_auc_score(y_te, probas):.4f}")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    path = os.path.join(MODELS_DIR, f"{asset.lower()}_ensemble.pkl")
    ensemble.save(path)


def main():
    import sys

    assets = sys.argv[1:] if len(sys.argv) > 1 else ASSETS
    for asset in assets:
        try:
            train_asset(asset)
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"[build] {asset} failed: {e}")
            traceback.print_exc()

    print("\n[build] Done. Models saved to:", MODELS_DIR)


if __name__ == "__main__":
    main()
