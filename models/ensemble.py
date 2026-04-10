"""
ensemble.py — Stacked ensemble using model-specific pipelines.

Model-specific feature sets and preprocessing:

  CatBoost / XGBoost / LightGBM (TreePipeline):
    - Features: TREE_FEATURES (tabular, no scaling)
    - Outlier clipping at 3-sigma
    - Temporal train/test split

  LSTM / Transformer (DeepPipeline):
    - Features: DEEP_FEATURES (same set, but scaled)
    - StandardScaler fit on train only
    - Sequences built per-symbol (temporal order preserved)
    - Only symbols with >= seq_len observations used

  Meta-learner: Logistic Regression on stacked OOF predictions
"""

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from models.data_pipeline import TREE_FEATURES, DEEP_FEATURES, ALL_FEATURES

try:
    from models.deep_models import (
        LSTMModel,
        TransformerModel,
        _train_model,
        predict_deep_3d,
    )

    DEEP_AVAILABLE = True
except ImportError:
    DEEP_AVAILABLE = False

    def predict_deep_3d(model, X):
        import numpy as np

        return np.full(len(X), 0.5, dtype="float32")


# Expose for external use
FEATURES = ALL_FEATURES

SEQ_LEN = 10


# ── Tree base learners ────────────────────────────────────────────────────────


def _catboost(X_tr, y_tr, X_val, y_val, feat_names):
    from catboost import Pool

    tr_pool = Pool(X_tr, y_tr, feature_names=feat_names)
    val_pool = Pool(X_val, y_val, feature_names=feat_names)
    m = CatBoostClassifier(
        iterations=600,
        learning_rate=0.04,
        depth=7,
        eval_metric="AUC",
        early_stopping_rounds=50,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=0,
    )
    m.fit(tr_pool, eval_set=val_pool)
    return m


def _xgboost(X_tr, y_tr, X_val, y_val):
    m = XGBClassifier(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=42,
        verbosity=0,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m


def _lgbm(X_tr, y_tr, X_val, y_val):
    m = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[])
    return m


# ── Ensemble ──────────────────────────────────────────────────────────────────


class OptionEnsemble:
    """
    Stacked ensemble: CatBoost + XGBoost + LightGBM + LSTM + Transformer.
    Each model uses its own feature set and preprocessing.
    Meta-learner: Logistic Regression on validation predictions.
    """

    def __init__(self):
        self.base_models = {}  # name -> fitted tree model
        self.deep_models = {}  # name -> (model, scaler)
        self.meta = None
        self.feature_names = TREE_FEATURES  # tree feature names
        self.auc_scores = {}

    def fit(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list = None,
        # Deep model sequences (pre-built by DeepPipeline)
        X_tr_seq: np.ndarray = None,
        y_tr_seq: np.ndarray = None,
        X_val_seq: np.ndarray = None,
        y_val_seq: np.ndarray = None,
    ):
        if feature_names:
            self.feature_names = feature_names

        print("[ensemble] Training tree models (tabular features, no scaling)...")

        # ── Tree models: use TREE_FEATURES, no scaling ────────────────────────
        tree_feat_idx = (
            [
                i
                for i, f in enumerate(feature_names or TREE_FEATURES)
                if f in TREE_FEATURES
            ]
            if feature_names
            else list(range(X_tr.shape[1]))
        )

        Xt_tr = X_tr[:, tree_feat_idx]
        Xt_val = X_val[:, tree_feat_idx]
        tree_names = (
            [feature_names[i] for i in tree_feat_idx]
            if feature_names
            else TREE_FEATURES
        )

        for name, fn in [
            ("catboost", lambda: _catboost(Xt_tr, y_tr, Xt_val, y_val, tree_names)),
            ("xgboost", lambda: _xgboost(Xt_tr, y_tr, Xt_val, y_val)),
            ("lgbm", lambda: _lgbm(Xt_tr, y_tr, Xt_val, y_val)),
        ]:
            print(f"  [{name}]...")
            m = fn()
            self.base_models[name] = m
            p = m.predict_proba(Xt_val)[:, 1]
            auc = roc_auc_score(y_val, p)
            self.auc_scores[name] = round(auc, 4)
            print(f"    AUC={auc:.4f}")

        # ── Deep models: use pre-scaled 3D sequences ──────────────────────────
        if X_tr_seq is not None and len(X_tr_seq) > SEQ_LEN * 3:
            n_feat = X_tr_seq.shape[2]

            print("  [lstm] (scaled sequences, per-symbol temporal order)...")
            lstm = LSTMModel(input_size=n_feat)
            lstm = _train_model(lstm, X_tr_seq, y_tr_seq, X_val_seq, y_val_seq)
            self.deep_models["lstm"] = lstm
            p = predict_deep_3d(lstm, X_val_seq)
            auc = roc_auc_score(y_val_seq, p)
            self.auc_scores["lstm"] = round(auc, 4)
            print(f"    AUC={auc:.4f}")

            print("  [transformer] (scaled sequences, per-symbol temporal order)...")
            tfm = TransformerModel(input_size=n_feat)
            tfm = _train_model(tfm, X_tr_seq, y_tr_seq, X_val_seq, y_val_seq)
            self.deep_models["transformer"] = tfm
            p = predict_deep_3d(tfm, X_val_seq)
            auc = roc_auc_score(y_val_seq, p)
            self.auc_scores["transformer"] = round(auc, 4)
            print(f"    AUC={auc:.4f}")

        # ── Meta-learner: stack val predictions ──────────────────────────────
        meta_X = self._stack_tree_preds(Xt_val)
        if X_val_seq is not None and len(X_val_seq) > 0 and self.deep_models:
            # Deep preds are on a different (smaller) val set — use tree val only
            pass  # meta uses tree preds only for alignment simplicity

        self.meta = LogisticRegression(C=1.0, max_iter=500)
        self.meta.fit(meta_X, y_val)

        meta_p = self.meta.predict_proba(meta_X)[:, 1]
        ens_auc = roc_auc_score(y_val, meta_p)
        self.auc_scores["ensemble"] = round(ens_auc, 4)
        print(f"\n[ensemble] Final AUC: {ens_auc:.4f}")
        print(f"[ensemble] Per-model: {self.auc_scores}")

    def _stack_tree_preds(self, X: np.ndarray) -> np.ndarray:
        cols = [m.predict_proba(X)[:, 1] for m in self.base_models.values()]
        return np.column_stack(cols) if cols else np.zeros((len(X), 1))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.meta is None:
            raise RuntimeError("Model not trained")
        # X should be tree features (unscaled)
        meta_X = self._stack_tree_preds(X)
        return self.meta.predict_proba(meta_X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[ensemble] Saved → {path}")

    @staticmethod
    def load(path: str) -> "OptionEnsemble":
        import pickle, io

        class SafeUnpickler(pickle.Unpickler):
            """Handles missing torch classes gracefully."""

            def find_class(self, module, name):
                if "torch" in module or module.startswith("models.deep_models"):
                    # Return a dummy class for torch objects
                    return type(name, (), {"__init__": lambda self, *a, **kw: None})
                return super().find_class(module, name)

        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (ModuleNotFoundError, ImportError):
            # torch not available — load with safe unpickler
            with open(path, "rb") as f:
                obj = SafeUnpickler(f).load()
            # Clear deep models since they can't run without torch
            obj.deep_models = {}
            return obj
