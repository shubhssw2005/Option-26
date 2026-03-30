"""
deep_models.py — LSTM and Transformer for sequential option data.

Input contract (from DeepPipeline):
  - X shape: (N, seq_len, n_features)  — already scaled by StandardScaler
  - y shape: (N,)                       — binary target
  - Sequences are built per-symbol, temporal order preserved
  - Scaler is fit on training data only (no leakage)

Device: CPU (MPS disabled — RNN segfault on PyTorch 2.2 + Apple Silicon)
OMP_NUM_THREADS=1 required to avoid macOS ARM deadlock
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

# torch is optional — needed for training only, not for inference on pre-trained pkl
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
    DEVICE = torch.device("cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    # Stub nn so class definitions don't fail at import time
    import types

    nn = types.SimpleNamespace(
        Module=object,
        LSTM=None,
        Linear=None,
        ReLU=None,
        Dropout=None,
        Sigmoid=None,
        BatchNorm1d=None,
        Sequential=None,
        TransformerEncoderLayer=None,
        TransformerEncoder=None,
        BCELoss=None,
    )


# ── Models (only defined when torch is available) ────────────────────────────

if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden=64, layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, num_layers=layers,
                                batch_first=True, dropout=dropout if layers>1 else 0.0)
            self.bn   = nn.BatchNorm1d(hidden)
            self.head = nn.Sequential(nn.Linear(hidden,32), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(32,1), nn.Sigmoid())
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(self.bn(out[:,-1,:])).squeeze(1)

    class TransformerModel(nn.Module):
        def __init__(self, input_size, d_model=32, nhead=4, num_layers=1, dropout=0.1):
            super().__init__()
            d_model = max(d_model, nhead) // nhead * nhead
            self.input_proj = nn.Linear(input_size, d_model)
            enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                      dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
            self.head = nn.Sequential(nn.Linear(d_model,16), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(16,1), nn.Sigmoid())
        def forward(self, x):
            return self.head(self.encoder(self.input_proj(x))[:,-1,:]).squeeze(1)

    def _train_model(model, X_tr, y_tr, X_val, y_val, epochs=30, lr=1e-3, batch=128):
        model = model.to(DEVICE)
        opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.BCELoss()
        X_tr_t = torch.tensor(np.asarray(X_tr, dtype=np.float32), dtype=torch.float32)
        y_tr_t = torch.tensor(np.asarray(y_tr, dtype=np.float32), dtype=torch.float32)
        X_v_t  = torch.tensor(np.asarray(X_val,dtype=np.float32), dtype=torch.float32).to(DEVICE)
        y_v_t  = torch.tensor(np.asarray(y_val,dtype=np.float32), dtype=torch.float32).to(DEVICE)
        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch, shuffle=True, num_workers=0)
        best_val, best_state, patience, wait = 1e9, None, 10, 0
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss_fn(model(xb).clamp(1e-7,1-1e-7), yb).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_v_t).clamp(1e-7,1-1e-7), y_v_t).item()
            if val_loss < best_val:
                best_val, best_state, wait = val_loss, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= patience: break
        if best_state: model.load_state_dict(best_state)
        return model.cpu()

else:
    # Stubs when torch not installed (inference uses pre-trained pkl, no training needed)
    class LSTMModel:
        def __init__(self, *a, **kw): pass
    class TransformerModel:
        def __init__(self, *a, **kw): pass
    def _train_model(*a, **kw):
        raise RuntimeError("torch not installed — cannot train deep models")


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_deep_3d(model, X: np.ndarray) -> np.ndarray:
    """
    Score pre-scaled 3D sequences.
    Returns probability array of shape (N,).
    Falls back to 0.5 if torch not available.
    """
    if not TORCH_AVAILABLE or model is None:
        return np.full(len(X), 0.5, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        return np.array(model(X_t).tolist(), dtype=np.float32)
