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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# CPU only — MPS causes RNN segfault on PyTorch 2.2
DEVICE = torch.device("cpu")


# ── LSTM ──────────────────────────────────────────────────────────────────────


class LSTMModel(nn.Module):
    """
    2-layer LSTM for sequential option feature prediction.
    Input: (batch, seq_len, n_features) — pre-scaled
    Output: scalar probability per sequence
    """

    def __init__(
        self, input_size: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.bn = nn.BatchNorm1d(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.bn(out[:, -1, :])
        return self.head(out).squeeze(1)


# ── Transformer ───────────────────────────────────────────────────────────────


class TransformerModel(nn.Module):
    """
    Transformer encoder for sequential option feature prediction.
    Input: (batch, seq_len, n_features) — pre-scaled
    Output: scalar probability per sequence
    d_model must be divisible by nhead.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Ensure d_model divisible by nhead
        d_model = max(d_model, nhead) // nhead * nhead
        self.input_proj = nn.Linear(input_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        out = self.encoder(x)
        return self.head(out[:, -1, :]).squeeze(1)


# ── Training ──────────────────────────────────────────────────────────────────


def _train_model(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    batch: int = 128,
) -> nn.Module:
    """
    Train a PyTorch model on pre-scaled 3D sequences.
    X shape: (N, seq_len, n_features) — already scaled
    Early stopping on validation loss (patience=10).
    """
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCELoss()

    X_tr_t = torch.tensor(np.asarray(X_tr, dtype=np.float32), dtype=torch.float32)
    y_tr_t = torch.tensor(np.asarray(y_tr, dtype=np.float32), dtype=torch.float32)
    X_v_t = torch.tensor(np.asarray(X_val, dtype=np.float32), dtype=torch.float32).to(
        DEVICE
    )
    y_v_t = torch.tensor(np.asarray(y_val, dtype=np.float32), dtype=torch.float32).to(
        DEVICE
    )

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch,
        shuffle=True,
        num_workers=0,
    )

    best_val, best_state, patience, wait = 1e9, None, 10, 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb).clamp(1e-7, 1 - 1e-7)  # guard against NaN/inf
            loss_fn(pred, yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v_t).clamp(1e-7, 1 - 1e-7)
            val_loss = loss_fn(val_pred, y_v_t).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model.cpu()


# ── Inference ─────────────────────────────────────────────────────────────────


def predict_deep_3d(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """
    Score pre-scaled 3D sequences.
    X shape: (N, seq_len, n_features) — already scaled by DeepPipeline.scaler
    Returns: probability array of shape (N,)
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        raw = model(X_t)
        return np.array(raw.tolist(), dtype=np.float32)
