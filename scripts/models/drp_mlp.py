"""
DRP Residual 예측용 소형 Tabular MLP (PyTorch).

- 학습 타깃: linear residual (y_DRP - prior). log residual 미사용.
- 입력: 기존 DRP 피처 + pred_TA + pred_EC + gems_distance_km (StandardScaler로 스케일링).
- 구조: (Linear->ReLU->Dropout) x 2~3 + output 1. Early stopping, weight decay, dropout.
- 사용처: benchmark_model.run_pipeline_drp 내부에서 fold별 fit, 제출 시 GBDT residual과 블렌딩.

주의:
- DL은 LB에서 오를 수 있으나 seed 변동성이 커서 blend weight(w)를 크게 하면 악화 가능. 0.05~0.2 권장.
- DRP 분포는 heavy-tail/0 많음 → residual 학습이 안정적.
- 과적합 방지: hidden 작게, dropout/weight_decay/early_stopping 필수.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from sklearn.preprocessing import StandardScaler


def set_seed(seed: int = 42) -> None:
    """numpy / random / torch / cudnn 시드 고정 (재현성)."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False    # type: ignore[attr-defined]


def _safe_array(X, copy: bool = True):
    """NaN/Inf 방어: np.nan_to_num. DataFrame이면 컬럼 순서 유지."""
    if isinstance(X, pd.DataFrame):
        arr = X.values.copy() if copy else X.values
        cols = list(X.columns)
    else:
        arr = np.asarray(X, dtype=np.float64)
        if copy:
            arr = arr.copy()
        cols = None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, cols


class _MLP(nn.Module):
    """Small MLP: (Linear->ReLU->Dropout) x num_layers + Linear(1)."""

    def __init__(self, input_dim: int, hidden: list[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)

    def forward(self, x):
        return self.out(self.layers(x)).squeeze(-1)


class DRPMLP:
    """
    DRP residual 예측용 MLP 래퍼.
    - fit(X_train, y_train, X_val=None, y_val=None): train만으로 scaler fit, val로 early stopping.
    - predict(X): DataFrame 또는 ndarray; 컬럼 정렬 후 스케일 -> 예측 (residual).
    """

    def __init__(
        self,
        hidden: list[int] | None = None,
        dropout: float = 0.2,
        weight_decay: float = 1e-5,
        epochs: int = 200,
        patience: int = 15,
        lr: float = 1e-3,
        batch_size: int = 64,
        seed: int = 42,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DRPMLP. pip install torch")
        self.hidden = hidden or [64, 32]
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.scaler_ = StandardScaler()
        self.feature_names_in_: list[str] | None = None
        self.model_: nn.Module | None = None
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
    ) -> DRPMLP:
        set_seed(self.seed)
        X_tr, cols = _safe_array(X_train)
        if cols is not None:
            self.feature_names_in_ = cols
        y_tr = np.asarray(y_train, dtype=np.float64).ravel()
        y_tr = np.nan_to_num(y_tr, nan=0.0, posinf=0.0, neginf=0.0)

        self.scaler_.fit(X_tr)
        X_tr_s = self.scaler_.transform(X_tr)

        has_val = X_val is not None and y_val is not None and len(y_val) > 0
        if has_val:
            X_va, _ = _safe_array(X_val)
            X_va_s = self.scaler_.transform(X_va)
            y_va = np.asarray(y_val, dtype=np.float64).ravel()
            y_va = np.nan_to_num(y_va, nan=0.0, posinf=0.0, neginf=0.0)

        input_dim = X_tr_s.shape[1]
        self.model_ = _MLP(input_dim, self.hidden, self.dropout).to(self.device_)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        best_state: dict | None = None
        wait = 0

        for ep in range(self.epochs):
            self.model_.train()
            perm = np.random.permutation(len(X_tr_s))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(perm), self.batch_size):
                idx = perm[start : start + self.batch_size]
                bx = torch.tensor(X_tr_s[idx], dtype=torch.float32, device=self.device_)
                by = torch.tensor(y_tr[idx], dtype=torch.float32, device=self.device_)
                opt.zero_grad()
                pred = self.model_(bx)
                loss = criterion(pred, by)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1
            train_loss = epoch_loss / max(n_batches, 1)

            if has_val:
                self.model_.eval()
                with torch.no_grad():
                    X_va_t = torch.tensor(X_va_s, dtype=torch.float32, device=self.device_)
                    pred_va = self.model_(X_va_t).cpu().numpy()
                val_loss = float(np.mean((pred_va - y_va) ** 2))
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                if wait >= self.patience:
                    break
            else:
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("DRPMLP not fitted")
        if isinstance(X, pd.DataFrame) and self.feature_names_in_ is not None:
            # 컬럼 순서/누락 정렬 (merge 기준만 사용, 행 순서 가정 금지)
            X = X.reindex(columns=self.feature_names_in_).fillna(0)
        X_arr, _ = _safe_array(X)
        X_s = self.scaler_.transform(X_arr)
        self.model_.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_s, dtype=torch.float32, device=self.device_)
            out = self.model_(x_t).cpu().numpy()
        return np.asarray(out, dtype=np.float64)
