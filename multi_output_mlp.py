"""
Multi-Output MLP: TA, EC, DRP를 한 번에 예측 (공통 은닉층 → Error Propagation 완화).

- 입력: 공통 피처만 (Landsat/TerraClimate/파생 등, pred_TA/pred_EC는 넣지 않음).
- 출력: (TA, EC, DRP) 3개. 각 타깃은 StandardScaler로 스케일 후 MSE 또는 Huber 가중합으로 학습.
- Loss 가중치: DRP 강조 시 (0.2, 0.2, 0.6). Huber 사용 시 이상치에 덜 민감해 R² 개선에 도움.

사용 예:
  from multi_output_mlp import MultiOutputMLPWrapper
  wrapper = MultiOutputMLPWrapper(loss_weights=(0.2, 0.2, 0.6), loss_type="huber", huber_beta=0.1)
  wrapper.fit(X_shared, y_ta, y_ec, y_drp)
  pred_ta, pred_ec, pred_drp = wrapper.predict(X_shared_test)
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

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
            torch.backends.cudnn.benchmark = False


def _safe_array(X, copy: bool = True):
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


def _huber_numpy(err: np.ndarray, beta: float) -> np.ndarray:
    """Huber loss element-wise. |e|<=beta -> 0.5*e^2/beta; else -> |e| - 0.5*beta."""
    abs_e = np.abs(err)
    return np.where(abs_e <= beta, 0.5 * err * err / beta, abs_e - 0.5 * beta)


def _r2_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² = 1 - SS_res / SS_tot. Single target."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


class ResidualBlock(nn.Module):
    """Linear -> BN -> ReLU -> Linear + skip (지름길 연결)."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.linear1(x)
        out = self.bn(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return torch.relu(x + out)


class MultiOutputMLP(nn.Module):
    """3개 출력(TA, EC, DRP). BatchNorm + Dropout."""

    def __init__(self, input_dim: int, hidden: tuple = (256, 128, 64), dropout: tuple = (0.2, 0.1), use_residual: bool = False):
        super().__init__()
        assert len(hidden) >= 2 and len(dropout) >= 1
        self.use_residual = use_residual
        if use_residual:
            # Proj: input -> hidden[0], then residual blocks on hidden[0], then out
            self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden[0]),
                nn.BatchNorm1d(hidden[0]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout[0]) if dropout[0] > 0 else nn.Identity(),
            )
            self.blocks = nn.ModuleList([
                ResidualBlock(hidden[0], dropout[1] if i < len(dropout) else 0.0)
                for i in range(min(2, len(hidden) - 1))
            ])
            self.out = nn.Linear(hidden[0], 3)
        else:
            layers = []
            prev = input_dim
            for i, h in enumerate(hidden):
                layers.append(nn.Linear(prev, h))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU(inplace=True))
                if i < len(dropout) and dropout[i] > 0:
                    layers.append(nn.Dropout(dropout[i]))
                prev = h
            self.layers = nn.Sequential(*layers)
            self.out = nn.Linear(prev, 3)
            self.blocks = nn.ModuleList()
            self.proj = nn.Identity()

    def forward(self, x):
        if self.use_residual:
            h = self.proj(x)
            for blk in self.blocks:
                h = blk(h)
            return self.out(h)
        return self.out(self.layers(x))


class MultiOutputMLPWithEmbedding(nn.Module):
    """Entity embedding + MLP. group_ids (int) -> embed -> concat to first layer."""

    def __init__(
        self,
        input_dim: int,
        n_embeddings: int,
        embed_dim: int = 16,
        hidden: tuple = (256, 128, 64),
        dropout: tuple = (0.2, 0.1),
        use_residual: bool = False,
    ):
        super().__init__()
        self.embed = nn.Embedding(n_embeddings, embed_dim)
        mlp_input = input_dim + embed_dim
        self.mlp = MultiOutputMLP(mlp_input, hidden=hidden, dropout=dropout, use_residual=use_residual)

    def forward(self, x: torch.Tensor, group_ids: torch.Tensor | None = None):
        if group_ids is not None:
            emb = self.embed(group_ids.clamp(0, self.embed.num_embeddings - 1))
            x = torch.cat([x, emb], dim=1)
        return self.mlp(x)


class MultiOutputMLPWrapper:
    """
    TA, EC, DRP 동시 예측 래퍼 (고성능 옵션).
    - Weighted Huber: 고농도 DRP 샘플에 가중치 (log1p+1).
    - Early stop: val R² 기준 조기 종료 (min_delta로 미세 출렁거림 무시).
    - Patience 20~30 권장, min_delta=1e-4로 노이즈성 개선 무시.
    - Best model restore: R² 최고 시점 가중치 자동 복구 (과적합 구간 버림).
    - Dynamic beta / Residual Block / Entity Embedding 선택 가능.
    """

    def __init__(
        self,
        hidden: tuple = (256, 128, 64),
        dropout: tuple = (0.2, 0.1),
        loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        loss_type: Literal["mse", "huber"] = "mse",
        huber_beta: float = 0.1,
        huber_beta_schedule: Literal["constant", "linear"] = "constant",
        huber_beta_end: float = 0.5,
        use_sample_weight_drp: bool = True,
        high_drp_extra_weight: float = 0.3,
        very_high_drp_threshold: float = 100.0,
        very_high_drp_multiplier: float = 2.0,
        early_stop_metric: Literal["loss", "r2"] = "r2",
        early_stop_min_delta: float = 1e-4,
        use_residual_blocks: bool = True,
        use_entity_embedding: bool = False,
        n_embeddings: int = 128,
        embed_dim: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 100,
        patience: int = 25,
        batch_size: int = 64,
        seed: int = 42,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required. pip install torch")
        self.hidden = hidden
        self.dropout = dropout
        self.loss_weights = loss_weights
        self.loss_type = loss_type
        self.huber_beta = huber_beta
        self.huber_beta_schedule = huber_beta_schedule
        self.huber_beta_end = huber_beta_end
        self.use_sample_weight_drp = use_sample_weight_drp
        self.high_drp_extra_weight = high_drp_extra_weight
        self.very_high_drp_threshold = very_high_drp_threshold
        self.very_high_drp_multiplier = very_high_drp_multiplier
        self.early_stop_metric = early_stop_metric
        self.early_stop_min_delta = early_stop_min_delta
        self.use_residual_blocks = use_residual_blocks
        self.use_entity_embedding = use_entity_embedding
        self.n_embeddings = n_embeddings
        self.embed_dim = embed_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.scaler_X_ = StandardScaler()
        self.scaler_ta_ = StandardScaler()
        self.scaler_ec_ = StandardScaler()
        self.scaler_drp_ = StandardScaler()
        self.feature_names_in_: list[str] | None = None
        self.model_: nn.Module | None = None
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.group_ids_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y_ta: np.ndarray | pd.Series,
        y_ec: np.ndarray | pd.Series,
        y_drp: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_ta_val=None,
        y_ec_val=None,
        y_drp_val=None,
        group_ids: np.ndarray | None = None,
    ) -> MultiOutputMLPWrapper:
        set_seed(self.seed)
        X_arr, cols = _safe_array(X)
        if cols is not None:
            self.feature_names_in_ = cols
        y_ta = np.asarray(y_ta, dtype=np.float64).ravel()
        y_ec = np.asarray(y_ec, dtype=np.float64).ravel()
        y_drp = np.asarray(y_drp, dtype=np.float64).ravel()
        y_ta = np.nan_to_num(y_ta, nan=0.0, posinf=0.0, neginf=0.0)
        y_ec = np.nan_to_num(y_ec, nan=0.0, posinf=0.0, neginf=0.0)
        y_drp = np.nan_to_num(y_drp, nan=0.0, posinf=0.0, neginf=0.0)
        Y = np.column_stack([y_ta, y_ec, y_drp])

        # Weighted Huber: 고농도 DRP에 더 큰 손실 가중치 (log1p + 1) + 상위 구간 + 100 이상 2배 처벌
        if self.use_sample_weight_drp:
            base = np.log1p(np.maximum(y_drp, 0)) + 1.0
            med = np.nanmedian(y_drp)
            extra = (1.0 + self.high_drp_extra_weight * (y_drp >= med)).astype(np.float64)
            sample_weight_drp = base * extra
            # 100 이상 고농도: 0~10 대비 Loss 2배 (포기하지 않게)
            high_mult = np.where(y_drp >= self.very_high_drp_threshold, self.very_high_drp_multiplier, 1.0)
            sample_weight_drp = (sample_weight_drp * high_mult).astype(np.float64)
        else:
            sample_weight_drp = np.ones(len(y_drp), dtype=np.float64)

        if group_ids is not None and self.use_entity_embedding:
            group_ids_arr = np.asarray(group_ids, dtype=np.int64).ravel()
            group_ids_arr = np.nan_to_num(group_ids_arr, nan=0, posinf=0, neginf=0)
            self.group_ids_ = np.clip(group_ids_arr, 0, self.n_embeddings - 1)
            if len(self.group_ids_) != len(X_arr):
                self.group_ids_ = np.zeros(len(X_arr), dtype=np.int64)
        else:
            self.group_ids_ = None

        self.scaler_X_.fit(X_arr)
        X_s = self.scaler_X_.transform(X_arr)
        self.scaler_ta_.fit(y_ta.reshape(-1, 1))
        self.scaler_ec_.fit(y_ec.reshape(-1, 1))
        self.scaler_drp_.fit(y_drp.reshape(-1, 1))
        Y_s = np.column_stack([
            self.scaler_ta_.transform(y_ta.reshape(-1, 1)).ravel(),
            self.scaler_ec_.transform(y_ec.reshape(-1, 1)).ravel(),
            self.scaler_drp_.transform(y_drp.reshape(-1, 1)).ravel(),
        ])

        has_val = (
            X_val is not None and y_ta_val is not None and y_ec_val is not None and y_drp_val is not None
            and len(y_ta_val) > 0
        )
        if has_val:
            X_va, _ = _safe_array(X_val)
            X_va_s = self.scaler_X_.transform(X_va)
            y_ta_v = np.nan_to_num(np.asarray(y_ta_val, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
            y_ec_v = np.nan_to_num(np.asarray(y_ec_val, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
            y_drp_v = np.nan_to_num(np.asarray(y_drp_val, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
            Y_va_s = np.column_stack([
                self.scaler_ta_.transform(y_ta_v.reshape(-1, 1)).ravel(),
                self.scaler_ec_.transform(y_ec_v.reshape(-1, 1)).ravel(),
                self.scaler_drp_.transform(y_drp_v.reshape(-1, 1)).ravel(),
            ])
            group_ids_va = np.zeros(len(X_va_s), dtype=np.int64) if self.use_entity_embedding else None

        input_dim = X_s.shape[1]
        if self.use_entity_embedding:
            self.model_ = MultiOutputMLPWithEmbedding(
                input_dim, self.n_embeddings, self.embed_dim,
                hidden=self.hidden, dropout=self.dropout, use_residual=self.use_residual_blocks,
            ).to(self.device_)
        else:
            self.model_ = MultiOutputMLP(
                input_dim, hidden=self.hidden, dropout=self.dropout, use_residual=self.use_residual_blocks,
            ).to(self.device_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        w_ta, w_ec, w_drp = self.loss_weights

        best_metric = float("-inf") if self.early_stop_metric == "r2" else float("inf")
        best_state: dict | None = None
        wait = 0

        for ep in range(self.epochs):
            beta_ep = self.huber_beta
            if self.loss_type == "huber" and self.huber_beta_schedule == "linear":
                t = ep / max(self.epochs - 1, 1)
                beta_ep = float(self.huber_beta + t * (self.huber_beta_end - self.huber_beta))
            if self.loss_type == "huber":
                criterion_ta_ec = nn.SmoothL1Loss(reduction="mean", beta=beta_ep)
                criterion_drp = nn.SmoothL1Loss(reduction="none", beta=beta_ep)
            else:
                criterion_ta_ec = nn.MSELoss()
                criterion_drp = nn.MSELoss(reduction="none")

            self.model_.train()
            perm = np.random.permutation(len(X_s))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(perm), self.batch_size):
                idx = perm[start : start + self.batch_size]
                bx = torch.tensor(X_s[idx], dtype=torch.float32, device=self.device_)
                by = torch.tensor(Y_s[idx], dtype=torch.float32, device=self.device_)
                sw = torch.tensor(sample_weight_drp[idx], dtype=torch.float32, device=self.device_)
                gid = torch.tensor(self.group_ids_[idx], dtype=torch.long, device=self.device_) if self.group_ids_ is not None else None
                optimizer.zero_grad()
                if self.use_entity_embedding and gid is not None:
                    out = self.model_(bx, gid)
                else:
                    out = self.model_(bx)
                loss_ta = criterion_ta_ec(out[:, 0], by[:, 0])
                loss_ec = criterion_ta_ec(out[:, 1], by[:, 1])
                loss_drp_raw = criterion_drp(out[:, 2], by[:, 2])
                loss_drp = (loss_drp_raw * sw).mean()
                total_loss = w_ta * loss_ta + w_ec * loss_ec + w_drp * loss_drp
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                n_batches += 1

            if has_val:
                self.model_.eval()
                with torch.no_grad():
                    x_va_t = torch.tensor(X_va_s, dtype=torch.float32, device=self.device_)
                    gid_va = torch.tensor(group_ids_va, dtype=torch.long, device=self.device_) if (self.use_entity_embedding and group_ids_va is not None) else None
                    if self.use_entity_embedding and gid_va is not None:
                        pred_va = self.model_(x_va_t, gid_va).cpu().numpy()
                    else:
                        pred_va = self.model_(x_va_t).cpu().numpy()
                if self.early_stop_metric == "r2":
                    r2_ta = _r2_numpy(Y_va_s[:, 0], pred_va[:, 0])
                    r2_ec = _r2_numpy(Y_va_s[:, 1], pred_va[:, 1])
                    r2_drp = _r2_numpy(Y_va_s[:, 2], pred_va[:, 2])
                    val_metric = (r2_ta + r2_ec + r2_drp) / 3.0
                    # min_delta: 미세한 출렁거림은 개선으로 치지 않음 (R²는 maximize)
                    if val_metric > best_metric + self.early_stop_min_delta:
                        best_metric = val_metric
                        best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                        wait = 0
                    else:
                        wait += 1
                else:
                    if self.loss_type == "huber":
                        b = beta_ep
                        val_loss = (
                            w_ta * np.mean(_huber_numpy(pred_va[:, 0] - Y_va_s[:, 0], b))
                            + w_ec * np.mean(_huber_numpy(pred_va[:, 1] - Y_va_s[:, 1], b))
                            + w_drp * np.mean(_huber_numpy(pred_va[:, 2] - Y_va_s[:, 2], b))
                        )
                    else:
                        val_loss = (
                            w_ta * np.mean((pred_va[:, 0] - Y_va_s[:, 0]) ** 2)
                            + w_ec * np.mean((pred_va[:, 1] - Y_va_s[:, 1]) ** 2)
                            + w_drp * np.mean((pred_va[:, 2] - Y_va_s[:, 2]) ** 2)
                        )
                    # min_delta: 미세한 개선은 무시 (loss는 minimize)
                    if val_loss < best_metric - self.early_stop_min_delta:
                        best_metric = val_loss
                        best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                        wait = 0
                    else:
                        wait += 1
                if wait >= self.patience:
                    break
            else:
                mean_loss = epoch_loss / max(n_batches, 1)
                if mean_loss < best_metric:
                    best_metric = mean_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray | pd.DataFrame, group_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """예측 (스케일 역변환 적용). 반환: (pred_ta, pred_ec, pred_drp). use_entity_embedding 시 group_ids 전달 가능."""
        if self.model_ is None or self.scaler_X_ is None:
            raise RuntimeError("MultiOutputMLPWrapper not fitted")
        if isinstance(X, pd.DataFrame) and self.feature_names_in_ is not None:
            X = X.reindex(columns=self.feature_names_in_).fillna(0)
        X_arr, _ = _safe_array(X)
        X_s = self.scaler_X_.transform(X_arr)
        n = len(X_s)
        if self.use_entity_embedding:
            gid = np.zeros(n, dtype=np.int64) if group_ids is None else np.clip(np.asarray(group_ids, dtype=np.int64).ravel()[:n], 0, self.n_embeddings - 1)
            gid_t = torch.tensor(gid, dtype=torch.long, device=self.device_)
        else:
            gid_t = None
        self.model_.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_s, dtype=torch.float32, device=self.device_)
            out = (self.model_(x_t, gid_t) if gid_t is not None else self.model_(x_t)).cpu().numpy()
        pred_ta = self.scaler_ta_.inverse_transform(out[:, 0].reshape(-1, 1)).ravel()
        pred_ec = self.scaler_ec_.inverse_transform(out[:, 1].reshape(-1, 1)).ravel()
        pred_drp = self.scaler_drp_.inverse_transform(out[:, 2].reshape(-1, 1)).ravel()
        pred_drp = np.maximum(pred_drp, 0.0)
        return (
            np.asarray(pred_ta, dtype=np.float64),
            np.asarray(pred_ec, dtype=np.float64),
            np.asarray(pred_drp, dtype=np.float64),
        )
