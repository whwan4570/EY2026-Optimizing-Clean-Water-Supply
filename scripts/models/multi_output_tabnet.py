"""
Multi-Output TabNet: TA, EC, DRP를 한 번에 예측 (MLP 대신 TabNet 사용).

- 인터페이스: MultiOutputMLPWrapper와 동일 (fit(X, y_TA, y_EC, y_DRP), predict(X) -> (ta, ec, drp)).
- pytorch-tabnet TabNetRegressor(output_dim=3) 사용.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    TabNetRegressor = None
    torch = None


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


class MultiOutputTabNetWrapper:
    """
    TA, EC, DRP 동시 예측 래퍼 (TabNet). MLP와 동일한 fit/predict 시그니처.
    """

    def __init__(
        self,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        loss_weights: tuple[float, float, float] = (0.1, 0.2, 0.7),
        max_epochs: int = 150,
        patience: int = 25,
        batch_size: int = 1024,
        virtual_batch_size: int = 128,
        lr: float = 1e-3,
        seed: int = 42,
        verbose: int = 0,
    ):
        if not TABNET_AVAILABLE:
            raise RuntimeError("pytorch-tabnet is required. pip install pytorch-tabnet")
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.loss_weights = loss_weights
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.lr = lr
        self.seed = seed
        self.verbose = verbose
        self.scaler_X_ = StandardScaler()
        self.scaler_ta_ = StandardScaler()
        self.scaler_ec_ = StandardScaler()
        self.scaler_drp_ = StandardScaler()
        self.feature_names_in_: list[str] | None = None
        self.tabnet_: TabNetRegressor | None = None

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
    ) -> MultiOutputTabNetWrapper:
        np.random.seed(self.seed)
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
        eval_set = None
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
            eval_set = [(X_va_s, Y_va_s)]

        input_dim = X_s.shape[1]
        # TabNetRegressor 생성자에는 max_epochs/patience/batch_size가 아니라,
        # fit() 인자로 max_epochs, patience, batch_size, virtual_batch_size를 넣어야 함.
        self.tabnet_ = TabNetRegressor(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            input_dim=input_dim,
            output_dim=3,
            seed=self.seed,
            verbose=self.verbose,
            optimizer_fn=torch.optim.Adam if torch is not None else None,
            optimizer_params=dict(lr=self.lr),
        )
        self.tabnet_.fit(
            X_train=X_s,
            y_train=Y_s,
            eval_set=eval_set,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
        )
        return self

    def predict(self, X: np.ndarray | pd.DataFrame, group_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """예측 (스케일 역변환). 반환: (pred_ta, pred_ec, pred_drp)."""
        if self.tabnet_ is None or self.scaler_X_ is None:
            raise RuntimeError("MultiOutputTabNetWrapper not fitted")
        if isinstance(X, pd.DataFrame) and self.feature_names_in_ is not None:
            X = X.reindex(columns=self.feature_names_in_).fillna(0)
        X_arr, _ = _safe_array(X)
        X_s = self.scaler_X_.transform(X_arr)
        out = self.tabnet_.predict(X_s)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        pred_ta = self.scaler_ta_.inverse_transform(out[:, 0].reshape(-1, 1)).ravel()
        pred_ec = self.scaler_ec_.inverse_transform(out[:, 1].reshape(-1, 1)).ravel()
        pred_drp = self.scaler_drp_.inverse_transform(out[:, 2].reshape(-1, 1)).ravel()
        pred_drp = np.maximum(pred_drp, 0.0)
        return (
            np.asarray(pred_ta, dtype=np.float64),
            np.asarray(pred_ec, dtype=np.float64),
            np.asarray(pred_drp, dtype=np.float64),
        )
