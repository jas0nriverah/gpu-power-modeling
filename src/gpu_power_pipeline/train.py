from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import TARGET_COL, build_preprocessor, infer_feature_columns

SPLIT_RANDOM = "random"
SPLIT_TIME = "time"
SPLIT_GROUPED_SESSION = "grouped"
VALID_SPLIT_STRATEGIES = {SPLIT_RANDOM, SPLIT_TIME, SPLIT_GROUPED_SESSION}

try:
    from sklearn.neural_network import MLPRegressor

    SKLEARN_MLP_AVAILABLE = True
except Exception:
    SKLEARN_MLP_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False


@dataclass
class ModelArtifacts:
    model_name: str
    pipeline: Pipeline
    metrics: Dict[str, float]
    y_test: np.ndarray
    y_pred: np.ndarray
    X_test_raw: pd.DataFrame
    feature_names: list[str]
    feature_importance: Optional[pd.DataFrame]
    permutation_importance: Optional[pd.DataFrame]
    residual_diagnostics: Dict[str, float]
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)

    @property
    def input_features(self) -> list[str]:
        """Raw input columns (pre-transform) the pipeline expects."""
        return self.numeric_features + self.categorical_features


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """Lightweight PyTorch MLP with sklearn-like API."""

    def __init__(
        self,
        hidden_dim: int = 64,
        hidden_dim2: int = 32,
        lr: float = 1e-3,
        epochs: int = 60,
        batch_size: int = 256,
        random_state: int = 42,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available in this environment.")
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model_: Optional[nn.Module] = None

    def _build(self, in_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim2, 1),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPRegressor":
        torch.manual_seed(self.random_state)
        x_np = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        X_t = torch.tensor(x_np, dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float32)

        self.model_ = self._build(X_t.shape[1])
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=min(self.batch_size, len(X_t)),
            shuffle=True,
        )
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model has not been fit.")
        self.model_.eval()
        with torch.no_grad():
            x_np = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            X_t = torch.tensor(x_np, dtype=torch.float32)
            preds = self.model_(X_t).cpu().numpy().reshape(-1)
        return preds


def _build_models(
    include_mlp: bool = False,
    include_torch_mlp: bool = False,
    include_xgboost: bool = False,
    include_lightgbm: bool = False,
    random_state: int = 42,
    model_params: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, object]:
    model_params = model_params or {}
    rf_params = {
        "n_estimators": 220,
        "max_depth": 14,
        "min_samples_leaf": 2,
        "random_state": random_state,
        "n_jobs": -1,
    }
    rf_params.update(model_params.get("random_forest", {}))
    gb_params = {
        "n_estimators": 220,
        "max_depth": 3,
        "learning_rate": 0.05,
        "random_state": random_state,
    }
    gb_params.update(model_params.get("gradient_boosting", {}))
    lin_params = {}
    lin_params.update(model_params.get("linear_regression", {}))
    models: Dict[str, object] = {
        "linear_regression": LinearRegression(**lin_params),
        "random_forest": RandomForestRegressor(**rf_params),
        "gradient_boosting": GradientBoostingRegressor(**gb_params),
    }
    if include_mlp and SKLEARN_MLP_AVAILABLE:
        models["mlp_regressor"] = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            max_iter=450,
            random_state=random_state,
            **model_params.get("mlp_regressor", {}),
        )
    if include_torch_mlp and TORCH_AVAILABLE:
        models["torch_mlp"] = TorchMLPRegressor(
            random_state=random_state,
            **model_params.get("torch_mlp", {}),
        )
    if include_xgboost:
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost is not installed. Install optional boosting dependencies first.")
        xgb_params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": random_state,
            "n_jobs": -1,
        }
        xgb_params.update(model_params.get("xgboost", {}))
        models["xgboost"] = XGBRegressor(**xgb_params)
    if include_lightgbm:
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("LightGBM is not installed. Install optional boosting dependencies first.")
        lgbm_params = {
            "n_estimators": 300,
            "max_depth": -1,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        lgbm_params.update(model_params.get("lightgbm", {}))
        models["lightgbm"] = LGBMRegressor(**lgbm_params)
    return models


def optional_model_status() -> dict[str, bool]:
    """Expose optional model availability for CLI help/tests."""
    return {
        "sklearn_mlp": SKLEARN_MLP_AVAILABLE,
        "torch_mlp": TORCH_AVAILABLE,
        "xgboost": XGBOOST_AVAILABLE,
        "lightgbm": LIGHTGBM_AVAILABLE,
    }


def _extract_feature_importance(pipeline: Pipeline, model_name: str, feature_names: list[str]) -> Optional[pd.DataFrame]:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        values = np.abs(coef)
    else:
        return None
    fi = pd.DataFrame({"feature": feature_names, "importance": values})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


def _safe_permutation_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    random_state: int,
) -> Optional[pd.DataFrame]:
    try:
        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=8,
            random_state=random_state,
            scoring="neg_root_mean_squared_error",
        )
    except Exception:
        return None
    features = X_test.columns.tolist()
    df = pd.DataFrame({"feature": features, "importance": result.importances_mean})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def get_train_test_indices(
    df: pd.DataFrame,
    split_strategy: str,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test indices for random, time, or grouped-session split."""
    if split_strategy not in VALID_SPLIT_STRATEGIES:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")
    n = len(df)
    all_idx = np.arange(n)
    if split_strategy == SPLIT_TIME and "timestamp" in df.columns:
        ordered_idx = np.argsort(pd.to_datetime(df["timestamp"], errors="coerce").to_numpy())
        cutoff = int((1.0 - test_size) * n)
        return ordered_idx[:cutoff], ordered_idx[cutoff:]
    if split_strategy == SPLIT_GROUPED_SESSION and "session_id" in df.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr, te = next(splitter.split(all_idx, groups=df["session_id"]))
        return all_idx[tr], all_idx[te]
    tr, te = train_test_split(all_idx, test_size=test_size, random_state=random_state)
    return np.asarray(tr), np.asarray(te)


def fit_full_pipeline(
    df: pd.DataFrame,
    model_name: str,
    random_state: int = 42,
    include_mlp: bool = False,
    include_torch_mlp: bool = False,
    include_xgboost: bool = False,
    include_lightgbm: bool = False,
    model_params: Optional[Dict[str, Dict[str, object]]] = None,
) -> tuple[Pipeline, list[str], list[str]]:
    """Fit a single model on the entire dataset for deployment/persistence.

    Unlike :func:`train_and_evaluate` (which holds out a test split for
    scoring), this trains on all rows so the saved artifact uses every
    available example. Returns the fitted pipeline and the raw numeric and
    categorical feature columns it expects.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from dataset.")

    numeric_cols, categorical_cols = infer_feature_columns(df, target_col=TARGET_COL)
    numeric_cols = [c for c in numeric_cols if c != "timestamp"]
    categorical_cols = [c for c in categorical_cols if c != "timestamp"]
    feature_cols = numeric_cols + categorical_cols

    models = _build_models(
        include_mlp=include_mlp,
        include_torch_mlp=include_torch_mlp,
        include_xgboost=include_xgboost,
        include_lightgbm=include_lightgbm,
        random_state=random_state,
        model_params=model_params,
    )
    if model_name not in models:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {sorted(models.keys())}"
        )

    preprocessor = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", models[model_name])])
    pipeline.fit(df[feature_cols].copy(), df[TARGET_COL].to_numpy())
    return pipeline, numeric_cols, categorical_cols


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    include_mlp: bool = False,
    include_torch_mlp: bool = False,
    include_xgboost: bool = False,
    include_lightgbm: bool = False,
    selected_features: Optional[list[str]] = None,
    split_strategy: str = "random",
    model_params: Optional[Dict[str, Dict[str, object]]] = None,
    model_names: Optional[list[str]] = None,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
) -> Dict[str, ModelArtifacts]:
    """Train model baselines and return metrics + diagnostics."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from dataset.")

    numeric_cols, categorical_cols = infer_feature_columns(df, target_col=TARGET_COL)
    if selected_features:
        keep = [c for c in selected_features if c in (numeric_cols + categorical_cols)]
        if not keep:
            raise ValueError("selected_features did not match any columns in dataset.")
        numeric_cols = [c for c in numeric_cols if c in keep]
        categorical_cols = [c for c in categorical_cols if c in keep]
    numeric_cols = [c for c in numeric_cols if c != "timestamp"]
    categorical_cols = [c for c in categorical_cols if c != "timestamp"]
    feature_cols = [c for c in (numeric_cols + categorical_cols) if c != "timestamp"]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].to_numpy()

    if train_indices is None or test_indices is None:
        train_indices, test_indices = get_train_test_indices(
            df=df,
            split_strategy=split_strategy,
            test_size=test_size,
            random_state=random_state,
        )
    X_train = X.iloc[train_indices].copy()
    y_train = y[train_indices]
    X_test = X.iloc[test_indices].copy()
    y_test = y[test_indices]

    preprocessor = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    models = _build_models(
        include_mlp=include_mlp,
        include_torch_mlp=include_torch_mlp,
        include_xgboost=include_xgboost,
        include_lightgbm=include_lightgbm,
        random_state=random_state,
        model_params=model_params,
    )
    if model_names:
        models = {k: v for k, v in models.items() if k in model_names}

    results: Dict[str, ModelArtifacts] = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
        feature_importance = _extract_feature_importance(pipeline, name, feature_names)
        perm = _safe_permutation_importance(
            pipeline=pipeline,
            X_test=X_test,
            y_test=y_test,
            random_state=random_state,
        )
        residuals = y_test - y_pred
        residual_diagnostics = {
            "residual_bias": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            "residual_p95_abs": float(np.percentile(np.abs(residuals), 95)),
        }
        results[name] = ModelArtifacts(
            model_name=name,
            pipeline=pipeline,
            metrics=metrics,
            y_test=y_test,
            y_pred=y_pred,
            X_test_raw=X_test,
            feature_names=feature_names,
            feature_importance=feature_importance,
            permutation_importance=perm,
            residual_diagnostics=residual_diagnostics,
            numeric_features=list(numeric_cols),
            categorical_features=list(categorical_cols),
        )
    return results
