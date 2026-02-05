from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pandas.api.types import is_numeric_dtype

from .metrics import expected_cost
from .utils import get_logger
from .dataio import REQUIRED_COLUMNS  # чтобы понять, что это Adult

log = get_logger("mlproj.pipeline")


# Явное разнесение по Adult (надёжнее, чем dtype-эвристики)
ADULT_NUM_COLS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
ADULT_CAT_COLS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq: float = 0.01, rare_label: str = "__RARE__"):
        self.min_freq = float(min_freq)
        self.rare_label = rare_label
        self.keep_levels_: dict = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.keep_levels_ = {}
        for col in X.columns:
            vc = X[col].astype("object").fillna("__NA__").value_counts(normalize=True)
            self.keep_levels_[col] = set(vc[vc >= self.min_freq].index.tolist())
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            keep = self.keep_levels_.get(col, set())
            s = X[col].astype("object").fillna("__NA__")
            X[col] = np.where(s.isin(list(keep)), s, self.rare_label)
        return X


def _infer_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Возвращает (num_cols, cat_cols) безопасным способом.

    1) Для Adult — используем явные списки, чтобы не зависеть от dtype (object/category).
    2) Для других датасетов — numeric -> num, всё остальное -> cat.
    """
    cols = list(X.columns)

    is_adult_like = all(c in cols for c in REQUIRED_COLUMNS)
    if is_adult_like:
        num_cols = [c for c in ADULT_NUM_COLS if c in cols]
        cat_cols = [c for c in ADULT_CAT_COLS if c in cols]
        # На всякий случай: всё неизвестное — в cat (чтобы не словить median на строке)
        leftover = [c for c in cols if c not in set(num_cols + cat_cols)]
        cat_cols = cat_cols + leftover
        return num_cols, cat_cols

    # Fallback-эвристика для любых других таблиц
    num_cols = [c for c in cols if is_numeric_dtype(X[c])]
    cat_cols = [c for c in cols if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(X: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[ColumnTransformer, List[str]]:
    num_cols, cat_cols = _infer_columns(X)
    rare_min_freq = float(cfg.get("preprocessing", {}).get("rare_min_freq", 0.01))

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("rare", RareCategoryGrouper(min_freq=rare_min_freq)),
            ("ohe", ohe),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, (num_cols + cat_cols)


def build_model(cfg: Dict[str, Any]) -> BaseEstimator:
    m = cfg.get("model", {}) or {}
    mtype = m.get("type", "hgb")
    params = m.get("params", {}) or {}

    if mtype == "logreg":
        default = dict(C=1.0, max_iter=1000, solver="lbfgs")
        default.update(params)
        return LogisticRegression(**default)

    if mtype == "hgb":
        default = dict(
            max_depth=6,
            learning_rate=0.06,
            max_leaf_nodes=31,
            random_state=cfg.get("seed", 42),
        )
        default.update(params)
        return HistGradientBoostingClassifier(**default)

    raise ValueError(f"Unknown model.type={mtype}")


class TemperatureScaler(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator: BaseEstimator):
        self.base_estimator = base_estimator
        self.temperature_: float = 1.0

    def fit(self, X, y):
        proba = self.base_estimator.predict_proba(X)[:, 1]
        proba = np.clip(proba, 1e-12, 1 - 1e-12)
        logits = np.log(proba / (1 - proba))

        Ts = np.linspace(0.5, 5.0, 91)
        best_ll = float("inf")
        best_T = 1.0
        for T in Ts:
            p = 1 / (1 + np.exp(-logits / T))
            ll = log_loss(y, np.vstack([1 - p, p]).T, labels=[0, 1])
            if ll < best_ll:
                best_ll = ll
                best_T = float(T)

        self.temperature_ = best_T
        return self

    def predict_proba(self, X):
        proba = self.base_estimator.predict_proba(X)[:, 1]
        proba = np.clip(proba, 1e-12, 1 - 1e-12)
        logits = np.log(proba / (1 - proba))
        p = 1 / (1 + np.exp(-logits / self.temperature_))
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@dataclass
class ModelBundle:
    pipeline: BaseEstimator
    threshold: float
    manual_band: Tuple[float, float]
    meta: Dict[str, Any]
    feature_names: List[str]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict_with_reject(self, X: pd.DataFrame):
        proba = self.predict_proba(X)[:, 1]
        low, high = self.manual_band
        decision = np.full_like(proba, fill_value=-1, dtype=int)
        reason = np.array(["manual_review"] * len(proba), dtype=object)

        auto_mask = (proba < low) | (proba > high)
        decision[auto_mask] = (proba[auto_mask] >= self.threshold).astype(int)
        reason[auto_mask] = np.where(proba[auto_mask] >= self.threshold, "auto", "below_threshold")
        return proba, decision, reason


def fit_bundle(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cfg: Dict[str, Any],
) -> ModelBundle:
    pre, _ = build_preprocessor(X_train, cfg)
    model = build_model(cfg)
    base_pipe = Pipeline([("pre", pre), ("model", model)])

    cal = cfg.get("calibration", {}) or {}
    cal_type = cal.get("type", "temperature")

    if cal_type in ("platt", "isotonic"):
        method = "sigmoid" if cal_type == "platt" else "isotonic"
        pipe = CalibratedClassifierCV(base_estimator=base_pipe, method=method, cv=3)
        pipe.fit(X_train, y_train)
        cal_meta = {"type": cal_type, "cv": 3}
        fitted_pre = pipe.base_estimator.named_steps["pre"]

    elif cal_type == "temperature":
        base_pipe.fit(X_train, y_train)
        pipe = TemperatureScaler(base_pipe).fit(X_valid, y_valid)
        cal_meta = {"type": "temperature"}
        fitted_pre = base_pipe.named_steps["pre"]

    else:
        raise ValueError(f"Unknown calibration.type={cal_type}")

    # feature names
    try:
        feature_names = list(fitted_pre.get_feature_names_out())
    except Exception:
        feature_names = list(X_train.columns)

    # threshold tuning on valid
    costs = cfg.get("costs", {}) or {}
    cost_fp = float(costs.get("fp", 1.0))
    cost_fn = float(costs.get("fn", 5.0))
    p_valid = pipe.predict_proba(X_valid)[:, 1]

    grid = np.linspace(0.01, 0.99, 99)
    best_cost = float("inf")
    best_thr = float(cfg.get("thresholding", {}).get("decision_threshold", 0.5))
    for t in grid:
        c = expected_cost(y_valid.values, p_valid, float(t), cost_fp=cost_fp, cost_fn=cost_fn)
        if c < best_cost:
            best_cost = c
            best_thr = float(t)

    mb = cfg.get("thresholding", {}).get("manual_band", [best_thr, best_thr])
    if not isinstance(mb, (list, tuple)) or len(mb) != 2:
        mb = [best_thr, best_thr]
    manual_band = (float(mb[0]), float(mb[1]))

    meta = {"calibration": cal_meta, "threshold_cost_valid": best_cost}
    return ModelBundle(
        pipeline=pipe,
        threshold=best_thr,
        manual_band=manual_band,
        meta=meta,
        feature_names=feature_names,
    )