from __future__ import annotations

from typing import Tuple

import pandas as pd
import pandera.pandas as pa
from sklearn.datasets import fetch_openml

from .utils import get_logger

log = get_logger("mlproj.dataio")

TARGET_COL = "target"

REQUIRED_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]

# Категориальные колонки Adult
CAT_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def adult_schema(require_target: bool) -> pa.DataFrameSchema:
    """
    Pandera-схема входных данных (строгая по колонкам).
    Категориальные признаки валидируем как `object`.
    Перед валидацией приводим их к `object`, чтобы не зависеть от того,
    пришли они как pandas `string`, `category` или обычные строки.
    """
    cols = {
        "age": pa.Column(int, pa.Check.between(17, 99), nullable=False),
        "workclass": pa.Column(object, nullable=True),
        "fnlwgt": pa.Column(int, pa.Check.ge(1), nullable=False),
        "education": pa.Column(object, nullable=True),
        "education-num": pa.Column(int, pa.Check.between(1, 16), nullable=False),
        "marital-status": pa.Column(object, nullable=True),
        "occupation": pa.Column(object, nullable=True),
        "relationship": pa.Column(object, nullable=True),
        "race": pa.Column(object, nullable=True),
        "sex": pa.Column(object, nullable=True),
        "capital-gain": pa.Column(int, pa.Check.between(0, 100000), nullable=False),
        "capital-loss": pa.Column(int, pa.Check.between(0, 5000), nullable=False),
        "hours-per-week": pa.Column(int, pa.Check.between(1, 99), nullable=False),
        "native-country": pa.Column(object, nullable=True),
    }
    if require_target:
        cols[TARGET_COL] = pa.Column(int, pa.Check.isin([0, 1]), nullable=False)
    return pa.DataFrameSchema(cols, strict=True)


def load_adult_openml(seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    # Internet required.
    ds = fetch_openml(name="adult", version=2, as_frame=True)
    df = ds.frame.copy()
    if "class" not in df.columns:
        raise RuntimeError("OpenML adult dataset missing 'class' column.")

    y = (df["class"].astype(str).str.strip() == ">50K").astype(int).rename(TARGET_COL)
    X = df.drop(columns=["class"])
    X.columns = [c.strip() for c in X.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in X.columns]
    if missing:
        raise RuntimeError(f"Adult dataset missing required columns: {missing}")

    X = X[REQUIRED_COLUMNS].copy()
    return X, y


def validate_input_df(df: pd.DataFrame, require_target: bool, missing_warn_pct: float = 0.2) -> None:
    """
    Валидация входного датафрейма:
    - нормализация dtype категориальных колонок к `object` (string/category -> object)
    - проверка схемы pandera (типы/диапазоны/обязательные колонки)
    - warning при высокой доле пропусков
    """
    # Нормализуем категориальные колонки: в CI они часто приходят как pandas "string"
    for c in CAT_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype("object")

    schema = adult_schema(require_target=require_target)
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        raise ValueError(str(e.failure_cases.head(50))) from e

    miss = df.isna().mean()
    high = miss[miss > missing_warn_pct]
    if len(high) > 0:
        log.warning("high_missing_rate", columns=high.to_dict(), threshold=missing_warn_pct)


def split_holdout(X: pd.DataFrame, y: pd.Series, mode: str, valid_size: float, seed: int = 42):
    n = len(X)
    n_valid = int(round(n * valid_size))
    n_valid = max(1, min(n - 1, n_valid))

    if mode == "temporal_holdout":
        return (
            X.iloc[:-n_valid].copy(),
            X.iloc[-n_valid:].copy(),
            y.iloc[:-n_valid].copy(),
            y.iloc[-n_valid:].copy(),
        )

    if mode == "random":
        from sklearn.model_selection import train_test_split

        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=valid_size, random_state=seed, stratify=y
        )
        return X_tr, X_va, y_tr, y_va

    raise ValueError(f"Unknown split.mode={mode}")