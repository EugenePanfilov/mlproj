import numpy as np
import pandas as pd

from src.mlproj.dataio import REQUIRED_COLUMNS, TARGET_COL

def make_synth(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(17, 80, size=n),
        "workclass": rng.choice(["Private", "Self-emp", "Gov", None], size=n, p=[0.6,0.15,0.2,0.05]),
        "fnlwgt": rng.integers(10000, 500000, size=n),
        "education": rng.choice(["HS-grad","Bachelors","Masters","Some-college",None], size=n),
        "education-num": rng.integers(1, 16, size=n),
        "marital-status": rng.choice(["Never-married","Married","Divorced",None], size=n),
        "occupation": rng.choice(["Tech","Sales","Exec","Service",None], size=n),
        "relationship": rng.choice(["Husband","Not-in-family","Own-child","Unmarried",None], size=n),
        "race": rng.choice(["White","Black","Asian-Pac-Islander","Other",None], size=n),
        "sex": rng.choice(["Male","Female",None], size=n, p=[0.49,0.49,0.02]),
        "capital-gain": rng.integers(0, 10000, size=n),
        "capital-loss": rng.integers(0, 2000, size=n),
        "hours-per-week": rng.integers(1, 80, size=n),
        "native-country": rng.choice(["United-States","Mexico","Canada","?",None], size=n),
    })
    # simple target with some signal
    score = (
        0.03*(df["age"].fillna(0)) +
        0.02*(df["hours-per-week"].fillna(0)) +
        0.0001*(df["capital-gain"].fillna(0)) -
        0.0002*(df["capital-loss"].fillna(0)) +
        rng.normal(0, 1.0, size=n)
    )
    p = 1/(1+np.exp(-0.03*(score-2)))
    y = (rng.random(n) < p).astype(int)
    df[TARGET_COL] = y
    return df
