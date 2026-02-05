from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .metrics import psi_numeric, js_divergence, ece_binary
from .utils import ensure_dir, get_logger

log = get_logger("mlproj.monitor")


def adversarial_auc(X_ref_t: np.ndarray, X_cur_t: np.ndarray, seed: int = 42) -> float:
    y = np.concatenate([np.zeros(len(X_ref_t)), np.ones(len(X_cur_t))]).astype(int)
    X = np.vstack([X_ref_t, X_cur_t])
    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, p))


def compute_feature_drift(X_ref: pd.DataFrame, X_cur: pd.DataFrame, top_k: int = 10, psi_bins: int = 10) -> pd.DataFrame:
    rows = []
    for col in X_ref.columns:
        a = X_ref[col]
        b = X_cur[col]
        if a.dtype == "object":
            ref_v = a.astype("object").fillna("__NA__").value_counts(normalize=True)
            cur_v = b.astype("object").fillna("__NA__").value_counts(normalize=True)
            idx = sorted(set(ref_v.index).union(set(cur_v.index)))
            rp = np.array([ref_v.get(i, 0.0) for i in idx])
            cp = np.array([cur_v.get(i, 0.0) for i in idx])
            rp = np.clip(rp, 1e-8, 1)
            cp = np.clip(cp, 1e-8, 1)
            psi = float(np.sum((cp - rp) * np.log(cp / rp)))
        else:
            psi = psi_numeric(a.values, b.values, n_bins=psi_bins)
        rows.append((col, psi))
    return pd.DataFrame(rows, columns=["feature", "psi"]).sort_values("psi", ascending=False).head(top_k)


def plot_score_hist(p_ref: np.ndarray, p_cur: np.ndarray, out_png: str) -> None:
    ensure_dir(str(Path(out_png).parent))
    plt.figure()
    plt.hist(p_ref, bins=30, alpha=0.6, label="reference", density=True)
    plt.hist(p_cur, bins=30, alpha=0.6, label="current", density=True)
    plt.xlabel("score (proba)")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_psi_bar(df_psi: pd.DataFrame, out_png: str) -> None:
    ensure_dir(str(Path(out_png).parent))
    plt.figure()
    plt.barh(df_psi["feature"][::-1], df_psi["psi"][::-1])
    plt.xlabel("PSI")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_reliability(y: np.ndarray, p: np.ndarray, out_png: str, n_bins: int = 10) -> None:
    ensure_dir(str(Path(out_png).parent))
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        xs.append(float(p[m].mean()))
        ys.append(float(y[m].mean()))
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.plot(xs, ys, marker="o")
    plt.xlabel("mean predicted proba")
    plt.ylabel("empirical positive rate")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def severity_from_rules(drift: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    mon = cfg.get("monitoring", {}) or {}
    psi_major = float(mon.get("psi_major", 0.25))
    adv_auc_major = float(mon.get("adv_auc_major", 0.70))
    ece_minor = float(mon.get("ece_drift_minor", 0.05))
    brier_minor = float(mon.get("brier_drift_minor", 0.01))

    major = False
    minor = False
    if drift.get("psi_max", 0.0) >= psi_major:
        major = True
    if drift.get("adv_auc", 0.0) >= adv_auc_major:
        major = True
    if abs(drift.get("delta_ece", 0.0)) >= ece_minor:
        minor = True
    if abs(drift.get("delta_brier", 0.0)) >= brier_minor:
        minor = True

    if major:
        return "MAJOR"
    if minor:
        return "MINOR"
    return "OK"
