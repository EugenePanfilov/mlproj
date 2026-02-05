from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss


def ece_binary(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        acc = y_true[m].mean()
        conf = p[m].mean()
        ece += (m.mean()) * abs(acc - conf)
    return float(ece)


def expected_cost(y_true: np.ndarray, p: np.ndarray, threshold: float, cost_fp: float, cost_fn: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p)
    pred = (p >= threshold).astype(int)
    fp = ((pred == 1) & (y_true == 0)).mean()
    fn = ((pred == 0) & (y_true == 1)).mean()
    return float(cost_fp * fp + cost_fn * fn)


def classification_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "logloss": float(log_loss(y_true, np.vstack([1 - p, p]).T, labels=[0, 1])),
        "ece": float(ece_binary(y_true, p)),
    }


def psi_numeric(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    qs = np.linspace(0, 1, n_bins + 1)
    cuts = np.quantile(ref, qs)
    cuts = np.unique(cuts)
    if len(cuts) < 3:
        return 0.0
    ref_counts, _ = np.histogram(ref, bins=cuts)
    cur_counts, _ = np.histogram(cur, bins=cuts)
    ref_p = ref_counts / max(1, ref_counts.sum())
    cur_p = cur_counts / max(1, cur_counts.sum())
    ref_p = np.clip(ref_p, 1e-8, 1)
    cur_p = np.clip(cur_p, 1e-8, 1)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def js_divergence(p: np.ndarray, q: np.ndarray, n_bins: int = 50) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    lo = float(min(np.min(p), np.min(q)))
    hi = float(max(np.max(p), np.max(q)))
    if hi <= lo:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    ph, _ = np.histogram(p, bins=bins, density=True)
    qh, _ = np.histogram(q, bins=bins, density=True)
    ph = ph / max(ph.sum(), 1e-12)
    qh = qh / max(qh.sum(), 1e-12)
    ph = np.clip(ph, 1e-12, 1)
    qh = np.clip(qh, 1e-12, 1)
    return float(jensenshannon(ph, qh, base=2.0) ** 2)
