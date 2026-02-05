#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.mlproj.dataio import TARGET_COL, validate_input_df
from src.mlproj.metrics import ece_binary, js_divergence
from src.mlproj.monitor import (
    adversarial_auc,
    compute_feature_drift,
    plot_psi_bar,
    plot_reliability,
    plot_score_hist,
    severity_from_rules,
)
from src.mlproj.utils import ensure_dir, get_logger, load_joblib, read_yaml, resolve_latest

log = get_logger("monitor_run")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Artifact directory or artifacts/LATEST")
    p.add_argument("--reference-csv", required=True)
    p.add_argument("--current-csv", required=True)
    return p.parse_args()


def _load_cfg(model_dir: str) -> dict:
    cfg_path = Path(model_dir) / "config.lock.yaml"
    return read_yaml(str(cfg_path)) if cfg_path.exists() else {}


def _get_preprocessor(bundle):
    """
    Try to locate the fitted preprocessor to transform X for adversarial AUC.
    Returns a callable transform(X) -> array or None.
    """
    candidates = [
        lambda b: b.pipeline.base_estimator.named_steps["pre"],  # CalibratedClassifierCV(base_estimator=pipe)
        lambda b: b.pipeline.base_estimator.base_estimator.named_steps["pre"],  # nested base_estimator
        lambda b: b.pipeline.base_estimator,  # maybe a Pipeline
        lambda b: getattr(b.pipeline, "base_estimator", None),
    ]
    for fn in candidates:
        try:
            pre = fn(bundle)
            if pre is None:
                continue
            # must have transform
            if hasattr(pre, "transform"):
                return pre
        except Exception:
            continue
    return None


def main(argv=None) -> int:
    args = parse_args()

    model_dir = args.model_dir
    if model_dir.endswith("LATEST"):
        model_dir = resolve_latest(str(Path(model_dir).parent))

    bundle = load_joblib(str(Path(model_dir) / "model.joblib"))
    cfg = _load_cfg(model_dir)

    ref = pd.read_csv(args.reference_csv)
    cur = pd.read_csv(args.current_csv)

    has_target_ref = TARGET_COL in ref.columns
    has_target_cur = TARGET_COL in cur.columns

    ref_X = ref.drop(columns=[TARGET_COL]) if has_target_ref else ref
    cur_X = cur.drop(columns=[TARGET_COL]) if has_target_cur else cur

    validate_input_df(ref_X, require_target=False)
    validate_input_df(cur_X, require_target=False)

    p_ref = bundle.predict_proba(ref_X)[:, 1]
    p_cur = bundle.predict_proba(cur_X)[:, 1]

    mon_cfg = cfg.get("monitoring", {}) or {}
    psi_bins = int(mon_cfg.get("psi_bins", 10))

    df_psi = compute_feature_drift(ref_X, cur_X, top_k=10, psi_bins=psi_bins)
    psi_max = float(df_psi["psi"].max()) if len(df_psi) else 0.0

    pre = _get_preprocessor(bundle)
    if pre is not None:
        try:
            X_ref_t = pre.transform(ref_X)
            X_cur_t = pre.transform(cur_X)
        except Exception:
            X_ref_t = ref_X.select_dtypes(exclude=["object"]).to_numpy()
            X_cur_t = cur_X.select_dtypes(exclude=["object"]).to_numpy()
    else:
        X_ref_t = ref_X.select_dtypes(exclude=["object"]).to_numpy()
        X_cur_t = cur_X.select_dtypes(exclude=["object"]).to_numpy()

    adv = float(adversarial_auc(np.asarray(X_ref_t), np.asarray(X_cur_t)))
    score_js = float(js_divergence(p_ref, p_cur, n_bins=50))

    drift = {
        "psi_max": psi_max,
        "adv_auc": adv,
        "score_js": score_js,
        # keep these for severity rules compatibility
        "delta_ece": 0.0,
        "delta_brier": 0.0,
        # extra fields for reporting
        "ece_ref": None,
        "ece_cur": None,
        "brier_ref": None,
        "brier_cur": None,
    }

    # Calibration / probability quality (correct reference vs current comparison)
    if has_target_ref:
        y_ref = ref[TARGET_COL].astype(int).values
        drift["ece_ref"] = float(ece_binary(y_ref, p_ref))
        drift["brier_ref"] = float(brier_score_loss(y_ref, p_ref))

    if has_target_cur:
        y_cur = cur[TARGET_COL].astype(int).values
        drift["ece_cur"] = float(ece_binary(y_cur, p_cur))
        drift["brier_cur"] = float(brier_score_loss(y_cur, p_cur))

    if has_target_ref and has_target_cur:
        drift["delta_ece"] = float(drift["ece_cur"] - drift["ece_ref"])
        drift["delta_brier"] = float(drift["brier_cur"] - drift["brier_ref"])

    severity = severity_from_rules(drift, cfg)

    reports_root = cfg.get("reports_dir", "reports/")
    out_dir = Path(reports_root) / "monitor" / Path(model_dir).name
    ensure_dir(str(out_dir))

    plot_score_hist(p_ref, p_cur, str(out_dir / "score_hist.png"))
    plot_psi_bar(df_psi, str(out_dir / "psi_top.png"))

    if has_target_cur:
        plot_reliability(cur[TARGET_COL].values, p_cur, str(out_dir / "reliability_current.png"))

    # Markdown report (ref/cur shown separately; delta only when valid)
    md = []
    md.append("# Offline monitoring report\n")
    md.append(f"**Model dir:** `{model_dir}`\n")
    md.append(f"**Severity:** **{severity}**\n")

    md.append("## Drift summary\n")
    md.append(f"- PSI max (top-10 features): `{drift['psi_max']:.4f}`\n")
    md.append(f"- Adversarial AUC: `{drift['adv_auc']:.4f}`\n")
    md.append(f"- Score JS divergence: `{drift['score_js']:.4f}`\n")

    if has_target_ref or has_target_cur:
        md.append("\n## Calibration / probability quality\n")
        if has_target_ref:
            md.append(f"- ECE (ref): `{drift['ece_ref']:.4f}`\n")
            md.append(f"- Brier (ref): `{drift['brier_ref']:.4f}`\n")
        else:
            md.append("- ECE (ref): `n/a` (no target in reference)\n")
            md.append("- Brier (ref): `n/a` (no target in reference)\n")

        if has_target_cur:
            md.append(f"- ECE (cur): `{drift['ece_cur']:.4f}`\n")
            md.append(f"- Brier (cur): `{drift['brier_cur']:.4f}`\n")
        else:
            md.append("- ECE (cur): `n/a` (no target in current)\n")
            md.append("- Brier (cur): `n/a` (no target in current)\n")

        if has_target_ref and has_target_cur:
            md.append(f"- ΔECE (cur - ref): `{drift['delta_ece']:.4f}`\n")
            md.append(f"- ΔBrier (cur - ref): `{drift['delta_brier']:.4f}`\n")
        else:
            md.append("- ΔECE / ΔBrier: `n/a` (delta requires target in both ref and cur)\n")

    md.append("\n## Top PSI features\n")
    md.append(df_psi.to_markdown(index=False))

    md.append("\n\n## Plots\n")
    md.append("- Score histogram: `score_hist.png`\n")
    md.append("- PSI bar: `psi_top.png`\n")
    if has_target_cur:
        md.append("- Reliability (current): `reliability_current.png`\n")

    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    log.info("monitor_done", out_dir=str(out_dir), severity=severity, drift=drift)

    return 2 if severity == "MAJOR" else 1 if severity == "MINOR" else 0


if __name__ == "__main__":
    sys.exit(main())