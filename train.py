#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.mlproj.dataio import load_adult_openml, validate_input_df, split_holdout, TARGET_COL
from src.mlproj.pipeline import fit_bundle
from src.mlproj.metrics import classification_metrics, expected_cost
from src.mlproj.utils import get_logger, make_run_id, ensure_dir, write_json, write_yaml, save_joblib, read_yaml, set_latest_pointer

log = get_logger("train")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-name", default="run")
    p.add_argument("--save-dir", default=None, help="Override artifacts_dir from config")
    return p.parse_args()


def main(argv=None) -> int:
    args = parse_args()
    cfg = read_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    if cfg.get("dataset") != "adult":
        raise SystemExit("Only dataset=adult is supported (uses fetch_openml; internet required).")

    X, y = load_adult_openml(seed=seed)
    df = X.copy()
    df[TARGET_COL] = y.values
    validate_input_df(df, require_target=True)

    split = cfg.get("split", {}) or {}
    mode = split.get("mode", "temporal_holdout")
    valid_size = float(split.get("valid_size", 0.2))
    X_tr, X_va, y_tr, y_va = split_holdout(X, y, mode=mode, valid_size=valid_size, seed=seed)

    bundle = fit_bundle(X_tr, y_tr, X_va, y_va, cfg)

    p_va = bundle.predict_proba(X_va)[:, 1]
    mets = classification_metrics(y_va.values, p_va)
    costs = cfg.get("costs", {}) or {}
    ec = expected_cost(y_va.values, p_va, bundle.threshold, float(costs.get("fp", 1.0)), float(costs.get("fn", 5.0)))
    mets["expected_cost@threshold"] = float(ec)
    mets["threshold"] = float(bundle.threshold)
    mets["manual_band_low"] = float(bundle.manual_band[0])
    mets["manual_band_high"] = float(bundle.manual_band[1])

    artifacts_root = args.save_dir or cfg.get("artifacts_dir", "artifacts/")
    artifacts_root = str(Path(artifacts_root))
    ensure_dir(artifacts_root)

    run_id = make_run_id(args.run_name)
    run_dir = str(Path(artifacts_root) / run_id)
    ensure_dir(run_dir)

    # lock config
    cfg_lock = dict(cfg)
    cfg_lock["resolved"] = {
        "run_id": run_id,
        "threshold": bundle.threshold,
        "manual_band": list(bundle.manual_band),
        "feature_names_n": len(bundle.feature_names),
        "meta": bundle.meta,
    }

    save_joblib(str(Path(run_dir) / "model.joblib"), bundle)
    write_json(str(Path(run_dir) / "feature_names.json"), bundle.feature_names)
    write_json(str(Path(run_dir) / "metrics.json"), mets)
    write_yaml(str(Path(run_dir) / "config.lock.yaml"), cfg_lock)

    set_latest_pointer(artifacts_root, run_dir)

    log.info("train_done", run_dir=run_dir, metrics=mets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
