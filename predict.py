#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.mlproj.dataio import validate_input_df, TARGET_COL
from src.mlproj.utils import get_logger, load_joblib, resolve_latest
from src.mlproj.pipeline import ModelBundle

log = get_logger("predict")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Artifact directory or artifacts/LATEST")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-csv", required=True)
    return p.parse_args()


def main(argv=None) -> int:
    args = parse_args()

    model_dir = args.model_dir
    if model_dir.endswith("LATEST"):
        # if user passed artifacts/LATEST, resolve content
        root = str(Path(model_dir).parent)
        model_dir = resolve_latest(root)

    bundle: ModelBundle = load_joblib(str(Path(model_dir) / "model.joblib"))

    df = pd.read_csv(args.input_csv)
    # target not required for predict
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    validate_input_df(df, require_target=False)

    proba, decision, reason = bundle.predict_with_reject(df)

    out = df.copy()
    out["proba"] = proba
    out["decision"] = decision
    out["decision_reason"] = reason

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    log.info("predict_done", n=len(out), output=args.output_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
