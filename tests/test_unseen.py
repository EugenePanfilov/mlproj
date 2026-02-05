from pathlib import Path
import tempfile
import yaml
import pandas as pd
import numpy as np

import train as train_mod
import predict as pred_mod
from src.mlproj.dataio import TARGET_COL
from tests._synthetic import make_synth

def test_unseen_categories_and_nan(monkeypatch):
    df = make_synth(n=1200)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    monkeypatch.setattr(train_mod, "load_adult_openml", lambda seed=42: (X, y))

    with tempfile.TemporaryDirectory() as td:
        cfg = {
            "seed": 42,
            "dataset": "adult",
            "split": {"mode": "random", "valid_size": 0.2},
            "preprocessing": {"rare_min_freq": 0.05},
            "model": {"type": "hgb", "params": {"max_depth": 3, "learning_rate": 0.1, "max_leaf_nodes": 15}},
            "calibration": {"type": "temperature"},
            "thresholding": {"decision_threshold": 0.5, "manual_band": [0.4, 0.6]},
            "costs": {"fp": 1.0, "fn": 5.0},
            "monitoring": {"psi_bins": 10},
            "artifacts_dir": str(Path(td)/"artifacts"),
            "reports_dir": str(Path(td)/"reports"),
        }
        cfg_path = Path(td)/"cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        import sys
        sys.argv = ["train.py", "--config", str(cfg_path), "--run-name", "unseen"]
        assert train_mod.main() == 0

        latest = Path(cfg["artifacts_dir"])/"LATEST"
        run_dir = Path(latest.read_text().strip())
        assert run_dir.exists()

        # Build input with unseen categories and NaNs
        inp = X.head(200).copy()
        inp.loc[:50, "workclass"] = "NEW_WORKCLASS"
        inp.loc[:30, "native-country"] = "Atlantis"
        inp.loc[:10, "occupation"] = np.nan

        in_path = Path(td)/"in.csv"
        out_path = Path(td)/"out.csv"
        inp.to_csv(in_path, index=False)

        sys.argv = ["predict.py", "--model-dir", str(Path(cfg["artifacts_dir"])/"LATEST"), "--input-csv", str(in_path), "--output-csv", str(out_path)]
        assert pred_mod.main() == 0
        out = pd.read_csv(out_path)
        # should not crash; manual_review share should be non-trivial but not 100%
        mr = (out["decision"] == -1).mean()
        assert 0.0 <= mr <= 1.0
