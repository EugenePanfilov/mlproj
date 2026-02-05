from pathlib import Path
import tempfile
import yaml
import numpy as np

import train as train_mod
from src.mlproj.utils import load_joblib
from src.mlproj.dataio import TARGET_COL
from tests._synthetic import make_synth

def test_serialize_predictions_identical(monkeypatch):
    df = make_synth(n=1400)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    monkeypatch.setattr(train_mod, "load_adult_openml", lambda seed=42: (X, y))

    with tempfile.TemporaryDirectory() as td:
        cfg = {
            "seed": 42,
            "dataset": "adult",
            "split": {"mode": "random", "valid_size": 0.2},
            "preprocessing": {"rare_min_freq": 0.02},
            "model": {"type": "logreg", "params": {"max_iter": 300}},
            "calibration": {"type": "platt"},
            "thresholding": {"decision_threshold": 0.5, "manual_band": [0.3, 0.7]},
            "costs": {"fp": 1.0, "fn": 5.0},
            "monitoring": {"psi_bins": 10},
            "artifacts_dir": str(Path(td)/"artifacts"),
            "reports_dir": str(Path(td)/"reports"),
        }
        cfg_path = Path(td)/"cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        import sys
        sys.argv = ["train.py", "--config", str(cfg_path), "--run-name", "ser"]
        assert train_mod.main() == 0

        run_dir = Path((Path(cfg["artifacts_dir"])/"LATEST").read_text().strip())
        b1 = load_joblib(str(run_dir/"model.joblib"))
        b2 = load_joblib(str(run_dir/"model.joblib"))
        p1 = b1.predict_proba(X.head(100))[:, 1]
        p2 = b2.predict_proba(X.head(100))[:, 1]
        assert np.max(np.abs(p1 - p2)) <= 1e-12
