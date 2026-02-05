from pathlib import Path
import tempfile
import yaml
import pandas as pd

import train as train_mod
import predict as pred_mod
from src.mlproj.dataio import TARGET_COL
from tests._synthetic import make_synth

def test_smoke_train_serialize_predict(monkeypatch):
    df = make_synth(n=1100)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # monkeypatch OpenML loader
    monkeypatch.setattr(train_mod, "load_adult_openml", lambda seed=42: (X, y))

    with tempfile.TemporaryDirectory() as td:
        cfg = {
            "seed": 42,
            "dataset": "adult",
            "split": {"mode": "random", "valid_size": 0.2},
            "preprocessing": {"rare_min_freq": 0.01},
            "model": {"type": "logreg", "params": {"max_iter": 200}},
            "calibration": {"type": "platt"},
            "thresholding": {"decision_threshold": 0.5, "manual_band": [0.2, 0.8]},
            "costs": {"fp": 1.0, "fn": 5.0},
            "monitoring": {"psi_bins": 10},
            "artifacts_dir": str(Path(td)/"artifacts"),
            "reports_dir": str(Path(td)/"reports"),
        }
        cfg_path = Path(td)/"cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        # run train
        import sys
        sys.argv = ["train.py", "--config", str(cfg_path), "--run-name", "smoke"]
        rc = train_mod.main()
        assert rc == 0

        latest = Path(cfg["artifacts_dir"])/"LATEST"
        run_dir = Path(latest.read_text().strip())
        assert (run_dir/"model.joblib").exists()

        # make input csv
        inp = X.head(50).copy()
        in_path = Path(td)/"in.csv"
        out_path = Path(td)/"out.csv"
        inp.to_csv(in_path, index=False)

        sys.argv = ["predict.py", "--model-dir", str(Path(cfg["artifacts_dir"])/"LATEST"), "--input-csv", str(in_path), "--output-csv", str(out_path)]
        rc2 = pred_mod.main()
        assert rc2 == 0
        out = pd.read_csv(out_path)
        assert "proba" in out.columns and "decision" in out.columns and "decision_reason" in out.columns
