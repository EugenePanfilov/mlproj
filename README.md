# mlproj — ml-drift-monitor (Adult / OpenML)

This repo implements a minimal but “alive” production-like contour:

**train → artifacts → batch predict → offline drift/calibration monitoring → CI**

## Requirements
- Python 3.11+
- Internet access for training (Adult is loaded via `fetch_openml`)

## Install
```bash
pip install -r requirements.txt
```

## Train
```bash
python train.py --config configs/config.yaml --run-name exp1
```

Artifacts are stored under:
```
artifacts/{YYYYMMDD_HHMM}_{gitsha}_{run-name}/
  model.joblib
  feature_names.json
  metrics.json
  config.lock.yaml
artifacts/LATEST   # pointer to the latest run dir
```

## Batch predict (CSV → CSV)
```bash
python predict.py --model-dir artifacts/LATEST --input-csv input.csv --output-csv out.csv
```

Outputs:
- `proba`
- `decision` (0/1) or `-1` (manual review)
- `decision_reason` (`auto|manual_review|below_threshold`)

## Offline monitoring
```bash
python monitor_run.py --model-dir artifacts/LATEST --reference-csv ref.csv --current-csv cur.csv
```

Generates:
```
reports/monitor/<run_id>/
  report.md
  score_hist.png
  psi_top.png
  reliability_current.png   # only if current has target
```

Exit codes:
- `0` OK
- `1` MINOR
- `2` MAJOR

## Makefile
```bash
make install
make train
make predict
make monitor
make test
```
