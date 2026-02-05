from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import subprocess
from typing import Any, Dict

import joblib
import yaml

try:
    import structlog  # type: ignore
except Exception:  # pragma: no cover
    structlog = None  # type: ignore


def get_logger(name: str = "mlproj"):
    if structlog is not None:
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer(),
            ]
        )
        return structlog.get_logger(name)

    import logging
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    class _Adapter:
        def __init__(self, lg):
            self._lg = lg

        def info(self, event, **kw):
            self._lg.info(f"{event} {kw}" if kw else str(event))

        def warning(self, event, **kw):
            self._lg.warning(f"{event} {kw}" if kw else str(event))

        def error(self, event, **kw):
            self._lg.error(f"{event} {kw}" if kw else str(event))

    return _Adapter(logger)


def ensure_dir(path: str | os.PathLike) -> str:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _git_sha_short() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "nogit"


def make_run_id(run_name: str) -> str:
    ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M")
    sha = _git_sha_short()
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_name)[:40]
    return f"{ts}_{sha}_{safe}"


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_yaml(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_joblib(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    joblib.dump(obj, path)


def load_joblib(path: str) -> Any:
    return joblib.load(path)


def set_latest_pointer(artifacts_root: str, run_dir: str) -> None:
    latest_path = os.path.join(artifacts_root, "LATEST")
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(run_dir)


def resolve_latest(artifacts_root: str) -> str:
    latest_path = os.path.join(artifacts_root, "LATEST")
    if os.path.isdir(latest_path):  # pragma: no cover
        return latest_path
    with open(latest_path, "r", encoding="utf-8") as f:
        return f.read().strip()
