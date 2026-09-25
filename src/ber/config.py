import json
from pathlib import Path

import yaml


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if "base" in cfg:
        parent = load_config(cfg.pop("base"))
        cfg = _deep_merge(parent, cfg)
    return cfg


def run_dir(cfg: dict) -> Path:
    d = Path(cfg["paths"]["artifacts"]) / cfg["run_name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_metrics(cfg: dict, section: str, data: dict) -> None:
    f = run_dir(cfg) / "metrics.json"
    metrics = json.loads(f.read_text()) if f.exists() else {}
    metrics[section] = data
    f.write_text(json.dumps(metrics, indent=2, sort_keys=True))
