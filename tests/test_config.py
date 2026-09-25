import json
from pathlib import Path

from ber.cli import main
from ber.config import load_config, log_metrics, run_dir


def test_base_inheritance_deep_merges(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("run_name: a\ntrain:\n  batch_size: 256\n  max_steps: 3000\npaths:\n  artifacts: " + str(tmp_path / "art") + "\n")
    child = tmp_path / "child.yaml"
    child.write_text(f"base: {base}\nrun_name: b\ntrain:\n  max_steps: 20\n")
    cfg = load_config(child)
    assert cfg["run_name"] == "b"
    assert cfg["train"] == {"batch_size": 256, "max_steps": 20}
    assert "base" not in cfg


def test_run_dir_and_metrics(tmp_path):
    cfg = {"run_name": "r", "paths": {"artifacts": str(tmp_path)}}
    d = run_dir(cfg)
    assert d == tmp_path / "r" and d.is_dir()
    log_metrics(cfg, "blocking", {"pc": 0.99})
    log_metrics(cfg, "split", {"n": 3})
    assert json.loads((d / "metrics.json").read_text()) == {"blocking": {"pc": 0.99}, "split": {"n": 3}}


def test_cli_show_config(capsys):
    main(["--config", "configs/base.yaml", "show-config"])
    assert json.loads(capsys.readouterr().out)["run_name"] == "v1"
