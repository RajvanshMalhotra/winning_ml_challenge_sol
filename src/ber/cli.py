import argparse
import importlib
import json

from ber.config import load_config

# name -> (module, help). Each module defines add_arguments(parser) and run(cfg, args).
COMMANDS: dict[str, tuple[str, str]] = {
    "show-config": ("ber.cli", "print the resolved config"),
    "normalize": ("ber.normalize", "build records_{family}.parquet"),
    "split": ("ber.split", "assign A_train/A_val/B splits and B folds"),
    "block-sparse": ("ber.blocking.sparse", "char TF-IDF top-k + key blocks per country"),
    "bakeoff": ("ber.contrastive.bakeoff", "bi-encoder bake-off: zeroshot | finetune | report"),
    "matcher": ("ber.matcher", "Model A: LightGBM matcher, out-of-fold on a B-split sample"),
    "predict": ("ber.predict", "score test candidates with Model A and write the submission files"),
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def run(cfg: dict, args: argparse.Namespace) -> None:
    print(json.dumps(cfg, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ber")
    parser.add_argument("--config", default="configs/base.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, (module, help_) in COMMANDS.items():
        importlib.import_module(module).add_arguments(sub.add_parser(name, help=help_))
    args = parser.parse_args(argv)
    importlib.import_module(COMMANDS[args.cmd][0]).run(load_config(args.config), args)
