import pandas as pd

from ber.contrastive.encoders import EncoderSpec
from ber.contrastive.train import finetune

TINY = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


def test_finetune_runs_and_saves(tmp_path):
    df = pd.DataFrame({"anchor": [f"name: a{i}" for i in range(16)],
                       "positive": [f"name: A{i}" for i in range(16)],
                       "negative": [f"name: z{i}" for i in range(16)]})
    cfg = {"batch_size": 8, "cached_mini_batch_size": 4, "max_steps": 2, "learning_rate": 2e-5,
           "lora_learning_rate": 1e-4, "warmup_ratio": 0.0}
    stats = finetune(EncoderSpec("tiny", TINY), df, tmp_path, cfg, mem_fraction=None, seed=0)
    assert (tmp_path / "model" / "config.json").exists()
    assert stats["steps"] == 2
