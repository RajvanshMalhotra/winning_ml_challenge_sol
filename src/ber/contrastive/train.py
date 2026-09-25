import time
from pathlib import Path

import pandas as pd

from ber.contrastive.encoders import EncoderSpec, load_encoder


def finetune(spec: EncoderSpec, train_df: pd.DataFrame, out_dir: Path, cfg_train: dict,
             mem_fraction: float | None, seed: int, base_model: str | None = None) -> dict:
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from sentence_transformers.training_args import BatchSamplers

    model = load_encoder(spec, path=base_model, mem_fraction=mem_fraction, for_training=True)
    lr = cfg_train["learning_rate"]
    if spec.lora:
        from peft import LoraConfig, TaskType

        model.add_adapter(LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32,
                                     lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        hf = model[0].auto_model
        hf.gradient_checkpointing_enable()
        hf.enable_input_require_grads()
        lr = cfg_train["lora_learning_rate"]
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=cfg_train["cached_mini_batch_size"])
    cuda = torch.cuda.is_available()
    args = SentenceTransformerTrainingArguments(
        output_dir=str(Path(out_dir) / "ckpt"),
        max_steps=cfg_train["max_steps"],
        per_device_train_batch_size=cfg_train["batch_size"],
        learning_rate=lr,
        warmup_ratio=cfg_train["warmup_ratio"],
        bf16=cuda,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    trainer = SentenceTransformerTrainer(
        model=model, args=args, loss=loss,
        train_dataset=Dataset.from_pandas(train_df[["anchor", "positive", "negative"]], preserve_index=False),
    )
    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0
    model.save(str(Path(out_dir) / "model"))
    return {"peak_train_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0,
            "steps": int(trainer.state.global_step), "train_seconds": elapsed}
