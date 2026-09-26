"""Embed every record of a family with the fine-tuned bi-encoder (cached .npy). usage: embed_family.py <family>"""
import sys

from ber.config import load_config
from ber.predict import record_embeddings

cfg = load_config("configs/v2.yaml")
emb, code = record_embeddings(cfg, sys.argv[1], "bgem3", "artifacts/v2/bakeoff/bgem3__finetune/model")
print("done", emb.shape, flush=True)
