from dataclasses import dataclass

import numpy as np

QWEN_INSTRUCT = "Instruct: Given a business name and address, retrieve records of the same business\nQuery: "


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    hf_id: str
    query_prefix: str = ""
    doc_prefix: str = ""
    trust_remote_code: bool = False
    lora: bool = False
    truncate_dim: int | None = None
    max_seq_length: int = 128
    padding_side: str | None = None


ENCODERS: dict[str, EncoderSpec] = {
    "gte": EncoderSpec("gte", "Alibaba-NLP/gte-multilingual-base", trust_remote_code=True),
    "nomic": EncoderSpec("nomic", "nomic-ai/nomic-embed-text-v2-moe", "search_query: ", "search_document: ",
                         trust_remote_code=True),
    "arctic": EncoderSpec("arctic", "Snowflake/snowflake-arctic-embed-l-v2.0", "query: ", ""),
    "bgem3": EncoderSpec("bgem3", "BAAI/bge-m3"),
    "qwen8b": EncoderSpec("qwen8b", "Qwen/Qwen3-Embedding-8B", QWEN_INSTRUCT, "", lora=True,
                          truncate_dim=1024, padding_side="left"),
}


def load_encoder(spec: EncoderSpec, path: str | None = None, mem_fraction: float | None = None,
                 for_training: bool = False):
    import torch
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available() and mem_fraction:
        torch.cuda.set_per_process_memory_fraction(mem_fraction)
    # bf16 weights for inference everywhere and for LoRA training; full fine-tunes keep fp32 master weights.
    dtype = torch.bfloat16 if (spec.lora or not for_training) else torch.float32
    model = SentenceTransformer(
        path or spec.hf_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=spec.trust_remote_code,
        truncate_dim=spec.truncate_dim,
        model_kwargs={"dtype": dtype},
        tokenizer_kwargs={"padding_side": spec.padding_side} if spec.padding_side else None,
    )
    model.max_seq_length = spec.max_seq_length
    return model


def encode(model, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
    emb = model.encode(texts, prompt=prefix or None, batch_size=batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    return emb.astype(np.float16)
