# Submission v12 (upload this)

- `matching_results.tsv` — 1,732,544 test Source-1 businesses. Official validator: **PASS**.
- Model: Model A v3 → **fine-tuned Qwen3-Reranker-4B** (LoRA r=32 on all layers, 1.04M A_train pairs incl. hard
  negatives: wrong-owner, same-name, same-address, no-address; hard positives repeated) reads every uncertain pair
  (0.01 ≤ p ≤ 0.99), **replacing bge-reranker** → stacker with v8 candidate-side features + v9 Model-A competition /
  cluster-support / address-competition features. τ = 0.75, one-owner rule.
- Out-of-fold macro F0.5 (150k B-split businesses): **0.9904** (singletons 0.9939, India 0.9913, US 0.9898)
  vs v8 0.9885 (LB 0.983), v9 0.9895, v10 0.9898.
- Reranker AUC on the uncertain band: Qwen fine-tuned 0.976 vs bge 0.907.
- Test sanity: 3.39 matches/business, 5.7% empty lists, France 3.38 (US 3.39, India 3.39).
- Scripts: `scripts/qwen_reranker.py`, `scripts/vllm_score.py` (vLLM venv `~/vllm-sol`), `scripts/v12_combo.py`.
