# Submission v3 (upload this)

- `matching_results.tsv` — 1,732,544 test Source-1 businesses. Official validator: **PASS**.
- Model: Model A v2 (LightGBM + fine-tuned bge-m3 cosine) **+ cross-encoder reranker** (bge-reranker-v2-m3, Apache-2.0,
  fine-tuned on A-train pairs) on the uncertain band p∈[0.01, 0.99] (3.22M test pairs), stacked with a small LightGBM.
- Cut-off τ = 0.70, one-owner rule.
- Out-of-fold macro F0.5 (50k B-split businesses): **0.9784** (singletons 0.9854, India 0.9712, US 0.9832) vs v2 0.9707 (LB 0.962).
- Test sanity: 3.30 matches/business, 6.1% empty lists, France 3.27 (US 3.36, India 3.27).
- `candidate_pairs.tsv` (2.5 GB) on the HPC: `~/winning_ml_challenge_sol/artifacts/submissions/v3/`.
