# Submission v5 (upload this)

- `matching_results.tsv` — 1,732,544 test Source-1 businesses. Official validator: **PASS**.
- Model: **Model A v3** (LightGBM, 48 features; candidates add the fine-tuned bge-m3 dense top-30 channel, blocking recall 99.9%)
  **+ cross-encoder reranker** (bge-reranker-v2-m3, fine-tuned on A-train pairs) on the uncertain band p∈[0.01, 0.99], stacked with a small LightGBM.
- Cut-off τ = 0.65, one-owner rule.
- Out-of-fold macro F0.5 (150k B-split businesses, 272k band pairs re-scored): **0.9834** (singletons 0.9781, India 0.9832, US 0.9835)
  vs Model A v3 alone 0.9779; v3 was 0.9784 OOF → LB 0.973.
- Test sanity: 3.37 matches/business, 5.9% empty lists, France 3.32 (US 3.37, India 3.38).
- `candidate_pairs.tsv` (3.0 GB) on the HPC: `~/winning_ml_challenge_sol/artifacts/submissions/v5/`.
