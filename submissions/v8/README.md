# Submission v8 (upload this)

- `matching_results.tsv` — 1,732,544 test Source-1 businesses. Official validator: **PASS**.
- Model: v5 (Model A v3 + bge reranker on the uncertain band) with a new stacker that adds **candidate-side features**:
  for each S2/S3 record, how this S1 ranks among ALL S1s that have it as a candidate (dense bge-m3 / TF-IDF / name-only),
  the margin to the best other S1, the number of competing S1s, how many S1s share the exact name, and whether the record has no address.
- Cut-off τ = 0.75, one-owner rule.
- Out-of-fold macro F0.5 (150k B-split businesses): **0.9885** (singletons 0.9906, India 0.9888, US 0.9884) vs v5 0.9834 (LB 0.979).
  No-address candidates: precision 86.7% → 97.0%, recall 44.0% → 50.2%.
- Test sanity: 3.38 matches/business, 5.7% empty lists, France 3.34 (US 3.38, India 3.38).
- `candidate_pairs.tsv` on the HPC: `~/winning_ml_challenge_sol/artifacts/submissions/v8/`.
