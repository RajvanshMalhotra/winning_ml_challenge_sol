# Submission v2 (upload this)

- `matching_results.tsv` — leaderboard file, 1,732,544 test Source-1 businesses. Official validator: **PASS**.
- Model: Model A v2 (LightGBM, 5 folds, 45 features incl. fine-tuned bge-m3 cosine), cut-off τ = 0.70, one-owner rule.
- Out-of-fold macro F0.5 on 50k B-split businesses: **0.9707** (India 0.9616, US 0.9767, singletons 0.9619).
- Test sanity: 3.31 matches/business, 5.9% empty lists, France 3.30 (US 3.37, India 3.27).
- `candidate_pairs.tsv` (2.5 GB, over GitHub's limit) is on the HPC: `~/winning_ml_challenge_sol/artifacts/submissions/v2/`.
- Reproduce: `python -m ber --config configs/v2.yaml predict --models matcher_emb --out output`
