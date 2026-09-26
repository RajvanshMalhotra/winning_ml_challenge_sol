# Submission v9 (upload this)

- `matching_results.tsv`: 1,732,544 test Source-1 businesses. Official validator: **PASS**.
- `candidate_pairs.tsv` is identical in content to v8's and stays on the HPC (2.3 GB): `~/winning_ml_challenge_sol/artifacts/submissions/v9/`.
- Model: v8 (Model A v3 + bge reranker + candidate-side stacker), plus new stacker features computed over **all** S1s from Model A's score (`scripts/cand_side_v9.py`):
  - for each S2/S3 record: this S1's rank among every S1 that has it as a candidate
  - the margin to the best other S1
  - this S1's share of the total score
  - cluster support and address competition. These got about 0% of the gain.
- Model A was run over all 285M train pairs (`predict --family train`), so the features mean the same thing on train and test.
- Cut-off τ = 0.70, one-owner rule.
- Out-of-fold macro F0.5 (150k B-split businesses): **0.9895** vs v8 0.9885 (LB 0.98299).
  - India 0.9898, US 0.9893, singletons 0.9903.
  - Misses 14,751 → 13,044; wrong accepts 1,300 → 1,323.
- Test vs v8: 46,593 rows changed, 40,740 matches added, 7,908 removed. 3.39 matches/business, 5.7% empty lists.
