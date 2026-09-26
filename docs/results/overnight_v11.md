# Overnight 2026-09-27: v11, the France investigation, and the OOF→leaderboard gap

CPU only; nothing else on the HPC was stopped. Code: `scripts/stacker_v11.py`, `scripts/sim_france.py`, `scripts/fr_records.py`,
`scripts/diag_neural.py`, `scripts/prior_shift.py`, France text fixes in `src/ber/text.py` and `src/ber/lexicons/`.

## 1. Submissions (all pass the official validator, on the HPC in `artifacts/submissions/`)

| Submission | What | OOF F0.5 (US+India) | Recommendation |
|---|---|---|---|
| **v11fr** | v11 ensemble stacker + France text fixes, τ 0.75 | **0.98982** | **upload first** |
| v11fr_t85 | same, cut-off 0.85 (Bayes correction for test's extra orphan records) | 0.98940 | 2nd upload: tests the "orphan" explanation of the gap |
| probe_v9_noFrance | v9 with every France row empty | – | diagnostic: gives France's real score |
| v11 | v11 without the France text fixes | 0.98982 | not needed (France text fixes change France p by 0.0003 on average) |
| v11fr_addr, v9_fraddr, v11fr_robust_addr | + France same-address rule | – | **not recommended** (see §4) |

Local copies: `submissions/v11fr`, `submissions/v11fr_t85`, `submissions/probe_v9_noFrance` (matching file only).

## 2. OOF comparison (150k B-split S1, same folds, 95% bootstrap CI vs v9)

| Model | OOF F0.5 | Δ vs v9 |
|---|---|---|
| v9 | 0.98948 | – |
| v10a (cleaned-name features) | 0.98973 | +0.00025 [+0.00014, +0.00036] |
| v11 (+ acronym feature) | 0.98977 | +0.00029 [+0.00018, +0.00042] |
| **v11 ensemble (4 LightGBMs)** | **0.98982** | **+0.00034 [+0.00022, +0.00046]** |
| v11 ensemble + expected-F0.5 per S1 | 0.98968 | worse (hurts singletons) |
| v11 robust (drops 9 France-shifted features) | 0.98933 | −0.0005 |

Precision / recall per stage: `artifacts/v2/stage_pr/report.md` (v10a: pair precision 99.79%, recall 97.31%).

## 3. France: what the gap is NOT

- **Région never extracted, `N°8` → house number lost:** fixed (région 0% → 96–100%, house found 90.7% → 92.3%).
  But simulating both gaps on labelled US/India OOF costs only **−0.0003** (`sim_france`), and re-scoring France with
  the fixed text moves Model A's p by 0.0003 on average. Real, but tiny.
- **Word-swap distractors** (`Pessac Centre` vs `Pessac Societe`): v9 accepts *fewer* of these in France (9.7% of accepts)
  than in the US (19%); on OOF they are 99.5% correct.
- **Same name + same house + different street:** France has 15× more of these per S1 (common names, small house
  numbers); v9 accepts 31%. They look like different businesses, but removing them makes France's count profile *less*
  like the US one, so no rule was applied.

## 4. The same-address pattern (investigated, rule NOT recommended)

S1 alone at its exact address, S2/S3 record at that address with a different name: in train 96.4% true; v9 accepts
94.8% (US), 97.2% (India), but only **56.7% in France** (renamed copies get realistic French names, e.g.
`Cercle Collectif Développement`). A rule accepting them adds 62,778 France matches.

Counter-evidence (label-free): the data generator's matches-per-S1 distribution is identical for US and India
(mean 3.44), and **France's predicted profile without the rule already matches the US profile** (mean 3.39 vs 3.38).
With the rule France jumps to 3.60 and 6.1% of S1s get 7+ matches (truth 4.0%), so most additions are probably
wrong. Break-even is ~62% true; expected gain +0.0025 only if ~90% true.

## 5. The OOF→leaderboard gap (~0.005)

- On test, the model gives **more matches per S1 than on OOF in every country** (US 3.389 vs 3.366; India 3.389 vs 3.375),
  though the generator's truth is the same. Test also has more S2/S3 records per S1 (5.76 vs 4.68): owned share
  74% in train vs ~59% on test if the match rate is unchanged.
- Either test has more orphan (unowned) records that the model wrongly accepts (precision loss in every country), or
  test simply has more copies per S1. `v11fr_t85` (Bayes cut-off for the orphan prior, OOF cost −0.0004) tests this.
- France's share of the gap is unknown until `probe_v9_noFrance` is scored:
  **France F0.5 ≈ (LB(v9) − LB(probe)) / 0.1498 + ~0.055** (0.055 = France singletons, which score 1 when empty).

## 6. Projection

v11fr ≈ LB(v8) + 0.85 × 0.0013 + 0.15 × (same gain on France) ≈ **0.984**. **0.99 is not reached** by CPU work.
The remaining levers need new information: a better reranker (GPU, in progress), and the probe results to know
whether to push France or precision.
