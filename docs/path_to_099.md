# Path to 0.99 on the leaderboard

_2026-09-27. Leaderboard best: **v14 = 0.98663** (v16 not scored yet). Target: **≥ 0.99**, i.e. **+0.0034**._

## 1. Where the missing points are

Test = 46.8% India, 38.2% US, **15.0% France** (France is not in train). Working back from our leaderboard (LB) scores:

| Part of test | F0.5 now | LB points lost vs 0.990 |
|---|---|---|
| US + India | ~0.989–0.990 | ~0.0000–0.0006 (test has more orphan records: ~60% owned vs 74% in train; v14 fixes part of it) |
| **France** | **~0.964–0.968** | **~0.0033–0.0040** |

- **US/India are nearly maxed.** Their remaining validation loss (0.0096) is mostly address-less copies whose name is shared by several businesses, which is unresolvable by name. Realistic gain: ≤ +0.001 on validation.
- **0.99 is only reachable by fixing France.** That is almost certainly where teams at 0.99+ are ahead, not in a better reranker.

## 2. What the leaderboard probes already proved

| Submission | LB | Conclusion |
|---|---|---|
| v12 | 0.9864 | baseline |
| v14 (hidden-owner stacker) | **0.98663** | the prior-shift fix works (+0.0002) |
| v12, cut-off 0.85 | 0.9863 | global cut-off changes don't help |
| v12_fraddr (+ France same-address records) | 0.980 | those records are **decoys** (train agrees: realistic-name same-address records are only 16–42% true) |
| v12_fr90 / v12_frp2 (fewer France matches) | worse | France is **not** over-accepting at the margin |
| v8fr (zero-shot Qwen for France) | 0.981 | zero-shot Qwen is worse than a fine-tuned one |

**France's loss is in pairs the model is confidently wrong about.** Adding or removing matches at the cut-off makes it worse in both directions, so no threshold change can fix it.

## 3. Key finding: on France, Model A is the bottleneck, not Qwen

French acronym copies (`CF` for `Commune Federation SAS`, `MSJ` for `Maison de Santé de la Jeanne`):

| Acronym S2/S3 records | France | India | US |
|---|---|---|---|
| per 1,000 businesses | **92** | 11 | 6 |
| true copies (train) | – | 99.8% | 98.5% |
| Model A score of the initials-matching business (median) | **0.27** | 0.997 | 0.997 |

What happens to France's 23,986 acronym records (v14):

| Model A score of the initials-matching business | Given to it | Given to another business | Not matched |
|---|---|---|---|
| 0.01–0.99 (**Qwen reads it**) | **14,285 (97%)** | 116 | 258 |
| < 0.01 (**Qwen never reads it**) | 0 | 1,025 | **3,956** |
| no such business among the candidates | 0 | 1,461 | 1,353 |

- **When Qwen reads the pair, it is right 97% of the time.**
- **Model A, trained only on US/India features, scores many French true pairs below 0.01**, so they never reach Qwen. Of the unmatched records with exactly one initials-matching business, 1,671 even share its house number.
- **So expanding the search space helps, but only for the reranker, not for blocking.** Blocking already keeps 99.9% of true pairs (only 519 lost, worth ≤ +0.0003). The learned filter (5.1 candidates per business) never accepts anything below 0.01 anyway, so widening it alone changes nothing.

## 4. Plan

| # | Step | Compute | Expected LB gain (rough) |
|---|---|---|---|
| 1 | **Acronym routing:** send each France acronym record's initials-matching businesses (~6k pairs) to Qwen; add an initials blocking key for the 1,353 records with no such candidate | GPU, minutes | +0.0003 to +0.0004 |
| 2 | **Measure how much Model A under-scores France in general:** Qwen scores a sample of France's best candidates below 0.01. If acronyms are just one case of this, it is the main France lever | GPU, ~1 h | sizes the remaining ~90% of France's gap |
| 3 | If step 2 confirms it: **widen Qwen's coverage for France** (top-k candidates per business and per record), with the stacker trained on the same extended coverage (Qwen scores for validation pairs below 0.01 as well) | GPU, several hours | potentially most of France's +0.0034 |
| 4 | Upload **v16** (3 rerankers, test-like stacker) | ready | pending |
| 5 | Keep hidden-owner training (v14/v16) for US/India | done | +0.0002 (measured) |

Together, steps 1–5 are the realistic route to **~0.989–0.991**. **0.99 depends on steps 2–3 working for France.**

**Note on the French reranker fine-tune (`chain_fr.sh`):** it trains on the proxy-France set, which was generated with US/India noise:
- it has **no acronym noise**
- it uses US/India decoy densities (France has 27× more realistic-name same-address decoys and 15× more "same name + same house, different street" pairs)

It will teach Qwen French text, but not France's actual problem types. Coverage (step 3) matters more than reading quality.

## 5. Already tested, no gain (don't repeat)

All on v12 validation, thresholds chosen on 4 folds and scored on the 5th (`scripts/` names in brackets):

| Idea | Result |
|---|---|
| Stage-2 recovery model on rejected pairs (`stage2_check.py`) | +0.00000; ceiling +0.0031 even if perfect |
| Normalize + one script + spelling correction + similarity threshold (`spellfix_check.py`) | 0 on nested CV; identical corrected names are only 5.6–9.3% true; spelling correction damages invented names (`Emote` → `remote`) |
| Unique-name rule for address-less / Indian-script misses (`unique_name_check.py`) | +0.00002 (noise); "exactly one business with that name" is only 53% true |
| Cleaned-name threshold rule (v10a, `clean_name_v10a.py`) | 0 as a rule; +0.00025 as stacker features |
| France région / `N°` text fixes (`sim_france.py`) | real but tiny: the gaps cost only ~0.0003 |
| Sibling-name signal for same-address records (`sibling_check.py`) | no separation on train |
| Per-business expected-F0.5 decisions | worse (0.99035 vs 0.99040) |

## 6. Rules to keep in mind

- **Final ranking uses the private leaderboard** (the rest of the test set). France is in both halves, so a clear France signal from the public board carries over, but avoid tuning many knobs against the public board.
- `candidate_pairs.tsv` must be the exact set the final model scores. Any France pairs added for Qwen must also appear there.
- Models must be MIT/Apache and ≤ 8B parameters. No external data.
