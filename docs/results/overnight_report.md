# Overnight Report: 25–26 Sep 2026

## TL;DR
- ✅ **We have a first valid submission.** `matching_results.tsv` + `candidate_pairs.tsv` for all **17,32,544** test businesses. The **official validator passes** ("Safe to submit").
  - **Leaderboard file on the laptop:** `~/ml_challenge/output/matching_results.tsv` (97 MB)
  - **Both files on the HPC:** `~/winning_ml_challenge_sol/artifacts/submissions/v1_2/`. `candidate_pairs.tsv` is 2.5 GB, too big for the laptop.
- **Honest validation score (out-of-fold, official macro F0.5): 0.9594** (India 0.9396, US 0.9726, singletons 0.9355), on 50,000 held-out-style B-split businesses.
- **Candidate search recall: 98.28%** (India 96.46%, US 99.51%), ~108 candidates per business. The ceiling (oracle F0.5) is ≈ 0.99.
- **bge-m3 (chosen bi-encoder), fine-tuned:** recall@50 **India 99.95%, Indian-script names 100%, US 99.97%** (zero-shot: 95.4% / 91.6% / 98.3%). Fine-tuning essentially solved the cross-script problem. ⚠ The evaluation pool has 1M sampled distractors per country, so real-pool recall will be somewhat lower.

## 1. Pipeline as it stands
```
clean (24M records, 22 columns incl. state, landmark, unit, skeleton/phonetic names)
  → candidate search per country + state group:
      char TF-IDF top-50 | key blocks | name-only (empty addresses) | Qwen-0.6B hard names | knowledge-graph expansion
  → Model A (LightGBM, 43 features, 5-fold out-of-fold on the B split)
  → decision: cut-off τ = 0.70 + each S2/S3 record goes to one owner only
  → matching_results.tsv / candidate_pairs.tsv → official validator: PASS
```

## 2. Candidate search (train, measured against ground truth)
| Channel | Candidates / business | Recall alone | Recall lost if removed |
|---|---|---|---|
| TF-IDF top-50 (state-grouped) | 50.0 | 95.82% | 0.97% |
| Key blocks (house number + rare word) | 30.1 | 87.08% | 0.37% |
| Name-only (empty-address records) | 19.6 | 3.92% | 0.24% |
| Qwen-0.6B hard names (Indian script / empty address) | 5.0 | 4.22% | 0.24% |
| Knowledge-graph expansion (neighbours of strong candidates) | 32.2 | 93.89% | 0.56% |
| **All channels** | **107.6** | **98.28%** (India 96.46%, US 99.51%) | |

- Speed-ups found tonight:
  - state groups: US search went from about 7 h (v1, never finished) to 7 min
  - parallel TF-IDF (all cores instead of one)
  - weighting fix: 25 min → seconds
  - parallel key blocks: 1.5 h → 35 s
- **Learned state groups:** only `Andhra Pradesh + Telangana` and `DC + Washington` get merged; every other state is its own group.
- **Remaining India misses, by reason:**
  - 57% names written in Indian scripts
  - 22% true duplicate pushed out of the top 50 by lookalikes
  - 8% empty address with a noisy name
  - 5% completely different (alias) name

  The fine-tuned bge-m3 is the main fix for the script problem (§4).

## 3. Model A: LightGBM matcher (out-of-fold, official macro F0.5)
| Version | Change | Macro F0.5 | India | US | Singletons |
|---|---|---|---|---|---|
| v1 | base features (fuzzy name/address, house/state/suffix, context) | 0.9458 | 0.9255 | 0.9593 | 0.9223 |
| v1.1 | + IDF-weighted name overlap, rarest non-shared name word, house-number distance | 0.9561 | 0.9352 | 0.9700 | 0.9298 |
| **v1.2** | + graph-expansion and hard-name candidate channels | **0.9594** | **0.9396** | **0.9726** | **0.9355** |

- Best cut-off τ = 0.70 (precision-leaning, as expected for F0.5). AUC ≈ 0.9997.
- **Top features:**
  1. `is_cand_best` (this S1 is the candidate's best S1: the one-owner / graph signal)
  2. `tfidf_sim`, `tfidf_rank`
  3. `key_hits`
  4. `house_num_diff`
  5. number overlap, fuzzy name scores, `name_idf_jacc`, `name_rare_mismatch`
- **Error analysis (v1 at τ = 0.70):**
  - pair precision **97.97%**, pair recall 89.84%
  - the main loss is **true matches the model rejects**: noisy duplicates with typos, reordering, or a house number off by one
  - wrong merges are mostly *same address, different business* (`Pune Retails` vs `Pune Aviation`)
- **Test prediction (v1.2):**
  - 19.2 crore test pairs scored in 45 min
  - **3.32 matches per business** (train truth ≈ 3.5), **5.8% empty lists** (train singletons 5.6%)
  - **France 3.35**, in line with the US (3.36) and India (3.28): no drift on the unseen country

## 4. bge-m3 bi-encoder (the selected model)
| Run | India R@50 | India Indian-script R@50 | US R@50 |
|---|---|---|---|
| Zero-shot (1M distractors per country) | 95.39% | 91.57% | 98.29% |
| **Fine-tuned** (A-train, 3000 steps, cross-script oversampling) | **99.95%** (R@10 99.69%) | **100.00%** (R@10 99.99%) | **99.97%** |
| Fine-tuned on US only → India (France stand-in) | *filled below* | *filled below* | — |

**Implication:** the fine-tuned bge-m3 should become a **dense candidate-search channel** (FAISS per state group). It should lift India's candidate recall well above the current 96.46% and handle Indian-script names directly.

## 5. Things that went wrong tonight (and how they were handled)
1. **Our conda env was wiped:** a teammate on the shared account ran `conda create -n ber` (Python 3.12). → We moved to our own env, **`ber-sol`**, documented in `CLAUDE.md`.
2. **v1 candidate search** ran 11 h without finishing the US; it was superseded by the state-grouped v2 (~20 min) and cancelled at your request.
3. **transformers 5.x broke gte/nomic** (their custom model code) → pinned `transformers<5`.
4. **`hpc.sh` sync deleted reports generated on the HPC** → fixed (`--exclude /docs/results`).
5. **My background watchers were silent for hours** (a zsh quoting bug) → all watchers now run under `bash -c`.
6. **The laptop moved to a hotspot** mid-run, so the HPC was unreachable. → Everything runs in **tmux**, so nothing was lost.
7. ⚠ **Security (please act):**
   - **Revoke the rclone Google token** at myaccount.google.com/permissions → rclone → Remove access. I accidentally printed it earlier. It's already deleted from the HPC.
   - Someone on the shared account is running **`ngrok tcp 22`** (a tmux session named `ngrok`). That exposes the HPC's SSH login to the internet. Please check who started it.

## 6. What's next
1. **Upload `~/ml_challenge/output/matching_results.tsv`** to the leaderboard. It's the first real score.
2. **Model A v2** = v1.2 + the fine-tuned bge-m3 similarity feature (running). If it beats 0.9594, embed the test pairs and re-submit.
3. **Singletons / "no match" model** and **S2↔S3 group decisions** (knowledge-graph Model B), aimed at precision.
4. **Train on more B-split businesses** (50k now; up to 13 lakh are available) once the GPU/CPU are less contended.
5. **Final zip:** package `code/business_entity_resolution/` (src, README, pinned requirements) + the documentation template.

## Where everything is
- **Branch:** `feat/foundations-bakeoff` (GitHub). Reports are in `docs/results/`.
- **HPC:** `~/winning_ml_challenge_sol`, env `ber-sol`, artifacts in `artifacts/v2/`, logs in `artifacts/logs/`.
- **tmux sessions:** `bakeoff` (bge-m3), `chain` (Model A v2); `ber` and `ngrok` are your teammate's.
