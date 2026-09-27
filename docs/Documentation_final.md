# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** TEAM_NAME
**Team Members:** TEAM_MEMBERS
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We use a multi-channel blocking stage that keeps 99.9% of true matches (≈130 candidates per business). A LightGBM matcher scores every candidate pair. A **fine-tuned Qwen3-Reranker-4B cross-encoder** (LoRA, Apache-2.0) then re-reads every uncertain pair. A final LightGBM stacker adds **candidate-side "competition" features**: it asks, from each S2/S3 record's point of view, which of *all* S1 businesses owns it. Because test contains far more records whose owner is absent from S1, the stacker is trained on **test-like frames** in which a share of training businesses is hidden. The final decision is a precision-oriented cut-off plus a one-owner rule. Out-of-fold macro F0.5 on held-out training businesses is **0.9904** (0.990 under test-like conditions); the best public leaderboard score is **0.98709** (v16).

---

## 2. Methodology

### 2.1 Problem Analysis
- **Noise is synthetic and systematic.** Names show typos, character swaps (`0`/`O`, `1`/`l`), fake accents, legal-suffix changes, words dropped or added, and sometimes **complete replacement by a gibberish token** while the address stays the same. Addresses show abbreviation changes, reordering, dropped components, **house-number digit typos** (`13825` → `1382`) and transliteration (Indian scripts).
- **Only S2/S3 records have empty addresses** (~3%). S1 is always complete, in train and test, France included. An address-less S2/S3 record belongs to *some* S1 **97.7%** of the time, versus 73% for records with an address. For these records the question is not "is it a match?" but "**which** S1 owns it?", which is an assignment problem.
- **Ambiguity comes from namesakes.** The matches that remain missed are mostly address-less records whose name is shared by several S1 businesses. Only ~3% of them have a unique name.
- **Country is an open set.** France (15% of test) never appears in train. All text models are multilingual, all features are country-agnostic, and nothing is hard-coded to {US, India}.
- **The metric is macro F0.5 per S1, with singletons.** False merges cost about twice as much as misses, and a wrong match on a singleton costs a whole point.

### 2.2 Solution Strategy
**Approach Type:** Hybrid: multi-channel blocking → gradient-boosted pair classifier as a learned filter (132 → 5 candidates per S1) → fine-tuned LLM cross-encoder → stacker with global (candidate-side) competition features → thresholded one-to-many assignment.

**Core Innovation:**
1. **Candidate-side competition.** For each (S1, record) pair we compute how this S1 ranks among **all** S1s that have the record as a candidate: by dense similarity, TF-IDF, name similarity and Model A probability, together with the margin to the best other S1 and the number of competitors. This turns per-pair scoring into an implicit global assignment. It raised precision on address-less records from 87% to 97% and gave the largest single gain (+0.005).
2. **Test-like training for the stacker.** In test only ~60% of S2/S3 records have their owner in S1 (74% in train), so a decoy often looks *uncontested*. We recompute the competition features with 20% of the (non-validation) training businesses hidden, in 3 random samples, and train the stacker on these frames. Under test-like conditions this lifts validation F0.5 from 0.9897 to 0.9901, mostly on singletons (0.9907 → 0.9932), and it improved the leaderboard (0.9864 → 0.98663).
3. **Fine-tuned Qwen3-Reranker-4B.** It is trained on hard negatives (wrong-owner records, namesakes, same-building businesses) and hard positives (address-less copies, gibberish names, digit typos). As a reranker its AUC on uncertain pairs is **0.976**, versus 0.907 for the fine-tuned bge-reranker it replaced.

---

## 3. Candidate Generation (Blocking)
Blocking runs per country and per learned *state group* (states that the data often confuses are merged), plus records with unknown state.

- **Blocking keys and channels:**
  1. Character 3–5-gram TF-IDF on name + address (parallel hashed vectoriser, `max_df=0.05`), top-50 per S1.
  2. Exact key blocks: rare-token and name-key combinations, postal code / PIN + house number, skeleton / phonetic name keys.
  3. **Name-only search** for records with empty addresses.
  4. **Hard-name search** with Qwen3-Embedding-0.6B for native-script / transliterated and empty-address records.
  5. **Graph expansion**: neighbours of strong candidates through record–record TF-IDF edges.
  6. **Dense retrieval**, top-30, with **bge-m3 fine-tuned** contrastively (cached multiple-negatives ranking loss) on training pairs, including extra English → Indian-script pairs.
- **Candidate pairs generated:** blocking yields 229,254,961 test pairs (≈132 per S1). A **learned filter** then keeps only pairs with Model A (LightGBM) probability ≥ 0.01, which leaves **8,828,474 pairs, 5.1 per S1** (true matches average 3.4). The final matcher (Qwen reranker + stacker) runs only on these, and `candidate_pairs.tsv` is exactly this set. Recall ceiling after the filter is 99.51% (validation); no pair the final model accepts is lost by the filter.
- **How true matches were not lost:** each channel was added only when it measurably raised recall on held-out training businesses. Blocking recall: TF-IDF + keys **98.3%** → with name-only / hard-name / graph channels → with dense bge-m3 **99.90%** (India 99.85%, US 99.93%). Fine-tuning bge-m3 only on US data still raised Indian-script recall@50 from 91.6% to 98.6%, which is evidence that it transfers to unseen languages such as French.

---

## 4. Matching Model

**Features used (Model A, 48 features):**
- **Name features:** RapidFuzz ratio, token-set, token-sort, partial ratio, Jaro–Winkler; skeleton and phonetic token-set; IDF-weighted token Jaccard and rare-token mismatch; name length ratio; legal-suffix equal / different / missing.
- **Address features:** token-set, ratio and core-address similarity, numeric-token similarity, house number equal / different / missing plus absolute numeric difference, state and unit comparisons, empty-address and native-script flags.
- **Other:**
  - blocking-channel signals: TF-IDF similarity and rank, key hits, name-channel similarity, dense bge-m3 similarity and rank, hard-name similarity and rank, graph paths
  - fine-tuned bge-m3 embedding cosine
  - context: candidates per S1, mutual-best flag, candidate degree
- **Stacker features (final model):**
  - Model A probability and within-S1 rank / gap
  - Qwen reranker score, rank and gap
  - **candidate-side competition**: rank, margin and number of competing S1s, by dense, TF-IDF, name and Model A score
  - share of the record's total Model A probability, number of other S1s with p ≥ 0.5
  - cluster support: name similarity of the record to the S1's other confident matches, and its competition
  - address similarity competition (for replaced names)
  - how many S1s share the exact name / address
  - no-address flag

**Model type:**
- **Model A:** LightGBM, 5-fold out-of-fold training on a 150k-business held-out sample.
- **Reranker:** Qwen3-Reranker-4B (Apache-2.0, 4.0B parameters).
  - LoRA r=32 on all attention and MLP projections; 1.04M training pairs from 90k businesses.
  - Pairs: true matches; hard negatives (look-alikes, wrong-owner, namesakes, same address, address-less); hard positives shown twice.
  - Loss: binary cross-entropy on the yes/no logit.
  - Applied to every pair with 0.01 ≤ Model A p ≤ 0.99 (3.7M test pairs, scored with vLLM).
  - Then trained further on 90k *new* businesses at half the learning rate (AUC 0.9764 → 0.9772).
- **Stacker:** LightGBM, same folds as Model A, trained on 3 test-like (hidden-owner) frames; the final probability is the mean of 3 × 5 models. It receives the two Qwen reranker scores **and** the bge-reranker score, plus their disagreement, so it can be cautious where the rerankers disagree.
- The bi-encoders (bge-m3, MIT; Qwen3-Embedding-0.6B, Apache-2.0) are used for blocking and features. **All models are MIT/Apache-2.0 and ≤ 8B parameters.**

**Threshold selection method:** the macro F0.5 cut-off τ is tuned directly on out-of-fold predictions, with the exact competition metric computed per S1 including singletons. τ = 0.75. Then a **one-owner rule**: each S2/S3 record is kept only for the S1 with the highest score above τ.

---

## 5. Results & Error Analysis

| Version | Main change | OOF macro F0.5 | Leaderboard |
|---|---|---|---|
| v2 | LightGBM + fine-tuned bge-m3 features | 0.9707 | 0.962 |
| v3 | + bge cross-encoder on uncertain pairs | 0.9784 | 0.973 |
| v5 | + dense bge-m3 blocking channel (recall 98.3% → 99.9%) | 0.9834 | 0.979 |
| v8 | + candidate-side competition features | 0.9885 | 0.983 |
| v12 | + Model-A competition / cluster support; fine-tuned Qwen3-Reranker-4B replaces bge | 0.9904 | 0.9864 |
| v14 | + stacker trained on test-like hidden-owner frames | 0.9901 (test-like) | 0.98663 |
| **v16** | + further-trained Qwen, three rerankers side by side, stronger stacker, 3 hidden samples | **0.9904 (test-like)** | **0.98709** |

- **F_0.5 Score (macro):** **0.9904** out-of-fold (singletons 0.9939, India 0.9913, US 0.9898); 0.9904 under test-like hidden-owner conditions for the final model.
- **Common false positives (wrong merges):**
  - an address-less record given to the wrong namesake S1
  - branches of a chain with near-identical names in different streets
  - different businesses in the same building

  Wrong accepts fell from 3,247 (v5) to 935 (v12) on the validation sample.
- **Common false negatives (missed matches):**
  - address-less copies of businesses whose name is shared by several S1s (the largest group; with no address the record carries no information that separates the namesakes)
  - names fully replaced by a gibberish token
  - house-number digit typos

  About 54% of address-less true pairs are found, at 96.6% precision.

---

## 6. Conclusion
High recall comes from combining complementary blocking channels. Precision comes from judging each record **against every business that competes for it**, not pair by pair, and from a large cross-encoder fine-tuned on this dataset's noise. A general-purpose 4B reranker used without training was far worse than a small fine-tuned one (AUC 0.69 vs 0.91). After fine-tuning, the same 4B model became the best component (0.976).

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/ber/`: the core package
  - normalisation, splits, sparse blocking, pair features, LightGBM matcher, prediction, metric
  - `contrastive/`: bi-encoder fine-tuning
- `src/scripts/`: pipeline stages
  - hard-name, graph and dense channels
  - reranker training and scoring (`qwen_reranker.py`, `vllm_score.py`)
  - candidate-side features (`cand_side.py`, `cand_side_v9.py`)
  - test-like hidden-owner frames (`hidden_sim.py`) and the final stacker (`v16.py`; `v12_combo.py` for the reference stacker)
- `src/configs/`: YAML configs.
- **Entry point:** `cd src && bash scripts/run_all.sh` runs every stage in order and writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (see `README.md`).

### B. Additional Results
- Reranker AUC on the 272k uncertain validation pairs: Model A 0.953 | bge-reranker (fine-tuned) 0.907 | Qwen3-Reranker-4B zero-shot 0.689 | **Qwen3-Reranker-4B fine-tuned 0.976**. On address-less pairs: bge 0.721 → Qwen 0.921.
- The prior probability that a record belongs to some S1 is lower in test (~60%) than in train (74%); simulating this explains part of the ~0.004 gap between out-of-fold and leaderboard scores.
- The rest of the gap is most likely France (15% of test, not in train): with US/India near 0.990, the leaderboard implies France ≈ 0.97. French businesses here are mostly associations with generic names ("club", "amicale", "comité des fêtes") and France shows 3× more borderline decisions. Rules that added or removed French matches were tested on the leaderboard and made it worse, so France is left to the learned model.
- Tested and rejected (no gain or worse): higher cut-offs (0.85: 0.9863), per-business expected-F0.5 decision rule (0.99035 vs 0.99040 OOF), a French same-address rule (LB 0.980), zero-shot Qwen for France (LB 0.981), lower cut-off for address-less pairs.
