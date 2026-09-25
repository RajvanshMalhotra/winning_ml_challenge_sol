# Approach Tracker: Parallel Work Board

Every approach from `docs/approaches.md` is listed here as its **own work item**, so team members can each pick one and work in parallel. Section numbers (§) point back to `approaches.md` for the details.

**Legend:** ✅ done 🔄 running / in progress ⬜ open, pick it up ❌ dropped (reason given)

**How to pick up an item:**
1. Put your name in the **Owner** column.
2. Check **Depends on**. Items marked `none` can start today.
3. Produce exactly the **Output** listed, using the shared formats below, so other people's work plugs in without coordination.
4. Report the **Done when** metric in the team channel, and flip the status.

_Last updated: 2026-09-25_

---

## 0. Shared formats: read this before starting anything

All artifacts live on the HPC under `~/winning_ml_challenge_sol/artifacts/v1/`. Build on these and don't change their columns. Add new columns instead.

| Artifact | Columns | Status |
|---|---|---|
| `records_{train,test}.parquet` | `entity_id, source, country, name_raw, addr_raw, name_norm, name_core, legal_suffix, name_domain, addr_norm, house_no, num_tokens, block_text` | ✅ |
| `splits.parquet` | `s1_id, country, n_matches, split (A_train/A_val/B), fold (0–4 for B, -1 otherwise)` | ✅ |
| `cand_sparse_{train,test}.parquet` | `s1_id, cand_id, tfidf_sim, tfidf_rank, key_hits` | 🔄 |
| **Any new blocker** → `cand_<name>_{train,test}.parquet` | `s1_id, cand_id, <name>_score, <name>_rank` | contract |
| **Any new feature group** → `feat_<name>_{train,test}.parquet` | `s1_id, cand_id, <feature columns…>` (keyed by the pair) | contract |
| **Any matcher** → `oof_<model>.parquet` (train B) + `pred_<model>_test.parquet` | `s1_id, cand_id, p` (+ `fold` for OOF) | contract |
| **Any decision method** → `pred_lists_<method>.parquet` | `s1_id, matched_ids` (list) | contract |

**Evaluation (use these, don't write your own):**
- `ber.evaluate.macro_f05`
- `ber.evaluate.pair_completeness`
- `ber.evaluate.candidates_per_entity`
- `ber.contrastive.retrieval.recall_at_k`

**Rules for everyone:**
- Report numbers on **B only**, broken down per country.
- Never train anything on B that later produces a feature for B, unless it's out-of-fold.
- Country is an open set: never hard-code US/India.
- No external data lookups.
- Models must be MIT/Apache and ≤8B parameters.

---

## 1. Data

### 1.1 Exploration and data checks
| ID | Status | Approach | Owner | Depends on | Output | Done when |
|---|---|---|---|---|---|---|
| E1 | ✅ | Size / country / source breakdown | | none | approaches §8a | done |
| E2 | ✅ | Singleton rate, matches per entity, per-source counts | | none | approaches §8a | 5.6% singletons |
| E3 | ✅ | One-owner check (does an S2/S3 record ever match >1 S1?) | | none | approaches §8a | 0 violations |
| E4 | ✅ | Cross-country link check | | none | approaches §8a | 0 links |
| E5 | ✅ | Postal-code coverage per source and country | | none | approaches §8a | US 10%, IN 1%, FR 0.5% |
| E6 | ✅ | Noise-pattern catalogue | | none | approaches §8a | done |
| E7 | ✅ | Lexicon gap report (unmapped states, abbreviations, native script) | | records | `scripts/lexicon_gaps.py` | gaps fixed |
| E8 | ⬜ | **Noise frequency per source** (how often S2 vs S3 is all-caps, `NULL`, domain names, truncated, …) | | records | short table | we know which noise to up-weight in augmentation |
| E9 | ⬜ | **Error analysis of blocking misses**: sample true pairs that blocking lost and classify why | | cand_sparse | table of miss reasons | top 3 miss reasons identified |
| E10 | ⬜ | **France test profile**: suffix, street-type and département/region patterns in the test set | | records_test | table | lexicon additions for FR listed |

### 1.2 Preprocessing / normalization (§1)
| ID | Status | Approach | Owner | Depends on | Output | Done when |
|---|---|---|---|---|---|---|
| N1 | ✅ | Unicode NFKC + accent/script folding (`anyascii`) | | none | `ber.text.fold` | tested |
| N2 | ✅ | Lowercase, punctuation cleanup, `NULL` removal | | none | `normalize_name/address` | tested |
| N3 | ✅ | Filler-word removal (`the`, `mr`, `m/s`, `shri`) + collapsing duplicates (`The The`) | | none | `name_norm` | tested |
| N4 | ✅ | Legal-suffix canonicalization US/IN/FR → `name_core` + `legal_suffix` | | none | columns | tested |
| N5 | ✅ | Website-style name splitting (`fafloonpetcare.com` → `fafloon pet care`) | | none | `name_domain` | tested |
| N6 | ✅ | Street-type abbreviation expansion (EN/FR) | | none | `addr_norm` | tested |
| N7 | ✅ | State abbreviation → full name, per country | | none | `addr_norm` | tested |
| N8 | ✅ | Native-script state names → English (17) | | none | `addr_norm` | verified on the data |
| N9 | ✅ | City aliases (`bangalore→bengaluru`, `hyd`, `kol`) | | none | `addr_norm` | done |
| N10 | ✅ | House-number extraction and canonicalization (`01130`→`1130`, `4-7/1`→`47/1`) | | none | `house_no`, `num_tokens` | tested |
| N11 | ⬜ | **DBA / trade-name splitting**: split on `dba`, `d/b/a`, `t/a`, `trading as`, `(…)`, FR `enseigne`, and match against each part | | none | new column `name_parts` | unit tests + number of records affected |
| N12 | ⬜ | **Landmark field**: split out `near X`, `opp X`, `behind X`, `nr X` into `landmark` and remove them from `addr_norm` | | none | new column `landmark` | unit tests; pair completeness doesn't drop |
| N13 | ⬜ | **Postal-code extraction as a feature** (IN 6-digit PIN, US 5/5+4 ZIP, FR 5-digit + département) | | none | new column `postal` | unit tests |
| N14 | ⬜ | **Unit / floor / PO box as separate fields** (so `Unit 609` vs no unit is not a mismatch) | | none | new column `unit` | unit tests |
| N15 | ⬜ | **Phonetic keys**: Double Metaphone of `name_core` tokens | | none | new column `name_phonetic` | unit tests |
| N16 | ⬜ | **Transliteration skeleton** for Indian names (`aa→a`, `ee→i`, `sh→s`, `v↔w`, `ph→f`) | | none | new column `name_skeleton` | unit tests |
| N17 | ⬜ | **Mojibake repair** (`Â`, `â` artifacts from bad UTF-8 decoding) | | none | fix in `fold` | unit tests |
| N18 | ⬜ | **Token-sorted name** (handles word-order swaps) | | none | new column `name_sorted` | unit tests |
| N19 | ❌ | libpostal address parsing | | | | dropped: risk under the "external data" rule |

### 1.3 Data augmentation (§4.4)
| ID | Status | Approach | Owner | Depends on | Output | Done when |
|---|---|---|---|---|---|---|
| A1 | ✅ | Noise operators: 9 for names, 8 for addresses | | none | `ber.noise` | tested |
| A2 | 🔄 | Augmentation inside bi-encoder training | | A1 | used by the bake-off | bake-off runs |
| A3 | ⬜ | **Noise weights from data**: set per-operator probabilities from E8 frequencies | | E8 | config weights | operator mix matches the data |
| A4 | ⬜ | **Synthetic positive pairs for the matcher** (augmented copies of S1 as extra positives) | | A1, M1 | extra training rows | OOF F0.5 change reported |
| A5 | ⬜ | **French-style augmentation**: operators that produce FR patterns (`Rue`↔`R.`, SARL↔S.A.R.L., département↔region), applied to US/IN records | | A1, E10 | new operators | leave-one-country-out recall change reported |

### 1.4 Feature engineering (§3.1): each group is independent → `feat_<name>` parquet
| ID | Status | Approach | Owner | Depends on | Output | Done when |
|---|---|---|---|---|---|---|
| F1 | ⬜ | **Edit-distance features**: Levenshtein ratio, Jaro-Winkler, Damerau on `name_core`, `addr_norm` (rapidfuzz) | | cand_sparse | `feat_edit` | features for all train and test pairs |
| F2 | ⬜ | **Token features**: Jaccard, token_set_ratio, token_sort_ratio, partial_ratio | | cand_sparse | `feat_token` | same |
| F3 | ⬜ | **Monge-Elkan (JW inner) + soft TF-IDF** (Cohen 2003) | | cand_sparse | `feat_soft` | same |
| F4 | ⬜ | **IDF-weighted token overlap**, IDF computed per country on that country's own text | | cand_sparse | `feat_idf` | same |
| F5 | ⬜ | **BM25 score per field** (name, address separately) | | cand_sparse | `feat_bm25` | same |
| F6 | ⬜ | **Structured equality**: house number (equal/unequal/missing), numeric-token Jaccard, legal-suffix equal/unequal/missing, postal (N13), unit (N14) | | cand_sparse (+N13/N14) | `feat_struct` | same |
| F7 | ⬜ | **Website-name similarity** (`name_domain` vs `name_core`) | | cand_sparse | `feat_domain` | same |
| F8 | ⬜ | **Phonetic / skeleton match** (N15/N16) | | N15, N16 | `feat_phonetic` | same |
| F9 | ⬜ | **Length ratios, token counts, source indicator (S2/S3)** | | cand_sparse | `feat_basic` | same |
| F10 | ⬜ | **Context features**: candidate rank within the S1's list, gap to best, how many S1s rank this candidate top-3, mutual-best flag | | cand_sparse | `feat_context` | same |
| F11 | ⬜ | **Embedding cosine + rank** from the chosen bi-encoder | | R-winner | `feat_embed` | same |
| F12 | ⬜ | **Feature importance / ablation report** (which groups matter) | | M1 | table | done |

---

## 2. Candidate generation / blocking (§2): each blocker is independent → `cand_<name>` parquet

Every blocker reports **pair completeness per country** and **mean/p95 candidates per S1** on train. The union (B13) is measured by how much each blocker adds on top of the others.

| ID | Status | Approach | Owner | Depends on | Output | Done when |
|---|---|---|---|---|---|---|
| B1 | ✅ | Hard partition by country (open set) | | none | in `block_family` | done |
| B2 | 🔄 | **Char TF-IDF (3–5-gram) top-k** | | records | `cand_sparse` | slow; testing `max_df=0.01` (~8× faster), recall check running |
| B3 | 🔄 | **Key blocks: house number + rare street word** | | records | `cand_sparse` (`key_hits`) | same job |
| B4 | 🔄 | **Key blocks: rare name word + rare address word** | | records | `cand_sparse` (`key_hits`) | same job |
| B5 | ⬜ | **Sorted neighbourhood** on `name_core` / `name_sorted` keys (Hernández & Stolfo) | | records | `cand_sortnb` | pair completeness reported |
| B6 | ⬜ | **Phonetic-key blocking** (Double Metaphone of the first/rarest name token) | | N15 | `cand_phonetic` | pair completeness reported |
| B7 | ⬜ | **BM25 top-k** per field | | records | `cand_bm25` | pair completeness reported |
| B8 | ⬜ | **MinHash LSH** on shingles (datasketch) | | records | `cand_minhash` | pair completeness + runtime vs B2 |
| B9 | ⬜ | **Dense ANN blocking** (FAISS on the fine-tuned bi-encoder, DeepBlocker / Sudowoodo style) | | R-winner | `cand_dense` | pair completeness reported |
| B10 | ⬜ | **Late-interaction (ColBERT / bge-m3 multi-vector) retrieval** | | R4 | `cand_colbert` | pair completeness reported |
| B11 | ⬜ | **bge-m3 sparse (lexical-weight) retrieval** | | R4 | `cand_bgesparse` | pair completeness reported |
| B12 | ⬜ | **Meta-blocking**: weight edges by how many blocks they share, prune weak ones (Papadakis) | | ≥2 blockers | pruned list | same completeness with fewer candidates |
| B13 | ⬜ | **Union of blockers** + per-S1 cap + marginal-recall report | | ≥2 blockers | `cand_union` | ≥99% pair completeness target |
| B14 | ⬜ | **Cheap prefilter** (small LightGBM on fast features, threshold at 99.5% out-of-fold recall) → final candidate set / `candidate_pairs.tsv` | | B13, F-cheap | `cand_final` | recall ≥99.5% of the union's true pairs |
| B15 | ❌ | Postal-code blocking | | | | dropped: codes present in only 0.5–10% of records |

---

## 3. Representation: contrastive bi-encoder (§2.3, §3.3)
| ID | Status | Approach | Owner | Depends on | Output | Done when |
|---|---|---|---|---|---|---|
| R0 | ✅ | Infrastructure: registry, triplet sampling (hard negatives), CachedMNRL fine-tuning, LoRA for Qwen, FAISS recall@k | | none | `ber.contrastive.*` | tested; all 5 models load on the H100 |
| R1 | 🔄 | **Bake-off: gte-multilingual-base** (zero-shot / fine-tuned / train-on-US→India) | | B2–B4 | `bakeoff/gte__*.json` | 3 result files |
| R2 | 🔄 | **Bake-off: nomic-embed-text-v2-moe** | | B2–B4 | `bakeoff/nomic__*.json` | 3 result files |
| R3 | 🔄 | **Bake-off: snowflake-arctic-embed-l-v2.0** | | B2–B4 | `bakeoff/arctic__*.json` | 3 result files |
| R4 | 🔄 | **Bake-off: bge-m3 (dense)** | | B2–B4 | `bakeoff/bgem3__*.json` | 3 result files |
| R5 | 🔄 | **Bake-off: Qwen3-Embedding-8B (LoRA)** ⚠ check the 8B limit with the organizers | | B2–B4 | `bakeoff/qwen8b__*.json` | 3 result files |
| R6 | ⬜ | **Qwen3-Embedding-4B** as the fallback if 8B is ruled over the limit | | R0 | `bakeoff/qwen4b__*.json` | 3 result files |
| R7 | ⬜ | **Matryoshka dimension sweep** (Qwen: 256/512/1024/4096) | | R5 | table | recall-vs-memory curve |
| R8 | ⬜ | **Full fine-tune of the winner** on all of A-train | | R1–R5 winner | model dir | A-val recall@50 reported |
| R9 | ⬜ | **SimCSE-style self-supervised pre-training on test records** (transductive, before supervised fine-tuning) | | R8 | model dir | leave-one-country-out recall change reported |
| R10 | ⬜ | **Symmetric vs asymmetric prefixes** (query/doc prompts vs the same prompt for both sides) | | R8 | table | recall change reported |

---

## 4. Matching models (§3): each matcher is independent → `oof_<model>` + `pred_<model>_test`
| ID | Status | Approach | Owner | Depends on | Output | Done when |
|---|---|---|---|---|---|---|
| M1 | ⬜ | **LightGBM matcher** on the feature groups + embedding (the combined version, main path), GroupKFold k=5 on B | | B14, F* | `oof_lgbm` | OOF macro F0.5 per country |
| M2 | ⬜ | **XGBoost / CatBoost** variants of M1 | | same as M1 | `oof_xgb`, `oof_cat` | OOF F0.5 vs M1 |
| M3 | ⬜ | **Bi-encoder cosine alone as the matcher** (the pure contrastive variant, baseline) | | R8, B14 | `oof_biencoder` | OOF F0.5 vs M1 |
| M4 | ⬜ | **Cross-encoder (Ditto-style)**: mdeberta-v3 / xlm-roberta on serialized pairs, with number/postal tagging | | B14 | `oof_crossenc` | OOF F0.5 vs M1 |
| M5 | ⬜ | **SupCon-EM / Sudowoodo**: contrastive pre-training + classifier head | | R0 | `oof_supcon` | OOF F0.5 vs M4 |
| M6 | ⬜ | **Research matchers**: HierGAT / JointBERT / Unicorn / AnyMatch (pick one) | | B14 | `oof_<name>` | OOF F0.5 vs M4 |
| M7 | ⬜ | **LLM judge (≤8B, MIT/Apache)**: Qwen2.5-7B / Qwen3-8B / Mistral-7B / Phi-4-mini, LoRA on yes/no, P(yes) read from the logits, **uncertain band only** | | M1 | `oof_llm` | F0.5 gain on the uncertain band |
| M8 | ⬜ | **Batched LLM prompting** (MatchGPT / BatchER: several pairs per prompt) to cut M7 cost | | M7 | runtime table | cost per pair vs M7 |
| M9 | ⬜ | **Fellegi–Sunter via Splink** (EM on comparison levels) | | B14 | `oof_splink` | OOF F0.5 vs M1 |
| M10 | ⬜ | **ZeroER** (unsupervised GMM over similarity features), used for France calibration | | F* | match-rate estimate for FR | FR predicted match rate vs M1 |
| M11 | ⬜ | **Stacker / blender** over the matcher outputs (logistic regression on out-of-fold p's) | | ≥2 of M* | `oof_stack` | OOF F0.5 vs best single model |

---

## 5. Decision layer (§5): each method is independent → `pred_lists_<method>`
| ID | Status | Approach | Owner | Depends on | Output | Done when |
|---|---|---|---|---|---|---|
| D0 | ✅ | One-owner constraint validated on the data (E3) | | none | | done |
| D1 | ⬜ | **Probability calibration** (isotonic / temperature) on OOF | | any M* | calibrator | reliability plot + Brier score |
| D2 | ⬜ | **Global threshold τ** tuned on OOF macro F0.5 (baseline decision) | | any M* | `pred_lists_thresh` | OOF F0.5 |
| D3 | ⬜ | **One-owner assignment**: each S2/S3 → its best S1 (greedy / Hungarian) | | any M* | `pred_lists_owner` | F0.5 gain over D2 |
| D4 | ⬜ | **"Has any match?" singleton model** (entity-level LightGBM on top-k probabilities, gaps, counts) + gate γ | | any M* | `pred_lists_gate` | F0.5 on singletons + overall |
| D5 | ⬜ | **Expected-F0.5-optimal list selection** per entity (Jansche 2007; Nan et al. 2012) | | D1 | `pred_lists_expf` | F0.5 gain over D2/D3 |
| D6 | ⬜ | **S2↔S3 collective clustering / graph consistency** (Bhattacharya & Getoor; correlation clustering) | | any M* + S2↔S3 scores | `pred_lists_graph` | F0.5 gain over D3 |
| D7 | ⬜ | **Per-country thresholds**, with a fallback rule for unseen countries | | D2 | config | F0.5 gain, and a documented rule for FR |

---

## 6. Training and validation protocol (§6)
| ID | Status | Approach | Owner | Depends on | Output | Done when |
|---|---|---|---|---|---|---|
| V1 | ✅ | A-train / A-val / B split, stratified by country and match count | | none | `splits.parquet` | done |
| V2 | ✅ | GroupKFold (k=5) fold ids on B | | V1 | `splits.parquet` | done |
| V3 | ✅ | Local scorer: official macro F0.5 + per-group breakdown | | none | `ber.evaluate` | matches the README example (0.714) |
| V4 | ✅ | Blocking metrics (pair completeness, candidates per entity) | | none | `ber.evaluate` | done |
| V5 | ⬜ | **Leave-one-country-out validation** for matchers (US→IN, IN→US) as the France proxy | | M1 | table | reported for every M* |
| V6 | ⬜ | **k-fold ensemble** (average of the fold models) for test predictions | | M1 | `pred_*_test` | done |
| V7 | ⬜ | **Retrain on all data** with tuned hyperparameters (vs V6), compared on A-val | | M1 | table | pick one |
| V8 | ⬜ | **Model soups** (weight averaging of fine-tuned transformers) | | R8 or M4 | model | recall/F0.5 vs a single model |

---

## 7. Generalizing to France (§4)
| ID | Status | Approach | Owner | Depends on | Output | Done when |
|---|---|---|---|---|---|---|
| G1 | ✅ | Country treated as an open set everywhere | | none | | done |
| G2 | ✅ | French lexicons (suffixes, street types, `rdc`) | | none | `lexicons/fr.yaml` | done |
| G3 | ✅ | Multilingual encoders only | | none | registry | done |
| G4 | ⬜ | **Per-country IDF** in all features (= F4) | | F4 | | |
| G5 | ⬜ | **Fit TF-IDF / IDF on train + test text** (unlabeled test is provided data) | | B2 | config flag | leave-one-country-out change reported |
| G6 | ⬜ | **Pseudo-labelling** high-confidence French test pairs, then a short fine-tune | | M1, R8 | model | FR predicted match rate stable, leave-one-country-out recall up |
| G7 | ⬜ | **Match-rate sanity check for FR** (predicted matches per entity vs US/IN, and vs ZeroER M10) | | M1 | table | no large drift |

---

## 8. Output and submission
| ID | Status | Approach | Owner | Depends on | Output | Done when |
|---|---|---|---|---|---|---|
| O1 | ⬜ | **Writer** for `matching_results.tsv` + `candidate_pairs.tsv` (matches ⊆ candidates) | | none (use a dummy prediction) | `ber.export` | official validator prints PASS on a dummy submission |
| O2 | ⬜ | **Validator as a pipeline step** (fail the run if it doesn't exit 0) | | O1 | CLI step | done |
| O3 | ⬜ | **End-to-end `python -m ber all`** | | all stages | CLI | one command regenerates both TSVs |
| O4 | ⬜ | **Submission zip**: `code/business_entity_resolution/{src,README.md,requirements.txt}` + `output/` | | O3 | zip | reproduced from a clean checkout |
| O5 | ⬜ | **Fill in `Documentation_template.md`** (method, blocking, model, features, error analysis) | | M1, D* | md | reviewed by the team |

---

## Infrastructure (done)
| ID | Status | Item |
|---|---|---|
| I1 | ✅ | `ber` package, YAML configs, `python -m ber <cmd>` CLI |
| I2 | ✅ | `scripts/hpc.sh` (laptop → HPC sync + run), SLURM `cpu`/`gpu` wrappers |
| I3 | ✅ | HPC conda env `ber`; transformers pinned <5 (remote-code models break on 5.x) |
| I4 | ✅ | 39 unit tests passing |

---

## Items anyone can start today (no dependencies)
**N11–N18** (normalization columns), **E8, E10** (data profiling), **O1** (writer + validator on a dummy submission), and **B5, B7, B8** (new blockers; they only need `records`).

Once `cand_sparse` lands, all of **F1–F10** can run in parallel, and so can **E9** and **B13**.
