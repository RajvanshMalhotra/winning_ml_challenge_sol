# Design: Bucketing + Contrastive + GBDT Entity-Resolution Pipeline (v1)

**Status:** approved (2026-09-25)
**Background:** `docs/approaches.md` (survey + EDA findings §8a)

## 1. Goal and success criteria

Build the pipeline the team proposed:

bucketing → filter out obvious negatives → a contrastive bi-encoder whose score feeds a LightGBM matcher → a has-match ("empty list") decision → k-fold ensembling

It produces `output/matching_results.tsv` and `output/candidate_pairs.tsv` for the test set.

Success means:
- Both output files pass `student_resource/utils/validate_submission.py`.
- A local out-of-fold (OOF) **macro F0.5** is reported: overall, per country, singletons vs non-singletons, and leave-one-country-out.
- **Blocking pair completeness ≥ 99%** on training entities, with the mean candidates per entity kept small. The exact cap will be tuned against completeness.
- The bi-encoder model is chosen by bake-off evidence (§5), not by leaderboard reputation.

## 2. Facts from EDA the design relies on
- Every S2/S3 record belongs to **at most one** S1 entity, and **no link crosses countries**.
- **5.6% singletons.** A typical entity has 3–5 matches, up to 5 in S2 and 6 in S3.
- About 26% of S2/S3 records are distractors that match nothing.
- Postal codes are almost absent. House number, street, city and name tokens carry the signal.
- The noise comes from a finite set of synthetic operators (listed in approaches §8a).
- The test set adds **France**, 15% of test S1, which never appears in train. Country is an open set of strings.

## 3. Data split (leakage control)

Training S1 entities are split **by S1 id**, stratified by country and by match-count bucket (0, 1, 2–3, 4–5, 6+):

| Split | Share | Used for |
|---|---|---|
| **A-train** | 35% | bi-encoder fine-tuning, and hard negatives for it |
| **A-val** | 5% | bi-encoder bake-off and model selection |
| **B** | 60% | prefilter + matcher + has-match training via **GroupKFold k=5**; all reported F0.5 numbers |

The bi-encoder never sees B entities, so its cosine feature on B behaves as it will on the test set.

**Search pool:** retrieval for any S1 entity searches **all** S2/S3 records of that entity's country in the same split family (train or test), distractors included. Matching density then mirrors the test set.

## 4. Pipeline stages

Each stage is a module under `src/ber/`. Each reads and writes Parquet under `artifacts/<run_name>/` (gitignored). All parameters come from one YAML config under `configs/`. A stage skips its work if its output already exists, unless `--force` is passed.

### 4.1 `normalize`
Input: raw TSVs. Output: `records.parquet`, one row per record, with these columns:
- `entity_id`, `source` (1/2/3), `country`
- `name_raw`, `addr_raw`
- `name_norm`: lowercased, NFKC, accents folded, punctuation normalized, `NULL` removed, filler words (`the`, `mr`, `m/s`, …) removed
- `name_core`: `name_norm` without the legal suffix
- `legal_suffix`: a canonical class such as `pvt_ltd`, `llc`, `inc`, `sarl`, `sas`, `sci`, …
- `name_domain_split`: `fafloonpetcare.com` → `fafloon pet care` via a greedy segmentation over a vocabulary built from S1 names in the same country
- `addr_norm`: street-type and state abbreviations expanded, `NULL` removed
- `house_no`: the first numeric house/door token, canonicalized (`01130`→`1130`, `4-7/1`→`47/1`)
- `num_tokens`: all numeric tokens
- `addr_tokens`
- `encoder_text` = `"name: {name_raw} | address: {addr_raw} | country: {country}"`. The encoder sees raw text, and augmentation supplies the robustness.

Lexicons live in `src/ber/lexicons/`, one file per language: `en.yaml`, `in.yaml`, `fr.yaml`. **Every lexicon is applied to every record regardless of country**, so an unseen country falls back gracefully. Normalization uses regexes only: no libpostal and no external lookups.

### 4.2 `blocking.sparse` (bucketing, part 1)
Candidate generators, each run **within one country string**:
1. **Character TF-IDF top-k.** Analyzer `char_wb` 3–5-grams over `name_core + addr_norm`, fit per country on S1+S2+S3 text of the split. Take top-k cosine neighbours of each S1 record among S2 ∪ S3, using a chunked sparse matmul (`sparse_dot_topn`).
2. **Key blocks.** Records sharing (`house_no`, any `addr_token` with document frequency < a cap), or (a rare `name_core` token, any shared place token). Blocks bigger than a size cap are dropped, since they aren't discriminative.

Output: `cand_sparse.parquet` with columns `s1_id`, `cand_id` and per-generator score/rank.

### 4.3 `contrastive` (bi-encoder)
- **Training data (A-train only):**
  - Positive pairs are all within-entity pairs: S1↔S2, S1↔S3, S2↔S3, S2↔S2, S3↔S3.
  - Hard negatives are sparse-blocking candidates of an A-train S1 that aren't its matches.
- **Augmentation (`src/ber/noise.py`)** applies stochastic noise operators to one side of a pair. The operators mirror the EDA: case change, `NULL` insertion, filler word, token drop or truncation, domain-style name, accent injection, char typo, house-number reformat, state or street-type abbreviate↔expand, component reorder, unit/PO box add or drop. Each operator is a pure function `(text, rng) -> text`.
- **Loss:** `MultipleNegativesRankingLoss` with hard negatives. For Qwen3-Embedding-8B use `CachedMultipleNegativesRankingLoss` (GradCache) + LoRA + gradient checkpointing.
- **Artifacts:** a fine-tuned model directory, plus `emb_<split>.npy` (float16) per record.
- The model is chosen by the bake-off (§5).

### 4.4 `blocking.dense` + union (bucketing, part 2)
- A FAISS index per country over S2 ∪ S3 embeddings. Exact inner product (`IndexFlatIP`, GPU if memory allows). Switch to HNSW or IVF only if exact search is too slow.
- Retrieve the top-k for each S1.
- Union with `cand_sparse`, keeping every generator's score and rank. Output: `cand_union.parquet`.
- Report pair completeness, mean and p95 candidates per entity, and each generator's marginal recall, per country.

### 4.5 `features`
Pairwise features for (S1, candidate):
- **Cheap set (prefilter):**
  - TF-IDF cosine
  - dense cosine
  - generator ranks
  - `house_no` equal / unequal / missing (three-state)
  - name-token Jaccard
  - address-token Jaccard
  - numeric-token Jaccard
- **Full set (matcher):** the cheap set plus
  - rapidfuzz ratio / partial / token_sort / token_set on name_core and addr_norm
  - Jaro-Winkler
  - IDF-weighted token overlap, with IDF computed per country on that country's text
  - legal-suffix class equal / unequal / missing
  - `name_domain_split` similarity
  - length ratios
  - candidate source (S2/S3)
  - **context features:** candidate rank within the S1's list, gap to that S1's best score, how many S1s rank this candidate in their top-3, and a mutual-best flag

Country is **never** a feature (it's an open set).

### 4.6 `prefilter` (filter out obvious negatives)
- A LightGBM model on the cheap features, trained on B with GroupKFold k=5.
- Pick the threshold on OOF predictions to keep **99.5% recall** of the true pairs present in `cand_union`.
- The surviving pairs are the **final candidate set**. On test they are written to `candidate_pairs.tsv`.

### 4.7 `matcher` (combined version)
- A LightGBM model on the full features, over prefiltered pairs of B, GroupKFold k=5 by S1 id.
- The OOF probabilities are **isotonic-calibrated**.
- Test prediction = the mean of the 5 fold models' probabilities, passed through the calibrator fitted on OOF.

### 4.8 `hasmatch` (empty lists)
- An entity-level LightGBM on B, using OOF matcher outputs. Features:
  - top-1, top-2 and top-3 probabilities
  - gap between top-1 and top-2
  - count of candidates above 0.3 and above 0.5
  - candidate count
  - the entity's own record features (address length, whether it has a `house_no`, `legal_suffix` present)
- Target: whether the entity has at least one true match.
- Its inputs are already OOF, so it is trained with the same GroupKFold split on B, and its own OOF output is what §4.9 tunes against.

### 4.9 `decide`
1. **One owner:** each S2/S3 candidate is kept only for the S1 with its highest calibrated probability. Ties go to the lowest `s1_id`.
2. **Gate:** if P(has match) < γ, the entity gets an empty list.
3. **Threshold:** otherwise, keep the owned candidates with p ≥ τ.
4. **Tuning:** grid-search τ and γ on B's OOF to maximize macro F0.5.
5. **Variant (config flag):** expected-F0.5-optimal prefix selection per entity (approaches §5.3), replacing the τ threshold.

### 4.10 `evaluate`
- `macro_f05(pred: dict[s1, set], truth: dict[s1, set]) -> float`. It follows the official definition exactly: per-entity F0.5, where an empty prediction on an empty truth scores 1.0 and any prediction on an empty truth scores 0.0, averaged over all entities.
- Breakdowns by country, by singleton status, and by match-count bucket.
- Blocking metrics: pair completeness and reduction ratio.

### 4.11 `export`
- Write both TSVs in the exact format: one row per test S1, comma-joined IDs, an empty string when there are none.
- Assert that matches ⊆ candidates, then run the official validator as a subprocess and fail the stage if it doesn't exit 0.

### 4.12 Orchestration
- `src/ber/cli.py` runs `python -m ber <stage> --config configs/<name>.yaml` and `python -m ber all`.
- `slurm/` holds sbatch wrappers. CPU stages go to the `batch` partition; bi-encoder training and embedding go to the `gpu` partition with the GPU memory capped by config, because the H100 is shared.

## 5. Bi-encoder bake-off (first build task)

**Candidates, all MIT/Apache 2.0:**
- Alibaba-NLP/gte-multilingual-base
- nomic-ai/nomic-embed-text-v2-moe
- Snowflake/snowflake-arctic-embed-l-v2.0
- BAAI/bge-m3 (dense output)
- Qwen/Qwen3-Embedding-8B (LoRA; MRL dims evaluated at 1024 and full)

Each model uses its documented query/document prefixes.

**Protocol:**
1. **Zero-shot:** embed A-val S1 and all train S2/S3 of the matching country. Report recall@{10, 50, 100} per country.
2. **Fine-tune** every candidate with the same recipe:
   - the same A-train sample (a fixed number of entities, set in config)
   - the same hard negatives, augmentation and step budget

   Qwen gets LoRA and GradCache. The others get full fine-tuning.
3. **Evaluate** fine-tuned recall@{10, 50, 100} on A-val per country.
4. **Leave-one-country-out:** fine-tune on A-train US only, then evaluate on A-val India.
5. **Record** encode throughput (records/s on the H100 at the configured memory cap), vector memory for the test set, and peak training memory.

**Selection rule:** highest mean fine-tuned recall@50 across countries. If the top models are within 0.5 points, the leave-one-country-out result decides, and then throughput. The results table is saved to `docs/results/bakeoff.md`.

## 6. Repository layout
```
src/ber/            normalize.py, lexicons/, noise.py, blocking/{sparse,dense}.py,
                    contrastive/{train,embed,bakeoff}.py, features.py, prefilter.py,
                    matcher.py, hasmatch.py, decide.py, evaluate.py, export.py, cli.py
configs/            base.yaml (+ per-experiment overrides)
slurm/              sbatch wrappers
tests/              pytest unit tests with tiny fixtures
artifacts/          (gitignored) intermediate parquet, models, embeddings
data/               (gitignored) student_resource/
requirements.txt    pinned
```
The repo root maps directly onto `code/business_entity_resolution/` in the final submission zip.

## 7. Testing
Pytest, with small hand-built fixtures and no real data. Tests cover:
- `evaluate`: the README example gives 0.714; the singleton rules hold; the empty/empty case scores 1.0.
- `normalize`: suffix, abbreviation and house-number canonicalization, `NULL`/filler removal, domain splitting.
- `noise`: every operator is deterministic under a seed and never returns an empty string.
- `decide`: the one-owner rule, the gate, and the threshold.
- `export`: fixture outputs pass the official validator, and every match appears among the candidates.
- `blocking`: on a tiny fixture, known pairs are retrieved and the country partition is respected.

Each stage on real data also logs its metrics (completeness, OOF F0.5) to `artifacts/<run>/metrics.json`.

## 8. Environment
- A dedicated conda env `ber` (Python 3.11) on the HPC, pinned by `requirements.txt`. Main packages: pandas, pyarrow, scikit-learn, sparse_dot_topn, rapidfuzz, lightgbm, faiss, torch, sentence-transformers, peft, pyyaml, pytest.
- Model weights are downloaded from the Hugging Face Hub. They are pretrained models, not entity lookups.
- The HPC has no GitHub credentials, so code is synced laptop→HPC with rsync. Git remains the source of truth on the laptop.

## 9. Out of scope for v1 (tracked for later)
- A cross-encoder matcher
- An LLM judge (≤8B) for the uncertain band
- S2↔S3 collective clustering / graph consistency
- Transductive self-training on test France
- bge-m3 sparse and ColBERT outputs

## 10. Open issues
- **Qwen3-Embedding-8B parameter count:** Qwen reports the Qwen3-8B base as about 8.2B, but the rule says "up to 8B". Confirm with the organizers before shipping it as the final model. If they say no, fall back to Qwen3-Embedding-4B.
