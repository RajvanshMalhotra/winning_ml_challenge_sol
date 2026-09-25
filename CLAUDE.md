# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is the working directory for the **Amazon ML Challenge: Business Entity Resolution**. It holds no code yet. The spec below comes from the official problem statement:
https://d8it4huxumps7.cloudfront.net/files/6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf

When code is added, update this file with the real commands and architecture.

## Task

Business records come from 3 independent sources. The records have noisy fields and share no common IDs. **Source 1 is the deduplicated reference source.** For each Source 1 entity, find every record in Source 2 and Source 3 that refers to the same real-world business. A Source 1 entity can match zero, one, or many records.

### Data layout (provided in `student_resource/`)
- `dataset/train/train_source{1,2,3}.tsv` and `dataset/train/train_ground_truth.tsv`
- `dataset/test/test_source{1,2,3}.tsv` (no labels)
- Source file columns: `entity_id` (prefix `S1-`/`S2-`/`S3-` gives the source; there is no separate source column), `business_name`, `business_address`, `country`
- Ground truth columns: `source1_entity_id`, `matched_entity_ids` (comma-separated, empty for singletons)

### Gotchas
- **Every file is TSV.** Always read with `pd.read_csv(path, sep="\t")`. Without `sep="\t"`, pandas silently loads each line as a single column. Addresses and ID lists contain commas.
- Watch for empty `matched_entity_ids` cells (NaN in pandas). Treat them as an empty list.
- **Country is an open set.** Train has only US and India. Test adds **France**, which never appears in train. Don't hard-code, filter, or one-hot encode country to {US, India}. Every test entity, France included, must appear in the output.
- Name noise: abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited), legal suffixes, DBA/trade names, `&` vs "and", word-order swaps, typos, transliterations.
- Address noise: Rd/Road and St/Street, transliteration, missing PIN code or state, landmark references ("Near SBI ATM"), different municipal numbering, reordered components.

## Metric

**F0.5, macro-averaged per Source 1 entity**, singletons included. Precision counts 2× as much as recall.
- A singleton scores 1.0 when you predict an empty list, and 0.0 when you predict any match. False merges on singletons are expensive.
- Set match thresholds for precision. No test labels exist, so hold out a validation split from train and compute this metric locally.

## Required outputs (`output/`)

Both files are tab-separated, with one row per test Source 1 entity and no duplicate rows:
1. `matching_results.tsv`: columns `source1_entity_id`, `matched_entity_ids`. This is the only file that gets scored, and it's what you upload to the portal.
2. `candidate_pairs.tsv`: columns `source1_entity_id`, `candidate_entity_ids`. This is the **final** candidate set the matching model runs inference on, after all blocking and filtering stages. It isn't scored, but the organizers use it to audit blocking (recall ceiling, reduction ratio).

Rules for both files:
- ID lists are comma-separated and unquoted.
- A cell is empty when there are no IDs.
- Lists contain only S2/S3 IDs that exist in the test set.
- No duplicates within a list.
- Matches must be a subset of candidates.

Validate before every submission (stdlib only, run from `student_resource/`):
```
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Exit 0 with `PASS` means the files are OK. This checks format only, not score.

## Intended pipeline shape

Data → normalization → **blocking / candidate generation** (this sets the recall ceiling) → pairwise features (Jaccard, Levenshtein, TF-IDF cosine on name and address; country-aware address handling) → match classifier → precision-tuned threshold → both output files.

## Hard constraints

- **Don't use external data lookups:** no entity-resolution APIs, government registries, geocoding APIs, or internet data augmentation. Any of these disqualifies the team. Use only the provided data.
- The final model must be **MIT or Apache 2.0 licensed and have at most 8B parameters.**
- The final zip must be reproducible end to end:
  ```
  <team_name>_submission.zip
  ├── output/{matching_results.tsv, candidate_pairs.tsv}
  ├── code/business_entity_resolution/{src/, README.md, requirements.txt (pinned)}
  └── Documentation_template.md   # methodology: approach, blocking strategy, model + features
  ```
  Keep all source code under `code/business_entity_resolution/src/`. Keep `requirements.txt` pinned.
