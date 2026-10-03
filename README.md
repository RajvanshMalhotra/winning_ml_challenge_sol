# Amazon ML Challenge 2026: Business Entity Resolution (Rank 313)

Team **Attention Seekers**. Final public leaderboard score: **0.987427** macro F0.5.

## Problem statement

Business records come from three independent sources with noisy fields and no shared IDs. Source 1 is a deduplicated reference set. For every Source 1 business, find all records in Source 2 and Source 3 that refer to the same real-world business. A business can match zero, one or many records.

- Fields: `business_name`, `business_address`, `country`.
- Name noise: abbreviations, legal suffixes, trade names, `&` vs "and", word swaps, typos, transliteration, and sometimes a name fully replaced by a gibberish token.
- Address noise: Rd/Road style abbreviations, reordering, dropped components, landmark references, house-number typos, missing PIN/state, and about 3% empty addresses.
- Train covers the US and India. Test adds **France**, which never appears in train.
- Metric: **F0.5, macro-averaged per Source 1 business**, singletons included. Precision counts twice as much as recall, and any wrong match on a singleton scores 0.
- Constraints: no external data or APIs; models must be MIT/Apache-2.0 and at most 8B parameters.

## Approach

```
normalise -> multi-channel blocking -> LightGBM pair filter -> fine-tuned Qwen3-Reranker-4B
          -> stacker with candidate-side competition features -> threshold + one-owner rule
```

1. **Blocking (99.9% recall, ~132 candidates per business).** Runs per country and learned state group. Channels: character 3-5-gram TF-IDF, exact keys (rare tokens, postal code + house number, phonetic name keys), name-only search for empty addresses, Qwen3-Embedding-0.6B search for transliterated names, graph expansion over record-record edges, and dense retrieval with a contrastively fine-tuned bge-m3.
2. **Model A (LightGBM, 48 features).** RapidFuzz and IDF name similarities, address/house-number/state/unit comparisons, legal-suffix flags, blocking-channel signals and bge-m3 cosine. Pairs with p >= 0.01 are kept: 5.1 candidates per business, 99.5% recall. This set is `candidate_pairs.tsv`.
3. **Reranker.** Qwen3-Reranker-4B fine-tuned with LoRA on 1M pairs, using hard negatives (namesakes, wrong owners, same building) and hard positives (address-less copies, gibberish names, digit typos). It scores every uncertain pair. AUC on uncertain pairs: 0.976, against 0.907 for a fine-tuned bge-reranker and 0.689 for zero-shot Qwen.
4. **Candidate-side competition.** For each pair the stacker sees how this business ranks among *all* businesses that want the same record, and by what margin. This turns pairwise scoring into an implicit assignment and gave the largest single gain (+0.005).
5. **Test-like stacker training.** Test has far more records whose owner is missing from Source 1. The stacker is trained on frames where 20% of training businesses are hidden, so it learns that an uncontested record can still be a decoy.
6. **France.** France has no labels, so the pipeline stays country-agnostic and multilingual. On French rows only, the base Qwen score is averaged with a copy fine-tuned on a synthetic French practice set, and acronym records (`AJ` -> *Association du Jeu*) are routed to the reranker.
7. **Decision.** The threshold (tau = 0.75) is tuned for macro F0.5 on out-of-fold predictions. Each Source 2/3 record is then assigned to at most one business.

## Results

| Version | Main change | OOF F0.5 | Leaderboard |
|---|---|---|---|
| v2 | LightGBM + fine-tuned bge-m3 features | 0.9707 | 0.962 |
| v3 | + bge cross-encoder on uncertain pairs | 0.9784 | 0.973 |
| v5 | + dense bge-m3 blocking (recall 98.3% -> 99.9%) | 0.9834 | 0.979 |
| v8 | + candidate-side competition features | 0.9885 | 0.983 |
| v12 | + fine-tuned Qwen3-Reranker-4B | 0.9904 | 0.9864 |
| v14 | + stacker trained on hidden-owner frames | 0.9901 | 0.98663 |
| v16 | + three rerankers, stronger stacker | 0.9904 | 0.98709 |
| **final** | + French-tuned Qwen blend, acronym routing | - | **0.987427** |

Ideas that were tried and did not help are listed in [docs/Documentation_final.md](docs/Documentation_final.md), along with a full write-up of the method and error analysis. Per-experiment notes are in [docs/results/](docs/results/).

## Repository layout

```
src/ber/          core package: normalisation, splits, blocking, features, LightGBM matcher, prediction, metric
src/ber/contrastive/  bi-encoder fine-tuning
scripts/          pipeline stages; run_all.sh runs them end to end
configs/          YAML configs
tests/            pytest suite
submissions/      submitted output files per version
docs/             methodology and experiment notes
```

## Reproducing

Requires Python 3.11, Linux, one 80 GB GPU and about 500 GB RAM for full-scale scoring.

```bash
pip install -r requirements.txt && pip install -e .
# put the official student_resource/ folder in place (see docs/package_README.md)
bash scripts/run_all.sh
pytest tests -q
```

Reranker scoring uses vLLM in a separate environment; see [docs/package_README.md](docs/package_README.md) for that setup and the data paths.

Models used: bge-m3 (MIT), bge-reranker-v2-m3, Qwen3-Embedding-0.6B and Qwen3-Reranker-4B (all Apache-2.0). No external data or APIs.
