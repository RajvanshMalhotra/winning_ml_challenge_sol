# Bi-Encoder Approach: Simple-Words Summary

## What is a bi-encoder?
A bi-encoder is a model that turns each business record into a list of numbers (a "vector"). It reads **one record at a time**:

```
"name: Galaxy Solutions Pvt Ltd | address: 47/1 Airport Road, Kolhapur | country: India"
      → model → [0.12, -0.03, …]   (768 or 1024 numbers)
```

We train it so that records of the **same business get similar vectors** and different businesses get different ones. To check whether two records match, we measure how close their vectors are.

**Why this kind of model?** It reads each record only once (about 12M records), rather than every pair (about 50M pairs). The vectors can then be searched very fast.

## How we train it
- **Positive examples (same business):** from the ground truth, the S1 record paired with each of its S2/S3 matches, and also S2/S3 records of the same business paired with each other.
- **Negative examples (different businesses):**
  - every other record in the same training batch
  - "hard negatives": lookalikes from the candidate search (same street, similar name, but a different business)
- **Fake noise:** we randomly add the same kinds of mess the data has (typos, ALL CAPS, `NULL`, accents, dropped words, `Rd`↔`Road`, `TX`↔`Texas`), so the model learns to ignore noise and doesn't memorize US/India names.
- **Training data:** only the A-train split (35% of training businesses). It never sees the B split, which the rest of the pipeline is tested on.

## Which model? (the bake-off)
We compare 5 open models (all MIT/Apache licensed):

| Model | Size |
|---|---|
| gte-multilingual-base | 305M |
| nomic-embed-text-v2-moe | 475M |
| snowflake-arctic-embed-l-v2.0 | 568M |
| bge-m3 | 568M |
| Qwen3-Embedding-8B (LoRA) | ~8B (⚠ confirm the 8B limit with the organizers) |

Each model is tested **before training (zero-shot)**, **after training**, and **trained on US only, then tested on India**. The best one goes into the pipeline.

## How we use it in the pipeline
It does **2 jobs**:
1. **Finding candidates:** for each S1 record, find the nearest S2/S3 records by vector. This catches matches that plain text matching misses (`fafloonpetcare.com` ↔ `Fafloon Pet Care Inc`).
2. **Giving a score:** its similarity score is one input to the final LightGBM matcher, alongside the spelling and house-number checks.

The bi-encoder is **one part** of the pipeline, not the whole thing:
```
clean data → find candidates (text match + keys + bi-encoder) → drop obvious non-matches
           → LightGBM decides match / no match (uses the bi-encoder score) → final lists
```

---

## How we test

| Stage | What's tested | Data | Score |
|---|---|---|---|
| Candidate search | each search method, and all of them combined | all training businesses | share of true matches kept; candidates per business |
| Bi-encoder bake-off | the 5 models | A-val: 5% of training businesses the model never trained on | recall@10/50/100 (share of true matches in the top 10/50/100), per country |
| Matcher (LightGBM) | the full pipeline | B: 60% of training businesses, in 5 parts; each part is scored by a model that didn't train on it | **macro F0.5**, the official score, per country and for no-match vs has-match businesses |
| France stand-in | the full pipeline | train on US only, test on India (and the reverse) | recall and F0.5 on the held-out country |

**What we finally ship** is the whole pipeline: clean data → find candidates → bi-encoder (the bake-off winner) → LightGBM → final lists. The bake-off only picks which embedding model goes inside.

## Testing France
**We can't score France directly.** France appears only in the test set, which has no answers, so only the leaderboard can score it. Instead we:
1. **Pretend a country is new:** train on US, test on India. This shows how well we handle a country we never trained on.
2. **Check that the French predictions look normal:** the number of matches per business and the share of "no match" businesses should look like US/India. If France gets far fewer matches, our cut-offs are off.

## Does France search everything?
**No. The search is cut down in two steps.**

```
France S1 (259k)  ──searches only──►  France S2+S3 (703k + 732k ≈ 1.43M)
                                       not all ~10M test records
```

1. **Split by country:** every French S2/S3 record is turned into a vector once (about 1.43M vectors) and stored in a **France-only search index**. US and India get their own indexes.
2. **Keep only the nearest ones:** each French S1 record searches only the French index and keeps its **top 50** closest records. The text-matching and key searches also run within France only.
3. **Merge and filter:** the candidate lists are combined, and a quick filter throws out obvious non-matches, usually leaving tens of candidates per business.
4. **Final decision:** LightGBM scores **only those candidates**, never all the vectors, and the final lists are built from its scores.

So the search goes from **10M → 1.4M** (country) → **about 50 per business** (nearest neighbours).

**Is splitting by country safe?** Yes. In the training data, **no business ever matches a record from another country**. We don't hard-code any country list either: we group by whatever country names appear, so France gets its own group automatically.
