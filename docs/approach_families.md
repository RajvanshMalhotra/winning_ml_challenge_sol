# Approach Families: The Big Picture

This is the plain-words overview of every family of approaches in `docs/approaches.md`, with the research behind each one and where we stand. For the detailed, assignable work items, see `docs/approach_tracker.md`. The **Tracker IDs** column links the two.

**Legend:** ✅ done 🔄 running / in progress ⬜ not started

_Last updated: 2026-09-25_

| # | Approach family | Plain-words idea | Key papers / tools | Role | Status | Tracker IDs |
|---|---|---|---|---|---|---|
| 1 | **Probabilistic record linkage** | For each field (name, address), learn how often it agrees when two records are the same business vs different ones; multiply the evidence into a match probability. No labels needed. | Fellegi & Sunter 1969; **Splink** (UK Ministry of Justice) | Full solution | ⬜ | M9 |
| 2 | **Unsupervised clustering of similarity scores** | Compute similarity scores for all pairs and let a statistical model split them into a "match" group and a "non-match" group. | **ZeroER** (Wu et al., 2020) | Full solution; useful for France (no labels) | ⬜ | M10 |
| 3 | **Hand-made similarity features + tree model** | Compute dozens of scores (spelling distance, word overlap, same house number…) and train a classifier like LightGBM. | **Magellan** (Konda et al., 2016); string metrics (Cohen et al., 2003) | Full solution (**our main path**) | 🔄 data cleanup ✅ done; features + LightGBM ⬜ next | N1–N18, F1–F12, M1–M2 |
| 4 | **Deep-learning matchers** | A neural network compares the two records field by field and learns what matters. | **DeepMatcher** (Mudgal et al., 2018) | Full solution (older, mostly replaced by #5) | ⬜ low priority | M6 |
| 5 | **Cross-encoder with a pretrained language model** | Feed both records together into a BERT-style model and ask "same or not?". It reads both at once, so it's very accurate but slow. | **Ditto** (Li et al., 2020); JointBERT, HierGAT (2021); **Unicorn** (2023); AnyMatch (2024) | Matcher | ⬜ | M4, M6 |
| 6 | **Contrastive learning** | Train a model to place records of the same business close together and different ones far apart. Fast search, and it copes well with noise. | **Sudowoodo** (Wang et al., 2023); **SupCon** (Peeters & Bizer, 2022); **DeepBlocker** (2021) | Candidate search + matcher input (**our bi-encoder**) | 🔄 code ✅ done, all 5 models load on the GPU ✅; bake-off waiting for blocking | R0–R10, B9–B11, F11, M3, M5 |
| 7 | **Large language models as judges** | Ask a small LLM (≤8B), fine-tuned on our labels, "are these the same business?". It generalizes well to new domains like France, but is expensive. | Peeters, Steiner & Bizer 2025; **MatchGPT / BatchER** (2024); Jellyfish (2024) | Matcher (unsure cases only) | ⬜ | M7, M8 |
| 8 | **Blocking research** | Clever ways to find likely pairs without comparing everything: alphabetical neighbours, hashing, pruning weak links. | Sorted neighbourhood (Hernández & Stolfo, 1995); **Meta-blocking** survey (Papadakis et al., 2020); MinHash LSH | Candidate generation | 🔄 country split ✅; TF-IDF + key blocks running on the full data; others ⬜ | B1–B15 |
| 9 | **Collective / graph entity resolution** | Don't decide each pair alone: use the whole network (if A=B and B=C then A=C) to fix inconsistent decisions. | **Collective ER** (Bhattacharya & Getoor, 2007); correlation clustering | Decision layer | ⬜ (the "each record belongs to one business" rule is confirmed on the data ✅) | D3, D6 |
| 10 | **Metric-optimized decisions** | Choose each business's match list to maximize the expected F0.5 score directly, rather than using a fixed 0.5 cut-off. | **Jansche 2007**; **Nan et al., ICML 2012** | Decision layer | ⬜ (official F0.5 scorer ✅ built) | D1, D2, D4, D5, D7 |
| 11 | **Ensembling / model soups** | Train several models (different data slices or settings) and average their predictions, or their weights. | k-fold bagging; **Model Soups** (Wortsman et al., 2022) | On top of any matcher | ⬜ (5-fold split ✅ ready) | V6–V8, M11 |
| 12 | **Domain adaptation (for France)** | Make models work on an unseen country: multilingual models, fake French-style noise, learning from the unlabeled French test data. | Ditto / Sudowoodo augmentation; self-training / pseudo-labelling; SimCSE | Add-on for generalization | 🔄 French word lists ✅, multilingual models ✅, noise generator ✅; US→India check runs in the bake-off; learning from test data ⬜ | A1–A5, G1–G7, R9 |

---

## What we're building first
A combination of the families above:
- **#3 (similarity features + LightGBM)** as the matcher
- **#6 (contrastive bi-encoder)** feeding it extra candidates and a similarity score
- **#8** for finding candidates
- **#9 + #10** for the final per-business decisions

```
clean data → find candidates (TF-IDF + keys + bi-encoder) → drop obvious non-matches
           → LightGBM on similarity features (+ bi-encoder score) → final lists (one-owner rule, empty-list model, F0.5-tuned cut-off)
```

## What the team can try in parallel
Each of these is a separate option. It works on the same candidate pairs, writes predictions in the shared format (see `approach_tracker.md` §0), and gets compared on the same F0.5 score, or blended with the rest:
- **#1 Splink**
- **#2 ZeroER**
- **#5 cross-encoder (Ditto-style)**
- **#7 LLM judge**, on unsure cases only
- **#11 blending** of all of the above

## Where things stand right now
- ✅ Data explored; all 24M records cleaned and standardized; train/validation split done.
- ✅ Official F0.5 scorer, noise generator and bi-encoder training code built; 39 tests passing.
- ✅ All 5 candidate embedding models (gte, nomic, arctic, bge-m3, Qwen3-8B) run on the H100.
- 🔄 Candidate search (TF-IDF + key blocks) running on the full data. It's slow, so we're testing a faster setting that drops very common letter-chunks.
- ⬜ Next: bi-encoder bake-off → similarity features → LightGBM matcher → final decision layer → submission files.
