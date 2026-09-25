# Business Entity Resolution: A Survey of Workable Approaches

This document is for brainstorming. It lists approaches from the research literature and from production entity-resolution (ER) systems, and says which ones fit this challenge. It ends with a recommended stack and the questions the data needs to answer before we pick a design.

Some constraints shape every choice below:
- The score is **macro F0.5 per Source 1 entity**. Precision counts 2× as much as recall, and each entity counts equally.
- **Singletons** (Source 1 entities with no match) score 1.0 or 0.0.
- **France** appears only in the test set.
- **External lookups are not allowed.**
- The final model must be **MIT or Apache 2.0 licensed and at most 8B parameters.**

---

## 0. How the problem breaks down

Every serious ER system, from research systems (Magellan, Ditto, Splink) to production ones (Amazon, Google Knowledge Graph, Senzing), splits the work into the same stages:

```
normalize → block (candidate generation) → pairwise match (score) → decide / cluster → output
             ↑ sets the recall ceiling      ↑ precision lives here   ↑ the metric is scored here
```

Your idea maps onto these stages like this:

| Your idea | Stage | Comment |
|---|---|---|
| Bucketing | Blocking | Yes. Use several complementary blockers, not one key (§2). |
| Score empty lists first | Decision | Good instinct, but "empty" can only be decided *after* scoring candidates. Make it an explicit decision layer (§5). |
| Filter obvious negatives | Cascade blocking / meta-blocking | Yes. A cheap classifier prunes, then an expensive model scores what's left (§2.5). |
| Contrastive learning | Embedding-based blocking + representation | Strongest in the blocking stage (§2.3). It can also be the matcher (§3.3). |
| k random subsets → merge into one model | Group k-fold ensembling / bagging | Yes. Split **by Source 1 entity** to avoid leakage. You also get out-of-fold predictions for tuning thresholds (§6). |

---

## 1. Normalization (cheap, high-leverage)

Almost every pipeline stage improves when its input is normalized first. Keep **both** raw and normalized versions, since some models (cross-encoders, LLMs) do better on raw text.

**Names**
- Lowercase, Unicode NFKC, strip accents (`unidecode`). Strip accents *carefully* for French (é→e): it helps matching, but keep the original text for the language models.
- Legal-suffix canonicalization. Build a lexicon: `pvt/private`, `ltd/limited`, `corp/corporation`, `inc/incorporated`, `llc`, `llp`, `co/company`, `&/and`. Add French forms up front: `SARL`, `SAS`, `SA`, `EURL`, `SCI`, `Société/Ste/Sté`, `Cie`, `et`. Libraries like `cleanco` (MIT) already have multi-country suffix lists. Two useful variants:
  - Produce a name **with the suffix removed**. This "core name" becomes a feature and a blocking key.
  - Produce a separate **suffix-class** feature, since a Ltd vs an LLC mismatch is itself evidence.
- DBA/trade names: split on `dba`, `d/b/a`, `t/a`, `trading as`, `(…)`, and on French `enseigne`. Match against each part.
- Word-order transpositions: use token-sorted strings, or token-set similarity.

**Addresses**
- Abbreviation expansion: `rd→road`, `st→street`, `ave`, `blvd`, `nr→near`, `opp→opposite`, `bldg`. French: `r→rue`, `av→avenue`, `bd→boulevard`, `pl→place`, `ch→chemin`, `bis/ter`.
- Pull out **structured tokens with regexes**. These are the strongest precision signals:
  - Indian PIN codes (6 digits)
  - US ZIP codes (5 or 5+4 digits)
  - French postal codes (5 digits, the first 2 give the département)
  - house and unit numbers
  - state codes
- Landmark phrases (`near X`, `opp X`, `behind X`) are high noise. Split them off as their own field so they don't pollute token overlap.
- **libpostal** (MIT) parses and normalizes addresses in many languages, French and Indian included. It runs fully offline. Its model was trained on OpenStreetMap data, though, so ⚠ ask the organizers whether a pre-trained parser counts as "external data". Treat it as optional, and use regex parsing as the fallback.

**Transliteration** (Indian names in particular)
- Phonetic encodings: Double Metaphone, and ideally an Indic-aware scheme (Soundex/NYSIIS work poorly on Hindi-in-Latin script).
- Collapse common romanization variants (`aa→a`, `ee→i`, `sh→s`, `v↔w`, `ph→f`) into a "skeleton" key.

---

## 2. Blocking / candidate generation

This stage sets the **recall ceiling**. Track two numbers for every change: **pair completeness** (recall of true pairs) and **reduction ratio**. The organizers audit these too, using `candidate_pairs.tsv`.

**Target:** at least 98–99% pair completeness with a candidate list per Source 1 entity that stays small (roughly 10–50).

### 2.1 Classic key-based blocking
- **Hard partition by country.** The ground truth almost certainly never matches across countries. Verify this on train. Treat country as an open label: group by whatever string appears, so France works automatically.
- **Token blocking:** index every rare token (high IDF) of the name and address. Two records are candidates if they share a rare token.
- **Postal-code blocking:** use the PIN/ZIP/postal code. Missing codes are common, so union this with the other blockers rather than relying on it alone.
- **Sorted neighborhood** on normalized name keys (Hernández & Stolfo, 1995).
- **Phonetic keys** (Double Metaphone of the first name token) catch typos and transliterations.

### 2.2 Similarity-join / sparse retrieval
- **TF-IDF over character 3–5-grams**, then top-k by cosine for each Source 1 record. `sparse_dot_topn` or sklearn with a sparse matrix product is fast. This is the most dependable baseline blocker and handles typos, reordering and transliteration well.
- **BM25** over tokens for each field.
- **MinHash LSH** (datasketch) on shingles scales to millions of records with sub-linear lookups. It's probably overkill here unless the sources are large.

### 2.3 Dense / learned blocking (your contrastive idea belongs here)
- **DeepBlocker** (Thirumuruganathan et al., VLDB 2021) showed that self-supervised embedding blocking beats hand-built blocking on dirty and textual data.
- **Sudowoodo** (Wang et al., ICDE 2023) uses contrastive self-supervised pre-training plus augmentations (token deletion, swapping, span shuffling). With little or no labeled data it's the state of the art for blocking and matching.
- **Our case:** we *have* labels, so train a **supervised contrastive bi-encoder**:
  - **Base model**, choosing among permissively licensed multilingual encoders (they have to handle French):
    - `intfloat/multilingual-e5-small/base` (MIT)
    - `BAAI/bge-m3` (MIT)
    - `Alibaba-NLP/gte-multilingual-base` (Apache 2.0)
    - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache 2.0)
  - **Input:** `"name: … | addr: … | country: …"`.
  - **Loss:** MultipleNegativesRankingLoss (InfoNCE) with in-batch negatives, plus **hard negatives** mined from the TF-IDF blocker (same bucket, not a match). Hard-negative mining is what makes contrastive ER actually work (see also ANCE, Xiong et al. 2021).
  - **Retrieval:** FAISS or hnswlib top-k per Source 1 record.
- **Late interaction (ColBERT-style)** is a middle ground between bi-encoders and cross-encoders. It's useful when token-level alignment matters (e.g. house numbers), but it adds complexity.

### 2.4 Combining blockers
Take the **union** of candidates from the sparse (char TF-IDF), dense (contrastive) and key-based (postal code, rare token) blockers, then cap it per entity. Sparse and dense blockers fail in different ways (typos vs semantic/abbreviation variants), so their union usually reaches much higher recall than either one alone.

### 2.5 Pruning obvious negatives (your "filter" step)
- **Meta-blocking** (Papadakis et al.) weights each candidate edge by how many blocks it co-occurs in, then prunes low-weight edges.
- A **cheap first-stage classifier** is simpler here: a tiny LightGBM on 5–10 fast features (char TF-IDF cosine, postal-code equality, rare-token overlap). Set its threshold to keep about 99.5% recall on out-of-fold data. Whatever survives is the "final candidate set" that goes into `candidate_pairs.tsv`.

---

## 3. Pairwise matching

### 3.1 Feature engineering + gradient-boosted trees (strong baseline, fast, interpretable)
This is Magellan / py_entitymatching style (Konda et al., VLDB 2016). Features for each field (name, core name, address, parsed pieces):

- **Edit-based:** Levenshtein ratio, Jaro-Winkler, Damerau. Use `rapidfuzz`, which is fast and MIT licensed.
- **Token-based:** Jaccard, token_set_ratio, token_sort_ratio, Monge-Elkan (JW inner), **soft TF-IDF** (Cohen et al. 2003).
- **IDF-weighted overlap:** matching rare tokens is strong evidence, matching "private limited" is none.
- **Char n-gram TF-IDF cosine and BM25 score.**
- **Structured equality:** postal code (equal / unequal / missing), house number, unit number, state. Keep *missing* as its own category, separate from *unequal*.
- **Embedding cosine** from the §2.3 bi-encoder.
- **Length ratios, numeric-token Jaccard, suffix-class match.**
- **Context features** (very strong, often overlooked). For a candidate *c* of Source 1 record *e*:
  - rank of *c* among *e*'s candidates
  - score gap to the best candidate
  - how many Source 1 records *c* is a top candidate for
  - whether *c* is *e*'s best **and** *e* is *c*'s best (mutual best match)
- **Source-pair indicator** (S1–S2 vs S1–S3). The noise profiles probably differ.
- **Country as an open string.** Don't one-hot encode it. If it's needed at all, use country-invariant features and let the model learn from structure.

**Model:** LightGBM, XGBoost or CatBoost (all Apache/MIT), trained with group k-fold by Source 1 id. In the literature this reaches about 90–95% of transformer quality on structured/dirty data, and it trains in minutes.

### 3.2 Cross-encoder (the transformer matcher)
- **Ditto** (Li et al., VLDB 2020) serializes a record pair as `[COL] name [VAL] … [SEP] [COL] name [VAL] …` and fine-tunes a BERT-style model as a binary classifier. It adds domain-knowledge injection (tagging numbers and postal codes) and data augmentation. For years it was the reference state of the art for supervised matching.
- **Our version:** fine-tune `xlm-roberta-base/large` (MIT), `mdeberta-v3-base` (MIT), or the multilingual-e5 / bge-m3 backbone as a cross-encoder on (Source 1, candidate) pairs from our blocker, with hard negatives. Multilingual pre-training is the main hedge for France.
- Related work: **HierGAT** (Yao et al., 2021), **JointBERT** (Peeters & Bizer, 2021), **Unicorn** (Tu et al., SIGMOD 2023, a unified multi-task matcher), and **AnyMatch** (Zhang et al., 2024, a small LM built for zero-shot matching).

### 3.3 Contrastive matching
**Sudowoodo** and **SupCon-EM** (Peeters & Bizer, 2022, "Supervised Contrastive Learning for Product Matching") pre-train with a contrastive loss and then fine-tune a classifier head. That second step bridges your contrastive idea into matching. In the product-matching benchmarks, SupCon clearly beat plain cross-entropy fine-tuning.

### 3.4 LLMs (≤8B, MIT/Apache)
- Evidence:
  - Peeters, Steiner & Bizer (EDBT 2025, "Entity Matching using Large Language Models") found that fine-tuned small LLMs are competitive with, and generalize better than, PLM matchers, and that they hold up better on unseen domains. That last point is relevant for France.
  - Jellyfish (Zhang et al., 2024) is a data-preprocessing LLM. ⚠ Check the base model's license.
  - MatchGPT / BatchER (Fan et al., 2024) show that batching several questions into one prompt lowers cost.
- **Allowed candidates:**
  - Qwen2.5-7B-Instruct and Qwen3-8B (Apache 2.0)
  - Mistral-7B-Instruct (Apache 2.0)
  - Phi-3.5/4-mini (MIT)
  - Qwen2.5-1.5B/3B (Apache 2.0) for cheap runs
- **How to use one:**
  - **LoRA fine-tune** on yes/no match prompts (or scoring prompts) with our labels.
  - Run it **only on the uncertain band** that the GBDT/cross-encoder produces, in a cascade.
  - Read the *probability* of "Yes" from the logits, not the generated text, so it can be calibrated.
- **Cost:** a 7B model over roughly 10⁵ pairs is feasible on one GPU with vLLM. Over 10⁷ pairs it isn't, which is why it only runs on the uncertain band.
- ⚠ **Constraint ambiguity:** "final model up to 8B" probably permits a cascade in which every component is ≤8B, but confirm this if the plan relies on it.

### 3.5 Probabilistic record linkage (principled, unsupervised fallback)
- **Fellegi–Sunter** (1969), as implemented in **Splink** (MIT, UK Ministry of Justice, used in production at national scale), fits m/u probabilities with EM on comparison levels.
- **ZeroER** (Wu et al., SIGMOD 2020) is an unsupervised Gaussian-mixture model over similarity features.
- Why it matters here: it's a **transductive calibration tool for France**. Fit it on test-set France features with no labels, and compare its match proportion with what the supervised model predicts. Any gap shows how far the thresholds have drifted.

---

## 4. Generalizing to France (the unseen country)

This is likely the difference between the public and private leaderboard. Options, from safest to most aggressive:
1. **Language-agnostic features:** char n-grams, numeric/postal equality, IDF overlap computed **per country on that country's own records**. The IDF of "rue" in French data then comes out low automatically, the same way "road" does in US data.
2. **Multilingual encoders** for dense and cross-encoder components.
3. **Prepare French lexicons ahead of time** (legal suffixes, street types) during normalization. This is domain knowledge, not data lookup.
4. **Noise-pattern augmentation (Ditto/Sudowoodo style):** apply the documented noise operators to train records to create extra positives. Abbreviating, dropping components, reordering, typos and suffix swaps make the model rely on structure instead of memorized US/Indian vocabulary. ⚠ Don't create synthetic French *entities* with an external LLM or data source. That drifts toward "external data augmentation".
5. **Transductive / self-training:** the unlabeled test records are provided data. Options include:
   - fitting TF-IDF/IDF on train+test text
   - self-supervised contrastive pre-training (SimCSE-style) on test records
   - pseudo-labeling high-confidence French pairs, then fine-tuning a little more

   All of these are standard domain adaptation.
6. **Leave-one-country-out validation:** train on US and validate on India, and vice versa. This is the best available **proxy for France** and should be a standard part of every evaluation.

---

## 5. Decision layer: turning scores into lists (optimizing the metric directly)

Most teams will threshold probabilities at 0.5. The scoring rule gives us something much better to exploit.

### 5.1 One-to-many assignment constraints
Source 1 is deduplicated, so **every S2 or S3 record should match at most one Source 1 entity.** Verify this on train. If it holds:
- Assign each S2/S3 record only to its **highest-scoring** Source 1 candidate, as long as that score clears the threshold.
- Use Hungarian matching or greedy assignment where scores conflict.

This single constraint removes a large share of false positives at almost no cost in recall. It's a standard move in production ER ("each record belongs to one cluster").

### 5.2 Graph consistency across S2 and S3
Also score S2↔S3 pairs. If S2-a → S1-x and S2-a ≈ S3-b with high confidence, that supports S3-b → S1-x, and the reverse holds too. This is **collective ER** (Bhattacharya & Getoor, 2007), also known as correlation clustering over the three-source graph. Transitivity checks catch inconsistent links.

### 5.3 Expected-F0.5-optimal list selection (per entity)
Given calibrated probabilities *p₁ ≥ p₂ ≥ … ≥ pₙ* for one entity's candidates, choose the list that **maximizes expected F0.5**:
- Consider the empty list and each top-k prefix.
- Compute the expected F0.5 of each option under independent Bernoulli outcomes. A few Monte-Carlo samples or exact DP is enough for small n.
- Pick the option with the highest expected score.

This follows Jansche (2007) and Nan, Chai, Lee & Chieu (ICML 2012, "Optimizing F-measure: A Tale of Two Approaches"). It handles the singleton case in a principled way. When every *p* is modest, the empty list's expected score (a sure 1.0 if it truly is a singleton) beats adding a risky match. **This is the formal version of your "score empty lists first" idea.**

It needs **calibrated** probabilities. Use isotonic regression or temperature scaling on out-of-fold predictions.

### 5.4 An explicit singleton / "has-match" model
A second classifier per Source 1 entity takes entity-level features:
- the max candidate probability
- the gap between the top two candidates
- the number of candidates above 0.3
- how complete the entity's address is
- whether its postal code has any candidate at all

It predicts P(entity has ≥1 match). Its output feeds §5.3 as a prior on the empty-list option, or acts as a gate. This is the cleanest way to realize your "score the empty lists first" intuition.

---

## 6. Training and validation protocol (your k-subsets idea, made rigorous)

- **GroupKFold by Source 1 entity id** (k = 5). All candidate pairs of one entity stay in the same fold. Otherwise the same entity appears in train and validation, and the score looks better than it really is.
- **Stratify** folds by country and by singleton/non-singleton.
- **Out-of-fold (OOF) predictions** for every train entity serve three purposes:
  - calibration (§5.3)
  - threshold tuning
  - training the stacker and singleton models without leakage
- **Final model:** average the k fold models' probabilities (bagging) or retrain on all data using the tuned hyperparameters. "Merging models" in the weight space is possible for transformers (model soups, Wortsman et al., ICML 2022), but averaging predictions is simpler and more robust.
- **Local scorer:** a script that reproduces the official **macro F0.5 per Source 1 entity**, singletons included. Report it overall, per country, singletons vs non-singletons, and in leave-one-country-out mode.
- **Blocking metrics:** pair completeness and reduction ratio, reported per country.

---

## 7. Recommended stack (tiered)

**Tier 1: strong baseline, 1–2 days, likely ~80% of the achievable score**
1. Normalization (§1) with regex-parsed postal and house numbers.
2. Blocking = country partition ∪ char TF-IDF top-k ∪ postal/rare-token keys (§2.1–2.2).
3. Handcrafted features + LightGBM, with group k-fold (§3.1).
4. One-to-many assignment + tuned threshold + expected-F0.5 list selection (§5.1, §5.3).

**Tier 2: the main gains**
5. Contrastive multilingual bi-encoder with hard negatives, added to blocking and used as a feature (§2.3).
6. Cross-encoder (mdeberta-v3 / xlm-r) fine-tuned on hard candidates. Its score becomes a GBDT feature, or the two are stacked (§3.2).
7. Singleton model + context features + S2↔S3 graph consistency (§5.2, §5.4).
8. Leave-one-country-out validation and noise augmentation for France (§4).

**Tier 3: if compute allows**
9. LoRA-fine-tuned ≤8B Apache/MIT LLM (Qwen2.5-7B / Qwen3-8B) as a cascade judge for the uncertain probability band only (§3.4).
10. Transductive self-training on test-set France records (§4.5).

---

## 8a. Data findings (EDA run on the HPC, `scripts/eda.py`)

| | S1 | S2 | S3 |
|---|---|---|---|
| Train | 2.21M (US 1.32M / IN 0.88M) | 5.03M | 5.29M |
| Test | 1.73M (IN 0.81M / US 0.66M / **FR 0.26M = 15%**) | 4.89M | 5.08M |

- **Singletons are only 5.6%** (same in both countries). The mean is about 3.7 matches per entity, spread as 2 (17%), 3 (24%), 4 (22%), 5 (15%) and 6+ (11%). **Recall matters a lot.** The metric is precision-weighted, but a typical entity has 3–5 true matches, so missing any of them costs real score.
- **Every S2/S3 record belongs to at most one S1 entity** (0 violations), and **there are no cross-country links.** The one-owner assignment constraint (§5.1) and a hard country partition are both **safe**.
- **Sources 2 and 3 contain internal duplicates.** One entity can have up to 5 S2 records and 6 S3 records. That makes S2↔S3 clustering (§5.2) a strong signal: find the group, then link the whole group.
- **About 26% of S2 and S3 records are distractors** that match nothing, so the matcher must learn to reject them.
- **Postal codes are almost absent** (US ≈10%, India ≈1%, France ≈0.5%). **Postal blocking is useless.** Block on house or street number, street tokens, city, and name tokens instead.
- The noise looks synthetic, with a fixed set of operators:
  - case changes (S2 is often all-caps)
  - `NULL` placeholder tokens
  - injected filler words (`The The`, `Mr`, `M/s`, `Center`, `Group`)
  - token deletion and truncation (`Southern`)
  - domain-style names (`fafloonpetcare.com`, `#southerneducational`)
  - accent injection even in US names (`Bérto`)
  - character typos (`S0lutions`, `Atlanat`)
  - house-number formatting (`01130`, `1130-`, `4-7/1`)
  - state abbreviated ↔ full (`TX`/`Texas`, `MH`/`Maharashtra`)
  - native script for state names (`ಕರ್ನಾಟಕ`)
  - reordered address components
  - PO box / unit number added or dropped

  **If we can reproduce these operators, we can generate training pairs for French-style records** (§4.4).
- France (test only) uses the same formats: legal suffixes `SARL/SAS/SASU/SCI/S.A.`, `& Fils`/`& Frères`, `R.`→`Rue`, `AV`→`Avenue`, `bis`, and region↔département (`Nouvelle-Aquitaine`/`Gironde`, `Hauts-de-France`/`Nord`).
- **Scale:** the test set has 1.73M S1 entities against ~10M S2/S3 records. With about 30 candidates each, that's roughly 50M pairs. A small cross-encoder can score that on the H100. A 7B LLM can only handle the uncertain band.
- With 2.2M labeled entities we have far more training data than a GBDT needs, so we can subsample. The large volume helps most when training the contrastive bi-encoder.

## 8. What the data needs to tell us before we lock the design

1. Record counts per source and per country. These decide whether a cross-encoder or LLM can score every candidate or only a subset.
2. Fraction of singletons in Source 1. This sets how much §5.3/§5.4 matter.
3. Distribution of matches per entity (1? 2–3? many?), split into S2 and S3 matches.
4. **Does any S2/S3 record ever match more than one Source 1 entity?** This decides whether §5.1 is valid.
5. Are there any cross-country matches? This decides whether a hard country partition is safe.
6. Recall ceiling of simple char-TF-IDF top-k at k = 5/10/20/50.
7. How often postal codes are present, per source and country.
8. Noise profile per source. Is S2 systematically different from S3?

---

## Key references
- Fellegi & Sunter, *A Theory for Record Linkage*, JASA 1969.
- Hernández & Stolfo, *The Merge/Purge Problem for Large Databases*, SIGMOD 1995.
- Cohen, Ravikumar & Fienberg, *A Comparison of String Distance Metrics for Name-Matching Tasks*, 2003.
- Bhattacharya & Getoor, *Collective Entity Resolution in Relational Data*, TKDD 2007.
- Jansche, *A Maximum Expected Utility Framework for Binary Sequence Labeling* (F-measure optimization), ACL 2007.
- Nan, Chai, Lee & Chieu, *Optimizing F-measure: A Tale of Two Approaches*, ICML 2012.
- Papadakis et al., *Blocking and Filtering Techniques for Entity Resolution: A Survey*, ACM CSUR 2020.
- Konda et al., *Magellan: Toward Building Entity Matching Management Systems*, VLDB 2016.
- Mudgal et al., *Deep Learning for Entity Matching: A Design Space Exploration* (DeepMatcher), SIGMOD 2018.
- Li et al., *Deep Entity Matching with Pre-Trained Language Models* (Ditto), VLDB 2020.
- Wu et al., *ZeroER: Entity Resolution using Zero Labeled Examples*, SIGMOD 2020.
- Thirumuruganathan et al., *Deep Learning for Blocking in Entity Matching: A Design Space Exploration* (DeepBlocker), VLDB 2021.
- Peeters & Bizer, *Supervised Contrastive Learning for Product Matching*, WWW 2022 Companion.
- Wang et al., *Sudowoodo: Contrastive Self-supervised Learning for Multi-purpose Data Integration*, ICDE 2023.
- Tu et al., *Unicorn: A Unified Multi-tasking Model for Supporting Matching Tasks in Data Integration*, SIGMOD 2023.
- Peeters, Steiner & Bizer, *Entity Matching using Large Language Models*, EDBT 2025.
- Wortsman et al., *Model Soups*, ICML 2022.
- Tools: Splink, dedupe, recordlinkage, py_entitymatching, rapidfuzz, libpostal, FAISS, sentence-transformers, LightGBM.
