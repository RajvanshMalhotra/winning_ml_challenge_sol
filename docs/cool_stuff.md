# Cool Stuff: Findings Worth Remembering

## 1. Fine-tuning did NOT bias the model toward the training languages; it transferred to unseen ones

**The worry:** if we fine-tune a multilingual model (bge-m3) only on US + India data, will it forget French (or get worse at scripts it didn't see), compared with using it as-is for inference?

**The experiment (leave-one-country-out):** fine-tune bge-m3 on **US data only** (English, no Indian records, no Indian scripts), then test on **India**, a country and scripts it never saw during fine-tuning.

| bge-m3 tested on India (recall@50) | All India | Indian-script names (Hindi/Tamil/Odia…) |
|---|---|---|
| Inference only (not fine-tuned) | 95.4% | 91.6% |
| **Fine-tuned on US only** (never saw India) | **99.2%** | **98.6%** |
| Fine-tuned on US + India | 99.95% | 100% |

**Fine-tuning on English/US alone improved Indian-script matching by +7 points** over the untouched model.

**Why it works:**
- The **languages come from pre-training** (100+ languages, incl. French and Indian scripts). Gentle fine-tuning (1 short pass, 3,000 steps, low learning rate) doesn't erase them.
- Fine-tuning changes **what "similar" means**: from *topically similar* (two different pizza shops look alike) to *same business* (ignore typos, ALL CAPS, `NULL`, extra words, reordered addresses; `Pvt Ltd` = `Private Limited`). Those rules are **language-independent**, so they transfer.

**France evidence (no labels, so proxies):**
- On test, French businesses get a near-certain top dense match (similarity ≥ 0.80) **94.3%** of the time, vs ~95% for US/India.
- Leaderboard went up with the fine-tuned models: v2 **0.962** → v3 (+ fine-tuned reranker) **0.973**, and the test includes France.

**Remaining risk and cheap safety net:** French *conventions* never seen in training (`SARL`, `& Fils`, `R.` = Rue, départements) may be handled slightly worse. Options:
- add the **inference-only (not fine-tuned) bge-m3 similarity as an extra feature** next to the fine-tuned one, and let LightGBM use it where the fine-tuned model is less reliable (≈2 GPU-hours per set to embed, no training)
- add **French-style noise** (SARL↔S.A.R.L., Rue↔R., département↔région) to the fine-tuning data

## 2. Cleaning vs language models: two text paths (by design)

| Uses the **cleaned** text (accents/apostrophes removed: `Sautéed` → `sauteed`, `L'Étoile` → `l etoile`) | Uses the **original** text (accents kept) |
|---|---|
| TF-IDF search, key blocks, name-only search, fuzzy/house-number/suffix/rare-word features | bge-m3 (dense search + similarity feature), bge-reranker, Qwen hard-name search |

- Letter-matching methods need `Sautéed` = `Sauteed` (the data's noise even *adds* fake accents: `Bérto`, `Córnerstone`).
- Multilingual models understand accents natively, so stripping them would throw information away.
- Small known flaws: possessive `'s` becomes a stray token (`Simmons's` → `simmons s`); `St.` always → `street` (sometimes it means *Saint*). Both are minor, because both sides of a pair are cleaned the same way.

## 3. An empty address is almost a guarantee of a match (train)

Only S2/S3 ever have empty addresses (~3%; S1 is always complete, in train **and** test, France included).

| S2/S3 record | Belongs to some S1 |
|---|---|
| **no address** | **97.7%** |
| has an address | 73.2% |

- The empty-address records look like deliberately degraded **copies** of an S1 business: name kept (62.7% identical core name, never zero shared words), address dropped. 97.9% of their S1s also have another match *with* an address.
- Yet they are v5's biggest error: **76% of the true matches v5 still misses** have an empty address on one side. The reranker usually says yes (median 0.84), but the stacker distrusts these pairs because half of the wrong accepts are also empty-address pairs.
- So the question for these records is not *"is it a match?"* (almost always yes) but *"**which** S1 does it belong to?"*: an assignment problem. Candidate-side features (how this S1 ranks among all S1s competing for the record, and the gap to the runner-up) answer that directly.
