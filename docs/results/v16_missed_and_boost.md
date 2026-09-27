# What the final model still misses, and how to recover or boost

_2026-09-27. Final submission: **v16frmixacr, leaderboard 0.987427** (v16 stacker; France = blended French Qwen + acronym routing).
Script: `scripts/v16_missed.py`._

## 1. US/India misses (validation, 150k B-split businesses, test-like frames)

v16's out-of-fold scores were rebuilt exactly as `v16.py final()` does. They reproduce F0.5 **0.99082** at cut-off 0.75.

- **Missed true pairs: 13,179** (12,660 scored too low + 519 never shortlisted). **Wrong accepts: 760** (v12: 935).
- vs v12: 758 misses fixed, 634 newly missed; 419 wrong accepts removed, 244 new.

| Cause | Missed | Final score 0.5–0.75 (near miss) | Model A < 0.01 (Qwen never reads) | Recoverable? |
|---|---|---|---|---|
| No address, 2+ businesses share the name | 5,823 | 952 | 606 | **No**: the name can't tell which namesake owns it |
| No address, name unique | 4,716 | 1,008 | 968 | **Mostly no**: an exact unique-name match is only 53% true |
| Names share no word (gibberish name) | 1,359 | 330 | 217 | Partly |
| Never a candidate (blocking) | 519 | – | – | ≤ +0.0003 |
| Other / similar text | 349 | 106 | 107 | Partly |
| House no. 1-digit typo | 289 | 89 | 47 | Partly |
| Indian script name | 105 | 18 | 60 | Partly |
| Acronym record | 19 | 5 | 6 | Solved in US/India |
| **Total** | **13,179** | **2,508** | **2,011** | |

By country: India 4,846, US 8,333.

- **80% (10,539) are address-less.** Their owner can't be identified from the name. Four recovery methods were tested (stage-2 model, spelling correction, cleaned-name threshold, unique-name rule): all gained 0 to +0.00003 (see `docs/path_to_099.md` §5).
- **The reachable pool is about 2,000–3,000 pairs** (near misses, pairs Qwen never reads, typos, gibberish names). Finding all of them would be worth about +0.0015 on validation.

## 2. France after the final submission (no labels, structure only)

| Test S2/S3 records | France | India | US |
|---|---|---|---|
| Records per S1 | 5.53 | 5.82 | 5.76 |
| Matched to some business | 61.4% | 58.1% | 58.9% |
| Matched, record has an address | 61.8% | 58.3% | 58.9% |
| Matched, address-less record | **48.2%** | 53.2% | 57.4% |
| Acronym records per 1,000 S1 | 92.4 | 11.4 | 5.8 |
| Acronym records matched | **89.7%** (v16: 77%) | 98.5% | 95.8% |
| Matches per S1 | 3.394 | 3.386 | 3.388 |

- **The acronym routing worked:** France acronyms went from 77% to 90% matched.
- **France's match volume now looks normal.** 61.4% of its records are matched, close to the ~62% expected if France has the same 3.44 matches per business as train. Yet France still scores about **0.970**, working back from the leaderboard (with US/India near 0.990). So France's remaining loss is mostly **records given to the wrong business**, which volume statistics can't show.

## 3. How to recover and boost (ranked)

| # | Idea | Targets | Compute | Expected LB gain (rough) |
|---|---|---|---|---|
| 1 | **Widen what Qwen reads, everywhere:** each business's top 2–3 candidates below Model A 0.01, plus each unmatched record's top 2 businesses. Score them on validation too, so the stacker learns from them | the 2,011 US/India misses below 0.01; in France likely many more (acronyms showed Model A underrates French pairs) | GPU, a few hours | +0.0005 to +0.0015 (mostly France) |
| 2 | **Listwise reranking:** show Qwen the record with its top 3–5 competing businesses and ask which one owns it (or none) | wrong-owner errors (760 in US/India, likely most of France's loss) and part of the 2,508 near misses | GPU: prompt change + short fine-tune | +0.0005 to +0.001, uncertain |
| 3 | **Test-time averaging for Qwen:** score each pair twice (swapped order or a second prompt) and average | near misses | GPU, inference only | +0.0001 to +0.0003 |
| 4 | **Better France proxy:** add acronym noise and France's decoy densities (27× more same-address look-alikes, ~6 businesses per name) to the practice set, then tune France's stacker and cut-off on it rather than on leaderboard probes | France wrong-owner errors | CPU + some GPU | uncertain, but the only way to *measure* France |
| 5 | Blocking recall | 519 misses | CPU | ≤ +0.0003 |

**Not worth more effort** (tested, at or below zero): name cleaning, spelling correction, same-address or unique-name rules, global cut-off changes, the France same-address rule (LB 0.980).

**Outlook:** steps 1 and 2 together might add about +0.001 to +0.002 on the leaderboard (≈ 0.9885–0.9895). **0.99 needs step 1 or 2 to work unusually well on France.**
