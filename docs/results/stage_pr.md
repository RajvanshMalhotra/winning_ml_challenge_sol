# Precision and recall at every stage (OOF, 150k B-split S1, 518,414 true pairs)

| stage | pair precision | pair recall | macro F0.5 | wrong accepts | missed | F0.5 US | F0.5 India | singletons |
|---|---|---|---|---|---|---|---|---|
| blocking (candidates) | 0.0266 | 0.9990 | nan | 18,958,222 | 519 | nan | nan | nan |
| Model A (LightGBM) | 0.9910 | 0.9575 | 0.9779 | 4,496 | 22,028 | 0.9791 | 0.9762 | 0.9699 |
| + bge reranker (v5) | 0.9936 | 0.9675 | 0.9834 | 3,247 | 16,863 | 0.9835 | 0.9832 | 0.9781 |
| + candidate-side (v8) | 0.9974 | 0.9705 | 0.9885 | 1,300 | 15,270 | 0.9884 | 0.9888 | 0.9906 |
| + Model-A competition (v9) | 0.9974 | 0.9738 | 0.9895 | 1,323 | 13,563 | 0.9893 | 0.9898 | 0.9903 |
| + cleaned names (v10a) | 0.9979 | 0.9731 | 0.9897 | 1,079 | 13,952 | 0.9895 | 0.9901 | 0.9928 |

Pair precision = accepted pairs that are true; pair recall = true pairs found (blocking misses count as not found).

# Where the remaining F0.5 is (v9 OOF 0.9895; each row alone)

| source | pairs | F0.5 if fixed | max gain |
|---|---|---|---|
| find all: no address, name unique (<=1 S1 with that core name) | 4,395 | 0.9922 | 0.0027 |
| find all: no address, 2+ S1s share the core name | 5,981 | 0.9932 | 0.0037 |
| find all: names share no word (gibberish name) | 1,537 | 0.9905 | 0.0010 |
| find all: other misses (typos, digit typos, script ...) | 1,131 | 0.9902 | 0.0008 |
| remove all wrong accepts | 1,323 | 0.9918 | 0.0023 |
| blocking: every true pair in the shortlist AND found | 519 | 0.9898 | 0.0003 |

# Test: France vs US / India (no labels; v9 test scores)

S1 share by country: India 46.8%, US 38.3%, France 15.0%
S2/S3 empty-address share: France 3.0%, India 2.4%, US 2.9%
| country | candidates | accepted | uncertain_pairs | median_top_score | no_confident_match |
|---|---|---|---|---|---|
| France | 171.5332 | 3.3985 | 0.1956 | 0.9995 | 0.0546 |
| India | 136.5521 | 3.3938 | 0.1006 | 0.9995 | 0.0572 |
| US | 111.8148 | 3.3936 | 0.1100 | 0.9995 | 0.0574 |

Implied France score if US/India score on the leaderboard what they score OOF (France = 15.0% of test S1):

| version | OOF (US+India) | LB | implied France |
|---|---|---|---|
| v2 | 0.9707 | 0.9620 | 0.9126 |
| v3 | 0.9784 | 0.9730 | 0.9423 |
| v5 | 0.9834 | 0.9790 | 0.9540 |
| v8 | 0.9885 | 0.9830 | 0.9517 |
