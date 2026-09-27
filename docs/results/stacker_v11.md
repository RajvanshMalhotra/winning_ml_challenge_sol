# v11 stacker: OOF comparison (150k B-split S1, same folds)

| model | decision | F0.5 | US | India | singletons | Δ vs v9 (95% CI) |
|---|---|---|---|---|---|---|
| v9 | global τ=0.7 | 0.9895 | 0.9893 | 0.9898 | 0.9903 | +0.00000 [+0.00000, +0.00000] |
| v10a | global τ=0.75 | 0.9897 | 0.9895 | 0.9901 | 0.9928 | +0.00025 [+0.00014, +0.00036] |
| v11 (v10a + acronym) | global τ=0.75 | 0.9898 | 0.9895 | 0.9902 | 0.9933 | +0.00029 [+0.00018, +0.00042] |
| v11 ensemble (4 LightGBMs) | global τ=0.75 | 0.9898 | 0.9896 | 0.9901 | 0.9927 | +0.00034 [+0.00022, +0.00046] |
| v11 ensemble | expected-F0.5 per S1 (calibrated; nested CV, picks [(1.0, 0.5), (1.0, 0.2), (0.8, 0.5), (1.0, 0.2), (1.0, 0.5)]) | 0.9897 | 0.9895 | 0.9900 | 0.9865 | +0.00020 [+0.00008, +0.00034] |

France-shifted features dropped in the robust model: []

## Cut-off under test-like orphan density

| model | tau | F0.5 (train density) | F0.5 (test-like orphans x2) | orphan wrong accepts | other wrong accepts |
|---|---|---|---|---|---|
| v9 | 0.5000 | 0.9881 | 0.9872 | 1161 | 1740 |
| v9 | 0.5500 | 0.9887 | 0.9879 | 1057 | 1248 |
| v9 | 0.6000 | 0.9891 | 0.9883 | 997 | 865 |
| v9 | 0.6500 | 0.9894 | 0.9886 | 906 | 641 |
| v9 | 0.7000 | 0.9895 | 0.9888 | 827 | 496 |
| v9 | 0.7500 | 0.9895 | 0.9889 | 745 | 385 |
| v9 | 0.8000 | 0.9894 | 0.9889 | 653 | 265 |
| v9 | 0.8500 | 0.9890 | 0.9886 | 587 | 163 |
| v9 | 0.9000 | 0.9884 | 0.9880 | 504 | 94 |
| v9 | 0.9500 | 0.9870 | 0.9867 | 397 | 49 |
| v11 ensemble (4 LightGBMs) | 0.5000 | 0.9885 | 0.9876 | 1138 | 1590 |
| v11 ensemble (4 LightGBMs) | 0.5500 | 0.9890 | 0.9882 | 1050 | 1108 |
| v11 ensemble (4 LightGBMs) | 0.6000 | 0.9893 | 0.9886 | 971 | 786 |
| v11 ensemble (4 LightGBMs) | 0.6500 | 0.9896 | 0.9889 | 892 | 574 |
| v11 ensemble (4 LightGBMs) | 0.7000 | 0.9898 | 0.9891 | 822 | 419 |
| v11 ensemble (4 LightGBMs) | 0.7500 | 0.9898 | 0.9892 | 736 | 296 |
| v11 ensemble (4 LightGBMs) | 0.8000 | 0.9897 | 0.9892 | 649 | 203 |
| v11 ensemble (4 LightGBMs) | 0.8500 | 0.9894 | 0.9889 | 582 | 137 |
| v11 ensemble (4 LightGBMs) | 0.9000 | 0.9888 | 0.9884 | 497 | 80 |
| v11 ensemble (4 LightGBMs) | 0.9500 | 0.9872 | 0.9869 | 398 | 39 |

robust stacker (dropped ['d_cmargin', 'n_cands', 'a_share', 'cn_cmargin', 'a_ncomp', 't_ncomp', 'n_ncomp', 't_cmargin', 'addr_cmargin']): τ=0.75 → OOF F0.5 0.98933
