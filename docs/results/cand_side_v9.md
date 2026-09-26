# v9 stacker: + Model-A competition, cluster support, address competition — out-of-fold (same folds, same pairs)

- v8: τ=0.75 → **0.9885** (singletons 0.9906, India 0.9888, US 0.9884)
- **v9: τ=0.70 → **0.9895** (singletons 0.9903, India 0.9898, US 0.9893)**

- v8 missed 14,751 → v9 missed 13,044 (fixed 2,007, newly lost 300)
- wrong accepts: v8 1,300 → v9 1,323
- no-address true pairs found: v8 50.9% → v9 54.2%; no-address precision v8 97.0% → v9 96.6%

Feature importance (gain %): p 89.1, a_share 4.0, a_cmargin 4.0, rr 1.9, d_cmargin 0.3, n_cmargin 0.3, p_gap 0.2, d_crank 0.1, addr_cmargin 0.0, sib_crank 0.0, n_cands 0.0, sib_cmargin 0.0, n_crank 0.0, t_cmargin 0.0, sib_name_max 0.0, d_ncomp 0.0, addr_crank 0.0, addr_sim 0.0, a_crank 0.0, p_rank 0.0, a_ncomp 0.0, s1_nconf 0.0, cand_noaddr 0.0, rr_gap 0.0, cand_name_cnt 0.0, s1_name_cnt 0.0, t_crank 0.0, a_nhi 0.0, t_ncomp 0.0, rr_rank 0.0, name_exact 0.0, s1_addr_cnt 0.0, cand_addr_s1cnt 0.0, has_rr 0.0, n_ncomp 0.0

| model | tau | macro_f05 | singletons | non_singletons | f05_India | f05_US |
|---|---|---|---|---|---|---|
| v8 | 0.5000 | 0.9872 | 0.9827 | 0.9875 | 0.9873 | 0.9871 |
| v8 | 0.5500 | 0.9877 | 0.9847 | 0.9879 | 0.9879 | 0.9876 |
| v8 | 0.6000 | 0.9881 | 0.9862 | 0.9882 | 0.9884 | 0.9879 |
| v8 | 0.6500 | 0.9883 | 0.9880 | 0.9884 | 0.9886 | 0.9882 |
| v8 | 0.7000 | 0.9885 | 0.9893 | 0.9884 | 0.9887 | 0.9883 |
| v8 | 0.7500 | 0.9885 | 0.9906 | 0.9884 | 0.9888 | 0.9884 |
| v8 | 0.8000 | 0.9885 | 0.9923 | 0.9882 | 0.9889 | 0.9882 |
| v8 | 0.8500 | 0.9881 | 0.9934 | 0.9878 | 0.9886 | 0.9878 |
| v8 | 0.9000 | 0.9874 | 0.9947 | 0.9870 | 0.9880 | 0.9870 |
| v9 | 0.5000 | 0.9881 | 0.9836 | 0.9884 | 0.9882 | 0.9880 |
| v9 | 0.5500 | 0.9887 | 0.9854 | 0.9889 | 0.9888 | 0.9887 |
| v9 | 0.6000 | 0.9891 | 0.9871 | 0.9892 | 0.9893 | 0.9889 |
| v9 | 0.6500 | 0.9894 | 0.9888 | 0.9894 | 0.9897 | 0.9891 |
| v9 | 0.7000 | 0.9895 | 0.9903 | 0.9894 | 0.9898 | 0.9893 |
| v9 | 0.7500 | 0.9895 | 0.9915 | 0.9893 | 0.9898 | 0.9893 |
| v9 | 0.8000 | 0.9894 | 0.9940 | 0.9891 | 0.9899 | 0.9891 |
| v9 | 0.8500 | 0.9890 | 0.9949 | 0.9887 | 0.9895 | 0.9887 |
| v9 | 0.9000 | 0.9884 | 0.9955 | 0.9880 | 0.9890 | 0.9880 |
