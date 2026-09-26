# Candidate-side features in the stacker — out-of-fold (same folds, same pairs)

- v5 stacker: τ=0.65 → **0.9834** (singletons 0.9781, India 0.9832, US 0.9835) | no-address recall 44.0%, precision 86.7%
- **v5 + candidate-side: τ=0.75 → **0.9885** (singletons 0.9906, India 0.9888, US 0.9884)** | no-address recall 50.2%, precision 97.0%

Feature importance (gain %): p 53.3, d_crank 18.5, s1_name_cnt 17.1, p_gap 5.6, p_rank 1.9, n_cands 1.7, rr 1.0, cand_name_cnt 0.4, d_cmargin 0.3, t_cmargin 0.1, n_cmargin 0.1, d_ncomp 0.0, rr_gap 0.0, n_crank 0.0, t_crank 0.0, cand_noaddr 0.0, t_ncomp 0.0, rr_rank 0.0, name_exact 0.0, n_ncomp 0.0, has_rr 0.0

| model | tau | macro_f05 | singletons | non_singletons | f05_India | f05_US |
|---|---|---|---|---|---|---|
| v5 stacker | 0.4000 | 0.9791 | 0.9474 | 0.9810 | 0.9788 | 0.9793 |
| v5 stacker | 0.4500 | 0.9807 | 0.9560 | 0.9822 | 0.9802 | 0.9811 |
| v5 stacker | 0.5000 | 0.9819 | 0.9623 | 0.9831 | 0.9816 | 0.9821 |
| v5 stacker | 0.5500 | 0.9825 | 0.9680 | 0.9834 | 0.9821 | 0.9827 |
| v5 stacker | 0.6000 | 0.9830 | 0.9756 | 0.9834 | 0.9827 | 0.9832 |
| v5 stacker | 0.6500 | 0.9834 | 0.9781 | 0.9837 | 0.9832 | 0.9835 |
| v5 stacker | 0.7000 | 0.9833 | 0.9833 | 0.9834 | 0.9833 | 0.9834 |
| v5 stacker | 0.7500 | 0.9833 | 0.9858 | 0.9832 | 0.9834 | 0.9833 |
| v5 stacker | 0.8000 | 0.9833 | 0.9887 | 0.9829 | 0.9834 | 0.9832 |
| v5 stacker | 0.8500 | 0.9829 | 0.9905 | 0.9825 | 0.9831 | 0.9828 |
| v5 stacker | 0.9000 | 0.9818 | 0.9929 | 0.9811 | 0.9822 | 0.9815 |
| v5 + candidate-side | 0.4000 | 0.9853 | 0.9772 | 0.9857 | 0.9853 | 0.9852 |
| v5 + candidate-side | 0.4500 | 0.9864 | 0.9803 | 0.9868 | 0.9866 | 0.9863 |
| v5 + candidate-side | 0.5000 | 0.9872 | 0.9827 | 0.9875 | 0.9873 | 0.9871 |
| v5 + candidate-side | 0.5500 | 0.9877 | 0.9847 | 0.9879 | 0.9879 | 0.9876 |
| v5 + candidate-side | 0.6000 | 0.9881 | 0.9862 | 0.9882 | 0.9884 | 0.9879 |
| v5 + candidate-side | 0.6500 | 0.9883 | 0.9880 | 0.9884 | 0.9886 | 0.9882 |
| v5 + candidate-side | 0.7000 | 0.9885 | 0.9893 | 0.9884 | 0.9887 | 0.9883 |
| v5 + candidate-side | 0.7500 | 0.9885 | 0.9906 | 0.9884 | 0.9888 | 0.9884 |
| v5 + candidate-side | 0.8000 | 0.9885 | 0.9923 | 0.9882 | 0.9889 | 0.9882 |
| v5 + candidate-side | 0.8500 | 0.9881 | 0.9934 | 0.9878 | 0.9886 | 0.9878 |
| v5 + candidate-side | 0.9000 | 0.9874 | 0.9947 | 0.9870 | 0.9880 | 0.9870 |
