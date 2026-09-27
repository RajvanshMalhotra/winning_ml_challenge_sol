# v14: v12 with the stacker retrained on test-like "hidden owner" features

In test only ~60% of S2/S3 records have their owner in S1 (74% in train). The competition features were recomputed with 20% of train S1s hidden (not in the OOF sample) and the stacker retrained on them. Test-like OOF: v12 stacker 0.98966 -> v14 0.99008 (+0.0004), singletons 0.9907 -> 0.9932. Same Qwen reranker, tau 0.75, small candidate set. Validator PASS.
