# v16 + France acronym routing

v16 (LB 0.98709) unchanged for US/India. France: acronym S2/S3 records (e.g. `AJ` -> `Association du Jeu`, same address) that v16 left unassigned are paired with France S1s whose initials match (blocking pool, or same house number + street word), read by the fine-tuned Qwen reranker, and added when Qwen prob >= 0.9 and >= 0.3 ahead of the runner-up: +3,393 matches on 3,322 France S1s, nothing removed. The 12,316 pairs Qwen read are added to candidate_pairs.tsv. Validator PASS. Idea: docs/path_to_099.md step 1.
