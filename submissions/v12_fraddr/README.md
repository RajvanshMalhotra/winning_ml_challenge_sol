# v12 + French address rule

v12 (OOF 0.9904, LB 0.9864) for US/India, byte-identical. France: +63,487 pairs from the rule "France S1 alone at its exact normalized address <-> every S2/S3 record at that address" (96.4% true in US/India train; our models rejected 38% of these in France), only when the record is not already given to another S1. No v12 match removed; 42,635 France rows changed. Candidates: Model A filter p>=0.01 plus the rule pairs = 5.12 per S1. Validator PASS.
