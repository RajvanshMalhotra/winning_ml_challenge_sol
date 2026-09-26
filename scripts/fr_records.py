"""Family `testfr`: test records with the France-only text fixes (région, N°, French abbreviations); same row order,
same candidates and embeddings as `test` (symlinked). Checks that non-France rows re-normalize byte-identically.
usage: CUDA_VISIBLE_DEVICES= python -u scripts/fr_records.py"""
import os
import numpy as np
import pandas as pd
from ber.config import load_config, run_dir
from ber.io import read_family
from ber.normalize import build_records, records_path

cfg = load_config("configs/v2.yaml"); RD = run_dir(cfg)
old = pd.read_parquet(records_path(cfg, "test"))
raw = read_family(cfg["paths"]["data_dir"], "test")
raw = raw.set_index("entity_id").loc[old.entity_id].reset_index()
fr = raw[raw.country == "France"]
chk = raw[raw.country != "France"].sample(300_000, random_state=0)
new_fr = build_records(fr, cfg["n_jobs"]).set_index("entity_id")
new_chk = build_records(chk, cfg["n_jobs"]).set_index("entity_id")
o = old.set_index("entity_id")
cols = [c for c in o.columns if c not in ("name_domain", "block_text")]  # both depend on the whole family's per-country vocabulary
diff = (new_chk[cols] != o.loc[new_chk.index, cols]).any(axis=1)
print(f"US/India check: {int(diff.sum())} of {len(new_chk):,} re-normalized rows differ")
assert diff.sum() == 0
new = o.copy()
for c in new_fr.columns:
    new.loc[new_fr.index, c] = new_fr[c]
ofr, nfr = o.loc[new_fr.index], new.loc[new_fr.index]
for c in ["addr_norm", "addr_core", "house_no", "num_tokens", "state", "name_domain"]:
    print(f"France {c}: changed {float((ofr[c] != nfr[c]).mean()):.1%}")
for s in (1, 2):
    m = nfr.source == 1 if s == 1 else nfr.source != 1
    print(f"France {'S1' if s == 1 else 'S2/S3'}: house found {float((ofr.house_no[m] != '').mean()):.3f} -> {float((nfr.house_no[m] != '').mean()):.3f}; "
          f"région found {float((ofr.state[m] != '').mean()):.3f} -> {float((nfr.state[m] != '').mean()):.3f}")
print("France régions:", nfr.state.value_counts().head(6).to_dict())
new.reset_index()[old.columns].to_parquet(records_path(cfg, "testfr"), index=False)
for f in ["cand_sparse_{}.parquet", "cand_hardname_{}.parquet", "kg_recrec_edges_{}.parquet", "cand_dense_{}.parquet",
          "emb_{}_bgem3_bgem3__finetune.npy"]:
    dst = RD / f.format("testfr")
    if not dst.exists():
        os.symlink(f.format("test"), dst)  # relative to the link's own folder
print("wrote records_testfr.parquet + symlinks")
