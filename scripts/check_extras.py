"""Fill rates and examples of the extra record fields on a real sample."""
import pandas as pd

from ber.io import read_family
from ber.normalize import build_records

raw = pd.concat([read_family("data/student_resource/dataset", f).sample(150_000, random_state=0) for f in ("train", "test")])
rec = build_records(raw, n_jobs=32)
pd.set_option("display.width", 220, "display.max_colwidth", 70)
for col in ["name_parts", "landmark", "postal", "unit"]:
    print(f"\n== {col}: filled per country", (rec[col] != "").groupby(rec.country).mean().round(4).to_dict())
    print(rec.loc[rec[col] != "", ["country", "name_raw" if col == "name_parts" else "addr_raw", col]].head(6).to_string(index=False))
print("\n== name keys")
print(rec[["name_core", "name_sorted", "name_phonetic", "name_skeleton"]].head(5).to_string(index=False))
