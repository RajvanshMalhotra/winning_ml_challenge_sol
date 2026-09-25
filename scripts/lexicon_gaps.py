"""Print frequent 2-3 letter address components and non-ASCII components per country (lexicon gaps)."""
import pandas as pd

for family in ("train", "test"):
    df = pd.read_parquet(f"artifacts/v1/records_{family}.parquet", columns=["country", "addr_raw"])
    comps = df.assign(c=df.addr_raw.str.split(",")).explode("c")
    comps["c"] = comps.c.str.strip()
    short = comps[comps.c.str.fullmatch(r"[A-Za-z]{2,3}", na=False)]
    nonascii = comps[comps.c.str.contains(r"[^\x00-\x7F]", regex=True, na=False)]
    for label, sub in (("short", short), ("non-ascii", nonascii)):
        print(f"== {family} {label}")
        print(sub.groupby("country").c.value_counts().groupby(level=0).head(40).to_string())
