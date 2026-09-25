"""Load and encode with every registered encoder, each in its own process (a CUDA assert poisons the context)."""
import subprocess
import sys

from ber.contrastive.encoders import ENCODERS

TEXT = "name: Galaxy Solutions Pvt Ltd | address: 47/1 Airport Road, Kolhapur | country: India"
CHILD = """
import sys
from ber.contrastive.encoders import ENCODERS, encode, load_encoder
spec = ENCODERS[sys.argv[1]]
model = load_encoder(spec, mem_fraction=0.4)
print("OK", spec.key, encode(model, [sys.argv[2]], spec.query_prefix, 8).shape, flush=True)
"""
for key in ENCODERS:
    r = subprocess.run([sys.executable, "-c", CHILD, key, TEXT], capture_output=True, text=True)
    ok = [l for l in r.stdout.splitlines() if l.startswith("OK")]
    print(ok[0] if ok else f"FAIL {key}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}", flush=True)
