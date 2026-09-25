from ber.contrastive.encoders import ENCODERS, encode, load_encoder

TEXT = "name: Galaxy Solutions Pvt Ltd | address: 47/1 Airport Road, Kolhapur | country: India"
for key, spec in ENCODERS.items():
    try:
        model = load_encoder(spec, mem_fraction=0.4)
        print("OK", key, encode(model, [TEXT], spec.query_prefix, 8).shape, flush=True)
        del model
    except Exception as ex:  # report every model, don't stop at the first failure
        print("FAIL", key, repr(ex), flush=True)
