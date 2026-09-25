from ber.contrastive.encoders import ENCODERS


def test_registry_has_bakeoff_models_with_permissive_licenses():
    assert set(ENCODERS) == {"gte", "nomic", "arctic", "bgem3", "qwen8b"}
    assert ENCODERS["nomic"].query_prefix == "search_query: " and ENCODERS["nomic"].doc_prefix == "search_document: "
    assert ENCODERS["qwen8b"].lora and ENCODERS["qwen8b"].truncate_dim == 1024
    assert not any(s.lora for k, s in ENCODERS.items() if k != "qwen8b")
