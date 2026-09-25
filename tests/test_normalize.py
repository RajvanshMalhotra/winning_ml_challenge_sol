import pandas as pd

from ber.normalize import build_records


def _raw():
    return pd.DataFrame({
        "entity_id": ["S1-1", "S2-1", "S3-1", "S1-2", "S2-2"],
        "business_name": ["Fafloon Pet Care Inc", "FAFLOON CARE INC", "fafloonpetcare.com", "Galaxy Solutions Pvt Ltd", "Galaxy S0lutions Pvt Ltd."],
        "business_address": ["1130 Regency Road, Atlanta, GA", "01130 REGENCY ROAD, ATL, GA", "1130- Regency Road, Atlanat, Georgia",
                             "47/1 Airport Road, Kolhapur, Maharashtra", "NULL, MH, 4-7/1 Airport Road"],
        "country": ["US", "US", "US", "India", "India"],
        "source": pd.Series([1, 2, 3, 1, 2], dtype="int8"),
    })


def test_columns_and_domain_split():
    rec = build_records(_raw(), n_jobs=1)
    assert list(rec.columns) == ["entity_id", "source", "country", "name_raw", "addr_raw", "name_norm", "name_core",
                                 "legal_suffix", "name_domain", "addr_norm", "house_no", "num_tokens", "block_text"]
    r = rec.set_index("entity_id")
    assert r.loc["S3-1", "name_domain"] == "fafloon pet care"
    assert r.loc["S1-1", "name_domain"] == ""
    assert r.loc["S3-1", "block_text"].startswith("fafloon pet care ")
    assert r.loc["S2-2", "house_no"] == r.loc["S1-2", "house_no"] == "47/1"


def test_parallel_matches_serial():
    raw = pd.concat([_raw()] * 20, ignore_index=True)
    pd.testing.assert_frame_equal(build_records(raw, n_jobs=1), build_records(raw, n_jobs=4))
