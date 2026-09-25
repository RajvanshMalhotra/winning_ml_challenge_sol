from ber.io import read_family, read_truth, truth_pairs

HDR = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def _write(d, family):
    (d / family).mkdir(parents=True)
    (d / family / f"{family}_source1.tsv").write_text(HDR + 'S1-1\tA "quoted" & Co\t1 Main St, X, TX\tUS\nS1-2\tB\t\tUS\n')
    (d / family / f"{family}_source2.tsv").write_text(HDR + "S2-1\tA Co\t1 MAIN ST\tUS\n")
    (d / family / f"{family}_source3.tsv").write_text(HDR + "S3-1\tB\t2 Rd\tUS\n")


def test_read_family_concatenates_with_source(tmp_path):
    _write(tmp_path, "train")
    df = read_family(tmp_path, "train")
    assert list(df.columns) == ["entity_id", "business_name", "business_address", "country", "source"]
    assert df.source.tolist() == [1, 1, 2, 3]
    assert df.business_name.iloc[0] == 'A "quoted" & Co'  # QUOTE_NONE keeps quotes
    assert df.business_address.iloc[1] == ""  # empty stays "", not NaN


def test_truth(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\t\n"
    )
    t = read_truth(tmp_path)
    assert t == {"S1-1": {"S2-1", "S3-1"}, "S1-2": set()}
    tp = truth_pairs(t)
    assert sorted(map(tuple, tp.values.tolist())) == [("S1-1", "S2-1"), ("S1-1", "S3-1")]
    assert len(truth_pairs(t, ["S1-2"])) == 0
