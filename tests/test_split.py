import pandas as pd

from ber.split import _bucket, assign_splits

CFG = {"a_train": 0.35, "a_val": 0.05, "n_folds": 5}


def _data(n=4000):
    truth = {f"S1-{i}": {f"S2-{i}-{j}" for j in range(i % 7)} for i in range(n)}
    country = pd.Series({f"S1-{i}": ("US" if i % 3 else "India") for i in range(n)})
    return truth, country


def test_shares_folds_and_determinism():
    truth, country = _data()
    a = assign_splits(truth, country, CFG, seed=1)
    b = assign_splits(truth, country, CFG, seed=1)
    pd.testing.assert_frame_equal(a, b)
    share = a.split.value_counts(normalize=True)
    assert abs(share["A_train"] - 0.35) < 0.01 and abs(share["A_val"] - 0.05) < 0.01
    assert (a[a.split != "B"].fold == -1).all()
    folds = a[a.split == "B"].fold.value_counts()
    assert set(folds.index) == set(range(5)) and folds.max() - folds.min() <= 10


def test_stratified_by_country_and_bucket():
    truth, country = _data()
    a = assign_splits(truth, country, CFG, seed=1)
    for key, grp in a.groupby([a.country, a.n_matches.map(_bucket)]):
        assert abs((grp.split == "B").mean() - 0.60) < 0.01, key
