import numpy as np
import pandas as pd

from ber.stage2 import stage2_features


def test_duplicate_support():
    # S1-a has a confident candidate x (p .95) and a weak candidate y (p .2); x and y are KG neighbours
    code = pd.Series(np.arange(5), index=["S1-a", "x", "y", "z", "S1-b"])
    scored = pd.DataFrame({"s1_id": ["S1-a", "S1-a", "S1-a", "S1-b"], "cand_id": ["x", "y", "z", "y"],
                           "p": [0.95, 0.2, 0.1, 0.4]})
    edges = pd.DataFrame({"a": [1, 2], "b": [2, 1]})        # x <-> y
    f = stage2_features(scored, edges, code).set_index(["s", "c"])
    assert f.loc[(0, 2), "sup_max"] == np.float32(0.95)    # y is supported by confident x (same S1)
    assert f.loc[(0, 3), "sup_max"] == 0                    # z has no neighbours
    assert f.loc[(4, 2), "sup_max"] == 0                    # support never crosses S1s
    assert f.loc[(0, 1), "rank_p"] == 1 and f.loc[(0, 1), "n_ge70"] == 1 and np.isclose(f.loc[(0, 2), "p_gap"], 0.75)
