import numpy as np

from ber.noise import ADDR_OPS, NAME_OPS, augment

NAME = "Galaxy Solutions Private Limited"
ADDR = "47 Airport Road, Kolhapur, Maharashtra"


def test_every_name_op_changes_text_for_some_seed():
    for op in NAME_OPS:
        outs = {op(NAME, np.random.default_rng(s)) for s in range(20)}
        assert outs - {NAME}, op.__name__
        assert all(o.strip() for o in outs), op.__name__


def test_every_addr_op_changes_text_for_some_seed():
    for op in ADDR_OPS:
        outs = {op(ADDR, "India", np.random.default_rng(s)) for s in range(20)}
        assert outs - {ADDR}, op.__name__


def test_augment_deterministic_and_name_never_empty():
    a = [augment(NAME, ADDR, "India", np.random.default_rng(7)) for _ in range(2)]
    assert a[0] == a[1]
    for s in range(300):
        n, _ = augment("X", "", "France", np.random.default_rng(s))
        assert n.strip()
