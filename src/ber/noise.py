"""Stochastic noise operators mirroring the patterns observed in the data (approaches.md §8a)."""
import re

import numpy as np

from ber.lexicons import load_lexicon

ACCENTS = {"a": "á", "e": "é", "o": "ó", "i": "í", "u": "ú"}
LOOKALIKE = {"o": "0", "l": "1", "i": "1", "s": "5", "e": "3"}
PREFIX_FILLERS = ["The", "Mr", "M/s", "The The"]
SUFFIX_FILLERS = ["Center", "Group", "Services"]
SUFFIX_SWAPS = [("Private Limited", "Pvt Ltd"), ("Limited", "Ltd"), ("Private", "Pvt"), ("Incorporated", "Inc"),
                ("Corporation", "Corp"), ("Company", "Co"), ("SARL", "S.A.R.L."), ("SAS", "S.A.S")]
STREET_SWAPS = [("Road", "Rd"), ("Street", "St"), ("Avenue", "Ave"), ("Boulevard", "Blvd"), ("Rue", "R."),
                ("Floor", "Fl"), ("Near", "Nr"), ("Opposite", "Opp")]


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _swap_pairs(text, pairs, rng):
    hits = [(a, b) for a, b in pairs if re.search(rf"\b{re.escape(a)}\b", text, re.I)]
    hits += [(b, a) for a, b in pairs if re.search(rf"(?<!\w){re.escape(b)}(?!\w)", text, re.I)]
    if not hits:
        return text
    a, b = _pick(rng, hits)
    return re.sub(rf"(?<!\w){re.escape(a)}(?!\w)", b, text, count=1, flags=re.I)


# ---- name operators: (name, rng) -> name ----
def upper_case(s, rng):
    return s.upper() if s != s.upper() else s.title()


def typo(s, rng):
    idx = [i for i, c in enumerate(s) if c.isalpha()]
    if not idx:
        return s
    i = _pick(rng, idx)
    kind = int(rng.integers(4))
    if kind == 0:
        return s[:i] + s[i + 1:] if len(s) > 1 else s
    if kind == 1 and i + 1 < len(s):
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if kind == 2 and s[i].lower() in LOOKALIKE:
        return s[:i] + LOOKALIKE[s[i].lower()] + s[i + 1:]
    return s[:i] + s[i] + s[i:]


def accent(s, rng):
    idx = [i for i, c in enumerate(s) if c.lower() in ACCENTS]
    if not idx:
        return s
    i = _pick(rng, idx)
    rep = ACCENTS[s[i].lower()]
    return s[:i] + (rep.upper() if s[i].isupper() else rep) + s[i + 1:]


def filler(s, rng):
    return f"{_pick(rng, PREFIX_FILLERS)} {s}" if rng.random() < 0.5 else f"{s} {_pick(rng, SUFFIX_FILLERS)}"


def drop_token(s, rng):
    toks = s.split()
    if len(toks) < 2:
        return s
    del toks[int(rng.integers(len(toks)))]
    return " ".join(toks)


def truncate(s, rng):
    toks = s.split()
    return " ".join(toks[: int(rng.integers(1, len(toks)))]) if len(toks) >= 2 else s


def domain_style(s, rng):
    body = re.sub(r"[^a-z0-9]", "", s.lower())
    return f"#{body}" if rng.random() < 0.3 else f"{body}.com"


def suffix_swap(s, rng):
    return _swap_pairs(s, SUFFIX_SWAPS, rng)


def reorder(s, rng):
    toks = s.split()
    return " ".join([toks[-1]] + toks[:-1]) if len(toks) >= 2 else s


NAME_OPS = [upper_case, typo, accent, filler, drop_token, truncate, domain_style, suffix_swap, reorder]


# ---- address operators: (addr, country, rng) -> addr ----
def _comps(a):
    return [c.strip() for c in a.split(",") if c.strip()]


def addr_upper(a, country, rng):
    return upper_case(a, rng)


def null_insert(a, country, rng):
    c = _comps(a)
    c.insert(int(rng.integers(len(c) + 1)), "NULL")
    return ", ".join(c)


def drop_component(a, country, rng):
    c = _comps(a)
    if len(c) < 2:
        return a
    del c[int(rng.integers(len(c)))]
    return ", ".join(c)


def shuffle_components(a, country, rng):
    c = _comps(a)
    if len(c) < 2:
        return a
    perm = rng.permutation(len(c))
    if (perm == np.arange(len(c))).all():
        perm = np.roll(perm, 1)
    return ", ".join(c[i] for i in perm)


def street_swap(a, country, rng):
    return _swap_pairs(a, STREET_SWAPS, rng)


def state_swap(a, country, rng):
    states = load_lexicon().states.get(country, {})
    c = _comps(a)
    for i, comp in enumerate(c):
        low = comp.lower()
        if low in states:
            c[i] = states[low].title()
            return ", ".join(c)
        for abbr, full in states.items():
            if low == full:
                c[i] = abbr.upper()
                return ", ".join(c)
    return a


def house_reformat(a, country, rng):
    m = re.search(r"\b\d+\b", a)
    if not m:
        return a
    n = m.group()
    new = _pick(rng, [f"0{n}", f"{n}-", f"{n[0]}-{n[1:]}" if len(n) > 1 else f"0{n}"])
    return a[: m.start()] + new + a[m.end():]


def unit_toggle(a, country, rng):
    c = _comps(a)
    units = [i for i, x in enumerate(c) if re.match(r"(?i)(unit|po box|#|suite|fl)\b", x)]
    if units:
        del c[units[0]]
    else:
        c.insert(min(1, len(c)), _pick(rng, [f"Unit {int(rng.integers(1, 999))}", f"PO BOX {int(rng.integers(100, 9999))}"]))
    return ", ".join(c)


ADDR_OPS = [addr_upper, null_insert, drop_component, shuffle_components, street_swap, state_swap, house_reformat, unit_toggle]
EMPTY_ADDR_P = 0.03


def augment(name: str, addr: str, country: str, rng: np.random.Generator, max_ops: int = 2) -> tuple[str, str]:
    for i in rng.choice(len(NAME_OPS), size=int(rng.integers(1, max_ops + 1)), replace=False):
        new = NAME_OPS[i](name, rng)
        name = new if new.strip() else name
    if rng.random() < EMPTY_ADDR_P:
        return name, ""
    if addr:
        for i in rng.choice(len(ADDR_OPS), size=int(rng.integers(1, max_ops + 1)), replace=False):
            addr = ADDR_OPS[i](addr, country, rng)
    return name, addr
