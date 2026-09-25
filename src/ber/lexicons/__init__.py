import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

FILES = ("en.yaml", "in.yaml", "fr.yaml")


@dataclass(frozen=True)
class Lexicon:
    fillers: frozenset[str]
    stopwords: frozenset[str]
    addr_fillers: frozenset[str]
    abbreviations: dict[str, str]
    states: dict[str, dict[str, str]]  # country label -> {abbr: full name}
    native: dict[str, str]  # native-script string -> English name
    postal: dict[str, re.Pattern]  # country label -> postal-code regex (applied to folded text)
    # first token -> [(pattern tokens, canonical)], longest pattern first
    legal_index: dict[str, list[tuple[tuple[str, ...], str]]]


@lru_cache(maxsize=1)
def load_lexicon() -> Lexicon:
    fillers, stop, addr_f = set(), set(), set()
    abbr, states, native, legal, postal = {}, {}, {}, {}, {}
    for name in FILES:
        d = yaml.safe_load((Path(__file__).parent / name).read_text())
        fillers |= set(d.get("fillers", []))
        stop |= set(d.get("stopwords", []))
        addr_f |= set(d.get("addr_fillers", []))
        abbr.update(d.get("abbreviations", {}))
        states.update(d.get("states", {}) or {})
        native.update(d.get("native", {}) or {})
        postal.update(d.get("postal", {}) or {})
        for canon, variants in d.get("legal_suffixes", {}).items():
            for v in variants:
                toks = tuple(v.split())
                legal.setdefault(toks[0], []).append((toks, canon))
    for pats in legal.values():
        pats.sort(key=lambda p: -len(p[0]))
    # str(k) guards state codes against YAML 1.1 booleans (yes/no/on/off); list entries like "no" must be quoted
    states = {c: {str(k): v for k, v in m.items()} for c, m in states.items()}
    postal = {c: re.compile(p) for c, p in postal.items()}
    return Lexicon(frozenset(fillers), frozenset(stop), frozenset(addr_f), abbr, states, native, postal, legal)
