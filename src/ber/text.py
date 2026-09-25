import re
import unicodedata

from anyascii import anyascii

from ber.lexicons import Lexicon

NUM_RE = re.compile(r"\d[\d/\-]*[a-z]?")
ADDR_TOKEN_RE = re.compile(r"#|[a-z0-9][a-z0-9/\-]*")
UNIT_WORDS = frozenset({"#", "unit", "suite", "apt", "apartment", "fl", "floor", "box", "lot",
                        "sector", "sec", "pin", "flat", "room", "ste"})
DOMAIN_RE = re.compile(r"#?(www\.)?[a-z0-9\-]+(\.[a-z]{2,4}){0,2}")


def fold(s: str) -> str:
    return anyascii(unicodedata.normalize("NFKC", s)).lower()


def _strip_legal(toks: list[str], lex: Lexicon) -> tuple[list[str], list[str]]:
    core, suffixes, i = [], [], 0
    while i < len(toks):
        for pat, canon in lex.legal_index.get(toks[i], ()):
            if tuple(toks[i:i + len(pat)]) == pat:
                suffixes.append(canon)
                i += len(pat)
                break
        else:
            core.append(toks[i])
            i += 1
    return core, suffixes


def normalize_name(raw: str, lex: Lexicon) -> tuple[str, str, str]:
    s = fold(raw)
    s = re.sub(r"\bm/s\b", " ", s).replace("&", " and ")
    toks = [t for t in re.sub(r"[^a-z0-9]+", " ", s).split() if t != "null" and t not in lex.fillers]
    toks = [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]]
    core, suffixes = _strip_legal(toks, lex)
    core = [t for t in core if t not in lex.stopwords]
    name_norm = " ".join(toks)
    return name_norm, " ".join(core) or name_norm, "+".join(sorted(set(suffixes)))


def _canon_num(tok: str) -> str:
    return tok.replace("-", "").strip("/").lstrip("0") or "0"


def normalize_address(raw: str, country: str, lex: Lexicon) -> tuple[str, str, str]:
    for native, english in lex.native.items():
        if native in raw:
            raw = raw.replace(native, english)
    states = lex.states.get(country, {})
    comps = [c.strip() for c in fold(raw).split(",")]
    s = " , ".join(states.get(c, c) for c in comps if c and c != "null")
    out, nums, house_no, prev = [], [], "", ""
    for t in ADDR_TOKEN_RE.findall(s):
        if NUM_RE.fullmatch(t):
            c = _canon_num(t)
            nums.append(c)
            if not house_no and prev not in UNIT_WORDS and prev != "po" and not (c.isdigit() and len(c) == 6):
                house_no = c
            out.append(c)
        elif t != "#":
            out.extend(p for p in t.split("-") if p)
        prev = t
    out = [lex.abbreviations.get(t, t) for t in out]
    out = [t for t in out if t not in lex.addr_fillers]
    return " ".join(out), house_no, " ".join(dict.fromkeys(nums))


def is_domain_name(raw: str) -> bool:
    f = fold(raw).strip()
    return len(f) >= 6 and ("." in f or f.startswith("#")) and DOMAIN_RE.fullmatch(f) is not None


def domain_body(raw: str) -> str:
    f = fold(raw).strip()
    f = re.sub(r"^#|^www\.", "", f)
    f = re.sub(r"(\.[a-z]{2,4})+$", "", f)
    return f.replace("-", "")


def segment(s: str, vocab: set[str], max_len: int = 20) -> list[str]:
    """Min-cost split of s into vocab words; digit runs are free words, unknown chunks cost 1+len."""
    n = len(s)
    best: list[tuple[float, int] | None] = [(0.0, 0)] + [None] * n
    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):
            if best[j] is None:
                continue
            w = s[j:i]
            cost = best[j][0] + (1 if (w in vocab or w.isdigit()) else 1 + len(w))
            if best[i] is None or cost < best[i][0]:
                best[i] = (cost, j)
    out, i = [], n
    while i > 0:
        j = best[i][1]
        out.append(s[j:i])
        i = j
    return out[::-1]


def record_text(name: str, addr: str, country: str) -> str:
    return f"name: {name} | address: {addr} | country: {country}"
