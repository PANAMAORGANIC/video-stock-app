"""Relevance ranking for Library footage search.

Stem + topic clusters, subject-over-place scoring. Optional cached Grok
query expansion. Does not call the network except llm_query_terms.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

STOPWORDS = {
    "el", "la", "los", "las", "de", "del", "un", "una", "unos", "unas",
    "y", "o", "en", "a", "al", "por", "para", "con", "que", "se", "su",
    "sus", "lo", "le", "es", "son", "me", "mi", "tu", "te", "si", "no",
    "ya", "muy", "mas", "más", "como", "over", "the", "a", "an", "of",
    "and", "or", "in", "on", "to", "for", "with", "from", "at", "by",
    "as", "is", "are", "be", "this", "that", "it", "its", "vs",
    "stock", "footage", "video", "videos", "clip", "clips", "b-roll",
    "broll", "b_roll", "youtube", "tiktok", "instagram", "dailymotion",
    "free", "hd", "4k", "shorts", "short", "official", "full", "watch",
    "nuevo", "new", "como", "how", "what", "when",
    "close", "closeup", "up",
}

# Longest suffix first. Remainder must stay >= 4 except plural -s/-es (>= 3)
# so "rios" -> "rio" still works.
_STEM_SUFFIXES = (
    "ciones", "mientos",
    "cion", "miento", "dades",
    "ados", "adas", "idos", "idas",
    "ando", "endo",
    "dad",
    "ado", "ada", "ido", "ida",
    "ar", "er", "ir",
    "es", "s",
)

TOPICS: Dict[str, Dict[str, List[str]]] = {
    "agua_contaminada": {
        "es": [
            "agua", "aguas", "sucia", "sucio", "sucias", "turbia", "turbio",
            "contaminacion", "contaminado", "contaminada", "contaminantes",
            "vertido", "vertimiento", "derrame", "residual", "residuales",
        ],
        "en": [
            "water", "waters", "dirty", "polluted", "contaminated",
            "pollution", "contamination", "wastewater", "spill", "runoff",
            "dumping",
        ],
    },
    "rio": {
        "es": [
            "rio", "rios", "quebrada", "arroyo", "cauce", "ribera",
        ],
        "en": [
            "river", "rivers", "creek", "stream", "streambed", "riverbank",
        ],
    },
    "sequia": {
        "es": ["sequia", "seco", "seca", "sediento", "bajante", "estiaje"],
        "en": ["drought", "dry", "parched", "receding", "low water"],
    },
    "quema_agricola": {
        "es": [
            "quema", "quemas", "humo", "humos", "fuego", "rastrojo",
            "canaveral", "canaverales", "zafra",
        ],
        "en": [
            "sugarcane", "burning", "smoke", "field", "fire", "stubble",
            "crop burning", "fume", "fumes", "smokes",
        ],
    },
    "ganaderia": {
        "es": ["ganado", "vaca", "res", "pasto", "potrero"],
        "en": ["cattle", "cow", "livestock", "pasture", "grazing"],
    },
    "residuos": {
        "es": [
            "basura", "desechos", "residuos", "plastico", "vertedero",
        ],
        "en": [
            "trash", "garbage", "waste", "rubbish", "plastic", "landfill",
            "dump", "residues",
        ],
    },
    "agricultura": {
        "es": [
            "agricultura", "cultivo", "siembra", "cosecha", "finca", "granja",
        ],
        "en": [
            "agriculture", "farming", "farm", "crop", "harvest",
        ],
    },
    "costa": {
        "es": [
            "playa", "mar", "oceano", "manglar", "manglares", "humedal",
            "pantano",
        ],
        "en": [
            "beach", "sea", "ocean", "mangrove", "mangroves", "wetland",
            "swamp", "marsh",
        ],
    },
    "suelo": {
        "es": ["suelo", "suelos", "tierra"],
        "en": ["soil", "soils", "land", "earth"],
    },
    "bosque": {
        "es": ["deforestacion"],
        "en": ["deforestation"],
    },
    "mineria": {
        "es": ["mineria"],
        "en": ["mining"],
    },
    "pesticidas": {
        "es": ["plaguicida", "pesticida"],
        "en": ["pesticide"],
    },
    "ambiente": {
        "es": ["ambiente", "ambiental"],
        "en": ["environment", "environmental"],
    },
    "documental": {
        "es": ["documental"],
        "en": ["documentary"],
    },
    "limpio": {
        "es": ["limpio"],
        "en": ["clean"],
    },
}

PLACES = {
    "panama", "azuero", "herrera", "chitre", "veraguas", "cocle", "chiriqui",
    "darien", "parita", "colon", "panameno", "panamanian",
}
PLACE_PHRASES = ("los santos", "la villa", "santa maria")

DEMOTE_RE = re.compile(
    r"(?i)\b("
    r"live\s*stream|en\s*vivo|24\s*/\s*7|hour\s*long|hora\s*completa|"
    r"music\s*video|official\s*video|lyrics|karaoke|nightcore|"
    r"full\s*album|podcast|storytime|vlog|compilation|"
    r"river\s*plate|partido\s+de|golazo|\bvs\.?\b|tiktok\s*compilation|"
    r"gameplay|reacts?|reaction|"
    r"honda\s*civic|farmear\s*aura"
    r")\b"
)

STOCK_BIAS = "stock footage b-roll"
MIN_SCORE = 2.0

BASE_DIR = Path(__file__).resolve().parent.parent
SEARCH_TERMS_CACHE = BASE_DIR / "storage" / "search-terms-cache.json"
_MEM_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_CACHE_CAP = 500


def fold(text: str) -> str:
    nk = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in nk if not unicodedata.combining(c)).lower()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", fold(text))


def stem(token: str) -> str:
    t = fold(token)
    if len(t) < 4:
        return t
    for suf in _STEM_SUFFIXES:
        if not t.endswith(suf):
            continue
        rest = t[: -len(suf)]
        min_rest = 3 if suf in {"s", "es"} else 4
        if len(rest) >= min_rest:
            return rest
    return t


def _index_topics() -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    stem_to: Dict[str, Set[str]] = {}
    topic_stems: Dict[str, Set[str]] = {}
    for name, spec in TOPICS.items():
        bag: Set[str] = set()
        for w in (spec.get("es") or []) + (spec.get("en") or []):
            f = fold(w)
            for piece in f.split():
                if len(piece) < 2:
                    continue
                bag.add(piece)
                bag.add(stem(piece))
                stem_to.setdefault(piece, set()).add(name)
                stem_to.setdefault(stem(piece), set()).add(name)
        topic_stems[name] = bag
    return stem_to, topic_stems


STEM_TO_TOPICS, TOPIC_STEMS = _index_topics()

GLOSSARY: Dict[str, List[str]] = {}
for _name, _spec in TOPICS.items():
    _members = [fold(w) for w in (_spec.get("es") or []) + (_spec.get("en") or []) if fold(w)]
    for _m in _members:
        GLOSSARY[_m] = [x for x in _members if x != _m]


def meaningful_tokens(query: str) -> List[str]:
    """Original query tokens minus stopwords. Order preserved, unique."""
    seen = set()
    out: List[str] = []
    for tok in tokenize(query):
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def expand_tokens(tokens: Sequence[str]) -> List[str]:
    """Original tokens plus topic-cluster synonyms. Originals stay first."""
    seen = set()
    out: List[str] = []
    for tok in tokens:
        f = fold(tok)
        if f and f not in seen:
            seen.add(f)
            out.append(f)
        for topic in STEM_TO_TOPICS.get(stem(f), ()):
            spec = TOPICS.get(topic) or {}
            for w in (spec.get("es") or []) + (spec.get("en") or []):
                fw = fold(w)
                if fw and fw not in seen and fw not in STOPWORDS:
                    seen.add(fw)
                    out.append(fw)
    return out


def activate_topics(query_stems: Sequence[str]) -> Dict[str, float]:
    """topic -> fraction of that topic's stems the query hit. Active at >= 1."""
    qset = {stem(fold(s)) for s in query_stems if s}
    out: Dict[str, float] = {}
    for name, members in TOPIC_STEMS.items():
        hits = sum(1 for s in qset if s in members)
        if hits >= 1:
            out[name] = hits / max(1, len(members))
    return out


def _place_stems() -> Set[str]:
    bag = {stem(p) for p in PLACES}
    bag.update(PLACES)
    for ph in PLACE_PHRASES:
        for w in ph.split():
            bag.add(fold(w))
            bag.add(stem(w))
    return bag


PLACE_STEMS = _place_stems()


def extract_entities(query: str) -> List[str]:
    raw = query or ""
    folded = fold(raw)
    found: List[str] = []
    seen = set()

    def add(tok: str) -> None:
        f = fold(tok)
        if f and f not in seen and f not in STOPWORDS:
            seen.add(f)
            found.append(f)

    for ph in PLACE_PHRASES:
        if ph in folded:
            add(ph)

    caps = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", raw)
    for i, tok in enumerate(caps):
        f = fold(tok)
        if f in PLACE_STEMS or f in PLACES:
            add(tok)
            continue
        if tok[:1].isupper() and not (i == 0 and f in STOPWORDS):
            if len(f) >= 3:
                add(tok)
    for tok in tokenize(raw):
        if tok in PLACES or stem(tok) in PLACE_STEMS:
            add(tok)
    return found


def understand_query(query: str) -> Dict[str, Any]:
    raw = " ".join((query or "").split())
    exact = meaningful_tokens(raw)
    terms: List[str] = []
    seen_t = set()
    for tok in exact:
        s = stem(tok)
        if s not in seen_t:
            seen_t.add(s)
            terms.append(s)
    entities = extract_entities(raw)
    topics = activate_topics(terms + exact)
    expansion_es: List[str] = []
    expansion_en: List[str] = []
    seen_es, seen_en = set(), set()
    ranked = sorted(topics, key=lambda n: topics[n], reverse=True)
    for name in ranked:
        spec = TOPICS.get(name) or {}
        for w in spec.get("es") or []:
            f = fold(w)
            if f and f not in seen_es:
                seen_es.add(f)
                expansion_es.append(f)
        for w in spec.get("en") or []:
            f = fold(w)
            if f and f not in seen_en:
                seen_en.add(f)
                expansion_en.append(f)
    search_strings: List[str] = []
    if ranked:
        words: List[str] = []
        for name in ranked[:2]:
            for w in (TOPICS.get(name) or {}).get("en") or []:
                if w not in words:
                    words.append(w)
        picked: List[str] = []
        seen_p = set()
        for w in words:
            for piece in fold(w).split():
                if piece in STOPWORDS or len(piece) < 3 or piece in seen_p:
                    continue
                seen_p.add(piece)
                picked.append(piece)
            if len(picked) >= 6:
                break
        if picked:
            search_strings.append(" ".join(picked[:6]))
    return {
        "raw": raw,
        "terms": terms,
        "exact": exact,
        "entities": entities,
        "topics": topics,
        "expansion_es": expansion_es,
        "expansion_en": expansion_en,
        "search_strings": search_strings,
        "llm_stems": [],
    }


def expand_search_query(query: str, stock_bias: bool = True) -> str:
    """Keep the user's words, append EN/ES topic synonyms and optional b-roll bias."""
    raw = " ".join((query or "").split())
    if not raw:
        return "stock footage"
    u = understand_query(raw)
    extras: List[str] = []
    seen = set(tokenize(raw))
    for w in u["expansion_en"] + u["expansion_es"]:
        s = fold(w)
        if s and s not in seen:
            seen.add(s)
            extras.append(w)
    parts = [raw]
    if extras:
        parts.append(" ".join(extras))
    folded_all = fold(" ".join(parts))
    if (
        stock_bias
        and "stock" not in folded_all
        and "b-roll" not in folded_all
        and "broll" not in folded_all
    ):
        parts.append(STOCK_BIAS)
    return " ".join(parts)


def parse_duration_sec(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    raw = str(value).strip().lower().replace(",", ".")
    raw = raw.replace("seconds", "").replace("second", "").replace("secs", "").replace("sec", "")
    raw = raw.replace("minutos", "").replace("minuto", "").replace("mins", "").replace("min", "")
    raw = raw.strip()
    if not raw:
        return None
    if raw.endswith("s") and raw[:-1].replace(".", "", 1).isdigit():
        raw = raw[:-1]
    if ":" in raw:
        parts = raw.split(":")
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return None
        if len(nums) == 3:
            return nums[0] * 3600 + nums[1] * 60 + nums[2]
        if len(nums) == 2:
            return nums[0] * 60 + nums[1]
        return None
    try:
        n = float(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def format_duration(value: Any) -> Optional[str]:
    sec = parse_duration_sec(value)
    if sec is None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None
    total = int(round(sec))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def duration_score(sec: Optional[float]) -> float:
    """Prefer documentary-length clips (5s–8min) over 30min+ vlogs/lives."""
    if sec is None:
        return 0.0
    if 5 <= sec <= 480:
        return 3.0
    if 480 < sec <= 900:
        return 1.5
    if 3 <= sec < 5:
        return 1.0
    if 900 < sec <= 1800:
        return -1.0
    if sec > 1800:
        return -3.0
    return -0.5


def item_text(item: Dict[str, Any]) -> str:
    bits = [
        str(item.get("title") or ""),
        str(item.get("description") or ""),
        str(item.get("channel") or ""),
    ]
    return " ".join(bits)


def _item_tokens(item: Dict[str, Any]) -> List[str]:
    return tokenize(item_text(item))


def _item_stems(item: Dict[str, Any]) -> Set[str]:
    return {stem(t) for t in _item_tokens(item)}


def token_hit_count(item: Dict[str, Any], tokens: Sequence[str]) -> int:
    blob = " " + fold(item_text(item)) + " "
    stems = _item_stems(item)
    hits = 0
    for tok in tokens:
        if len(tok) < 2:
            continue
        f = fold(tok)
        if f" {f} " in blob or blob.startswith(f + " ") or blob.endswith(" " + f):
            hits += 1
            continue
        if re.search(r"(?<![a-z0-9])" + re.escape(f) + r"(?![a-z0-9])", blob):
            hits += 1
            continue
        if stem(f) in stems:
            hits += 1
    return hits


def _exact_hit(blob: str, tok: str) -> bool:
    f = fold(tok)
    if len(f) < 2:
        return False
    if f" {f} " in blob or blob.startswith(f + " ") or blob.endswith(" " + f):
        return True
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(f) + r"(?![a-z0-9])", blob))


def _platform_score(platform: str) -> float:
    p = (platform or "").lower()
    if p in {"youtube", "commons", "archive", "openverse"}:
        return 0.8
    if p == "dailymotion":
        return 0.3
    if p in {"tiktok", "instagram"}:
        return -2.0
    return 0.0


def _query_phrases(u: Dict[str, Any]) -> List[str]:
    exact = u.get("exact") or []
    out = []
    for i in range(len(exact) - 1):
        out.append(exact[i] + " " + exact[i + 1])
    return out


def _is_entity_tok(tok: str, u: Dict[str, Any]) -> bool:
    f = fold(tok)
    ents = set(u.get("entities") or [])
    return f in ents or f in PLACES or stem(f) in PLACE_STEMS


def _as_understood(original: Any, expanded: Any = None) -> Dict[str, Any]:
    if isinstance(original, dict) and "exact" in original:
        return original
    q = " ".join(str(t) for t in (original or []))
    u = understand_query(q)
    if expanded:
        for t in expanded:
            f = fold(str(t))
            if f and f not in u["expansion_en"] and f not in u["exact"]:
                u["expansion_en"].append(f)
    return u


def merge_llm_into_understood(u: Dict[str, Any], llm: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not llm or not isinstance(llm, dict):
        return u
    llm_stems: List[str] = []
    for key in ("terms_es", "terms_en", "visuals"):
        vals = llm.get(key) or []
        if isinstance(vals, str):
            vals = [vals]
        target = "expansion_es" if key == "terms_es" else "expansion_en"
        bag = u[target]
        seen = set(bag)
        for w in vals:
            f = fold(str(w))
            if not f:
                continue
            llm_stems.append(stem(f))
            if f not in seen:
                seen.add(f)
                bag.append(f)
    existing = set(u.get("llm_stems") or [])
    for s in llm_stems:
        if s not in existing:
            existing.add(s)
            u.setdefault("llm_stems", []).append(s)
    topic = fold(str(llm.get("topic") or ""))
    if topic:
        for name in TOPICS:
            if fold(name.replace("_", " ")) in topic or topic in fold(name):
                u.setdefault("topics", {})
                if name not in u["topics"]:
                    u["topics"][name] = 0.5
    return u


def relevance_score(item, original, expanded=None):
    u = _as_understood(original, expanded)
    blob = " " + fold(item_text(item)) + " "
    item_stems = _item_stems(item)
    exact_toks = list(u.get("exact") or [])
    entities = list(u.get("entities") or [])
    topics = u.get("topics") or {}
    llm_stems = set(u.get("llm_stems") or [])

    subject_toks = [t for t in exact_toks if not _is_entity_tok(t, u)]
    score = 0.0
    covered = 0
    subject_hits = 0

    topic_members_in_item: Set[str] = set()
    for name in topics:
        topic_members_in_item |= item_stems & (TOPIC_STEMS.get(name) or set())

    for tok in exact_toks:
        if _is_entity_tok(tok, u):
            continue
        if _exact_hit(blob, tok):
            score += 5.0
            covered += 1
            subject_hits += 1
        elif stem(tok) in item_stems:
            score += 3.0
            covered += 1
            subject_hits += 1

    for tok in entities:
        if _exact_hit(blob, tok) or stem(tok) in item_stems:
            score += 2.5

    for name, strength in topics.items():
        members = TOPIC_STEMS.get(name) or set()
        if item_stems & members:
            score += 2.0 * float(strength or 0)

    for s in llm_stems:
        if s and s in item_stems:
            score += 2.0

    title_fold = fold(str(item.get("title") or ""))
    for ph in _query_phrases(u):
        if ph and ph in title_fold:
            score += 2.0
            break

    total_subj = max(1, len(subject_toks) if subject_toks else len(exact_toks) or 1)
    score += (covered / total_subj) * 3.0
    no_subject = bool(subject_toks) and subject_hits == 0 and not topic_members_in_item
    if no_subject:
        score -= 4.0

    sec = parse_duration_sec(
        item.get("duration_sec") if item.get("duration_sec") is not None
        else item.get("duration")
    )
    if not no_subject:
        score += duration_score(sec)
    score += _platform_score(str(item.get("platform") or ""))
    title = str(item.get("title") or "")
    if DEMOTE_RE.search(title):
        score -= 4.0
    folded = fold(title)
    if "live" in folded.split() or "en vivo" in folded:
        score -= 2.0
    return score


def _has_any_hit(item: Dict[str, Any], u: Dict[str, Any]) -> bool:
    blob = " " + fold(item_text(item)) + " "
    stems = _item_stems(item)
    for tok in (u.get("exact") or []) + (u.get("entities") or []):
        if _exact_hit(blob, tok) or stem(tok) in stems:
            return True
    for name in (u.get("topics") or {}):
        if stems & (TOPIC_STEMS.get(name) or set()):
            return True
    for s in u.get("llm_stems") or []:
        if s in stems:
            return True
    for w in (u.get("expansion_en") or []) + (u.get("expansion_es") or []):
        if _exact_hit(blob, w) or stem(w) in stems:
            return True
    return False


def dedupe_items(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        vid = str(it.get("video_id") or "").strip().lower()
        url = str(it.get("url") or "").split("?")[0].strip().lower().rstrip("/")
        key = (it.get("platform") or "", vid or url)
        if not key[1] or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _xai_key_for_search() -> Optional[str]:
    return os.environ.get("XAI_API_KEY") or os.environ.get("xai_api_key")


def _load_disk_cache() -> Dict[str, Any]:
    try:
        raw = SEARCH_TERMS_CACHE.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_disk_cache(data: Dict[str, Any]) -> None:
    try:
        SEARCH_TERMS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SEARCH_TERMS_CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(SEARCH_TERMS_CACHE)
    except Exception:
        pass


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    if key in _MEM_CACHE:
        _MEM_CACHE.move_to_end(key)
        return _MEM_CACHE[key]
    disk = _load_disk_cache()
    row = disk.get(key)
    if isinstance(row, dict) and ("terms_en" in row or "terms_es" in row or "topic" in row):
        val = {k: row[k] for k in ("topic", "terms_es", "terms_en", "visuals") if k in row}
        _MEM_CACHE[key] = val
        return val
    if isinstance(row, dict) and "val" in row and isinstance(row["val"], dict):
        _MEM_CACHE[key] = row["val"]
        return row["val"]
    return None


def _cache_put(key: str, val: Dict[str, Any]) -> None:
    _MEM_CACHE[key] = val
    _MEM_CACHE.move_to_end(key)
    while len(_MEM_CACHE) > _CACHE_CAP:
        _MEM_CACHE.popitem(last=False)
    disk = _load_disk_cache()
    disk[key] = {"ts": time.time(), "val": val, **val}
    if len(disk) > _CACHE_CAP:
        ordered = sorted(
            disk.items(),
            key=lambda kv: float((kv[1] or {}).get("ts") or 0) if isinstance(kv[1], dict) else 0,
        )
        for old_k, _ in ordered[: max(0, len(disk) - _CACHE_CAP)]:
            disk.pop(old_k, None)
    _save_disk_cache(disk)


def llm_query_terms(query: str, fetch: bool = True) -> Optional[Dict[str, Any]]:
    """Optional Grok expansion. None on missing key, timeout, or bad JSON."""
    q = " ".join((query or "").split())
    if not q:
        return None
    key_id = fold(q)
    hit = _cache_get(key_id)
    if hit is not None:
        return hit
    if not fetch:
        return None
    api_key = _xai_key_for_search()
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1", timeout=8.0)
        resp = client.chat.completions.create(
            model=os.environ.get("XAI_MODEL", "grok-4.6"),
            temperature=0.2,
            max_tokens=200,
            timeout=8.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Footage search helper. Return JSON only: "
                        '{"topic":str,"terms_es":[str],"terms_en":[str],"visuals":[str]}. '
                        "Short visual search terms for b-roll. No markdown."
                    ),
                },
                {"role": "user", "content": q[:200]},
            ],
        )
        text = ((resp.choices or [None])[0].message.content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        raw = json.loads(text)
        if not isinstance(raw, dict):
            return None
        out = {
            "topic": str(raw.get("topic") or "")[:80],
            "terms_es": [str(x)[:40] for x in (raw.get("terms_es") or []) if str(x).strip()][:12],
            "terms_en": [str(x)[:40] for x in (raw.get("terms_en") or []) if str(x).strip()][:12],
            "visuals": [str(x)[:40] for x in (raw.get("visuals") or []) if str(x).strip()][:12],
        }
        _cache_put(key_id, out)
        return out
    except Exception:
        return None


def filter_and_rank(
    items: Sequence[Dict[str, Any]],
    query: str,
    max_results: int = 12,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Drop zero-token misses, rank the rest. Returns (kept, dropped, raw)."""
    raw_list = [it for it in items if isinstance(it, dict) and (it.get("url") or it.get("video_id"))]
    raw = len(raw_list)
    u = understand_query(query)
    llm = llm_query_terms(query, fetch=False)
    u = merge_llm_into_understood(u, llm)
    if not u.get("exact") and not u.get("terms"):
        return [], raw, raw
    scored: List[Tuple[float, Dict[str, Any]]] = []
    dropped = 0
    for it in dedupe_items(raw_list):
        if not _has_any_hit(it, u):
            dropped += 1
            continue
        s = relevance_score(it, u)
        if s < MIN_SCORE:
            dropped += 1
            continue
        subject_toks = [t for t in (u.get("exact") or []) if not _is_entity_tok(t, u)]
        blob = " " + fold(item_text(it)) + " "
        stems = _item_stems(it)
        exact_stem = 0
        for tok in subject_toks:
            if _exact_hit(blob, tok) or stem(tok) in stems:
                exact_stem += 1
        topic_hits = set()
        for name in (u.get("topics") or {}):
            topic_hits |= stems & (TOPIC_STEMS.get(name) or set())
        entity_hit = any(
            _exact_hit(blob, e) or stem(e) in stems for e in (u.get("entities") or [])
        )
        # Two+ subject tokens: need an exact/stem hit or two topic members
        # (EN "river pollution" for "río contaminación"). One member is not
        # enough — that is the fish-tank/"water" case.
        if len(subject_toks) >= 2 and exact_stem == 0 and len(topic_hits) < 2:
            dropped += 1
            continue
        # Place-heavy queries (subject tokens are all entities, or mixed with
        # places): a lone EN synonym like "river" in a Minecraft title must
        # not clear the floor without a subject or place hit.
        if (
            (u.get("entities") or [])
            and exact_stem == 0
            and not entity_hit
            and len(topic_hits) < 2
        ):
            dropped += 1
            continue
        scored.append((s, it))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    n = max(1, int(max_results) or 8)
    kept = [it for _s, it in scored[:n]]
    return kept, dropped, raw
