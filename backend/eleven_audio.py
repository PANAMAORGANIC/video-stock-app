"""
ElevenLabs: voiceover TTS, voice pick, and optional vocal isolation.
Used by Video Creation export and Director IA audio edits.
"""

from __future__ import annotations

import html
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from video_tools import ffmpeg_bin, _run

_BASE = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env_path = _BASE / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except Exception:
        pass

API = "https://api.elevenlabs.io/v1"
MODEL = "eleven_multilingual_v2"

# Style → search names in the user's ElevenLabs library (fallback to first voice).
STYLE_HINTS = {
    "documentary": ["daniel", "george", "adam", "chris", "brian"],
    "warm": ["sarah", "rachel", "bella", "charlotte", "alice"],
    "energetic": ["josh", "antoni", "sam", "will"],
    "ad": ["matilda", "elli", "domi", "freya"],
    "calm": ["rachel", "lily", "alice", "emma"],
}

_VOICE_CACHE: Optional[List[Dict[str, Any]]] = None


def eleven_key() -> Optional[str]:
    _load_dotenv()
    return os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVEN_API_KEY")


def _headers(key: str) -> dict:
    return {"xi-api-key": key, "Accept": "application/json", "Content-Type": "application/json"}


def list_voices() -> List[Dict[str, Any]]:
    global _VOICE_CACHE
    if _VOICE_CACHE is not None:
        return _VOICE_CACHE
    key = eleven_key()
    if not key:
        return []
    try:
        r = httpx.get(f"{API}/voices", headers={"xi-api-key": key}, timeout=30.0)
        r.raise_for_status()
        voices = r.json().get("voices") or []
        _VOICE_CACHE = [
            {"voice_id": v.get("voice_id"), "name": v.get("name") or ""}
            for v in voices
            if v.get("voice_id")
        ]
        return _VOICE_CACHE
    except Exception:
        return []


def pick_voice(style: str = "documentary") -> Optional[str]:
    voices = list_voices()
    if not voices:
        return None
    hints = STYLE_HINTS.get((style or "documentary").lower(), STYLE_HINTS["documentary"])
    for v in voices:
        name = (v.get("name") or "").lower()
        if any(h in name for h in hints):
            return v["voice_id"]
    return voices[0]["voice_id"]


def synthesize_voiceover(
    text: str,
    dest: Path,
    style: str = "documentary",
    voice_id: Optional[str] = None,
) -> Tuple[bool, str]:
    """TTS → mp3. Editar el texto y volver a generar es el flujo Studio de ElevenLabs."""
    key = eleven_key()
    if not key:
        return False, "Falta ELEVENLABS_API_KEY en .env"
    script = (text or "").strip()
    if not script:
        return False, "No hay narración para convertir en voz"
    vid = voice_id or pick_voice(style)
    if not vid:
        return False, "No hay voces en la cuenta ElevenLabs"
    stability = {"energetic": 0.35, "ad": 0.4, "calm": 0.7, "warm": 0.55}.get((style or "").lower(), 0.5)
    try:
        r = httpx.post(
            f"{API}/text-to-speech/{vid}",
            headers={
                "xi-api-key": key,
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
            },
            json={
                "text": script,
                "model_id": MODEL,
                "voice_settings": {
                    "stability": stability,
                    "similarity_boost": 0.75,
                },
            },
            timeout=90.0,
        )
        if r.status_code >= 400:
            return False, (r.text or f"ElevenLabs {r.status_code}")[:300]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        if dest.stat().st_size < 200:
            dest.unlink(missing_ok=True)
            return False, "ElevenLabs devolvió audio vacío"
        return True, str(dest)
    except Exception as e:
        return False, str(e)[:300]


def isolate_vocals(src_audio: Path, dest: Path) -> Tuple[bool, str]:
    """Voice Isolator: quita música/ambiente y deja la voz."""
    key = eleven_key()
    if not key:
        return False, "Falta ELEVENLABS_API_KEY"
    if not src_audio.exists():
        return False, "No hay audio para aislar"
    try:
        with src_audio.open("rb") as fh:
            r = httpx.post(
                f"{API}/audio-isolation",
                headers={"xi-api-key": key},
                files={"audio": (src_audio.name, fh, "application/octet-stream")},
                timeout=120.0,
            )
        if r.status_code >= 400:
            return False, (r.text or f"Isolation {r.status_code}")[:300]
        dest.write_bytes(r.content)
        return True, str(dest)
    except Exception as e:
        return False, str(e)[:300]


_PROFANITY = {
    "fuck", "shit", "bitch", "asshole", "dick", "pussy", "cunt",
    "mierda", "carajo", "puta", "puto", "cabron", "cabrón", "verga", "coño",
}


def apply_transcript_controls(
    text: str,
    punctuation: bool = True,
    title_case: bool = False,
    show_profanity: bool = False,
) -> str:
    out = (text or "").strip()
    if not show_profanity:
        words = []
        for w in out.split(" "):
            core = "".join(ch for ch in w.lower() if ch.isalpha())
            words.append("***" if core in _PROFANITY else w)
        out = " ".join(words)
    if not punctuation:
        out = "".join(ch for ch in out if ch.isalnum() or ch.isspace() or ch in "áéíóúüñÁÉÍÓÚÜÑ")
        out = " ".join(out.split())
    if title_case:
        out = out.title()
    return out


def transcribe_audio(
    src: Path,
    language: str = "es",
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """ElevenLabs Scribe STT with word timestamps."""
    key = eleven_key()
    if not key:
        return False, "Falta ELEVENLABS_API_KEY en .env", []
    if not src.exists() or src.stat().st_size < 200:
        return False, "Audio demasiado corto para transcribir", []
    lang = (language or "es").lower()
    if lang in ("es", "spa", "spanish"):
        lang = "es"
    elif lang in ("en", "eng", "english"):
        lang = "en"
    try:
        with src.open("rb") as fh:
            r = httpx.post(
                f"{API}/speech-to-text",
                headers={"xi-api-key": key},
                data={
                    "model_id": "scribe_v1",
                    "language_code": lang,
                    "timestamps_granularity": "word",
                    "tag_audio_events": "false",
                },
                files={"file": (src.name, fh, "application/octet-stream")},
                timeout=180.0,
            )
        if r.status_code >= 400:
            return False, (r.text or f"Scribe {r.status_code}")[:400], []
        data = r.json()
        words = []
        for w in data.get("words") or []:
            if (w.get("type") or "word") not in ("word",):
                continue
            txt = (w.get("text") or "").strip()
            if not txt:
                continue
            try:
                start = float(w.get("start") or 0)
                end = float(w.get("end") or (start + 0.3))
            except (TypeError, ValueError):
                continue
            words.append({"text": txt, "start": start, "end": max(end, start + 0.08)})
        if not words and (data.get("text") or "").strip():
            words.append({"text": data["text"].strip(), "start": 0.0, "end": 4.0})
        return True, (data.get("text") or "").strip(), words
    except Exception as e:
        return False, str(e)[:400], []


# 0:05 / 00:05 / 1:02:03 / 00:00:01,500 / 00:00:01.500
_CLOCK = r"(?:(?:\d{1,2}:)?\d{1,2}:\d{1,2}(?:[.,]\d{1,3})?)"
_CLOCK_RE = re.compile(
    r"^(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:[.,](?P<ms>\d{1,3}))?$"
)
_LINE_STAMP = re.compile(
    r"^(?P<junk>[\s\[\(\{\-\–\—\*>•·\"'«»]*(?:\d+[\.\)]\s+)?(?:[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9 .]{1,24}:\s+)?)?"
    r"(?P<a>" + _CLOCK + r")"
    r"(?:\s*(?:-->|->|–|—|-|to)\s*(?P<b>" + _CLOCK + r"))?"
    r"(?:\s*/\s*" + _CLOCK + r")?"
    r"\s*[\]\):\-–—]?\s*"
    r"(?P<rest>.*)$",
    re.I,
)
_INLINE_STAMP = re.compile(
    r"(?:(?<=^)|(?<=[\s\[\(]))(" + _CLOCK + r")"
    r"(?:\s*(?:-->|->|–|—|-)\s*(" + _CLOCK + r"))?"
    r"(?=[\s\]\):]|$)"
)
_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+")


def parse_clock(token: str) -> Optional[float]:
    raw = (token or "").strip().replace(",", ".")
    if not raw:
        return None
    m = _CLOCK_RE.match(raw)
    if not m:
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None
    h = int(m.group("h") or 0)
    mi = int(m.group("m") or 0)
    s = int(m.group("s") or 0)
    ms = m.group("ms") or "0"
    frac = int(ms.ljust(3, "0")[:3]) / 1000.0
    if m.group("h") is None and mi > 59:
        # "75:02" as mm:ss still ok; keep as minutes
        pass
    return h * 3600.0 + mi * 60.0 + s + frac


def pretty_clip_label(name: str) -> str:
    stem = Path(str(name or "")).stem
    stem = re.sub(r"[_\-]+", " ", stem)
    return re.sub(r"\s+", " ", stem).strip() or "Clip"


def looks_like_timestamped(text: str) -> bool:
    if not text:
        return False
    hits = len(re.findall(r"(?:^|[\s\[\(])" + _CLOCK, text, re.M))
    return hits >= 1 and (
        "-->" in text
        or hits >= 2
        or bool(re.search(r"^\s*[\[\(]?" + _CLOCK, text, re.M))
    )


def _split_instagram_window(
    text: str,
    start: float,
    end: float,
    max_words: int = 6,
) -> List[Dict[str, Any]]:
    words = [w for w in (text or "").split() if w]
    if not words:
        return []
    start = float(start)
    end = max(start + 0.4, float(end))
    chunks = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    slice_d = (end - start) / max(1, len(chunks))
    out = []
    for i, ch in enumerate(chunks):
        a = start + i * slice_d
        b = start + (i + 1) * slice_d if i < len(chunks) - 1 else end
        out.append({"start": round(a, 2), "end": round(b, 2), "text": ch})
    return out


def _clean_caption_src(raw: str) -> str:
    text = html.unescape(raw or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if text.upper().lstrip().startswith("WEBVTT"):
        text = re.sub(r"^\s*WEBVTT[^\n]*\n+", "", text, flags=re.I)
    return text.strip()


def _content_words(text: str) -> List[str]:
    stripped = re.sub(_CLOCK, " ", text or "")
    return _WORD_RE.findall(stripped.lower())


def _split_inline_stamps(blob: str, base_start: float, base_end: Optional[float]) -> List[Dict[str, Any]]:
    matches = list(_INLINE_STAMP.finditer(blob or ""))
    if len(matches) < 2:
        body = (blob or "").strip()
        return [{"start": base_start, "end": base_end, "text": body}] if body else []
    out: List[Dict[str, Any]] = []
    for idx, m in enumerate(matches):
        start = parse_clock(m.group(1))
        end = parse_clock(m.group(2)) if m.group(2) else None
        if start is None:
            continue
        if idx == 0 and start > base_start + 0.2:
            head = blob[: m.start()].strip(" \t[]()|-–—\n")
            if head:
                out.append({"start": base_start, "end": start, "text": head})
        a = m.end()
        b = matches[idx + 1].start() if idx + 1 < len(matches) else len(blob)
        body = blob[a:b].strip(" \t[]()|-–—\n")
        if not body:
            continue
        out.append({"start": start, "end": end, "text": body})
    if not out:
        body = (blob or "").strip()
        return [{"start": base_start, "end": base_end, "text": body}] if body else []
    if base_end is not None and out[-1]["end"] is None:
        out[-1]["end"] = base_end
    return out


def parse_timestamped_transcript(
    raw: str,
    punctuation: bool = True,
    title_case: bool = False,
    show_profanity: bool = False,
    duration: float = 0,
) -> List[Dict[str, Any]]:
    """SRT / VTT / YouTube / bullets / speaker labels. Never drops pasted sentences."""
    text = _clean_caption_src(raw)
    if not text:
        return []

    events: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    def flush() -> None:
        nonlocal cur
        if cur and (cur.get("text") or "").strip():
            events.append(cur)
        cur = None

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if (
            line.isdigit()
            or line.upper().startswith("WEBVTT")
            or line.upper().startswith("NOTE")
            or line.upper().startswith("STYLE")
            or line.upper().startswith("KIND:")
            or line.upper().startswith("LANGUAGE:")
        ):
            continue
        m = _LINE_STAMP.match(line)
        if m and parse_clock(m.group("a")) is not None:
            junk = (m.group("junk") or "").strip()
            junk_words = _WORD_RE.findall(junk)
            rest = (m.group("rest") or "").strip()
            if junk_words and len(junk_words) > 3:
                m = None
            else:
                start = parse_clock(m.group("a"))
                end = parse_clock(m.group("b")) if m.group("b") else None
                if len(junk_words) == 1 and junk_words[0].isdigit():
                    extra = rest
                elif junk_words:
                    extra = " ".join(junk_words + ([rest] if rest else [])).strip()
                else:
                    extra = rest
                extra = extra.strip(" []()|-–—")
                flush()
                cur = {"start": start, "end": end, "text": extra}
                continue
        if cur is None:
            cur = {"start": 0.0, "end": None, "text": line}
        else:
            cur["text"] = (cur["text"] + " " + line).strip() if cur.get("text") else line
    flush()

    if not events:
        inline = _split_inline_stamps(text, 0.0, None)
        events = inline or [{"start": 0.0, "end": None, "text": re.sub(r"\s+", " ", text).strip()}]

    exploded: List[Dict[str, Any]] = []
    for ev in events:
        pieces = _split_inline_stamps(ev["text"], float(ev["start"]), ev.get("end"))
        exploded.extend(pieces or [ev])
    events = exploded

    def meaningful(words: List[str]) -> List[str]:
        return [w for w in words if not w.isdigit()]

    missing = Counter(meaningful(_content_words(text))) - Counter(
        meaningful(_content_words(" ".join(ev["text"] for ev in events)))
    )
    leftover = " ".join(missing.elements())
    if leftover:
        if events:
            events[-1]["text"] = (events[-1]["text"] + " " + leftover).strip()
        else:
            events.append({"start": 0.0, "end": None, "text": leftover})

    dur = float(duration or 0)
    for idx, ev in enumerate(events):
        if ev["end"] is None:
            if idx + 1 < len(events):
                ev["end"] = events[idx + 1]["start"]
            else:
                nwords = max(1, len((ev["text"] or "").split()))
                guess = ev["start"] + max(1.4, min(12.0, nwords / 2.2))
                ev["end"] = max(guess, dur) if dur > ev["start"] else guess
        if dur and idx == len(events) - 1:
            ev["end"] = max(float(ev["end"]), dur)
        if ev["end"] <= ev["start"]:
            ev["end"] = ev["start"] + 1.2

    cues: List[Dict[str, Any]] = []
    for ev in events:
        body = apply_transcript_controls(ev["text"], punctuation, title_case, show_profanity)
        if not body:
            continue
        cues.extend(_split_instagram_window(body, ev["start"], ev["end"], max_words=14))
    return cues


def cues_from_untimed_script(
    raw: str,
    duration: float,
    punctuation: bool = True,
    title_case: bool = False,
    show_profanity: bool = False,
    max_words: int = 6,
) -> List[Dict[str, Any]]:
    cleaned = re.sub(
        r"\[pausa(?:\s+(\d+(?:\.\d+)?)\s*s?)?\]",
        " ",
        str(raw or ""),
        flags=re.IGNORECASE,
    )
    text = apply_transcript_controls(cleaned, punctuation, title_case, show_profanity)
    if not text.strip():
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+|\n+", text) if p.strip()]
    if not parts:
        parts = [text.strip()]
    chunks: List[str] = []
    for p in parts:
        words = p.split()
        if not words:
            continue
        for i in range(0, len(words), max_words):
            chunks.append(" ".join(words[i:i + max_words]))
    if not chunks:
        return []
    dur = max(1.0, float(duration or len(chunks) * 2.0))
    slice_d = dur / len(chunks)
    return [
        {
            "start": round(i * slice_d, 2),
            "end": round(dur if i == len(chunks) - 1 else (i + 1) * slice_d, 2),
            "text": ch,
        }
        for i, ch in enumerate(chunks)
    ]


def cues_from_clip_scripts(
    clips: List[Any],
    punctuation: bool = True,
    title_case: bool = False,
    show_profanity: bool = False,
    max_words: int = 6,
) -> List[Dict[str, Any]]:
    """Captions from real narration/overlay only. Never uses the clip filename."""
    def get(obj: Any, key: str, default: Any = "") -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    blob_parts: List[str] = []
    for c in clips or []:
        raw = (get(c, "narration") or "").strip() or (get(c, "text") or "").strip()
        if raw:
            blob_parts.append(raw)
    blob = "\n".join(blob_parts).strip()
    if not blob:
        return []
    if looks_like_timestamped(blob):
        return parse_timestamped_transcript(blob, punctuation, title_case, show_profanity)

    cues: List[Dict[str, Any]] = []
    for c in clips or []:
        start = float(get(c, "timelineStart", 0) or 0)
        in_p = float(get(c, "inPoint", 0) or 0)
        out_p = float(get(c, "outPoint", in_p + 4) or (in_p + 4))
        dur = max(0.8, out_p - in_p)
        raw = (get(c, "narration") or "").strip() or (get(c, "text") or "").strip()
        if not raw:
            continue
        raw = apply_transcript_controls(raw, punctuation, title_case, show_profanity)
        cues.extend(_split_instagram_window(raw, start, start + dur, max_words))
    return cues


def format_cues_transcript(cues: List[Dict[str, Any]]) -> str:
    lines = []
    for c in cues or []:
        t = float(c.get("start") or 0)
        m = int(t // 60)
        s = t - m * 60
        if abs(s - round(s)) < 0.05:
            stamp = f"{m}:{int(round(s)):02d}"
        else:
            stamp = f"{m}:{s:04.1f}"
        lines.append(f"{stamp} {c.get('text') or ''}".rstrip())
    return "\n".join(lines)


def words_to_cues(
    words: List[Dict[str, Any]],
    max_chars: int = 32,
    max_words: int = 6,
) -> List[Dict[str, Any]]:
    """Group words into Instagram-style caption lines (~6 words / line).

    A gap of ~0.75s (a [pausa] hole) starts a new cue so CC goes blank.
    """
    cues: List[Dict[str, Any]] = []
    buf: List[str] = []
    start = None
    end = 0.0
    limit_words = max(1, int(max_words or 6))
    for w in words:
        t = (w.get("text") or "").strip()
        if not t:
            continue
        wst = float(w["start"])
        if start is None:
            start = wst
        elif buf and (wst - end) > 0.75:
            cues.append({"start": start, "end": end, "text": " ".join(buf)})
            buf = [t]
            start = wst
            end = float(w["end"])
            continue
        nxt = " ".join(buf + [t])
        if buf and (len(nxt) > max_chars or len(buf) >= limit_words):
            cues.append({"start": start, "end": end, "text": " ".join(buf)})
            buf = [t]
            start = wst
        else:
            buf.append(t)
        end = float(w["end"])
    if buf and start is not None:
        cues.append({"start": start, "end": end, "text": " ".join(buf)})
    return cues


def extract_audio(video: Path, dest: Path) -> Tuple[bool, str]:
    ff = ffmpeg_bin()
    if not ff:
        return False, "FFmpeg no disponible"
    cmd = [
        ff, "-y", "-i", str(video),
        "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
        str(dest),
    ]
    r = _run(cmd)
    if r.returncode != 0 or not dest.exists():
        return False, (r.stderr or "extract audio failed")[-300:]
    return True, str(dest)
