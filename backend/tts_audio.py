"""Clearview TTS dispatcher: Edge neural by default, ElevenLabs opt-in.

No account for Edge. ElevenLabs only when ELEVENLABS_API_KEY is set and
voiceEngine is "eleven".
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eleven_audio import eleven_key, synthesize_voiceover as synthesize_elevenlabs

EDGE_VOICES: List[Dict[str, str]] = [
    {"id": "es-PA-MargaritaNeural", "label": "Margarita (Panamá)"},
    {"id": "es-PA-RobertoNeural", "label": "Roberto (Panamá)"},
    {"id": "es-MX-DaliaNeural", "label": "Dalia (México)"},
    {"id": "es-MX-JorgeNeural", "label": "Jorge (México)"},
    {"id": "es-US-AlonsoNeural", "label": "Alonso (US)"},
    {"id": "es-ES-AlvaroNeural", "label": "Álvaro (España)"},
    {"id": "en-US-AndrewNeural", "label": "Andrew (EN)"},
    {"id": "en-US-AriaNeural", "label": "Aria (EN)"},
]
EDGE_VOICE_IDS = {v["id"] for v in EDGE_VOICES}

# documentary → Roberto slower; warm → Margarita; energetic → Jorge faster;
# ad → Dalia; calm → softer/slower (Margarita).
STYLE_EDGE = {
    "documentary": {"voice": "es-PA-RobertoNeural", "rate": "-12%", "volume": "+0%"},
    "warm": {"voice": "es-PA-MargaritaNeural", "rate": "-4%", "volume": "+0%"},
    "energetic": {"voice": "es-MX-JorgeNeural", "rate": "+14%", "volume": "+4%"},
    "ad": {"voice": "es-MX-DaliaNeural", "rate": "+6%", "volume": "+2%"},
    "calm": {"voice": "es-PA-MargaritaNeural", "rate": "-16%", "volume": "-6%"},
}
VOICE_ENGINES = {"edge", "eleven"}
VO_FX = {"none", "booth", "radio", "telephone"}
TTS_CHUNK_MAX = 3000
SCRIPT_CAP = 50000
PAUSA_RE = re.compile(r"\[pausa(?:\s+(\d+(?:\.\d+)?)\s*s?)?\]", re.IGNORECASE)
PAUSA_DEFAULT_SEC = 2.0


def edge_style_opts(style: str = "documentary") -> Dict[str, str]:
    key = (style or "documentary").strip().lower()
    return dict(STYLE_EDGE.get(key) or STYLE_EDGE["documentary"])


def pick_edge_voice(style: str = "documentary", voice_name: Optional[str] = None) -> str:
    name = (voice_name or "").strip()
    if name in EDGE_VOICE_IDS:
        return name
    return edge_style_opts(style)["voice"]


def sanitize_engine(engine: str = "") -> str:
    e = (engine or "").strip().lower()
    if e in {"elevenlabs", "11labs", "11"}:
        e = "eleven"
    if e not in VOICE_ENGINES:
        return "edge"
    return e


def resolve_engine(engine: str = "") -> str:
    """Eleven only if a key exists and the scene asked for it. Else Edge."""
    e = sanitize_engine(engine)
    if e == "eleven" and eleven_key():
        return "eleven"
    return "edge"


def sanitize_voice_name(name: Any) -> str:
    n = str(name or "").strip()
    return n if n in EDGE_VOICE_IDS else ""


def sanitize_vo_fx(kind: Any) -> str:
    k = str(kind or "none").strip().lower()
    return k if k in VO_FX else "none"


def sanitize_vo_rate(value: Any) -> float:
    try:
        r = float(value)
    except (TypeError, ValueError):
        return 1.0
    if r != r:  # NaN
        return 1.0
    return max(0.5, min(1.5, r))


def sanitize_script(value: Any) -> str:
    t = str(value or "").strip()
    if len(t) > SCRIPT_CAP:
        t = t[:SCRIPT_CAP]
    return t


def parse_pausa_seconds(raw: Any) -> float:
    if raw is None or str(raw).strip() == "":
        return PAUSA_DEFAULT_SEC
    try:
        n = float(raw)
    except (TypeError, ValueError):
        return PAUSA_DEFAULT_SEC
    if n != n or n < 0:
        return PAUSA_DEFAULT_SEC
    return max(0.05, min(60.0, n))


def strip_pausa_markers(text: str) -> str:
    out = PAUSA_RE.sub(" ", str(text or ""))
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def split_pausa_parts(text: str) -> List[Tuple[str, float]]:
    """[(spoken_text, silence_after_seconds), ...]. Last silence is 0. Marker is not spoken."""
    raw = str(text or "")
    parts: List[Tuple[str, float]] = []
    last = 0
    for m in PAUSA_RE.finditer(raw):
        parts.append((raw[last:m.start()], parse_pausa_seconds(m.group(1))))
        last = m.end()
    tail = raw[last:]
    if tail or not parts:
        parts.append((tail, 0.0))
    return parts


def split_tts_chunks(text: str, max_chars: int = TTS_CHUNK_MAX) -> List[str]:
    """Split long VO text on paragraphs/sentences under max_chars. No silent truncate."""
    raw = (text or "").strip()
    if not raw:
        return []
    max_chars = max(200, int(max_chars or TTS_CHUNK_MAX))
    if len(raw) <= max_chars:
        return [raw]
    paras = re.split(r"\n\s*\n+", raw)
    units: List[str] = []
    for p in paras:
        p = (p or "").strip()
        if not p:
            continue
        if len(p) <= max_chars:
            units.append(p)
            continue
        parts = re.split(r"(?<=[\.!?…])\s+", p)
        buf = ""
        for sent in parts:
            sent = (sent or "").strip()
            if not sent:
                continue
            if len(sent) > max_chars:
                piece = ""
                for w in sent.split():
                    cand = (piece + " " + w).strip() if piece else w
                    if len(cand) > max_chars and piece:
                        units.append(piece)
                        piece = w
                    else:
                        piece = cand
                if piece:
                    if buf:
                        units.append(buf)
                        buf = ""
                    units.append(piece)
                continue
            cand = (buf + " " + sent).strip() if buf else sent
            if len(cand) > max_chars and buf:
                units.append(buf)
                buf = sent
            else:
                buf = cand
        if buf:
            units.append(buf)
    chunks: List[str] = []
    buf = ""
    for u in units:
        cand = (buf + "\n\n" + u).strip() if buf else u
        if len(cand) > max_chars and buf:
            chunks.append(buf)
            buf = u
        else:
            buf = cand
    if buf:
        chunks.append(buf)
    return chunks or [raw[:max_chars]]


def split_script_blocks(text: str) -> List[str]:
    t = (text or "").replace("\r\n", "\n").strip()
    if not t:
        return []
    numbered = re.split(r"\n(?=\s*\d+[.)]\s+)", t)
    numbered = [re.sub(r"^\s*\d+[.)]\s+", "", b).strip() for b in numbered]
    numbered = [b for b in numbered if b]
    if len(numbered) > 1:
        return numbered
    paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    return paras or [t]


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.!?…])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def split_script_to_scenes(text: str, durations: List[float]) -> List[str]:
    """1:1 if blank-line/numbered blocks match scene count; else duration-weighted sentences."""
    n = len(durations or [])
    if n < 1:
        return []
    blocks = split_script_blocks(text)
    if len(blocks) == n:
        return blocks
    sentences = _split_sentences(text)
    if not sentences:
        return [""] * n
    weights = [max(1, len(s.split())) for s in sentences]
    total_w = sum(weights) or 1
    total_d = sum(max(0.2, float(d or 0)) for d in durations) or 1.0
    out: List[str] = []
    si = 0
    for i, d in enumerate(durations):
        if i == n - 1:
            out.append(" ".join(sentences[si:]).strip())
            break
        target = max(1, int(round(total_w * (max(0.2, float(d or 0)) / total_d))))
        acc = 0
        chunk: List[str] = []
        while si < len(sentences) and (acc < target or not chunk):
            chunk.append(sentences[si])
            acc += weights[si]
            si += 1
            if acc >= target:
                break
        out.append(" ".join(chunk).strip())
    while len(out) < n:
        out.append("")
    return out[:n]


def edge_rate_percent(vo_rate: Any = 1.0) -> str:
    """Map 1.0 → +0%, 1.5 → +50%, 0.5 → -50%. Signed percent for edge-tts."""
    r = sanitize_vo_rate(vo_rate)
    pct = int(round((r - 1.0) * 100))
    pct = max(-50, min(100, pct))
    if pct == 0:
        return "+0%"
    return f"{pct:+d}%"


def _edge_percent_to_mult(value: str) -> float:
    raw = str(value or "+0%").strip().replace("%", "")
    try:
        return 1.0 + float(raw) / 100.0
    except (TypeError, ValueError):
        return 1.0


def combined_edge_rate(style: str = "documentary", vo_rate: Any = 1.0) -> str:
    """STYLE_EDGE pace × the voRate slider. Explicit voices keep the style's rate."""
    base = _edge_percent_to_mult(edge_style_opts(style).get("rate") or "+0%")
    return edge_rate_percent(base * sanitize_vo_rate(vo_rate))


def _run_coro(coro):
    asyncio.run(coro)


def _ticks_to_seconds(value: Any) -> float:
    try:
        n = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    if n != n or n < 0:
        return 0.0
    return n / 10_000_000.0


def word_from_boundary(chunk: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(chunk, dict):
        return None
    if str(chunk.get("type") or "") != "WordBoundary":
        return None
    text = str(chunk.get("text") or "").strip()
    if not text:
        return None
    start = _ticks_to_seconds(chunk.get("offset"))
    dur = _ticks_to_seconds(chunk.get("duration"))
    end = start + max(0.04, dur)
    return {"text": text, "start": round(start, 3), "end": round(end, 3)}


def shift_words(words: Optional[List[Dict[str, Any]]], offset: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    off = float(offset or 0)
    for w in words or []:
        if not isinstance(w, dict):
            continue
        text = str(w.get("text") or "").strip()
        if not text:
            continue
        try:
            a = float(w.get("start") or 0)
            b = float(w.get("end") or 0)
        except (TypeError, ValueError):
            continue
        out.append({
            "text": text,
            "start": round(a + off, 3),
            "end": round(max(a, b) + off, 3),
        })
    return out


def _as_synth_result(result: Any) -> Tuple[bool, str, List[Dict[str, Any]]]:
    if not isinstance(result, tuple) or len(result) < 2:
        return False, "TTS error", []
    words: List[Dict[str, Any]] = []
    if len(result) >= 3 and result[2]:
        words = [w for w in result[2] if isinstance(w, dict)]
    return bool(result[0]), str(result[1]), words


def _audio_duration_seconds(path: Path) -> float:
    try:
        from video_tools import probe_duration
        d = probe_duration(Path(path))
        if d and float(d) > 0:
            return float(d)
    except Exception:
        pass
    return 0.0


def _edge_synthesize(
    text: str,
    dest: Path,
    voice: str,
    rate: str = "+0%",
    volume: str = "+0%",
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """Stream WordBoundary while writing mp3. Tests mock Communicate.stream."""
    try:
        import edge_tts
    except ImportError:
        return False, "Falta el paquete edge-tts. En PowerShell: pip install edge-tts", []

    words: List[Dict[str, Any]] = []

    async def _go() -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"rate": rate, "volume": volume}
        try:
            comm = edge_tts.Communicate(text, voice, boundary="WordBoundary", **kwargs)
        except TypeError:
            comm = edge_tts.Communicate(text, voice, **kwargs)
        audio = bytearray()
        async for chunk in comm.stream():
            if not isinstance(chunk, dict):
                continue
            kind = chunk.get("type")
            if kind == "audio":
                data = chunk.get("data") or b""
                if data:
                    audio.extend(data)
            else:
                w = word_from_boundary(chunk)
                if w:
                    words.append(w)
        dest.write_bytes(bytes(audio))

    try:
        _run_coro(_go())
    except Exception as e:
        return False, str(e)[:300], []
    if not dest.exists() or dest.stat().st_size < 200:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        return False, "Edge TTS devolvió audio vacío", []
    return True, str(dest), words


def _atempo_vo_file(dest: Path, vo_rate: float) -> Tuple[bool, str]:
    """Stretch an already-generated VO (ElevenLabs has no native rate)."""
    rate = sanitize_vo_rate(vo_rate)
    if abs(rate - 1.0) < 0.01:
        return True, str(dest)
    from video_tools import ffmpeg_bin, _run, _atempo_chain
    ff = ffmpeg_bin()
    if not ff or not dest.exists():
        return True, str(dest)
    tmp = dest.with_name(dest.stem + ".__rate__" + dest.suffix)
    cmd = [
        ff, "-y", "-i", str(dest),
        "-filter:a", _atempo_chain(rate),
        "-vn", str(tmp),
    ]
    result = _run(cmd)
    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 200:
        tmp.unlink(missing_ok=True)
        return True, str(dest)
    try:
        dest.unlink(missing_ok=True)
        tmp.rename(dest)
    except Exception:
        from shutil import move
        move(str(tmp), str(dest))
    return True, str(dest)


def _probe_audio_format(path: Path) -> Tuple[int, str]:
    """Sample rate and channel layout of a spoken TTS file."""
    try:
        from video_tools import get_video_info
        info = get_video_info(Path(path))
        for st in info.get("streams") or []:
            if st.get("codec_type") != "audio":
                continue
            try:
                rate = int(float(st.get("sample_rate") or 0))
            except (TypeError, ValueError):
                rate = 0
            layout = str(st.get("channel_layout") or "mono")
            if rate > 0:
                return rate, layout if layout in {"mono", "stereo"} else "mono"
    except Exception:
        pass
    return 24000, "mono"


def _make_silence_mp3(
    dest: Path,
    seconds: float,
    sample_rate: int = 24000,
    channel_layout: str = "mono",
) -> Tuple[bool, str]:
    from video_tools import ffmpeg_bin, _run
    ff = ffmpeg_bin()
    if not ff:
        return False, "FFmpeg no disponible"
    sec = parse_pausa_seconds(seconds)
    try:
        rate = int(sample_rate or 24000)
    except (TypeError, ValueError):
        rate = 24000
    if rate < 8000 or rate > 192000:
        rate = 24000
    layout = channel_layout if channel_layout in {"mono", "stereo"} else "mono"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ff, "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout={layout}:sample_rate={rate}",
        "-t", f"{sec:.3f}",
        "-c:a", "libmp3lame", "-q:a", "4",
        str(dest),
    ]
    result = _run(cmd)
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size < 80:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        return False, ((result.stderr or "No se pudo crear silencio")[-300:])
    return True, str(dest)


def _concat_audio_files(parts: List[Path], dest: Path) -> Tuple[bool, str]:
    from video_tools import ffmpeg_bin, _run
    ff = ffmpeg_bin()
    if not ff:
        return False, "FFmpeg no disponible"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lst = dest.with_name(dest.stem + ".__concat.txt")
    lines = []
    for p in parts:
        posix = Path(p).resolve().as_posix().replace("'", r"'\''")
        lines.append(f"file '{posix}'")
    lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [ff, "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(dest)]
    result = _run(cmd)
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size < 200:
        cmd = [
            ff, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
            "-c:a", "libmp3lame", "-q:a", "4", str(dest),
        ]
        result = _run(cmd)
    try:
        lst.unlink(missing_ok=True)
    except Exception:
        pass
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size < 200:
        return False, ((result.stderr or "No se pudo unir el audio")[-300:])
    return True, str(dest)


def _synth_chunks_then_concat(
    chunks: List[str],
    dest: Path,
    one_fn,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    dest = Path(dest)
    all_words: List[Dict[str, Any]] = []
    offset = 0.0
    if len(chunks) == 1:
        return _as_synth_result(one_fn(chunks[0], dest))
    parts: List[Path] = []
    try:
        for i, ch in enumerate(chunks):
            p = dest.with_name(f"{dest.stem}.__c{i:02d}{dest.suffix or '.mp3'}")
            ok, msg, words = _as_synth_result(one_fn(ch, p))
            if not ok:
                return False, msg, []
            parts.append(p)
            all_words.extend(shift_words(words, offset))
            dur = _audio_duration_seconds(p)
            if dur <= 0 and words:
                try:
                    dur = max(float(w.get("end") or 0) for w in words)
                except (TypeError, ValueError):
                    dur = 0.0
            offset += max(0.0, dur)
        ok, msg = _concat_audio_files(parts, dest)
        if not ok:
            return False, msg, []
        return True, str(dest), all_words
    finally:
        for p in parts:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


def _synth_script_then_concat(
    script: str,
    dest: Path,
    one_fn,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """TTS each side of [pausa] markers; insert silence; shift later word times."""
    dest = Path(dest)
    segments = split_pausa_parts(script)
    spoken_any = any((t or "").strip() for t, _ in segments)
    if not spoken_any:
        return False, "No hay narración para convertir en voz", []
    if len(segments) == 1 and segments[0][1] <= 0:
        return _synth_chunks_then_concat(split_tts_chunks(segments[0][0]), dest, one_fn)

    timeline: List[Tuple[Any, ...]] = []
    parts: List[Path] = []
    try:
        for spoken, pause in segments:
            spoken = (spoken or "").strip()
            if spoken:
                for ch in split_tts_chunks(spoken):
                    p = dest.with_name(f"{dest.stem}.__t{len(parts):02d}{dest.suffix or '.mp3'}")
                    ok, msg, words = _as_synth_result(one_fn(ch, p))
                    if not ok:
                        return False, msg, []
                    parts.append(p)
                    timeline.append(("speech", p, words))
            if pause > 0.04:
                timeline.append(("pause", float(pause)))
        if not any(item[0] == "speech" for item in timeline):
            return False, "No hay narración para convertir en voz", []
        rate, layout = 24000, "mono"
        for item in timeline:
            if item[0] == "speech":
                rate, layout = _probe_audio_format(item[1])
                break
        concat_parts: List[Path] = []
        all_words: List[Dict[str, Any]] = []
        offset = 0.0
        pause_i = 0
        for item in timeline:
            if item[0] == "pause":
                sp = dest.with_name(f"{dest.stem}.__p{pause_i:02d}.mp3")
                pause_i += 1
                ok, msg = _make_silence_mp3(
                    sp, item[1], sample_rate=rate, channel_layout=layout
                )
                if not ok:
                    return False, msg, []
                parts.append(sp)
                concat_parts.append(sp)
                offset += float(item[1])
                continue
            p = item[1]
            words = item[2]
            concat_parts.append(p)
            all_words.extend(shift_words(words, offset))
            dur = _audio_duration_seconds(p)
            if dur <= 0 and words:
                try:
                    dur = max(float(w.get("end") or 0) for w in words)
                except (TypeError, ValueError):
                    dur = 0.0
            offset += max(0.0, dur)
        if not concat_parts:
            return False, "No hay narración para convertir en voz", []
        ok, msg = _concat_audio_files(concat_parts, dest)
        if not ok:
            return False, msg, []
        return True, str(dest), all_words
    finally:
        for p in parts:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


def synthesize_edge(
    text: str,
    dest: Path,
    style: str = "documentary",
    voice_name: Optional[str] = None,
    vo_rate: Any = 1.0,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    script = (text or "").strip()
    if not script:
        return False, "No hay narración para convertir en voz", []
    opts = edge_style_opts(style)
    voice = pick_edge_voice(style, voice_name)
    rate = combined_edge_rate(style, vo_rate)

    def one(chunk: str, path: Path):
        return _edge_synthesize(chunk, path, voice, rate, opts["volume"])

    return _synth_script_then_concat(script, Path(dest), one)


def synthesize_voiceover(
    text: str,
    dest: Path,
    style: str = "documentary",
    voice_id: Optional[str] = None,
    voice_name: Optional[str] = None,
    engine: str = "",
    vo_rate: Any = 1.0,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """Dispatcher: Edge when no ElevenLabs key; ElevenLabs only as opt-in.

    Returns (ok, path_or_error, words) with times on the concatenated VO.
    Edge WordBoundary already includes voRate (passed as Communicate rate).
    """
    script = (text or "").strip()
    if not script:
        return False, "No hay narración para convertir en voz", []
    eng = resolve_engine(engine)
    dest = Path(dest)
    if eng == "eleven":
        rate = sanitize_vo_rate(vo_rate)

        def one(chunk: str, path: Path):
            ok, msg, words = _as_synth_result(
                synthesize_elevenlabs(chunk, path, style, voice_id)
            )
            if not ok:
                return ok, msg, words
            ok, msg = _atempo_vo_file(path, rate)
            if abs(rate - 1.0) > 0.01 and words:
                words = [
                    {
                        "text": w["text"],
                        "start": round(float(w["start"]) / rate, 3),
                        "end": round(float(w["end"]) / rate, 3),
                    }
                    for w in words
                ]
            return ok, msg, words if ok else []

        return _synth_script_then_concat(script, dest, one)
    return synthesize_edge(script, dest, style, voice_name, vo_rate)


def tts_status() -> Dict[str, Any]:
    has_eleven = bool(eleven_key())
    return {
        "ok": True,
        "engine": "edge",
        "elevenlabs": has_eleven,
        "voices": list(EDGE_VOICES),
    }
