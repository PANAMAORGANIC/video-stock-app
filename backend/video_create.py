"""
Clearview Assemble: the director writes a scene plan and fits Library clips.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from video_tools import CLIPS_DIR, UPLOADS_DIR, TEMP_DIR, probe_duration, safe_filename, ffmpeg_bin
from tts_audio import (
    sanitize_engine,
    sanitize_voice_name,
    sanitize_vo_fx,
    sanitize_vo_rate,
    sanitize_script,
)

CAP_POS = {"lower-third", "top", "bottom", "center"}


def _sanitize_cap_pos(value) -> str:
    p = str(value or "").strip().lower().replace("_", "-")
    if p in {"igual", "inherit", "global", "default"}:
        return ""
    return p if p in CAP_POS else ""


def _sanitize_cap_off(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _scene_span(s) -> Tuple[float, float]:
    start = _safe_float(s.get("timelineStart"), 0.0, 0.0, 86400.0)
    dur = _safe_float(s.get("duration"), 0.0, 0.0, 86400.0)
    if dur <= 0:
        dur = max(
            0.2,
            _safe_float(s.get("outPoint"), 0.0, 0.0, 86400.0)
            - _safe_float(s.get("inPoint"), 0.0, 0.0, 86400.0),
        )
    return start, start + max(0.05, dur)


def scene_at_time(scenes, t: float):
    t = float(t or 0)
    last = None
    for s in scenes or []:
        if not isinstance(s, dict):
            continue
        start, end = _scene_span(s)
        last = s
        if start <= t < end:
            return s
    return last


def scene_covering(scenes, t: float):
    """Scene that contains t. No last-scene fallback — None if t is in a gap."""
    t = float(t or 0)
    for s in scenes or []:
        if not isinstance(s, dict):
            continue
        start, end = _scene_span(s)
        if start <= t < end:
            return s
    return None


def _capoff_windows(scenes) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for s in scenes or []:
        if not isinstance(s, dict):
            continue
        if not _sanitize_cap_off(s.get("capOff")):
            continue
        a, b = _scene_span(s)
        if b > a:
            out.append((a, b))
    return out


def _subtract_intervals(
    start: float, end: float, holes: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    segs = [(start, end)]
    for h0, h1 in holes:
        nxt: List[Tuple[float, float]] = []
        for a, b in segs:
            if h1 <= a or h0 >= b:
                nxt.append((a, b))
                continue
            if a < h0:
                nxt.append((a, min(b, h0)))
            if b > h1:
                nxt.append((max(a, h1), b))
        segs = [(a, b) for a, b in nxt if b - a > 0.05]
    return segs


def caption_export_texts(cues, scenes, default_pos: str = "lower-third"):
    """Drop or clip cues that overlap capOff scenes; use scene capPos when set."""
    pos0 = _sanitize_cap_pos(default_pos) or "lower-third"
    holes = _capoff_windows(scenes)
    out = []
    for c in cues or []:
        if not isinstance(c, dict):
            continue
        start = _safe_float(c.get("start"), 0.0, 0.0, 86400.0)
        end = _safe_float(c.get("end"), start + 0.4, 0.0, 86400.0)
        if end <= start:
            continue
        text = str(c.get("text") or c.get("content") or "").strip()
        if not text:
            continue
        for a, b in _subtract_intervals(start, end, holes):
            if any(h0 <= a < h1 for h0, h1 in holes):
                continue
            scene = scene_covering(scenes, a)
            if scene and _sanitize_cap_off(scene.get("capOff")):
                continue
            pos = _sanitize_cap_pos(scene.get("capPos") if scene else "") or pos0
            out.append({
                "content": text,
                "start": round(a, 3),
                "end": round(b, 3),
                "position": pos,
            })
    return out

BASE_DIR = Path(__file__).resolve().parent.parent
FORMATS = {
    "documentary": "documental con narración clara, tono informativo y ritmo pausado",
    "movie": "cortometraje con arco (inicio, conflicto, cierre), tono cinematográfico",
    "explainer": "vídeo explicativo, pasos claros, textos en pantalla",
    "ad": "anuncio corto, gancho al inicio y cierre con mensaje",
    "youtube": "vídeo para YouTube, ganchos, secciones y cierre",
}

# Director playbook: Clearview Cut + Color, InVideo Magic Box, Seedance.
DIRECTOR_PLAYBOOK = """
Eres Grok, director-productor de Clearview (Cut + Color + InVideo Magic Box + Seedance).
Hablas como un colaborador inteligente: entiendes, asimilas, agrupas ideas, aclaras
y preguntas cuando falta información. No eres un ejecutor ciego de regex.
Este programa se llama Clearview. NO eres Adobe Premiere ni DaVinci Resolve:
usas las herramientas de Clearview y las traduces si el usuario habla con esos nombres.

CONVERSACIÓN
- Si el pedido es vago, pregunta 1 o 2 cosas concretas (tono, duración, audiencia, plataforma).
- Si ya puedes actuar, actúa y explica qué hiciste.
- Si falta footage, NO inventes archivos: pide búsqueda de footage web (search_stock).
- Puedes opinar sobre ritmo, continuidad, ganchos y CTA como un director real.
- reply: lenguaje natural, en el idioma del usuario, hasta ~8 frases. No seas telegráfico.
- Vocabulario: “Clips” = librería del usuario. “Footage web” / “web footage” = vídeo bajado de YouTube o Dailymotion. “Imágenes web” = fotos de Wikimedia/Openverse convertidas a stills. No llames stock a lo de la web.

FOOTAGE WEB (YouTube / Dailymotion, no la librería del usuario)
- El usuario puede pedir “busca X”, “footage web de ríos”, “necesito un close-up de humo”, “from the web”.
- Devuelve search_stock: {query, platform: youtube|dailymotion|all}.
- query en el idioma que mejor encuentre footage (suele ir bien en inglés + español).
- No descargues tú. El usuario elige de los resultados y se guardan en Clips.
- Si la librería está vacía, ofrece buscar footage web antes de armar escenas.

IMÁGENES WEB
- “busca imágenes de X”, “fotos de ríos”, “stills of smoke”, “image search”.
- Devuelve search_images: {query}. No inventes archivos. El usuario elige y se guardan como still MP4 (img_*) en Clips.

INVIDEO (Magic Box / script breakdown)
- Edición quirúrgica: cambia SOLO lo pedido. El resto del plan se conserva.
- Comandos típicos: “extiende escena 3”, “cambia el texto de la 2”, “borra la escena 1”,
  “haz el intro más enérgico”, “cierre con llamada a la acción”, “cambia el clip de la 4”,
  “pon lower-third”, “reordena: gancho, demo, cierre”.
- Desglosa el guion en beats (hook → desarrollo → giro → cierre), no en párrafos.
- Empareja cada beat con el clip de librería más semánticamente cercano (título/archivo).
- Textos en pantalla: escribe las palabras EXACTAS, cortas (máx ~8 palabras).
- Si un pedido es ambiguo, pregunta UNA cosa en reply y no inventes.
- En reply di qué cambió, escena por escena, como el Magic Box.

SEEDANCE (de “generar clips” a “dirigir”)
- Cada escena = UN plano con evento principal y estado final visible.
- visual describe cámara + acción, no poesía: wide establishing / medium / close-up /
  push-in / pan / tracking / hold. Un movimiento de cámara por escena.
- Arco multi-shot: escena 1 establece, las del medio desarrollan, la última resuelve o CTA.
- Continuidad: mismo sujeto, luz y estilo de un plano al siguiente (los clips son referencias bloqueadas).
- No recortes un clip a 3s por defecto: inPoint=0, outPoint=duration_sec del archivo,
  salvo que el usuario pida acortar/alargar un beat.
- No inventes archivos. Solo library.file. Si no encaja, reutiliza el más cercano y dilo en reply.
- Narración = lo que se CUENTA; text = overlay; visual = cómo se FILMA ese clip.

VOICEOVER (TTS: Edge por defecto, ElevenLabs si hay clave)
- narration = guion HABLADO (TTS). Editar narration y regenerar voz = editar la grabación.
- text = captions/overlay, distinto de la VO salvo que el usuario pida “mismo texto en pantalla”.
- voiceover=true pone voz sobre el plano. muted/duckOriginal baja o silencia el audio del clip.
- voiceStyle: documentary | warm | energetic | ad | calm (entrega: estabilidad, energía, documental).
- Comandos: “pon voiceover en todas las escenas”, “reescribe la VO de la 2 más documental”,
  “silencia el original y deja solo la voz”, “isola la voz / quita la música”,
  “VO más enérgica”, “misma narración en overlay”.
- Una frase por escena, ritmo hablado (no párrafos). Español natural.
- No inventes un voice_id. Solo voiceStyle + narration.

INSTAGRAM EDITS (captions / CC)
- captions = subtítulos automáticos sobre el vídeo (no son la VO).
- El usuario pide: “pon captions”, “genera captions”, “estilo bold/classic/retro/playful/handwritten/bubble/modern/literature”,
  “lower-third / centro / arriba / abajo”, “highlight amarillo”, “opacidad 80”, “sin puntuación”,
  “title case”, “oculta groserías”, “captions de la narración”, “apaga captions”.
- Devuelve también un objeto captions (no reescribas escenas si solo pidió captions):
  {on, style, position, color, highlight, opacity, source: original|narration, generate: bool,
   punctuation, title_case, show_profanity}.
- generate=true cuando pidan auto-CC / transcribir / generar captions.
- source=narration si piden captions del guion hablado; original si del audio del clip.
- voice (global): {all: bool, style, duck, muteOriginal} para VO en todas las escenas.
- Si solo cambian captions/voz, deja scenes igual o omite scenes.

CLEARVIEW CUT (estilo Premiere, nombres propios)
- Salas: Library (busca/guarda), Edit (timeline de un clip), Assemble (montaje de escenas + export).
- Corte: I Mark In, O Mark Out, C razor/split, Q ripple trim inicio, W ripple trim final,
  J/K/L shuttle (atrás / pausa / adelante, L L = 2x), M marker, Delete ripple delete,
  extrae el rango I–O (cierra el hueco).
- Comandos: “haz split en la 2”, “acorta el inicio de la 3”, “ripple delete escena 1”,
  “marca in/out”, “más ritmo (recorta)”. Si dicen “como Premiere / razor / ripple / JKL”,
  aplícalo con estas herramientas de Clearview.

CLEARVIEW COLOR (estilo Resolve, nombres propios)
- Primaries por escena: lift (sombras), gamma (medios), gain (altas luces),
  sat (0–3), temp (−1 frío / +1 cálido), contrast.
- Looks: neutral, film, night, golden, punch.
- Comandos: “escena 2 más cálida”, “look film en todas”, “desatura la 3”,
  “más contraste en la 1”, “reset grade”, “lift down escena 4”, “golden en el cierre”.
- Devuelve grade por escena: {lift, gamma, gain, sat, temp, contrast}.
- Si piden “como DaVinci / Resolve / color wheels / etalonaje”, traduce a Clearview Color.
- No copies la UI de Premiere ni de Resolve.

UNDO
- “undo”, “deshacer”, “deshaz”, “vuelve atrás”, “Ctrl+Z” = revertir el ÚLTIMO cambio del montaje (escenas, duraciones, textos, clips, grade).
- “rehacer” / “redo” = reaplicar lo deshecho.
- No reescribas el guion cuando pidan undo: es una orden de historial, no un recorte nuevo.
- Si no hay historial, di que no hay nada que deshacer.
""".strip()


def _default_grade() -> Dict[str, float]:
    return {"lift": 0.0, "gamma": 1.0, "gain": 1.0, "sat": 1.0, "temp": 0.0, "contrast": 1.0}


def _clamp_grade(g: Dict[str, float]) -> Dict[str, float]:
    def num(key: str, default: float) -> float:
        try:
            return float(g.get(key) if g.get(key) is not None else default)
        except (TypeError, ValueError):
            return default

    return {
        "lift": max(-0.4, min(0.4, num("lift", 0.0))),
        "gamma": max(0.4, min(2.2, num("gamma", 1.0))),
        "gain": max(0.4, min(2.2, num("gain", 1.0))),
        "sat": max(0.0, min(3.0, num("sat", 1.0))),
        "temp": max(-1.0, min(1.0, num("temp", 0.0))),
        "contrast": max(0.4, min(2.2, num("contrast", 1.0))),
    }


def _normalize_grade(raw: Any) -> Dict[str, float]:
    g = dict(_default_grade())
    if isinstance(raw, dict):
        for key in g:
            if raw.get(key) is not None:
                try:
                    g[key] = float(raw[key])
                except (TypeError, ValueError):
                    pass
    return _clamp_grade(g)


GRADE_LOOKS: Dict[str, Dict[str, float]] = {
    "neutral": _default_grade(),
    "film": {"lift": 0.04, "gamma": 1.05, "gain": 0.96, "sat": 0.88, "temp": 0.12, "contrast": 1.08},
    "night": {"lift": -0.08, "gamma": 0.92, "gain": 1.05, "sat": 0.82, "temp": -0.35, "contrast": 1.12},
    "golden": {"lift": 0.02, "gamma": 1.04, "gain": 1.06, "sat": 1.12, "temp": 0.42, "contrast": 1.06},
    "punch": {"lift": -0.05, "gamma": 1.0, "gain": 1.08, "sat": 1.18, "temp": 0.08, "contrast": 1.22},
}


def _look_from_msg(low: str) -> Optional[str]:
    pairs = [
        ("golden", r"golden|dorado|sunset|atardecer|look c[aá]lido"),
        ("night", r"night|noche|luna|moonlight|look fr[ií]o"),
        ("punch", r"punch|contraste alto|high contrast|vivid grade"),
        ("film", r"\bfilm\b|cinem|pel[ií]cula|teal"),
        ("neutral", r"neutral|reset grade|sin grade|quita (el )?grade|rec\.?709"),
    ]
    for name, pat in pairs:
        if re.search(pat, low):
            return name
    return None


def _mutate_grade(g: Dict[str, float], low: str) -> tuple:
    look = _look_from_msg(low)
    if look:
        return _normalize_grade(GRADE_LOOKS[look]), "look " + look
    out = dict(_normalize_grade(g))
    bits: List[str] = []
    if re.search(r"c[aá]lid|warm|naranja", low):
        out["temp"] = min(1.0, out["temp"] + 0.28)
        bits.append("más cálido")
    if re.search(r"fr[ií]o|cool|azul", low):
        out["temp"] = max(-1.0, out["temp"] - 0.28)
        bits.append("más frío")
    if re.search(r"b&w|blanco y negro|sin satur", low):
        out["sat"] = 0.0
        bits.append("sin saturación")
    elif re.search(r"desatura|menos color", low):
        out["sat"] = max(0.0, min(0.4, out["sat"] * 0.45))
        bits.append("menos saturación")
    elif re.search(r"m[aá]s satur|m[aá]s color|\bsatura", low):
        out["sat"] = min(2.2, out["sat"] + 0.25)
        bits.append("más saturación")
    if re.search(r"menos contraste", low):
        out["contrast"] = max(0.5, out["contrast"] - 0.15)
        bits.append("menos contraste")
    elif re.search(r"m[aá]s contraste|\bcontraste\b", low):
        out["contrast"] = min(2.0, out["contrast"] + 0.15)
        bits.append("más contraste")
    if re.search(r"lift (up|arriba|sube)|sombras m[aá]s claras|abre sombras", low):
        out["lift"] = min(0.35, out["lift"] + 0.08)
        bits.append("lift +")
    if re.search(r"lift (down|abajo|baja)|sombras m[aá]s oscuras|cierra sombras", low):
        out["lift"] = max(-0.35, out["lift"] - 0.08)
        bits.append("lift -")
    if re.search(r"\bgain\b|highlights|altas luces", low):
        if re.search(r"baja|menos|down", low):
            out["gain"] = max(0.5, out["gain"] - 0.08)
            bits.append("gain -")
        else:
            out["gain"] = min(1.8, out["gain"] + 0.08)
            bits.append("gain +")
    if re.search(r"\bgamma\b|medios", low) and re.search(r"grade|color|lift|gain", low):
        if re.search(r"baja|menos|down", low):
            out["gamma"] = max(0.5, out["gamma"] - 0.08)
            bits.append("gamma -")
        elif re.search(r"sube|m[aá]s|up", low):
            out["gamma"] = min(1.8, out["gamma"] + 0.08)
            bits.append("gamma +")
    return _clamp_grade(out), (", ".join(bits) or "grade")


def _safe_float(val: Any, default: float, lo: float, hi: float) -> float:
    try:
        n = float(val)
    except (TypeError, ValueError):
        n = default
    if n != n:  # NaN
        n = default
    return max(lo, min(hi, n))


def _image_query_from_msg(msg: str) -> Optional[str]:
    low = (msg or "").lower()
    if not re.search(r"im[aá]genes?|images?|fotos?|photos?|stills?|pictures?", low):
        return None
    q = re.sub(
        r"\b(busca(r)?|search|encuentra|trae|necesito|quiero|pon|add|online|web|"
        r"im[aá]genes?|images?|fotos?|photos?|stills?|pictures?|de|del|para|the|of|from)\b",
        " ",
        low,
        flags=re.I,
    )
    q = re.sub(r"[¿?¡!.,;:]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q[:80] or None


def _image_search_reply(msg: str, current_plan: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    low = (msg or "").lower()
    if not re.search(r"im[aá]genes?|images?|fotos?|photos?|stills?|pictures?", low):
        return None
    if not re.search(r"busca|search|encuentra|trae|necesito|quiero|add|online", low) and not re.search(
        r"im[aá]genes? (de|of)|fotos? (de|of)", low
    ):
        q = _image_query_from_msg(msg)
        if not q:
            return None
    else:
        q = _image_query_from_msg(msg)
    if not q:
        idea = ""
        if isinstance(current_plan, dict):
            idea = str(current_plan.get("title") or current_plan.get("summary") or "")
        q = (idea or "documentary").strip()[:80]
    return {
        "ai": True,
        "reply": f"Busco imágenes web de «{q}». Elige abajo y Añadir; se guardan como stills en Clips.",
        "plan": None,
        "local": True,
        "search_images": {"query": q},
    }


def _wants_load_clips(low: str) -> bool:
    return bool(re.search(
        r"load all|cargar todos|todos los clips|clip[s]?.{0,40}(a |into |as |en )(escenas?|scenes?)|"
        r"añade (todos )?los clips|add (all |the |selected )?clips|"
        r"seleccionad.{0,20}(escena|scene)|escenas? (con|from|desde) (los )?clips",
        low,
    ))


def _plan_from_library(
    current_plan: Dict[str, Any],
    clips: List[Dict[str, Any]],
) -> Dict[str, Any]:
    safe = [c for c in (clips or []) if isinstance(c, dict) and c.get("name")]
    if not safe:
        return {
            "ai": True,
            "reply": "No hay clips para cargar. Marca clips a la izquierda o importa footage web.",
            "plan": None,
            "local": True,
        }
    scenes: List[Dict[str, Any]] = []
    for c in safe[:40]:
        dur = _safe_float(c.get("duration"), 5.0, 0.4, 3600.0)
        scenes.append({
            "clip": c["name"],
            "inPoint": 0.0,
            "outPoint": round(dur, 2),
            "narration": "",
            "text": "",
            "textPosition": "lower-third",
            "visual": "",
            "voiceover": False,
            "voiceStyle": "documentary",
            "duckOriginal": False,
            "grade": _default_grade(),
        })
    packed = _pack_director_plan(
        f"{len(scenes)} clip(s) loaded as scenes, in library order.",
        current_plan if isinstance(current_plan, dict) else {},
        safe,
        scenes,
    )
    packed["reply"] = f"Loaded {len(scenes)} clip(s) as scenes. Drag to reorder, or tell me the cut order."
    return packed


def _wants_grade(low: str) -> bool:
    if re.search(r"voiceover|voz en off|\bvo\b|narraci|voice style|captions|subtit", low) and not re.search(
        r"grade|color|look |etalon|lift|gain|gamma|wheels", low
    ):
        return False
    return bool(re.search(
        r"\bgrade\b|etalon|look film|look night|look golden|look punch|look neutral|"
        r"c[aá]lid|fr[ií]o|desatura|saturac|\bcontraste\b|\blift\b|\bgain\b|"
        r"color grade|color de (la )?escena|da\s*vinci|resolve|wheels",
        low,
    ))


def _load_dotenv() -> None:
    env_path = BASE_DIR / ".env"
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


def _xai_key() -> Optional[str]:
    _load_dotenv()
    return os.environ.get("XAI_API_KEY") or os.environ.get("xai_api_key")


_LIB_CACHE: Dict[str, Any] = {"sig": None, "items": []}


def list_library_clips() -> List[Dict[str, Any]]:
    clips: List[Dict[str, Any]] = []
    if not CLIPS_DIR.exists():
        return clips
    files = [
        f for f in CLIPS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    sig = tuple((f.name, int(f.stat().st_mtime), f.stat().st_size) for f in files)
    cached = _LIB_CACHE.get("items") or []
    if _LIB_CACHE.get("sig") == sig and cached:
        return [dict(x) for x in cached]
    for f in files:
        dur = probe_duration(f) or 5.0
        clips.append({
            "name": f.name,
            "title": f.stem,
            "url": f"/clips/{f.name}",
            "duration": round(float(dur), 2),
            "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
        })
    _LIB_CACHE["sig"] = sig
    _LIB_CACHE["items"] = [dict(x) for x in clips]
    return clips


def _heuristic_plan(
    prompt: str,
    fmt: str,
    target_seconds: float,
    clips: List[Dict[str, Any]],
) -> Dict[str, Any]:
    clips = [c for c in (clips or []) if isinstance(c, dict) and c.get("name")]
    if not clips:
        raise ValueError("No hay clips en la librería. Guarda clips en Edit primero.")

    sentences = [s.strip() for s in re.split(r"[.\n!?]+", prompt) if s.strip()]
    if not sentences:
        sentences = [prompt.strip() or "Secuencia de footage"]

    n = max(1, min(len(clips), max(3, len(sentences)), 10))
    scenes = []
    t = 0.0
    for i in range(n):
        clip = clips[i % len(clips)]
        clip_dur = _safe_float(clip.get("duration"), 5.0, 0.4, 3600.0)
        use = max(0.4, clip_dur)
        line = sentences[i % len(sentences)]
        shots = (
            "wide establishing, hold",
            "medium shot, slow push-in",
            "close-up, slight pan",
            "tracking shot, follow action",
            "wide, hold for CTA",
        )
        scenes.append({
            "order": i + 1,
            "narration": line,
            "visual": shots[i % len(shots)] + f" · {clip.get('title') or clip['name']}",
            "clip": clip["name"],
            "inPoint": 0.0,
            "outPoint": round(use, 2),
            "text": line[:48],
            "textPosition": "lower-third" if i % 2 == 0 else "center",
            "timelineStart": round(t, 2),
            "grade": _default_grade(),
        })
        t += use

    title = (prompt[:60] or "Video").strip()
    return {
        "title": title,
        "format": fmt,
        "duration": round(t, 2),
        "ai": False,
        "summary": "Plan básico (sin XAI_API_KEY). Ordena tus clips y pone textos del prompt.",
        "scenes": scenes,
    }


def _llm_plan(
    prompt: str,
    fmt: str,
    target_seconds: float,
    clips: List[Dict[str, Any]],
    language: str,
) -> Dict[str, Any]:
    from openai import OpenAI

    key = _xai_key()
    if not key:
        raise RuntimeError("missing_key")

    catalog = [
        {
            "file": c["name"],
            "title": c.get("title"),
            "duration_sec": c.get("duration"),
        }
        for c in clips
        if isinstance(c, dict) and c.get("name")
    ]
    fmt_help = FORMATS.get(fmt, FORMATS["youtube"])
    system = (
        DIRECTOR_PLAYBOOK
        + f"\nIdioma de narración y textos: {language}. "
        "Primera pasada: arma el montaje completo con los clips de library. "
        "Devuelves SOLO JSON válido, sin markdown."
    )
    user = {
        "task": "Armar un video con los clips del usuario",
        "format": fmt,
        "format_notes": fmt_help,
        "target_seconds": target_seconds,
        "idea": prompt,
        "library": catalog,
        "json_schema": {
            "title": "string",
            "summary": "string",
            "duration": "number (suma aproximada de escenas)",
            "scenes": [
                {
                    "order": "int",
                    "narration": "string (voz en off / lo que se cuenta)",
                    "visual": "string (cámara + acción Seedance, ej: wide establishing, slow push-in)",
                    "clip": "filename exacto de library",
                    "inPoint": "float segundos desde el inicio del archivo",
                    "outPoint": "float > inPoint, no mayor que duration del clip",
                    "text": "string overlay corto",
                    "textPosition": "center|top|bottom|lower-third",
                    "voiceover": "bool (TTS con narration)",
                    "voiceStyle": "documentary|warm|energetic|ad|calm",
                    "duckOriginal": "bool (baja el audio del clip bajo la VO)",
                    "grade": {
                        "lift": "float -0.4..0.4 sombras",
                        "gamma": "float 0.4..2.2 medios",
                        "gain": "float 0.4..2.2 altas",
                        "sat": "float 0..3",
                        "temp": "float -1 frío .. +1 cálido",
                        "contrast": "float 0.4..2.2",
                    },
                }
            ],
        },
    }
    client = OpenAI(api_key=key, base_url="https://api.x.ai/v1", timeout=45.0)
    resp = client.chat.completions.create(
        model=os.environ.get("XAI_MODEL", "grok-4.6"),
        temperature=0.4,
        timeout=45.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )
    choices = getattr(resp, "choices", None) or []
    if not choices or not getattr(choices[0], "message", None):
        raise RuntimeError("empty_llm")
    text = (choices[0].message.content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("bad_json") from exc
    return data


def filename_from_src(src: Any) -> str:
    s = str(src or "").strip().split("?")[0].replace("\\", "/")
    if not s:
        return ""
    return Path(s.rstrip("/")).name


def _normalize_crop(raw: Any) -> Optional[Dict[str, float]]:
    if not isinstance(raw, dict):
        return None
    x = _safe_float(raw.get("x"), 0.0, 0.0, 1.0)
    y = _safe_float(raw.get("y"), 0.0, 0.0, 1.0)
    w = _safe_float(raw.get("w"), 1.0, 0.0, 1.0)
    h = _safe_float(raw.get("h"), 1.0, 0.0, 1.0)
    if w < 0.05 or h < 0.05:
        return None
    if w >= 0.995 and h >= 0.995 and x <= 0.005 and y <= 0.005:
        return None
    return {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)}


def _media_url(clip_name: str, hint: Any = "") -> str:
    h = str(hint or "").replace("\\", "/")
    if "/uploads/" in h:
        return f"/uploads/{clip_name}"
    return f"/clips/{clip_name}"


def _sanitize_plan(
    raw: Dict[str, Any],
    clips: List[Dict[str, Any]],
    fmt: str,
    remap_missing: bool = True,
) -> Dict[str, Any]:
    by_name = {
        str(c.get("name") or ""): c
        for c in (clips or [])
        if isinstance(c, dict) and c.get("name")
    }
    names = list(by_name.keys())

    scenes_in = raw.get("scenes") or []
    if isinstance(scenes_in, dict):
        scenes_in = [scenes_in[k] for k in sorted(scenes_in, key=lambda x: str(x))]
    if not isinstance(scenes_in, list):
        scenes_in = []
    scenes_in = [s for s in scenes_in if isinstance(s, dict)]

    if not names:
        if remap_missing:
            raise ValueError("No hay clips en la librería.")
        for s in scenes_in:
            n = filename_from_src(
                s.get("clip") or s.get("src") or s.get("url") or s.get("name") or s.get("serverSrc")
            )
            if n and n not in by_name:
                dur = _safe_float(s.get("outPoint"), 5.0, 0.4, 3600.0)
                by_name[n] = {"name": n, "duration": dur}
        names = list(by_name.keys())
        if not names and not scenes_in:
            title = str(raw.get("title") or "Video").strip()[:80] or "Video"
            return {
                "title": title,
                "format": fmt,
                "duration": 0.0,
                "summary": str(raw.get("summary") or "").strip(),
                "script": sanitize_script(raw.get("script")),
                "scenes": [],
                "export_name": safe_filename(title)[:60],
            }
        if not names:
            raise ValueError("No hay clips en la librería.")

    if not scenes_in:
        if remap_missing:
            raise ValueError("La IA no devolvió escenas válidas.")
        title = str(raw.get("title") or "Video").strip()[:80] or "Video"
        return {
            "title": title,
            "format": fmt,
            "duration": 0.0,
            "summary": str(raw.get("summary") or "").strip(),
            "script": sanitize_script(raw.get("script")),
            "scenes": [],
            "export_name": safe_filename(title)[:60],
        }

    scenes = []
    t = 0.0
    for i, s in enumerate(scenes_in[:40]):
        raw_clip = s.get("clip")
        if isinstance(raw_clip, int):
            clip_name = names[raw_clip] if 0 <= raw_clip < len(names) else names[i % len(names)]
        elif isinstance(raw_clip, (list, dict)) or raw_clip is None:
            clip_name = ""
        else:
            clip_name = str(raw_clip).strip()
        if not clip_name:
            clip_name = filename_from_src(
                s.get("src") or s.get("url") or s.get("name") or s.get("serverSrc") or ""
            )
        if not clip_name:
            clip_name = names[i % len(names)] if names else ""
        if clip_name and clip_name not in by_name:
            stem = Path(str(clip_name)).stem.lower()
            match = next(
                (n for n in names if Path(n).stem.lower() == stem or stem in n.lower()),
                None,
            )
            if match:
                clip_name = match
            elif remap_missing and names:
                clip_name = names[i % len(names)]
            else:
                by_name[clip_name] = {
                    "name": clip_name,
                    "duration": _safe_float(s.get("outPoint"), 5.0, 0.4, 3600.0),
                }
                names.append(clip_name)
        meta = by_name.get(clip_name) or (by_name[names[0]] if names else {"duration": 5.0})
        max_d = _safe_float(meta.get("duration"), 5.0, 0.4, 3600.0)
        in_p = max(0.0, _safe_float(s.get("inPoint"), 0.0, 0.0, max_d))
        raw_out = s.get("outPoint")
        out_p = _safe_float(raw_out, max_d, 0.0, max_d) if raw_out is not None else max_d
        if out_p <= in_p + 0.2:
            in_p = 0.0
            out_p = max_d
        out_p = min(max_d, out_p)
        if out_p <= in_p:
            in_p = 0.0
            out_p = max_d
        pos = s.get("textPosition") or "lower-third"
        if pos not in {"center", "top", "bottom", "lower-third"}:
            pos = "lower-third"
        speed = _safe_float(s.get("speed"), 1.0, 0.25, 4.0)
        freeze = bool(s.get("freeze"))
        filt = str(s.get("filter") or "none").strip().lower()
        if filt not in {"none", "bw", "sepia", "vivid", "cool", "warm"}:
            filt = "none"
        fit = str(s.get("fit") or "contain").strip().lower()
        if fit not in {"contain", "cover"}:
            fit = "contain"
        src_hint = s.get("src") or s.get("url") or s.get("serverSrc") or meta.get("url") or ""
        url = _media_url(clip_name, src_hint)
        if freeze:
            play = _safe_float(s.get("duration"), 2.0, 0.2, 3600.0)
        else:
            play = max(0.2, (out_p - in_p) / speed)
        sid = str(s.get("id") or "").strip()[:24]
        if not sid:
            sid = uuid.uuid4().hex[:12]
        scenes.append({
            "order": i + 1,
            "id": sid,
            "narration": str(s.get("narration") or "").strip(),
            "visual": str(s.get("visual") or "").strip(),
            "clip": clip_name,
            "src": url,
            "inPoint": round(in_p, 2),
            "outPoint": round(out_p, 2),
            "text": str(s.get("text") or "").strip()[:80],
            "textPosition": pos,
            "timelineStart": round(t, 2),
            "url": url,
            "voiceover": bool(s.get("voiceover")),
            "voiceStyle": (
                str(s.get("voiceStyle") or "documentary").strip().lower()
                if str(s.get("voiceStyle") or "").strip().lower()
                in {"documentary", "warm", "energetic", "ad", "calm"}
                else "documentary"
            ),
            "voiceEngine": sanitize_engine(s.get("voiceEngine") or "edge"),
            "voiceName": sanitize_voice_name(s.get("voiceName")),
            "voFx": sanitize_vo_fx(s.get("voFx")),
            "voRate": sanitize_vo_rate(s.get("voRate")),
            "capOff": _sanitize_cap_off(s.get("capOff")),
            "capPos": _sanitize_cap_pos(s.get("capPos")),
            "duckOriginal": bool(s.get("duckOriginal") if s.get("duckOriginal") is not None else s.get("voiceover")),
            "grade": _normalize_grade(s.get("grade")),
            "muted": bool(s.get("muted")),
            "speed": speed,
            "rotation": int(_safe_float(s.get("rotation"), 0, 0, 359)) % 360,
            "crop": _normalize_crop(s.get("crop")),
            "filter": filt,
            "fit": fit,
            "freeze": freeze,
            "fadeIn": bool(s.get("fadeIn")),
            "fadeOut": bool(s.get("fadeOut")),
            "volume": _safe_float(s.get("volume"), 1.0, 0.0, 2.0),
            "duration": round(play, 2),
        })
        t += play

    title = str(raw.get("title") or "Video").strip()[:80] or "Video"
    return {
        "title": title,
        "format": fmt,
        "duration": round(t, 2),
        "summary": str(raw.get("summary") or "").strip(),
        "script": sanitize_script(raw.get("script")),
        "scenes": scenes,
        "export_name": safe_filename(title)[:60],
    }


SEQUENCE_VERSION = 1
SHRINK_MIN_SCENES = 4
SHRINK_RATIO = 0.5
_save_sequence_lock = threading.Lock()
_MEDIA_EXT = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}


def _autosave_dir(folder: Optional[Path] = None) -> Path:
    d = Path(folder) if folder else (BASE_DIR / "storage" / "autosave")
    d.mkdir(parents=True, exist_ok=True)
    return d


def sequence_paths(folder: Optional[Path] = None) -> Tuple[Path, Path]:
    d = _autosave_dir(folder)
    return d / "sequence.json", d / "sequence.bak.json"


SEQUENCE_VO_URL = "/api/sequence/vo"
SEQUENCE_MUSIC_URL = "/api/sequence/music"


def sequence_vo_path(folder: Optional[Path] = None) -> Path:
    return _autosave_dir(folder) / "vo.mp3"


def resolve_vo_source(url: str = "", folder: Optional[Path] = None) -> Optional[Path]:
    """Map a VO url to a file on disk. /api/sequence/vo is storage/autosave/vo.mp3."""
    raw = str(url or "").strip().split("?")[0]
    name = Path(raw).name if raw else ""
    candidates: List[Path] = []
    if (not raw) or raw.rstrip("/").endswith("/sequence/vo") or name in {"vo", "vo.mp3"}:
        candidates.append(sequence_vo_path(folder))
    if name and name not in {".", "..", "vo"}:
        candidates.append(UPLOADS_DIR / name)
        candidates.append(TEMP_DIR / name)
    seen = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except OSError:
            rp = p
        if rp in seen:
            continue
        seen.add(rp)
        try:
            if p.is_file() and p.stat().st_size >= 200:
                return p
        except OSError:
            continue
    return None


def sequence_vo_exists(folder: Optional[Path] = None) -> bool:
    p = sequence_vo_path(folder)
    try:
        return p.is_file() and p.stat().st_size >= 200
    except OSError:
        return False


def persist_sequence_vo(src: Path, folder: Optional[Path] = None) -> Optional[Path]:
    """Copy a generated mp3 to storage/autosave/vo.mp3. Never stores bytes in JSON."""
    src = Path(src)
    try:
        if not src.is_file() or src.stat().st_size < 200:
            return None
    except OSError:
        return None
    dest = sequence_vo_path(folder)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name("vo.mp3.tmp")
    try:
        if dest.exists() and dest.resolve() == src.resolve():
            return dest
        shutil.copy2(src, tmp)
        tmp.replace(dest)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return None
    return dest if dest.is_file() and dest.stat().st_size >= 200 else None


def sequence_music_path(folder: Optional[Path] = None) -> Path:
    return _autosave_dir(folder) / "music.mp3"


def sequence_music_exists(folder: Optional[Path] = None) -> bool:
    p = sequence_music_path(folder)
    try:
        return p.is_file() and p.stat().st_size >= 200
    except OSError:
        return False


def persist_sequence_music(src: Path, folder: Optional[Path] = None) -> Optional[Path]:
    """Copy/transcode a song to storage/autosave/music.mp3. Never stores bytes in JSON."""
    src = Path(src)
    try:
        if not src.is_file() or src.stat().st_size < 200:
            return None
    except OSError:
        return None
    dest = sequence_music_path(folder)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name("music.mp3.tmp")
    try:
        if dest.exists() and dest.resolve() == src.resolve():
            return dest
        ext = src.suffix.lower()
        if ext == ".mp3":
            shutil.copy2(src, tmp)
        else:
            ff = ffmpeg_bin()
            if not ff:
                return None
            flags = 0x08000000 if os.name == "nt" else 0
            r = subprocess.run(
                [ff, "-y", "-i", str(src), "-vn", "-c:a", "libmp3lame", "-q:a", "4", str(tmp)],
                capture_output=True,
                timeout=180,
                creationflags=flags,
            )
            if r.returncode != 0 or not tmp.is_file() or tmp.stat().st_size < 200:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
                return None
        tmp.replace(dest)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return None
    return dest if dest.is_file() and dest.stat().st_size >= 200 else None


def clear_sequence_music(folder: Optional[Path] = None) -> None:
    p = sequence_music_path(folder)
    try:
        if p.is_file():
            p.unlink()
    except OSError:
        pass


def _sanitize_ui_music(raw: Any, *, has_file: bool = False) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    name = str(raw.get("name") or "").replace("\\", "/").split("/")[-1].strip()[:120]
    vol = _safe_float(raw.get("volume"), 0.25, 0.0, 1.0)
    mute = bool(raw.get("mute"))
    if not has_file and not name:
        mute = bool(raw.get("mute")) if "mute" in raw else False
    return {
        "url": SEQUENCE_MUSIC_URL,
        "name": name,
        "volume": round(vol, 3),
        "mute": mute,
    }


def _sanitize_ui_vo(
    raw: Any,
    *,
    script: str = "",
    has_file: bool = False,
) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    sig = str(raw.get("sig") or "")[:8000]
    try:
        dur = float(raw.get("duration") or 0)
    except (TypeError, ValueError):
        dur = 0.0
    if dur != dur or dur < 0:
        dur = 0.0
    url = str(raw.get("url") or "").strip().split("?")[0]
    if url and url != SEQUENCE_VO_URL:
        url = SEQUENCE_VO_URL
    if not url:
        url = SEQUENCE_VO_URL
    if "follow" in raw:
        follow = bool(raw.get("follow"))
    else:
        follow = bool(sig or dur > 0 or (script or "").strip() or has_file)
    return {
        "url": url,
        "sig": sig,
        "duration": round(min(dur, 86400.0), 3),
        "follow": follow,
    }


def _sanitize_ui_captions(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    cap = dict(raw)
    cues = []
    for c in cap.get("cues") or []:
        if not isinstance(c, dict):
            continue
        text = str(c.get("text") or c.get("content") or "").strip()[:200]
        if not text:
            continue
        start = _safe_float(c.get("start"), 0.0, 0.0, 86400.0)
        end = _safe_float(c.get("end"), start + 0.4, 0.0, 86400.0)
        cues.append({"start": round(start, 2), "end": round(end, 2), "text": text})
    cap["cues"] = cues[:400]
    cap["on"] = cap.get("on") is not False
    if cap.get("source") is not None:
        cap["source"] = str(cap.get("source") or "")[:40]
    if cap.get("style") is not None:
        cap["style"] = str(cap.get("style") or "")[:40]
    return cap


def _sanitize_ui(
    ui: Any,
    *,
    script: str = "",
    has_file: bool = False,
    has_music: bool = False,
) -> Dict[str, Any]:
    if not isinstance(ui, dict):
        return {}
    out = dict(ui)
    if "vo" in out:
        out["vo"] = _sanitize_ui_vo(out.get("vo"), script=script, has_file=has_file)
    if "captions" in out:
        out["captions"] = _sanitize_ui_captions(out.get("captions"))
    if "music" in out:
        out["music"] = _sanitize_ui_music(out.get("music"), has_file=has_music)
    return out


def _seq_rev(raw: Any) -> int:
    """Hands token. Stale tabs send an older rev and must not overwrite disk."""
    if not isinstance(raw, dict):
        return 0
    val = raw.get("rev", raw.get("baseRev"))
    if val is None and isinstance(raw.get("plan"), dict):
        val = raw["plan"].get("rev")
    try:
        return max(0, int(val or 0))
    except (TypeError, ValueError):
        return 0


def empty_sequence() -> Dict[str, Any]:
    return {
        "version": SEQUENCE_VERSION,
        "rev": 0,
        "savedAt": 0,
        "title": "Video",
        "format": "youtube",
        "summary": "",
        "duration": 0.0,
        "script": "",
        "scenes": [],
        "markers": [],
        "texts": [],
        "markIn": None,
        "markOut": None,
        "ui": {},
    }


def _normalize_markers(raw: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for m in raw:
        if not isinstance(m, dict):
            continue
        t = _safe_float(m.get("t"), 0.0, 0.0, 86400.0)
        out.append({"t": round(t, 2), "label": str(m.get("label") or "")[:40]})
    out.sort(key=lambda x: x["t"])
    return out[:80]


def _normalize_texts(raw: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for t in raw:
        if not isinstance(t, dict):
            continue
        content = str(t.get("content") or t.get("text") or "").strip()
        if not content:
            continue
        pos = str(t.get("position") or "center")
        if pos not in {"center", "top", "bottom", "lower-third"}:
            pos = "center"
        out.append({
            "content": content[:200],
            "size": int(_safe_float(t.get("size"), 42, 12, 120)),
            "position": pos,
            "start": _safe_float(t.get("start"), 0.0, 0.0, 86400.0),
            "end": _safe_float(t.get("end"), 4.0, 0.0, 86400.0),
            "color": str(t.get("color") or "white")[:20],
        })
    return out[:40]


def sequence_library(scenes: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    lib = list_library_clips()
    by = {str(c.get("name") or ""): c for c in lib if c.get("name")}
    try:
        if UPLOADS_DIR.exists():
            for f in UPLOADS_DIR.iterdir():
                if f.is_file() and f.suffix.lower() in _MEDIA_EXT and f.name not in by:
                    dur = probe_duration(f) or 5.0
                    by[f.name] = {
                        "name": f.name,
                        "title": f.stem,
                        "url": f"/uploads/{f.name}",
                        "duration": round(float(dur), 2),
                    }
    except Exception:
        pass
    for s in scenes or []:
        if not isinstance(s, dict):
            continue
        name = filename_from_src(
            s.get("clip") or s.get("src") or s.get("url") or s.get("name") or s.get("serverSrc")
        )
        if name and name not in by:
            dur = _safe_float(s.get("outPoint") or s.get("duration"), 5.0, 0.4, 3600.0)
            hint = str(s.get("src") or s.get("url") or "")
            by[name] = {"name": name, "duration": dur, "url": _media_url(name, hint)}
    return list(by.values())


def editor_clip_to_scene(c: Dict[str, Any]) -> Dict[str, Any]:
    src = str(c.get("src") or c.get("serverSrc") or "")
    name = filename_from_src(c.get("name") or src)
    out_p = c.get("outPoint")
    if c.get("freeze"):
        in_p = _safe_float(c.get("inPoint"), 0.0, 0.0, 3600.0)
        out_p = in_p + _safe_float(c.get("duration"), 2.0, 0.2, 3600.0)
    out = {
        "id": c.get("id") or "",
        "clip": name,
        "src": src or _media_url(name, src),
        "name": name,
        "inPoint": c.get("inPoint"),
        "outPoint": out_p,
        "duration": c.get("duration"),
        "speed": c.get("speed"),
        "volume": c.get("volume"),
        "muted": c.get("muted"),
        "rotation": c.get("rotation"),
        "filter": c.get("filter"),
        "freeze": c.get("freeze"),
        "fadeIn": c.get("fadeIn"),
        "fadeOut": c.get("fadeOut"),
        "crop": c.get("crop"),
        "grade": c.get("grade"),
        "text": c.get("text") or "",
        "textPosition": c.get("textPosition") or "lower-third",
        "voiceover": bool(c.get("voiceover")),
        "duckOriginal": bool(c.get("duckOriginal")),
    }
    if "narration" in c:
        out["narration"] = c.get("narration") or ""
    if "voiceStyle" in c:
        out["voiceStyle"] = c.get("voiceStyle")
    if "voiceEngine" in c:
        out["voiceEngine"] = c.get("voiceEngine")
    if "voiceName" in c:
        out["voiceName"] = c.get("voiceName")
    if "voFx" in c:
        out["voFx"] = c.get("voFx")
    if "voRate" in c:
        out["voRate"] = c.get("voRate")
    if "capOff" in c:
        out["capOff"] = c.get("capOff")
    if "capPos" in c:
        out["capPos"] = c.get("capPos")
    return out


def migrate_legacy(data: Any) -> Dict[str, Any]:
    """Turn editor/assemble/recovered payloads into a sequence v1 dict (unsanitized wrapper)."""
    if not isinstance(data, dict):
        return {}
    if int(data.get("version") or 0) >= SEQUENCE_VERSION and isinstance(data.get("scenes"), list):
        return data
    if isinstance(data.get("clips"), list):
        return {
            "title": data.get("name") or data.get("title") or "Video",
            "format": "youtube",
            "summary": "",
            "scenes": [editor_clip_to_scene(c) for c in data["clips"] if isinstance(c, dict)],
            "markers": data.get("markers") or [],
            "texts": data.get("texts") or [],
            "markIn": data.get("markIn"),
            "markOut": data.get("markOut"),
            "savedAt": data.get("savedAt") or 0,
            "rev": _seq_rev(data),
            "ui": {
                "currentTime": data.get("currentTime"),
                "selectedClipId": data.get("selectedClipId"),
                "pxPerSec": data.get("pxPerSec"),
                "exportName": data.get("exportName"),
                "exportRes": data.get("exportRes"),
                "previewH": data.get("previewH"),
                **({"music": data.get("music")} if data.get("music") is not None else {}),
            },
        }
    plan = data.get("plan") if isinstance(data.get("plan"), dict) else None
    if plan and isinstance(plan.get("scenes"), list):
        ui = {
            "idea": data.get("idea"),
            "fmt": data.get("fmt") or plan.get("format"),
            "dur": data.get("dur"),
            "exportName": data.get("exportName"),
            "exportRes": data.get("exportRes"),
            "captions": data.get("captions"),
            "chatHistory": data.get("chatHistory"),
            "selectedSceneIndex": data.get("selectedSceneIndex"),
            "currentTime": data.get("currentTime"),
        }
        if data.get("vo") is not None:
            ui["vo"] = data.get("vo")
        if data.get("music") is not None:
            ui["music"] = data.get("music")
        return {
            "title": plan.get("title") or data.get("title") or "Video",
            "format": plan.get("format") or data.get("fmt") or "youtube",
            "summary": plan.get("summary") or "",
            "scenes": plan.get("scenes") or [],
            "script": plan.get("script") or data.get("script") or "",
            "markers": data.get("markers") or plan.get("markers") or [],
            "texts": data.get("texts") or plan.get("texts") or [],
            "markIn": data.get("markIn"),
            "markOut": data.get("markOut"),
            "savedAt": data.get("savedAt") or 0,
            "rev": _seq_rev(data),
            "ui": ui,
        }
    if isinstance(data.get("scenes"), list):
        return data
    return {}


def sanitize_sequence(
    raw: Any,
    library: Optional[List[Dict[str, Any]]] = None,
    folder: Optional[Path] = None,
) -> Dict[str, Any]:
    migrated = migrate_legacy(raw) if isinstance(raw, dict) else {}
    scenes = migrated.get("scenes") if isinstance(migrated.get("scenes"), list) else []
    fmt = str(migrated.get("format") or "youtube").lower()
    if fmt not in FORMATS:
        fmt = "youtube"
    lib = library if library is not None else sequence_library(scenes)
    plan = _sanitize_plan(
        {
            "title": migrated.get("title") or "Video",
            "summary": migrated.get("summary") or "",
            "scenes": scenes,
            "script": migrated.get("script") or "",
        },
        lib,
        fmt,
        remap_missing=False,
    )
    ui = _sanitize_ui(
        migrated.get("ui") if isinstance(migrated.get("ui"), dict) else {},
        script=plan.get("script") or "",
        has_file=sequence_vo_exists(folder) if folder is not None else False,
        has_music=sequence_music_exists(folder) if folder is not None else False,
    )
    saved_at = migrated.get("savedAt") or 0
    try:
        saved_at = int(saved_at)
    except (TypeError, ValueError):
        saved_at = 0
    rev = _seq_rev(migrated)
    return {
        "version": SEQUENCE_VERSION,
        "rev": rev,
        "savedAt": saved_at or int(time.time() * 1000),
        "title": plan.get("title") or "Video",
        "format": plan.get("format") or fmt,
        "summary": plan.get("summary") or "",
        "script": plan.get("script") or "",
        "duration": plan.get("duration") or 0,
        "scenes": plan.get("scenes") or [],
        "markers": _normalize_markers(migrated.get("markers")),
        "texts": _normalize_texts(migrated.get("texts")),
        "markIn": migrated.get("markIn"),
        "markOut": migrated.get("markOut"),
        "ui": ui,
        "export_name": plan.get("export_name") or "",
    }


def sequence_as_assemble(seq: Dict[str, Any]) -> Dict[str, Any]:
    ui = seq.get("ui") if isinstance(seq.get("ui"), dict) else {}
    return {
        "savedAt": seq.get("savedAt") or 0,
        "rev": _seq_rev(seq),
        "idea": ui.get("idea") or seq.get("summary") or "",
        "fmt": ui.get("fmt") or seq.get("format") or "documentary",
        "dur": ui.get("dur") or "60",
        "exportName": ui.get("exportName") or seq.get("export_name") or "",
        "exportRes": ui.get("exportRes") or "720",
        "plan": {
            "title": seq.get("title") or "Video",
            "summary": seq.get("summary") or "",
            "format": seq.get("format") or "youtube",
            "duration": seq.get("duration") or 0,
            "script": seq.get("script") or "",
            "scenes": seq.get("scenes") or [],
        },
        "selectedSceneIndex": ui.get("selectedSceneIndex") or 0,
        "currentTime": ui.get("currentTime") or 0,
        "chatHistory": ui.get("chatHistory") or [],
        "captions": ui.get("captions"),
        "vo": ui.get("vo"),
        "music": ui.get("music"),
        "markers": seq.get("markers") or [],
        "texts": seq.get("texts") or [],
    }


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file() or path.stat().st_size < 8:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _legacy_candidate_files(folder: Path) -> List[Path]:
    storage = folder.parent if folder.name == "autosave" else folder
    names = [
        folder / "sequence.json",
        folder / "sequence.bak.json",
        folder / "video-creation.json",
        folder / "video-creation.bak.json",
        storage / "pvs-video-creation-v1.recovered.json",
        storage / "pvs-clip-edition-v1.recovered.json",
    ]
    out: List[Path] = []
    seen = set()
    for p in names:
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def load_sequence(
    folder: Optional[Path] = None,
    library: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    d = _autosave_dir(folder)
    seq_path, bak_path = sequence_paths(d)
    for path, label in ((seq_path, "sequence.json"), (bak_path, "sequence.bak.json")):
        data = _read_json(path)
        if not data:
            continue
        seq = sanitize_sequence(data, library=library, folder=d)
        if seq.get("scenes") or path == seq_path:
            return seq, label
    best: Optional[Tuple[int, int, Dict[str, Any], str]] = None
    for path in _legacy_candidate_files(d):
        if path.name.startswith("sequence"):
            continue
        data = _read_json(path)
        if not data:
            continue
        migrated = migrate_legacy(data)
        n = len(migrated.get("scenes") or [])
        if n < 1:
            continue
        try:
            ts = int(migrated.get("savedAt") or 0)
        except (TypeError, ValueError):
            ts = 0
        cand = (n, ts, migrated, path.name)
        if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
            best = cand
    if not best:
        return empty_sequence(), None
    seq = sanitize_sequence(best[2], library=library, folder=d)
    return seq, "migrated:" + best[3]


_ASSEMBLE_KEEP = (
    "capOff",
    "capPos",
    "narration",
    "voiceover",
    "voiceStyle",
    "voiceEngine",
    "voiceName",
    "voFx",
    "voRate",
    "duckOriginal",
    "fadeIn",
    "fadeOut",
    "freeze",
)


def _is_editor_payload(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    if int(raw.get("version") or 0) >= SEQUENCE_VERSION:
        return False
    if isinstance(raw.get("plan"), dict) and isinstance(raw["plan"].get("scenes"), list):
        return False
    return isinstance(raw.get("clips"), list)


def _incoming_scenes_raw(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    migrated = migrate_legacy(raw)
    scenes = migrated.get("scenes")
    if not isinstance(scenes, list):
        return []
    return [s for s in scenes if isinstance(s, dict)]


def _incoming_script(raw: Any) -> Optional[str]:
    if not isinstance(raw, dict):
        return None
    if "script" in raw:
        return str(raw.get("script") or "")
    plan = raw.get("plan")
    if isinstance(plan, dict) and "script" in plan:
        return str(plan.get("script") or "")
    return None


def _match_existing_scene(
    new_s: Dict[str, Any], index: int, old_scenes: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    nid = str(new_s.get("id") or "").strip()
    if nid:
        for o in old_scenes:
            if isinstance(o, dict) and str(o.get("id") or "").strip() == nid:
                return o
    if 0 <= index < len(old_scenes) and isinstance(old_scenes[index], dict):
        return old_scenes[index]
    return None


def _merge_existing_sequence(seq: Dict[str, Any], existing: Dict[str, Any], raw: Any) -> Dict[str, Any]:
    """Keep Assemble-only fields when Edit omits them. Never default capOff because the key is missing."""
    if not existing or not isinstance(existing, dict):
        return seq
    editor = _is_editor_payload(raw)
    raw_scenes = _incoming_scenes_raw(raw)
    old_scenes = existing.get("scenes") if isinstance(existing.get("scenes"), list) else []
    new_scenes = seq.get("scenes") if isinstance(seq.get("scenes"), list) else []
    for i, ns in enumerate(new_scenes):
        if not isinstance(ns, dict):
            continue
        raw_s = raw_scenes[i] if i < len(raw_scenes) else {}
        if not isinstance(raw_s, dict):
            raw_s = {}
        old_s = _match_existing_scene(ns, i, old_scenes)
        if not old_s:
            continue
        for k in _ASSEMBLE_KEEP:
            if editor or (k not in raw_s):
                if k in old_s:
                    ns[k] = old_s[k]
    incoming_script = _incoming_script(raw)
    if incoming_script is None and existing.get("script"):
        seq["script"] = existing.get("script")
    return seq


def save_sequence(
    raw: Any,
    folder: Optional[Path] = None,
    library: Optional[List[Dict[str, Any]]] = None,
    force: bool = False,
) -> Tuple[Dict[str, Any], str]:
    with _save_sequence_lock:
        return _save_sequence_locked(raw, folder=folder, library=library, force=force)


def _save_sequence_locked(
    raw: Any,
    folder: Optional[Path] = None,
    library: Optional[List[Dict[str, Any]]] = None,
    force: bool = False,
) -> Tuple[Dict[str, Any], str]:
    d = _autosave_dir(folder)
    seq_path, bak_path = sequence_paths(d)
    existing = _read_json(seq_path)
    disk_rev = _seq_rev(existing) if existing else 0
    incoming_rev = _seq_rev(raw)
    incoming_scenes = _incoming_scenes_raw(raw)
    if existing and (existing.get("scenes") or []) and not force:
        if not incoming_scenes:
            return sanitize_sequence(existing, library=library, folder=d), "protected"
        if incoming_rev < disk_rev:
            return sanitize_sequence(existing, library=library, folder=d), "stale"
        old_n = len(existing.get("scenes") or [])
        new_n = len(incoming_scenes)
        if old_n >= SHRINK_MIN_SCENES and new_n < old_n * SHRINK_RATIO:
            return sanitize_sequence(existing, library=library, folder=d), "shrink"
    seq = sanitize_sequence(raw, library=library, folder=d)
    if not seq.get("scenes") and not force:
        if existing and (existing.get("scenes") or []):
            return sanitize_sequence(existing, library=library, folder=d), "protected"
    if seq_path.is_file() and seq_path.stat().st_size > 20:
        try:
            shutil.copy2(seq_path, bak_path)
        except Exception:
            pass
    if existing:
        seq = _merge_existing_sequence(seq, existing, raw)
    if existing and isinstance(existing.get("ui"), dict):
        merged = dict(existing["ui"])
        incoming_ui = seq.get("ui") or {}
        merged.update(incoming_ui)
        old_vo = existing["ui"].get("vo") if isinstance(existing["ui"].get("vo"), dict) else None
        inc_vo = incoming_ui.get("vo") if isinstance(incoming_ui.get("vo"), dict) else None
        if old_vo and old_vo.get("sig") and (inc_vo is None or not inc_vo.get("sig")):
            kept = dict(old_vo)
            if isinstance(inc_vo, dict) and "follow" in inc_vo:
                kept["follow"] = bool(inc_vo.get("follow"))
            merged["vo"] = kept
        if "captions" not in incoming_ui and existing["ui"].get("captions"):
            merged["captions"] = existing["ui"]["captions"]
        seq["ui"] = _sanitize_ui(
            merged,
            script=seq.get("script") or "",
            has_file=sequence_vo_exists(d),
            has_music=sequence_music_exists(d),
        )
    seq["rev"] = disk_rev + 1
    seq["savedAt"] = int(time.time() * 1000)
    seq["version"] = SEQUENCE_VERSION
    payload = json.dumps(seq, ensure_ascii=False)
    tmp = seq_path.with_name(seq_path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(seq_path)
    return seq, "sequence.json"


def _project_slug(name: str) -> str:
    raw = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return (raw or "proyecto")[:48]


def _projects_root() -> Path:
    d = BASE_DIR / "storage" / "projects"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _projects_index_path() -> Path:
    return _projects_root() / "index.json"


def _project_folder(pid: str) -> Path:
    safe = Path(str(pid or "")).name
    return _projects_root() / safe


def _clips_from_scenes(scenes: Optional[List[Dict[str, Any]]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for s in scenes or []:
        if not isinstance(s, dict):
            continue
        name = filename_from_src(
            s.get("clip") or s.get("src") or s.get("url") or s.get("name") or ""
        )
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def load_projects_index() -> Dict[str, Any]:
    path = _projects_index_path()
    data = _read_json(path) if path.is_file() else None
    if not isinstance(data, dict):
        data = {}
    projects: List[Dict[str, Any]] = []
    for p in data.get("projects") or []:
        if not isinstance(p, dict) or not str(p.get("id") or "").strip():
            continue
        clips: List[str] = []
        for c in p.get("clips") or []:
            name = filename_from_src(c)
            if name and name not in clips:
                clips.append(name)
        try:
            updated = int(p.get("updatedAt") or 0)
        except (TypeError, ValueError):
            updated = 0
        projects.append({
            "id": str(p["id"]).strip(),
            "name": str(p.get("name") or p["id"]).strip() or str(p["id"]).strip(),
            "updatedAt": updated,
            "clips": clips,
        })
    active = str(data.get("activeId") or "").strip()
    ids = {p["id"] for p in projects}
    if projects and active not in ids:
        active = projects[0]["id"]
    if not projects:
        active = ""
    return {"activeId": active, "projects": projects}


def _write_projects_index(idx: Dict[str, Any]) -> None:
    path = _projects_index_path()
    path.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")


def ensure_project_clip_membership() -> Dict[str, Any]:
    """Fill each project's clip list from that project's sequence. Does not touch scenes."""
    idx = load_projects_index()
    for p in idx["projects"]:
        folder = _project_folder(p["id"])
        seq_path = folder / "sequence.json"
        if not seq_path.is_file():
            p["clips"] = list(p.get("clips") or [])
            continue
        seq, _src = load_sequence(folder)
        names = _clips_from_scenes(seq.get("scenes") or [])
        if names:
            p["clips"] = names
        else:
            p["clips"] = list(p.get("clips") or [])
    _write_projects_index(idx)
    return idx


def create_project(name: str) -> Dict[str, Any]:
    title = (name or "").strip()
    if not title:
        raise ValueError("Pon un nombre al proyecto")
    idx = ensure_project_clip_membership()
    slug = _project_slug(title)
    ids = {p["id"] for p in idx["projects"]}
    pid = slug
    n = 2
    while pid in ids:
        pid = f"{slug}-{n}"
        n += 1
    folder = _project_folder(pid)
    folder.mkdir(parents=True, exist_ok=True)
    seq, _src = save_sequence(
        {"title": title, "scenes": []},
        folder=folder,
        force=True,
    )
    project = {
        "id": pid,
        "name": title[:80],
        "updatedAt": int(time.time() * 1000),
        "clips": [],
    }
    idx["projects"].append(project)
    idx["activeId"] = pid
    _write_projects_index(idx)
    return {"ok": True, "activeId": pid, "project": project, "sequence": seq}


def active_project_clip_names() -> List[str]:
    idx = load_projects_index()
    aid = idx.get("activeId") or ""
    for p in idx["projects"]:
        if p["id"] == aid:
            return list(p.get("clips") or [])
    return []


def attach_clip_to_active_project(name: str) -> List[str]:
    clip = filename_from_src(name)
    if not clip:
        return []
    idx = load_projects_index()
    if not idx["projects"]:
        return []
    aid = idx.get("activeId") or idx["projects"][0]["id"]
    for p in idx["projects"]:
        if p["id"] != aid:
            continue
        clips = list(p.get("clips") or [])
        if clip not in clips:
            clips.append(clip)
        p["clips"] = clips
        p["updatedAt"] = int(time.time() * 1000)
        idx["activeId"] = aid
        _write_projects_index(idx)
        return clips
    return []


def switch_project(pid: str) -> Dict[str, Any]:
    idx = load_projects_index()
    want = str(pid or "").strip()
    match = next((p for p in idx["projects"] if p["id"] == want), None)
    if not match:
        return {"ok": False, "activeId": idx.get("activeId") or ""}
    idx["activeId"] = match["id"]
    for p in idx["projects"]:
        if p["id"] == match["id"]:
            p["updatedAt"] = int(time.time() * 1000)
    _write_projects_index(idx)
    return {"ok": True, "activeId": match["id"]}


def plan_video(
    prompt: str,
    fmt: str = "documentary",
    target_seconds: float = 60,
    clip_names: Optional[List[str]] = None,
    language: str = "es",
) -> Dict[str, Any]:
    fmt = (fmt or "documentary").lower().strip()
    if fmt not in FORMATS:
        fmt = "documentary"
    target_seconds = max(15.0, min(240.0, float(target_seconds or 60)))
    library = list_library_clips()
    if clip_names:
        allow = set(clip_names)
        library = [c for c in library if c["name"] in allow] or library
    if not library:
        raise ValueError(
            "No hay clips en storage/clips. Busca footage, recórtalo en Edit y vuelve."
        )

    used_ai = False
    llm_error: Optional[BaseException] = None
    try:
        raw = _llm_plan(prompt, fmt, target_seconds, library, language)
        used_ai = True
    except Exception as e:
        llm_error = e
        print(f"Video Creation LLM fallback: {e}")
        raw = _heuristic_plan(prompt, fmt, target_seconds, library)

    plan = _sanitize_plan(raw, library, fmt)
    plan["ai"] = used_ai
    if not used_ai:
        if isinstance(llm_error, TimeoutError):
            plan["summary"] = "El director no respondió. Plan local con tus clips."
        else:
            plan["summary"] = (
                plan.get("summary")
                or "Plan local (sin XAI_API_KEY). Pon la clave en .env para un guion estilo InVideo."
            )
    return plan


def _compact_scenes(scenes: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for i, s in enumerate(scenes or []):
        if not isinstance(s, dict):
            continue
        out.append({
            "order": i + 1,
            "clip": s.get("clip") or "",
            "inPoint": s.get("inPoint"),
            "outPoint": s.get("outPoint"),
            "narration": s.get("narration") or "",
            "text": s.get("text") or "",
            "textPosition": s.get("textPosition") or "lower-third",
            "visual": s.get("visual") or "",
            "voiceover": bool(s.get("voiceover")),
            "voiceStyle": s.get("voiceStyle") or "documentary",
            "voiceEngine": sanitize_engine(s.get("voiceEngine") or "edge"),
            "voiceName": sanitize_voice_name(s.get("voiceName")),
            "voFx": sanitize_vo_fx(s.get("voFx")),
            "voRate": sanitize_vo_rate(s.get("voRate")),
            "capOff": _sanitize_cap_off(s.get("capOff")),
            "capPos": _sanitize_cap_pos(s.get("capPos")),
            "duckOriginal": bool(s.get("duckOriginal")),
            "grade": _normalize_grade(s.get("grade")),
            "muted": bool(s.get("muted")),
            "speed": _safe_float(s.get("speed"), 1.0, 0.25, 4.0),
        })
    return out[:40]


def _scene_index(msg: str, count: int) -> Optional[int]:
    m = re.search(r"(?:escena|scene)\s*#?\s*(\d+)", msg, re.I)
    if not m:
        return None
    n = int(m.group(1))
    if n < 1 or n > count:
        return None
    return n - 1


def _parse_seconds(msg: str) -> Optional[float]:
    m = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(?:s(?:eg(?:undos?)?)?|seconds?)\b",
        msg,
        re.I,
    )
    if not m:
        return None
    try:
        return max(0.4, float(m.group(1).replace(",", ".")))
    except ValueError:
        return None


def _director_media_command(msg: str) -> Optional[Dict[str, Any]]:
    """Captions (Instagram Edits) + VO ElevenLabs without rewriting the whole plan."""
    low = (msg or "").lower()
    captions: Dict[str, Any] = {}
    voice: Dict[str, Any] = {}
    bits: List[str] = []

    styles = [
        ("literature", r"literature|literari|serif"),
        ("handwritten", r"handwritten|manuscrit|escrito a mano"),
        ("playful", r"playful|jugueton|divertid"),
        ("bubble", r"\bbubble\b|burbuja"),
        ("modern", r"\bmodern"),
        ("retro", r"\bretro\b"),
        ("bold", r"\bbold\b|negrita"),
        ("classic", r"classic|cl[aá]sico"),
    ]
    if re.search(r"caption|subtit|subt[ií]tul|\bcc\b|sin fondo|fondo (de )?(los )?(caption|subtit)", low):
        if re.search(r"(apaga|quita|hide|oculta|off|sin captions|sin subtit)", low):
            captions["on"] = False
            bits.append("Captions apagados.")
        else:
            captions["on"] = True
            captions["generate"] = True
            bits.append("Genero captions.")
        if re.search(r"narraci|voiceover|\bvo\b|guion hablado", low):
            captions["source"] = "narration"
            bits.append("Origen: narración.")
        elif re.search(r"audio|original|del video|del v[ií]deo", low):
            captions["source"] = "original"
            bits.append("Origen: audio del clip.")
        for sid, pat in styles:
            if re.search(pat, low):
                captions["style"] = sid
                bits.append("Estilo " + sid + ".")
                break
        if re.search(r"lower.?third|tercio inferior", low):
            captions["position"] = "lower-third"
            bits.append("Posición lower-third.")
        elif re.search(r"\barriba\b|\btop\b", low):
            captions["position"] = "top"
            bits.append("Posición arriba.")
        elif re.search(r"\babajo\b|\bbottom\b", low):
            captions["position"] = "bottom"
            bits.append("Posición abajo.")
        elif re.search(r"centro|center", low):
            captions["position"] = "center"
            bits.append("Posición centro.")
        if re.search(r"sin fondo|quita (el )?fondo|sin highlight|sin caja|no background|fondo off", low):
            captions["bgOn"] = False
            captions["highlight"] = ""
            bits.append("Fondo de captions quitado.")
        elif re.search(r"fondo negro|black (box|bg)|highlight negro", low):
            captions["bgOn"] = True
            captions["highlight"] = "#000000"
            bits.append("Fondo negro.")
        elif re.search(r"amarillo|yellow", low) and re.search(r"highlight|fondo|caja|box", low):
            captions["bgOn"] = True
            captions["highlight"] = "#facc15"
            bits.append("Fondo amarillo.")
        elif re.search(r"fondo blanco|highlight blanco", low):
            captions["bgOn"] = True
            captions["highlight"] = "#ffffff"
            bits.append("Fondo blanco.")
        if re.search(r"sin puntu", low):
            captions["punctuation"] = False
            bits.append("Sin puntuación.")
        px = re.search(r"(\d{2})\s*px", low)
        if px:
            captions["size"] = max(22, min(96, int(px.group(1))))
            bits.append("Tamaño " + str(captions["size"]) + " px.")
        elif re.search(r"\bxl\b|enorme|muy grande", low):
            captions["size"] = 64
            bits.append("Tamaño XL.")
        elif re.search(r"letra grande|tama[nñ]o l\b|m[aá]s grande", low):
            captions["size"] = 48
            bits.append("Tamaño L.")
        elif re.search(r"letra peque|tama[nñ]o s\b|m[aá]s chiqu", low):
            captions["size"] = 28
            bits.append("Tamaño S.")
        elif re.search(r"tama[nñ]o m\b|mediano", low):
            captions["size"] = 36
            bits.append("Tamaño M.")
        if re.search(r"title case|tipo t[ií]tulo", low):
            captions["title_case"] = True
            bits.append("Title case.")
        if re.search(r"groser|palabrot|profan|malas palabras", low):
            captions["show_profanity"] = not bool(re.search(r"oculta|esconde|quita|sin |hide", low))
            bits.append("Groserías: " + ("sí" if captions["show_profanity"] else "ocultas") + ".")
        op = re.search(r"opacidad\s*(\d{1,3})", low)
        if op:
            captions["opacity"] = max(0.2, min(1.0, int(op.group(1)) / 100.0))
            bits.append("Opacidad " + op.group(1) + "%.")

    vo_all = re.search(r"(voiceover|voz en off|\bvo\b)", low) and re.search(
        r"(todas|all|every|en todas)", low
    )
    if vo_all or (re.search(r"(voiceover|voz en off|\bvo\b)", low) and not re.search(r"escena|scene", low)):
        if re.search(r"(quita|apaga|off|sin vo|desactiva)", low):
            voice["all"] = False
            bits.append("Voiceover apagada.")
        else:
            voice["all"] = True
            voice["duck"] = True
            bits.append("Voiceover ElevenLabs en todas las escenas.")
        for sid, pat in (
            ("energetic", r"en[eé]rgic"),
            ("warm", r"c[aá]lid|warm"),
            ("ad", r"\bad\b|anuncio"),
            ("calm", r"calm"),
            ("documentary", r"documental|documentary"),
        ):
            if re.search(pat, low):
                voice["style"] = sid
                bits.append("Voz " + sid + ".")
                break
        if re.search(r"solo voz|silencia el original|mute original", low):
            voice["muteOriginal"] = True
            bits.append("Audio original silenciado.")

    if not captions and not voice:
        return None
    return {
        "ai": True,
        "reply": " ".join(bits) or "Ajusté audio/captions.",
        "plan": None,
        "local": True,
        "captions": captions or None,
        "voice": voice or None,
    }


def _pretty_label(name: str) -> str:
    stem = Path(str(name or "")).stem
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", stem)).strip() or "Clip"


def _pack_director_plan(
    reply: str,
    current_plan: Dict[str, Any],
    clips: List[Dict[str, Any]],
    changed: List[Dict[str, Any]],
) -> Dict[str, Any]:
    raw = {
        "title": current_plan.get("title") or "Video",
        "summary": current_plan.get("summary") or "",
        "scenes": changed,
        "reply": reply,
    }
    plan = _sanitize_plan(raw, clips, current_plan.get("format") or "documentary")
    plan["ai"] = True
    return {"ai": True, "reply": reply, "plan": plan, "local": True}


def _cap_first_and_bridge_stills(
    current_plan: Dict[str, Any],
    clips: List[Dict[str, Any]],
    scenes: List[Dict[str, Any]],
    by_name: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """First 3 scenes ≤ 3s, short stock VIDEO between those cuts."""
    safe_clips = [c for c in (clips or []) if isinstance(c, dict) and c.get("name")]
    n = min(3, len(scenes))
    if n < 1:
        return {
            "ai": True,
            "reply": "No hay escenas para recortar. Añade clips y repite.",
            "plan": None,
            "local": True,
        }
    changed: List[Dict[str, Any]] = []
    for i, raw in enumerate(scenes):
        if not isinstance(raw, dict):
            continue
        scene = dict(raw)
        if i < n:
            in_p = max(0.0, float(scene.get("inPoint") or 0))
            meta = by_name.get(str(scene.get("clip") or ""))
            if not isinstance(meta, dict):
                meta = {}
            max_d = float(meta.get("duration") or 0)
            cap = 3.0
            if max_d > 0:
                cap = min(3.0, max(0.4, max_d - in_p))
            scene["outPoint"] = round(in_p + cap, 2)
        changed.append(scene)
    packed = _pack_director_plan(
        f"First {n} scenes are now 3 seconds.",
        current_plan if isinstance(current_plan, dict) else {},
        safe_clips,
        changed,
    )
    packed["search_stock"] = {"query": "documentary footage", "platform": "youtube"}
    packed["fetch_bridges"] = max(0, n - 1)
    packed["reply"] = (
        f"First {n} scenes capped at 3s. Pulling web footage from YouTube "
        "into Clips, then inserting it between those cuts…"
    )
    return packed


def _global_director_edit(
    message: str,
    current_plan: Dict[str, Any],
    clips: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Edits that do not name a scene number — applied without waiting for Grok."""
    msg = (message or "").strip()
    low = msg.lower()
    if not isinstance(current_plan, dict):
        return None
    scenes = [dict(s) for s in (current_plan.get("scenes") or []) if isinstance(s, dict)]
    if not scenes:
        return None
    by_name = {
        str(c.get("name") or ""): c
        for c in (clips or [])
        if isinstance(c, dict) and c.get("name")
    }

    def full_clip(scene: Dict[str, Any]) -> Dict[str, Any]:
        clip_name = scene.get("clip") or ""
        max_d = float((by_name.get(clip_name) or {}).get("duration") or 0)
        scene["inPoint"] = 0.0
        scene["outPoint"] = round(max_d if max_d > 0.4 else float(scene.get("outPoint") or 4), 2)
        return scene

    cap_first = re.search(
        r"(?:f?irst|primeras?|first three|1\s*[–-]\s*3)\s*3|(?:escenas?|scenes?)\s*(?:1\s*(?:,|y|and|&)\s*2\s*(?:,|y|and|&)\s*3|1\s*[–-]\s*3)",
        low,
    ) or re.search(r"f?irst\s*3\s*scenes|\b3\s*scenes", low)
    wants_3s = bool(re.search(r"(?:no longer than |máximo |max(?:imo)?\s*|a\s+)?3\s*(?:s|sec|seconds?|seg)", low))
    wants_bridge = bool(re.search(
        r"web footage|footage web|from the web|online footage|stock (?:image|video|clip)|imagen(?:es)? stock|between (?:the )?cuts|entre (?:los )?cortes|b-?roll|insert(?:a|ar)? stock",
        low,
    ))
    if (cap_first and wants_3s) or (wants_bridge and cap_first) or (wants_3s and wants_bridge):
        return _cap_first_and_bridge_stills(current_plan, clips, scenes, by_name)

    if re.search(r"(clip[s]?\s+enteros|usa cada clip|cada clip entero|full clips?|whole clips?)", low):
        return _pack_director_plan(
            "Cada escena usa el clip entero.",
            current_plan,
            clips,
            [full_clip(s) for s in scenes],
        )

    energetic = bool(re.search(r"en[eé]rgic|punch|impacto|din[aá]mic", low))
    intro = bool(re.search(r"intro|inicio|primera|hook|gancho", low))
    if energetic and scenes and (intro or not re.search(r"escena|scene", low)):
        first = dict(scenes[0])
        in_p = max(0.0, float(first.get("inPoint") or 0))
        out_p = float(first.get("outPoint") or (in_p + 4))
        if out_p - in_p > 6:
            first["outPoint"] = round(in_p + 6.0, 2)
        first["voiceStyle"] = "energetic"
        label = _pretty_label(first.get("clip") or "")
        if not (first.get("narration") or "").strip():
            first["narration"] = label
        if not (first.get("text") or "").strip():
            first["text"] = " ".join(label.split()[:6])
        scenes[0] = first
        if not intro:
            for i, s in enumerate(scenes):
                s["voiceStyle"] = "energetic"
                scenes[i] = s
        return _pack_director_plan(
            "Intro más enérgico: plano 1 más corto, estilo energetic y gancho en pantalla.",
            current_plan,
            clips,
            scenes,
        )

    if re.search(r"(cierre|cta|llamada a la acci[oó]n|call to action)", low):
        last = dict(scenes[-1])
        last["text"] = (last.get("text") or "").strip() or "Sígueme para más"
        last["narration"] = (last.get("narration") or "").strip() or "Si te gustó, sígueme para más."
        last["textPosition"] = "center"
        scenes[-1] = last
        return _pack_director_plan(
            "Cierre con llamada a la acción en la última escena.",
            current_plan,
            clips,
            scenes,
        )

    if re.search(r"(m[aá]s r[aá]pido|acelera|acorta (todo|el montaje|el video)|ritmo m[aá]s (alto|r[aá]pido))", low):
        changed = []
        for s in scenes:
            in_p = max(0.0, float(s.get("inPoint") or 0))
            out_p = float(s.get("outPoint") or (in_p + 4))
            dur = max(0.8, (out_p - in_p) * 0.7)
            s["outPoint"] = round(in_p + dur, 2)
            changed.append(s)
        return _pack_director_plan("Ritmo más rápido: recorté cada escena un 30%.", current_plan, clips, changed)

    if re.search(r"(m[aá]s lento|m[aá]s largo|alarga (todo|el montaje)|ritmo m[aá]s (bajo|lento))", low):
        return _pack_director_plan(
            "Ritmo más amplio: cada escena usa el clip entero.",
            current_plan,
            clips,
            [full_clip(s) for s in scenes],
        )

    if re.search(r"(reescribe|escribe).{0,20}(narraci|guion|voz)", low) or re.search(
        r"narraci[oó]n (de los clips|con los nombres|desde los clips)", low
    ):
        changed = []
        for s in scenes:
            label = _pretty_label(s.get("clip") or "")
            if not (s.get("narration") or "").strip():
                s["narration"] = label
            if not (s.get("text") or "").strip():
                s["text"] = " ".join(label.split()[:6])
            changed.append(s)
        return _pack_director_plan(
            "Narración y textos de pantalla a partir de cada clip.",
            current_plan,
            clips,
            changed,
        )

    if _wants_grade(low):
        changed = []
        notes = []
        for s in scenes:
            g, note = _mutate_grade(_normalize_grade(s.get("grade")), low)
            s["grade"] = g
            notes.append(note)
            changed.append(s)
        return _pack_director_plan(
            "Color en todas las escenas: " + (notes[0] if notes else "grade") + ".",
            current_plan,
            clips,
            changed,
        )

    return None


def apply_director_command(
    message: str,
    current_plan: Dict[str, Any],
    clips: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Ediciones directas (duración, borrar, mute, VO, captions) sin reescribir todo el JSON."""
    msg = (message or "").strip()
    if not msg:
        return None
    low = msg.lower().strip()
    if re.search(r"\b(undo|deshacer|deshaz|ctrl\s*\+?\s*z|vuelve\s+atr[aá]s|echar\s+atr[aá]s)\b", low) and not re.search(
        r"\b(escena|scene)\s*#?\s*\d+", low
    ):
        if re.search(r"\b(rehacer|redo)\b", low):
            return {"ai": True, "reply": "Rehacer el último cambio.", "plan": None, "redo": True, "local": True}
        return {"ai": True, "reply": "Deshacer el último cambio.", "plan": None, "undo": True, "local": True}
    if re.search(r"^\s*(rehacer|redo)\b", low):
        return {"ai": True, "reply": "Rehacer el último cambio.", "plan": None, "redo": True, "local": True}
    if _wants_load_clips(low):
        return _plan_from_library(current_plan if isinstance(current_plan, dict) else {}, clips)
    img = _image_search_reply(msg, current_plan if isinstance(current_plan, dict) else {})
    if img:
        return img
    scenes = [s for s in (current_plan.get("scenes") or []) if isinstance(s, dict)]
    if not scenes:
        return _director_media_command(msg)
    by_name = {
        str(c.get("name") or ""): c
        for c in (clips or [])
        if isinstance(c, dict) and c.get("name")
    }
    idx = _scene_index(msg, len(scenes))
    low = msg.lower()

    def pack(reply: str, changed: List[Dict[str, Any]]) -> Dict[str, Any]:
        raw = {
            "title": current_plan.get("title") or "Video",
            "summary": current_plan.get("summary") or "",
            "scenes": changed,
            "reply": reply,
        }
        plan = _sanitize_plan(raw, clips, current_plan.get("format") or "documentary")
        plan["ai"] = True
        return {"ai": True, "reply": reply, "plan": plan, "local": True}

    media = _director_media_command(msg)
    if idx is None:
        if media:
            return media
        global_edit = _global_director_edit(msg, current_plan, clips)
        if global_edit:
            return global_edit
        if re.search(r"(?:escena|scene)\s*#?\s*\d+", msg, re.I):
            n = re.search(r"(\d+)", msg)
            return {
                "ai": True,
                "reply": f"No hay escena {n.group(1) if n else '?'}. Hay {len(scenes)} escena(s).",
                "plan": None,
                "local": True,
            }
        return None

    scene = dict(scenes[idx])
    n = idx + 1
    clip_name = scene.get("clip") or ""
    max_d = _safe_float((by_name.get(clip_name) or {}).get("duration"), 0.0, 0.0, 3600.0) or None
    in_p = max(0.0, _safe_float(scene.get("inPoint"), 0.0, 0.0, 3600.0))
    out_p = _safe_float(scene.get("outPoint"), in_p + 4, 0.0, 3600.0)
    file_left = (max_d - in_p) if max_d else None

    secs = _parse_seconds(msg)
    wants_dur = bool(secs) and re.search(
        r"(reduc|acort|trim|cort|limit|duraci|deja|poner|pon\s|a\s+\d|to\s+\d|en\s+\d|alarg|extend|aument)",
        low,
    )
    if wants_dur and secs is not None:
        cap = file_left if file_left and file_left > 0.4 else secs
        use = min(secs, cap)
        scene["outPoint"] = round(in_p + use, 2)
        scenes[idx] = scene
        extra = ""
        if file_left is not None and secs > file_left + 0.05:
            extra = f" El clip solo tiene {max_d:.1f}s desde este punto, así que quedó en {use:.1f}s."
        return pack(f"Escena {n} ahora dura {use:.1f}s.{extra}", scenes)

    if re.search(r"(borra|elimina|quita|delete|remove)\s+(?:la\s+)?(?:escena|scene)", low):
        if len(scenes) < 2:
            return {"ai": True, "reply": "Deja al menos una escena.", "plan": None, "local": True}
        del scenes[idx]
        return pack(f"Escena {n} eliminada.", scenes)

    if re.search(r"(silenc|mute)", low):
        scene["muted"] = True
        scenes[idx] = scene
        return pack(f"Escena {n} silenciada.", scenes)

    if re.search(r"(voiceover|voz en off|\bvo\b)", low) and re.search(r"(activa|pon|añade|add|enable)", low):
        scene["voiceover"] = True
        scene["duckOriginal"] = True
        scenes[idx] = scene
        return pack(f"Voiceover ElevenLabs activada en la escena {n}.", scenes)

    if _wants_grade(low):
        g, note = _mutate_grade(_normalize_grade(scene.get("grade")), low)
        scene["grade"] = g
        scenes[idx] = scene
        return pack(f"Escena {n} color: {note}.", scenes)

    if media:
        return media
    return None


def _wants_director_chat(msg: str) -> bool:
    low = (msg or "").lower()
    if len(msg or "") > 70 or "?" in (msg or "") or "¿" in (msg or ""):
        return True
    return bool(re.search(
        r"busca|stock|footage|web footage|encuentra|necesito (video|clip|plano)|trae clips|"
        r"investiga|qu[eé] opinas|c[oó]mo|por qu[eé]|explica|aclara|sugiere|"
        r"\bidea\b|guion|gui[oó]n|audiencia|tono|plataforma|invideo|"
        r"premiere|resolve|da\s*vinci|jkl|ripple|razor|grade|etalon|"
        r"im[aá]genes?|images?|fotos?|photos?|stills?",
        low,
    ))


def chat_edit_plan(
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
    current_plan: Optional[Dict[str, Any]] = None,
    clip_names: Optional[List[str]] = None,
    fmt: str = "documentary",
    language: str = "es",
    idea: str = "",
    captions_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Chat estilo InVideo: corrige, reescribe o crea el guion sobre el plan actual."""
    msg = (message or "").strip()
    if not msg:
        raise ValueError("Escribe un mensaje para la IA")
    fmt = (fmt or "documentary").lower().strip()
    if fmt not in FORMATS:
        fmt = "documentary"
    library = list_library_clips()
    if clip_names:
        allow = set(clip_names)
        filtered = [c for c in library if c["name"] in allow]
        if filtered:
            library = filtered

    current = current_plan if isinstance(current_plan, dict) else {}
    try:
        if _wants_load_clips(msg.lower()) or library or (current.get("scenes")):
            local = apply_director_command(msg, current, library)
            if local:
                return local
    except Exception as e:
        print(f"director local fail: {e}")
        if _wants_load_clips((msg or "").lower()):
            try:
                return _plan_from_library(current, library)
            except Exception as e2:
                return {
                    "ai": False,
                    "reply": "No pude cargar los clips a escenas: " + str(e2)[:180],
                    "plan": None,
                    "local": True,
                }
    try:
        scenes_now = _compact_scenes(current.get("scenes"))
    except Exception:
        scenes_now = []
    catalog = [
        {"file": c["name"], "title": c.get("title"), "duration_sec": c.get("duration")}
        for c in library
    ]
    key = _xai_key()
    if not key:
        out = {
            "ai": False,
            "reply": "Falta XAI_API_KEY en .env. Sin eso no puedo dirigir el montaje. Usa Footage web a la izquierda.",
            "plan": None,
        }
        img = _image_search_reply(msg, current)
        if img:
            return img
        if re.search(r"busca|stock|footage|encuentra", msg, re.I):
            out["search_stock"] = {"query": msg[:120], "platform": "youtube"}
            out["reply"] = "Sin clave Grok, busco footage web con tu texto. Elige resultados a la izquierda y Añadir."
        return out

    hist = []
    for h in (history or [])[-16:]:
        if not isinstance(h, dict):
            continue
        role = (h.get("role") or "").strip()
        content = (h.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            hist.append({"role": role, "content": content[:1800]})

    system = (
        DIRECTOR_PLAYBOOK
        + f"\nHablas con el usuario en {language}. "
        "Si solo conversas o preguntas, omite scenes. "
        "Si cambias el montaje, devuelve el guion COMPLETO (todas las escenas). "
        "Si falta footage, llena search_stock (vídeo YouTube/Dailymotion) o search_images (fotos web). No inventes filenames. "
        "Devuelves SOLO JSON válido, sin markdown."
    )
    payload = {
        "idea_original": idea or current.get("summary") or "",
        "format": fmt,
        "format_notes": FORMATS.get(fmt, ""),
        "library_empty": not bool(catalog),
        "plan_actual": {
            "title": current.get("title") or "",
            "summary": current.get("summary") or "",
            "scenes": scenes_now,
        },
        "library": catalog,
        "pedido": msg,
        "captions_actuales": captions_state or {},
        "json_schema": {
            "reply": "string (respuesta al usuario, natural)",
            "questions": ["pregunta de aclaración, opcional"],
            "search_stock": {
                "query": "string de búsqueda de footage",
                "platform": "youtube|dailymotion|all",
            },
            "search_images": {
                "query": "string de búsqueda de imágenes",
            },
            "title": "string",
            "summary": "string",
            "captions": {
                "on": "bool",
                "style": "classic|bold|retro|playful|handwritten|bubble|modern|literature",
                "position": "lower-third|center|top|bottom",
                "color": "#hex",
                "highlight": "#hex or empty",
                "opacity": "0-1",
                "source": "original|narration",
                "generate": "bool",
                "punctuation": "bool",
                "title_case": "bool",
                "show_profanity": "bool",
            },
            "voice": {
                "all": "bool",
                "style": "documentary|warm|energetic|ad|calm",
                "duck": "bool",
                "muteOriginal": "bool",
            },
            "scenes": [
                {
                    "order": "int",
                    "narration": "string",
                    "visual": "string (cámara + acción Seedance)",
                    "clip": "filename exacto de library",
                    "inPoint": "float",
                    "outPoint": "float",
                    "text": "string overlay",
                    "textPosition": "center|top|bottom|lower-third",
                    "voiceover": "bool",
                    "voiceStyle": "documentary|warm|energetic|ad|calm",
                    "duckOriginal": "bool",
                    "grade": {
                        "lift": "float",
                        "gamma": "float",
                        "gain": "float",
                        "sat": "float",
                        "temp": "float",
                        "contrast": "float",
                    },
                }
            ],
        },
    }
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url="https://api.x.ai/v1", timeout=45.0)
    messages = [{"role": "system", "content": system}]
    messages.extend(hist)
    messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("XAI_MODEL", "grok-4.6"),
            temperature=0.55,
            messages=messages,
            timeout=45.0,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        fallback = _global_director_edit(msg, current, library)
        if fallback:
            fallback["reply"] = (fallback.get("reply") or "Listo.") + " (Grok no respondió a tiempo; apliqué el cambio en local.)"
            return fallback
        return {
            "ai": False,
            "reply": (
                "El director tardó demasiado. Prueba una orden concreta: "
                "«escena 1 a 6 segundos», «usa cada clip entero», «genera captions», «intro más enérgico»."
            ),
            "plan": None,
            "error": str(e)[:200],
        }
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return {"ai": True, "reply": "La IA no devolvió respuesta. Prueba de nuevo.", "plan": None}
    text = (choices[0].message.content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        raw = json.loads(text)
    except Exception:
        return {"ai": True, "reply": (text or "Listo.")[:2000], "plan": None}
    if not isinstance(raw, dict):
        return {"ai": True, "reply": str(text)[:2000], "plan": None}
    reply = str(raw.get("reply") or "Listo.").strip()[:2000]
    qs = raw.get("questions") or raw.get("ask")
    if isinstance(qs, str) and qs.strip():
        qs = [qs.strip()]
    if isinstance(qs, list):
        qs = [str(q).strip() for q in qs if str(q).strip()][:3]
        if qs:
            reply = (reply + ("\n" if reply else "") + " ".join("→ " + q for q in qs)).strip()
    plan = None
    if raw.get("scenes") and library:
        try:
            plan = _sanitize_plan(raw, library, fmt)
            plan["ai"] = True
        except Exception as e:
            reply = (reply + f"\n(No pude aplicar el guion: {e})").strip()
            try:
                if _wants_load_clips((msg or "").lower()):
                    return _plan_from_library(current, library)
            except Exception:
                pass
    out = {"ai": True, "reply": reply, "plan": plan}
    if isinstance(raw.get("captions"), dict):
        out["captions"] = raw["captions"]
    if isinstance(raw.get("voice"), dict):
        out["voice"] = raw["voice"]
    stock = raw.get("search_stock")
    if isinstance(stock, dict) and (stock.get("query") or "").strip():
        plat = str(stock.get("platform") or "youtube").lower()
        if plat not in {"youtube", "dailymotion", "all"}:
            plat = "youtube"
        out["search_stock"] = {"query": str(stock.get("query")).strip()[:120], "platform": plat}
    elif isinstance(stock, str) and stock.strip():
        out["search_stock"] = {"query": stock.strip()[:120], "platform": "youtube"}
    images = raw.get("search_images")
    if isinstance(images, dict) and (images.get("query") or "").strip():
        out["search_images"] = {"query": str(images.get("query")).strip()[:120]}
    elif isinstance(images, str) and images.strip():
        out["search_images"] = {"query": images.strip()[:120]}
    return out


def grok_caption_cues(
    idea: str,
    language: str,
    scenes: List[Dict[str, Any]],
    duration: float,
    transcript: str = "",
) -> List[Dict[str, Any]]:
    """Timed spoken captions. Never the clip filename."""
    key = _xai_key()
    if not key:
        return []
    catalog = []
    for i, s in enumerate(scenes or []):
        catalog.append({
            "order": i + 1,
            "timelineStart": round(float(s.get("timelineStart") or 0), 2),
            "duration": round(max(0.4, float(s.get("outPoint") or 0) - float(s.get("inPoint") or 0)), 2),
            "narration": (s.get("narration") or "")[:400],
            "what_is_on_screen": (s.get("text") or s.get("visual") or "")[:200],
            "clip_hint": Path(str(s.get("clip") or s.get("src") or "")).stem.replace("_", " ")[:80],
        })
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url="https://api.x.ai/v1", timeout=22.0)
    payload = {
        "language": language or "es",
        "idea": idea or "",
        "duration_sec": round(float(duration or 0), 2),
        "script_or_transcript": (transcript or "")[:6000],
        "scenes": catalog,
        "rules": [
            "Devuelve SOLO JSON {cues:[{start,end,text}]}",
            "start/end en segundos absolutos del vídeo (0 = inicio del montaje)",
            "text = lo que se LEE en pantalla, estilo Instagram Edits: 3 a 6 palabras",
            "NUNCA pongas el nombre del archivo ni el título del clip como caption",
            "Usa el transcript/guion si existe; si no, escribe frases habladas sobre lo que se ve",
            "Cubre todo el duration_sec, sin huecos largos",
        ],
    }
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("XAI_MODEL", "grok-4.6"),
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres el motor de captions de Instagram Edits. "
                        "Frases cortas, tiempos reales. JSON puro, sin markdown."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            timeout=22.0,
            response_format={"type": "json_object"},
        )
        text = ((resp.choices or [None])[0].message.content or "").strip() if resp.choices else ""
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        raw = json.loads(text)
        items = raw.get("cues") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            return []
        cues = []
        for c in items:
            if not isinstance(c, dict):
                continue
            body = str(c.get("text") or "").strip()
            if not body:
                continue
            try:
                a = float(c.get("start") or 0)
                b = float(c.get("end") or (a + 1.5))
            except (TypeError, ValueError):
                continue
            if b <= a:
                b = a + 1.2
            cues.append({"start": round(a, 2), "end": round(b, 2), "text": body})
        return cues
    except Exception as e:
        print(f"grok captions fail: {e}")
        return []
