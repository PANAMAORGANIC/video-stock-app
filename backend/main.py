"""
Clearview — personal footage NLE (Library → Edit → Assemble).
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
from urllib.parse import unquote
import asyncio
import json
import shutil
import uuid
import uvicorn

from search_youtube import get_transcript_with_timestamps
from search_platforms import search_footage_videos, search_footage_pack
from search_images import (
    search_web_images,
    import_web_image,
    import_local_image,
    classify_upload_name,
    image_ext_for_upload,
    IMAGE_UPLOAD_EXT,
    AUDIO_UPLOAD_EXT,
)
from instagram_account import (
    whoami as ig_whoami,
    save_sessionid,
    save_cookies_file,
    clear_session as ig_clear_session,
    import_from_browser as ig_import_from_browser,
    load_cookie_map,
)
from clip_processor import suggest_segments
from video_tools import (
    download_video_segment,
    check_ffmpeg,
    ffmpeg_bin,
    probe_duration,
    render_editor_export,
    safe_filename,
    unique_path,
    strip_captions,
    publish_edited_clip,
    delogo_region,
    load_clip_peaks,
    move_peaks_sidecar,
    delete_peaks_sidecar,
    ensure_clip_poster,
    poster_sidecar,
    poster_is_fresh,
    move_poster_sidecars,
    delete_poster_sidecars,
    ensure_clip_proxy,
    proxy_needed,
    proxy_path,
    proxy_is_fresh,
    move_clip_proxy,
    delete_clip_proxy,
    PROXIES_DIR,
    CLIPS_DIR,
    UPLOADS_DIR,
)
from video_create import (
    plan_video,
    chat_edit_plan,
    _xai_key,
    grok_caption_cues,
    load_sequence,
    save_sequence,
    sanitize_sequence,
    sequence_as_assemble,
    empty_sequence,
    persist_sequence_vo,
    persist_sequence_music,
    clear_sequence_music,
    sequence_vo_path,
    sequence_music_path,
    sequence_music_exists,
    resolve_vo_source,
    SEQUENCE_VO_URL,
    SEQUENCE_MUSIC_URL,
)
from eleven_audio import (
    eleven_key,
    list_voices,
    pick_voice,
    transcribe_audio,
    words_to_cues,
    apply_transcript_controls,
    extract_audio,
    cues_from_clip_scripts,
    parse_timestamped_transcript,
    cues_from_untimed_script,
    looks_like_timestamped,
    format_cues_transcript,
)
from tts_audio import synthesize_voiceover, tts_status, pick_edge_voice, resolve_engine, sanitize_vo_rate, sanitize_script
from hardsub_ai import remove_hardsubs, remove_overlays

app = FastAPI(
    title="Clearview",
    description="Clearview — library, edit, assemble. Cut like Premiere, grade like Resolve.",
    version="clearview"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
REFS_DIR = STORAGE_DIR / "references"
FRONTEND_DIR = BASE_DIR / "frontend"
TEMP_DIR = STORAGE_DIR / "temp"

for d in [CLIPS_DIR, REFS_DIR, TEMP_DIR, UPLOADS_DIR, PROXIES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/clips", StaticFiles(directory=str(CLIPS_DIR)), name="clips")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/proxies", StaticFiles(directory=str(PROXIES_DIR)), name="proxies")

ALLOWED_MEDIA_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


# ─── Models ───────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    max_results: int = 8
    platform: str = "all"  # youtube | dailymotion | tiktok | instagram | all


class SegmentSuggestion(BaseModel):
    start: float
    end: float
    text: str
    confidence: float
    reason: str


class VideoResult(BaseModel):
    video_id: str
    title: str
    url: str
    thumbnail: str
    duration: Optional[str] = None
    channel: Optional[str] = None
    platform: str = "youtube"
    suggested_segments: List[SegmentSuggestion] = []


class ImageSearchRequest(BaseModel):
    query: str
    max_results: int = 8


class ImageImportRequest(BaseModel):
    url: str
    title: str = ""
    seconds: float = 4.0


class SaveClipRequest(BaseModel):
    video_id: str
    video_url: str
    title: str
    start: float
    end: float
    custom_name: str
    remove_soft_subs: bool = True


class CropBox(BaseModel):
    x: float = 0
    y: float = 0
    w: float = 1
    h: float = 1


class ExportClip(BaseModel):
    src: str
    inPoint: float = Field(..., ge=0)
    outPoint: float
    timelineStart: float = 0
    duration: Optional[float] = None
    speed: float = 1
    volume: float = 1
    muted: bool = False
    rotation: int = 0
    filter: str = "none"
    freeze: bool = False
    fadeIn: bool = False
    fadeOut: bool = False
    crop: Optional[CropBox] = None
    voiceover: bool = False
    narration: str = ""
    voiceStyle: str = "documentary"
    duckOriginal: bool = False
    voiceEngine: str = "edge"
    voiceName: str = ""
    voFx: str = "none"
    voRate: float = 1.0
    grade: Optional[dict] = None


class ExportText(BaseModel):
    content: str
    size: int = 42
    position: str = "center"
    start: float = 0
    end: float = 4
    color: str = "white"
    highlight: str = ""
    opacity: float = 1
    boxOpacity: float = 0.85
    font: str = "sans"


class ExportRequest(BaseModel):
    name: str
    resolution: str = "720"
    clips: List[ExportClip]
    texts: List[ExportText] = []
    script: str = ""


class CreatePlanRequest(BaseModel):
    prompt: str
    format: str = "documentary"
    target_seconds: float = 60
    clips: List[str] = []
    language: str = "es"


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatScene(BaseModel):
    model_config = {"extra": "allow"}

    clip: str = ""
    inPoint: float = 0
    outPoint: float = 0
    narration: str = ""
    text: str = ""
    textPosition: str = "lower-third"
    visual: str = ""
    voiceover: bool = False
    voiceStyle: str = "documentary"
    duckOriginal: bool = False
    muted: bool = False
    speed: float = 1
    rotation: int = 0
    grade: Optional[dict] = None


class ChatPlanState(BaseModel):
    title: str = ""
    summary: str = ""
    scenes: List[ChatScene] = []


class CreateChatRequest(BaseModel):
    message: str
    history: List[ChatTurn] = []
    plan: Optional[ChatPlanState] = None
    clips: List[str] = []
    format: str = "documentary"
    language: str = "es"
    idea: str = ""
    captions: Optional[dict] = None


# ─── Routes ───────────────────────────────────────────────

NO_STORE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
}


def html_page(path: Path, missing: str, extra_head: str = "") -> HTMLResponse:
    if path.exists():
        content = path.read_text(encoding="utf-8")
        if extra_head:
            content = content.replace("</head>", extra_head + "\n</head>", 1)
        return HTMLResponse(content=content, headers=NO_STORE)
    return HTMLResponse(missing)


INCOMING_PATH = STORAGE_DIR / "autosave" / "incoming.json"
_CLIP_DUR_CACHE: dict = {}


def cached_clip_duration(path: Path):
    try:
        key = (str(path), path.stat().st_mtime, path.stat().st_size)
    except OSError:
        return None
    if key in _CLIP_DUR_CACHE:
        return _CLIP_DUR_CACHE[key]
    dur = probe_duration(path)
    _CLIP_DUR_CACHE[key] = dur
    return dur


def read_incoming_clips():
    if not INCOMING_PATH.is_file():
        return []
    try:
        data = json.loads(INCOMING_PATH.read_text(encoding="utf-8"))
        return [n for n in (data.get("names") or []) if isinstance(n, str) and n.strip()]
    except Exception:
        return []


def write_incoming_clips(names):
    folder = STORAGE_DIR / "autosave"
    folder.mkdir(parents=True, exist_ok=True)
    clean = [n for n in names if n]
    INCOMING_PATH.write_text(
        json.dumps({"names": clean}, ensure_ascii=False),
        encoding="utf-8",
    )


def load_create_autosave():
    """Return the last montaje as saved, without reordering scenes.

    A later 1-clip send is ignored if an older copy has a full board.
    """
    folder = STORAGE_DIR / "autosave"
    candidates = []
    for name in ("video-creation.json", "video-creation.bak.json"):
        path = folder / name
        if path.is_file() and path.stat().st_size > 20:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                scenes = ((data.get("plan") or {}).get("scenes")) or []
                if scenes:
                    candidates.append((len(scenes), int(data.get("savedAt") or 0), data, name))
            except Exception:
                continue
    rec = STORAGE_DIR / "pvs-video-creation-v1.recovered.json"
    if rec.is_file():
        try:
            data = json.loads(rec.read_text(encoding="utf-8"))
            scenes = ((data.get("plan") or {}).get("scenes")) or []
            if scenes:
                candidates.append((len(scenes), int(data.get("savedAt") or 0), data, rec.name))
        except Exception:
            pass
    if not candidates:
        return None, None
    max_n = max(c[0] for c in candidates)
    threshold = max(2, int(max_n * 0.5))
    substantial = [c for c in candidates if c[0] >= threshold] or candidates
    _n, _ts, base, source = max(substantial, key=lambda x: x[1])
    return base, source


@app.get("/", response_class=HTMLResponse)
async def home():
    return html_page(FRONTEND_DIR / "index.html", "<h1>Video Stock App</h1>")


@app.get("/review", response_class=HTMLResponse)
async def review_page():
    return html_page(FRONTEND_DIR / "review.html", "<h1>Review page missing</h1>")


@app.get("/editor", response_class=HTMLResponse)
@app.get("/editor/", response_class=HTMLResponse)
@app.get("/clip-edition", response_class=HTMLResponse)
@app.get("/clip-edition/", response_class=HTMLResponse)
async def editor_page():
    return html_page(FRONTEND_DIR / "editor" / "index.html", "<h1>Edit page missing</h1>")


@app.get("/create", response_class=HTMLResponse)
@app.get("/create/", response_class=HTMLResponse)
@app.get("/video-creation", response_class=HTMLResponse)
async def create_page(request: Request):
    names = []
    q = request.query_params.get("clips") or ""
    if q:
        names.extend([unquote(s.strip()) for s in q.split(",") if s.strip()])
    for n in read_incoming_clips():
        if n not in names:
            names.append(n)
    extra = ""
    if names:
        blob = json.dumps(names, ensure_ascii=False).replace("<", "\\u003c")
        extra = "<script>window.__PVS_INCOMING=" + blob + ";</script>"
    return html_page(
        FRONTEND_DIR / "create" / "index.html",
        "<h1>Assemble page missing</h1>",
        extra,
    )


@app.post("/api/search")
async def search_footage(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")

    platform = (req.platform or "all").lower().strip()
    allowed = {"youtube", "dailymotion", "tiktok", "instagram", "all"}
    if platform not in allowed:
        raise HTTPException(status_code=400, detail=f"Plataforma no soportada: {platform}")

    videos, platform_warnings = await asyncio.to_thread(
        search_footage_pack, req.query, platform, req.max_results
    )
    results = []

    for v in videos or []:
        if not isinstance(v, dict) or not v.get("url"):
            continue
        segments = []
        plat = v.get("platform") or "youtube"
        if plat == "youtube" and v.get("video_id"):
            try:
                transcript = get_transcript_with_timestamps(v["video_id"])
                if transcript:
                    segments = suggest_segments(req.query, transcript)
            except Exception as e:
                print(f"Error transcript {v.get('video_id')}: {e}")

        results.append(VideoResult(
            video_id=str(v.get("video_id") or ""),
            title=v.get("title") or "Sin título",
            url=v.get("url") or "",
            thumbnail=v.get("thumbnail") or "",
            duration=v.get("duration"),
            channel=v.get("channel"),
            platform=plat,
            suggested_segments=segments
        ))

    warning = " · ".join(w for w in (platform_warnings or []) if w)
    ig_on = "sessionid" in load_cookie_map()
    if not warning:
        if platform == "instagram" and not results:
            if ig_on:
                warning = (
                    "Con tu cuenta no hubo reels para esa búsqueda. "
                    "Prueba un hashtag, un @usuario o pega la URL del reel."
                )
            else:
                warning = (
                    "Instagram no busca sin tu cuenta. Pulsa “Conectar Instagram”, "
                    "o pega la URL del reel en la caja de búsqueda."
                )
        elif platform == "all" and not any(r.platform == "instagram" for r in results):
            warning = (
                "Instagram no devolvió resultados. Conecta tu cuenta o pega una URL de reel."
            )

    return {
        "query": req.query,
        "results": results,
        "platform": platform,
        "warning": warning or None,
    }


class IgSessionRequest(BaseModel):
    sessionid: str
    csrftoken: Optional[str] = None


class IgBrowserRequest(BaseModel):
    browser: str = "chrome"


@app.get("/api/instagram/status")
async def instagram_status():
    return ig_whoami()


@app.post("/api/instagram/session")
async def instagram_session(req: IgSessionRequest):
    try:
        save_sessionid(req.sessionid, req.csrftoken)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ig_whoami()


@app.post("/api/instagram/cookies")
async def instagram_cookies(file: UploadFile = File(...)):
    raw = (await file.read()).decode("utf-8", errors="replace")
    try:
        save_cookies_file(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ig_whoami()


@app.post("/api/instagram/from-browser")
async def instagram_from_browser(req: IgBrowserRequest):
    ok, msg = await asyncio.to_thread(ig_import_from_browser, req.browser)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    info = ig_whoami()
    if not info.get("message"):
        info["message"] = msg
    return info


@app.delete("/api/instagram/session")
async def instagram_logout():
    ig_clear_session()
    return {"connected": False, "username": None, "message": "Sesión de Instagram eliminada."}


@app.post("/api/save-clip")
async def save_clip(req: SaveClipRequest):
    """
    Descarga el segmento, quita subtítulos suaves si se pide,
    y lo guarda con el nombre personalizado.
    """
    if not check_ffmpeg():
        raise HTTPException(status_code=500, detail="FFmpeg no está disponible en el servidor")

    if req.end <= req.start:
        raise HTTPException(status_code=400, detail="El tiempo final debe ser mayor que el inicial")

    if not req.custom_name.strip():
        raise HTTPException(status_code=400, detail="Debes poner un nombre al clip")

    success, message, path = download_video_segment(
        video_url=req.video_url,
        start=req.start,
        end=req.end,
        output_name=req.custom_name,
        remove_soft_subs=req.remove_soft_subs,
    )

    if not success:
        raise HTTPException(status_code=500, detail=message)

    return {
        "success": True,
        "message": message,
        "filename": path.name if path else None,
        "path": f"/clips/{path.name}" if path else None,
        "start": req.start,
        "end": req.end,
    }


def _parse_duration_sec(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    raw = str(value).strip()
    if not raw:
        return None
    parts = raw.replace(",", ".").split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (TypeError, ValueError):
        return None


@app.post("/api/stock/search")
async def stock_search(req: SearchRequest):
    """Búsqueda rápida de footage web (YouTube/Dailymotion, sin transcripciones)."""
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Escribe qué footage web necesitas")
    platform = (req.platform or "youtube").lower().strip()
    n = max(4, min(12, int(req.max_results or 8)))
    if platform in {"images", "image", "photos", "photo", "fotos"}:
        packed = await asyncio.to_thread(search_web_images, q, n)
        return {
            "query": q,
            "platform": "images",
            "kind": "image",
            "results": packed.get("results") or [],
            "source": packed.get("source"),
            "error": packed.get("error"),
        }
    if platform not in {"youtube", "dailymotion", "tiktok", "instagram", "all"}:
        platform = "youtube"
    videos = await asyncio.to_thread(search_footage_videos, q, platform, n)
    out = []
    for v in videos:
        dur = _parse_duration_sec(v.get("duration"))
        out.append({
            "video_id": v.get("video_id") or "",
            "title": v.get("title") or "Sin título",
            "url": v.get("url") or "",
            "thumbnail": v.get("thumbnail") or "",
            "duration": v.get("duration"),
            "duration_sec": dur,
            "channel": v.get("channel"),
            "platform": v.get("platform") or "youtube",
        })
    return {"query": q, "platform": platform, "results": out}


@app.post("/api/stock/import")
async def stock_import(req: SaveClipRequest):
    """Descarga un segmento de footage web (YouTube/Dailymotion) a Clips."""
    if not check_ffmpeg():
        raise HTTPException(status_code=500, detail="FFmpeg no está disponible")
    url = (req.video_url or "").strip()
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="URL de footage web no válida")
    start = max(0.0, float(req.start or 0))
    end = float(req.end or 0)
    if start <= 0.05:
        start = 3.0
    if end <= start:
        end = start + 6.0
    end = min(end, start + 12.0)
    name = (req.custom_name or req.title or "stock").strip()
    success, message, path = await asyncio.to_thread(
        download_video_segment,
        url,
        start,
        end,
        name,
        True,
    )
    if not success:
        raise HTTPException(status_code=500, detail=message)
    duration = probe_duration(path) if path else None
    return {
        "ok": True,
        "message": message,
        "filename": path.name if path else None,
        "path": f"/clips/{path.name}" if path else None,
        "duration": round(duration, 2) if duration else None,
        "start": start,
        "end": end,
    }


@app.post("/api/images/search")
async def images_search(req: ImageSearchRequest):
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Escribe qué imagen necesitas")
    n = max(4, min(12, int(req.max_results or 12)))
    packed = await asyncio.to_thread(search_web_images, q, n)
    body = {
        "query": q,
        "kind": "image",
        "results": packed.get("results") or [],
        "source": packed.get("source"),
        "error": packed.get("error"),
    }
    if body["results"]:
        return body
    if packed.get("hard_fail"):
        raise HTTPException(
            status_code=502,
            detail=packed.get("error") or "Commons y Openverse fallaron.",
        )
    return body


@app.post("/api/images/import")
async def images_import(req: ImageImportRequest):
    if not check_ffmpeg():
        raise HTTPException(status_code=500, detail="FFmpeg no está disponible")
    url = (req.url or "").strip()
    seconds = max(1.0, min(12.0, float(req.seconds or 4)))
    ok, message, path = await asyncio.to_thread(
        import_web_image, url, req.title or "imagen", seconds
    )
    if not ok:
        raise HTTPException(status_code=500, detail=message)
    duration = probe_duration(path) if path else seconds
    return {
        "ok": True,
        "message": message,
        "filename": path.name if path else None,
        "path": f"/clips/{path.name}" if path else None,
        "duration": round(duration, 2) if duration else seconds,
        "kind": "image",
    }


@app.get("/api/clips")
async def list_clips():
    """Lista los clips ya guardados en storage/clips (incluye exports)."""
    clips = []
    files = [
        f for f in CLIPS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in ALLOWED_MEDIA_EXT
    ]
    for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
        duration = cached_clip_duration(f)
        clips.append({
            "name": f.name,
            "title": f.stem,
            "url": f"/clips/{f.name}",
            "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            "duration": round(duration, 2) if duration else None,
        })
    return {"clips": clips}


@app.get("/api/clips/peaks")
async def clip_peaks(name: str = ""):
    """Small peak array for the Edit A1 waveform. Cached next to the media file."""
    safe = Path(name or "").name
    if not safe or safe in {".", ".."}:
        raise HTTPException(status_code=400, detail="Falta el nombre del clip")
    path = None
    for folder in (CLIPS_DIR, UPLOADS_DIR):
        cand = (folder / safe)
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved.parent != folder.resolve():
            continue
        if resolved.is_file():
            path = resolved
            break
    if path is None:
        raise HTTPException(status_code=404, detail=f"No se encuentra: {safe}")
    data = await asyncio.to_thread(load_clip_peaks, path)
    return {
        "name": path.name,
        "peaks": data.get("peaks") or [],
        "hasAudio": bool(data.get("hasAudio")),
        "duration": float(data.get("duration") or 0),
    }


def _resolve_named_media(name: str) -> Path:
    safe = Path(name or "").name
    if not safe or safe in {".", ".."}:
        raise HTTPException(status_code=400, detail="Falta el nombre del clip")
    for folder in (CLIPS_DIR, UPLOADS_DIR):
        cand = folder / safe
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved.parent != folder.resolve():
            continue
        if resolved.is_file():
            return resolved
    raise HTTPException(status_code=404, detail=f"No se encuentra: {safe}")


@app.get("/api/clips/poster")
async def clip_poster(name: str = "", t: float = 0.0, make: int = 0):
    """One JPEG thumb at t (or 0). Cached next to the media like peaks. make=1 generates."""
    path = _resolve_named_media(name)
    dest = poster_sidecar(path, t)
    if poster_is_fresh(path, dest):
        return FileResponse(str(dest), media_type="image/jpeg")
    if not make:
        raise HTTPException(status_code=404, detail="Sin poster")
    made = await asyncio.to_thread(ensure_clip_poster, path, t)
    if not made or not made.is_file():
        raise HTTPException(status_code=404, detail="No se pudo generar el poster")
    return FileResponse(str(made), media_type="image/jpeg")


def _proxy_info(path: Path) -> dict:
    dest = proxy_path(path.name)
    if not proxy_needed(path):
        return {"ready": False, "skipped": True, "url": None, "name": path.name}
    if proxy_is_fresh(path, dest):
        return {"ready": True, "skipped": False, "url": f"/proxies/{path.name}", "name": path.name}
    return {"ready": False, "skipped": False, "url": None, "name": path.name}


@app.get("/api/clips/proxy")
async def clip_proxy_status(name: str = ""):
    """Preview proxy status. Original stays in storage/clips; this is optional 540p."""
    path = _resolve_named_media(name)
    return _proxy_info(path)


@app.post("/api/clips/proxy")
async def clip_proxy_make(background_tasks: BackgroundTasks, name: str = ""):
    path = _resolve_named_media(name)
    info = _proxy_info(path)
    if info["skipped"] or info["ready"]:
        return info

    def _job(p: Path) -> None:
        ensure_clip_proxy(p)

    background_tasks.add_task(_job, path)
    info["started"] = True
    return info


class ClipNameRequest(BaseModel):
    name: str
    new_name: Optional[str] = None


class StripCaptionsRequest(BaseModel):
    src: str
    inPoint: float = 0
    outPoint: Optional[float] = None
    mode: str = "hardsubs"  # hardsubs | watermark | both
    x: Optional[float] = None
    y: Optional[float] = None
    w: Optional[float] = None
    h: Optional[float] = None
    normalized: bool = True


class DelogoRequest(BaseModel):
    src: str
    inPoint: float = 0
    outPoint: Optional[float] = None
    x: float
    y: float
    w: float
    h: float
    normalized: bool = True


class SendToCreateItem(BaseModel):
    src: str
    inPoint: float = 0
    outPoint: float
    name: str = ""
    rotation: int = 0
    filter: str = "none"
    crop: Optional[CropBox] = None
    grade: Optional[dict] = None
    speed: float = 1
    muted: bool = False
    freeze: bool = False
    duration: Optional[float] = None
    fadeIn: bool = False
    fadeOut: bool = False
    volume: float = 1
    voiceover: bool = False
    duckOriginal: bool = False
    id: str = ""
    text: str = ""
    textPosition: str = "lower-third"


class SendToCreateRequest(BaseModel):
    clips: List[SendToCreateItem] = []
    texts: List[dict] = []
    markers: List[dict] = []
    title: str = ""
    force: bool = False


def _clip_file(name: str) -> Path:
    safe = Path(name).name
    if Path(safe).suffix.lower() not in ALLOWED_MEDIA_EXT:
        raise HTTPException(status_code=400, detail="Archivo no válido")
    path = (CLIPS_DIR / safe).resolve()
    if path.parent != CLIPS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Ruta no permitida")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No se encuentra: {safe}")
    return path


@app.post("/api/clips/rename")
async def rename_clip(req: ClipNameRequest):
    if not req.new_name or not req.new_name.strip():
        raise HTTPException(status_code=400, detail="Pon el nombre nuevo")
    src = _clip_file(req.name)
    stem = safe_filename(Path(req.new_name.strip()).stem)
    ext = src.suffix.lower()
    dest = CLIPS_DIR / f"{stem}{ext}"
    if dest.resolve() == src.resolve():
        return {"name": src.name, "url": f"/clips/{src.name}", "title": src.stem}
    if dest.exists():
        dest = unique_path(CLIPS_DIR, stem, ext)
    src.rename(dest)
    move_peaks_sidecar(src, dest)
    move_poster_sidecars(src, dest)
    move_clip_proxy(src, dest)
    return {"name": dest.name, "url": f"/clips/{dest.name}", "title": dest.stem, "old_name": src.name}


@app.post("/api/clips/delete")
async def delete_clip(req: ClipNameRequest):
    path = _clip_file(req.name)
    delete_peaks_sidecar(path)
    delete_poster_sidecars(path)
    delete_clip_proxy(path)
    path.unlink()
    return {"ok": True, "deleted": path.name}


@app.post("/api/strip-captions")
async def api_strip_captions(req: StripCaptionsRequest):
    """Quita pistas de subtítulos / closed captions del archivo en disco."""
    if not check_ffmpeg():
        raise HTTPException(status_code=500, detail="FFmpeg no está disponible")
    path = _resolve_media_src(req.src)
    ok, message, out = await asyncio.to_thread(strip_captions, path, None, True)
    if not ok:
        raise HTTPException(status_code=500, detail=message)
    return {
        "ok": True,
        "message": message,
        "url": req.src.split("?")[0],
        "filename": (out.name if out else path.name),
    }


@app.post("/api/send-to-create")
async def send_to_create(req: SendToCreateRequest):
    """Open the current Edit cuts as the shared sequence in Assemble. Does not bake MP4s."""
    if not req.clips:
        raise HTTPException(status_code=400, detail="No hay clips para enviar")
    scenes = []
    for item in req.clips:
        if item.outPoint <= item.inPoint and not item.freeze:
            raise HTTPException(status_code=400, detail="Cada clip debe tener fin mayor que inicio")
        blob = item.model_dump()
        blob["clip"] = item.name or Path(item.src).name
        if item.crop:
            blob["crop"] = item.crop.model_dump()
        scenes.append(blob)
    seq, source = save_sequence(
        {
            "title": req.title or "Video",
            "scenes": scenes,
            "texts": req.texts or [],
            "markers": req.markers or [],
        },
        force=True,
    )
    names = [s.get("clip") for s in (seq.get("scenes") or []) if s.get("clip")]
    write_incoming_clips(names)
    return {
        "ok": True,
        "baked": False,
        "source": source,
        "sequence": seq,
        "clips": [{"filename": n, "url": f"/clips/{n}", "title": Path(n).stem} for n in names],
        "redirect": "/create",
    }


@app.post("/api/create/incoming/ack")
async def ack_incoming():
    try:
        if INCOMING_PATH.exists():
            INCOMING_PATH.unlink()
    except Exception:
        pass
    return {"ok": True}


@app.post("/api/remove-hardsubs")
async def api_remove_hardsubs(req: StripCaptionsRequest):
    """Quita hardsubs con Grok Imagine Video Edit (tramo in/out, máx ~26s)."""
    if not _xai_key():
        raise HTTPException(
            status_code=400,
            detail="Falta XAI_API_KEY en .env. Sin eso no se puede usar Imagine.",
        )
    if not check_ffmpeg():
        raise HTTPException(status_code=500, detail="FFmpeg no está disponible")
    path = _resolve_media_src(req.src)
    end = req.outPoint if req.outPoint is not None else 0
    mode = (req.mode or "hardsubs").lower().strip()
    if mode not in {"hardsubs", "watermark", "both"}:
        mode = "hardsubs"
    region = None
    if req.w is not None and req.h is not None and req.w > 0 and req.h > 0:
        region = {"x": req.x or 0, "y": req.y or 0, "w": req.w, "h": req.h, "normalized": req.normalized}
    ok, message, out = await asyncio.to_thread(remove_overlays, path, req.inPoint, end, mode, region)
    if not ok:
        raise HTTPException(status_code=500, detail=message)
    return {
        "ok": True,
        "message": message,
        "filename": out.name if out else None,
        "url": f"/clips/{out.name}" if out else None,
        "name": out.stem if out else None,
        "mode": mode,
    }


@app.post("/api/remove-watermarks")
async def api_remove_watermarks(req: StripCaptionsRequest):
    req.mode = "watermark"
    return await api_remove_hardsubs(req)


@app.post("/api/remove-watermark-region")
async def api_remove_watermark_region(req: DelogoRequest):
    """Tapa un recuadro marcado a mano con FFmpeg delogo (sin IA, sin créditos)."""
    if not check_ffmpeg():
        raise HTTPException(status_code=500, detail="FFmpeg no está disponible")
    if req.w <= 0 or req.h <= 0:
        raise HTTPException(status_code=400, detail="Marca un recuadro sobre el watermark")
    path = _resolve_media_src(req.src)
    end = req.outPoint if req.outPoint is not None else 0
    ok, message, out = await asyncio.to_thread(
        delogo_region,
        path,
        req.inPoint,
        end,
        req.x,
        req.y,
        req.w,
        req.h,
        req.normalized,
    )
    if not ok:
        raise HTTPException(status_code=500, detail=message)
    return {
        "ok": True,
        "message": message,
        "filename": out.name if out else None,
        "url": f"/clips/{out.name}" if out else None,
        "name": out.stem if out else None,
    }


@app.post("/api/upload-media")
async def upload_media(file: UploadFile = File(...)):
    """Sube un video o imagen local. Imágenes se guardan como still MP4 en Clips."""
    original = file.filename or "media"
    ctype = file.content_type or ""
    ext = Path(original).suffix.lower()
    kind = classify_upload_name(original, ctype)
    if kind is None and ext not in ALLOWED_MEDIA_EXT and ext not in AUDIO_UPLOAD_EXT:
        raise HTTPException(status_code=400, detail=f"Formato no soportado: {ext or '(sin extensión)'}")
    if kind == "image":
        ext = image_ext_for_upload(original, ctype)

    stem = safe_filename(Path(original).stem) or "drop"
    tmp = TEMP_DIR / f"up_{uuid.uuid4().hex[:10]}{ext or '.bin'}"
    try:
        with tmp.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"No se pudo guardar el archivo: {e}")
    finally:
        await file.close()

    if not tmp.exists() or tmp.stat().st_size < 80:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="El archivo está vacío")

    if kind == "audio" or ext in AUDIO_UPLOAD_EXT:
        stored = persist_sequence_music(tmp)
        tmp.unlink(missing_ok=True)
        if not stored:
            raise HTTPException(status_code=500, detail="No se pudo guardar la canción")
        duration = probe_duration(stored)
        return {
            "filename": stored.name,
            "url": SEQUENCE_MUSIC_URL,
            "name": Path(original).name,
            "title": Path(original).name,
            "duration": round(duration, 2) if duration else None,
            "size_mb": round(stored.stat().st_size / (1024 * 1024), 2),
            "kind": "audio",
        }

    if kind == "image" or ext in IMAGE_UPLOAD_EXT:
        ok, message, dest = await asyncio.to_thread(import_local_image, tmp, stem, 4.0)
        tmp.unlink(missing_ok=True)
        if not ok or not dest:
            raise HTTPException(status_code=500, detail=message)
        duration = probe_duration(dest)
        return {
            "filename": dest.name,
            "url": f"/clips/{dest.name}",
            "name": dest.stem,
            "title": Path(original).name,
            "duration": round(duration, 2) if duration else 4.0,
            "size_mb": round(dest.stat().st_size / (1024 * 1024), 2),
            "kind": "image",
        }

    dest = unique_path(CLIPS_DIR, stem, ext)
    try:
        shutil.copy2(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    duration = probe_duration(dest)
    return {
        "filename": dest.name,
        "url": f"/clips/{dest.name}",
        "name": Path(original).stem,
        "duration": round(duration, 2) if duration else None,
        "size_mb": round(dest.stat().st_size / (1024 * 1024), 2),
        "kind": "video",
    }


def _resolve_media_src(src: str) -> Path:
    """Solo permite archivos bajo /clips o /uploads. Bloquea path traversal."""
    raw = (src or "").strip().split("?")[0]
    for _ in range(3):
        nxt = unquote(raw)
        if nxt == raw:
            break
        raw = nxt
    if raw.startswith("blob:") or raw.startswith("http://") or raw.startswith("https://"):
        raise HTTPException(
            status_code=400,
            detail="El video local no se ha subido al servidor. Vuelve a cargarlo o espera a que termine la subida.",
        )

    if raw.startswith("/clips/"):
        folder = CLIPS_DIR
        prefix = "/clips/"
    elif raw.startswith("/uploads/"):
        folder = UPLOADS_DIR
        prefix = "/uploads/"
    elif "/" not in raw.replace("\\", "/").strip("/"):
        folder = CLIPS_DIR
        prefix = ""
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Fuente no válida: {raw[:80]}. Usa un clip de Stock o un video subido.",
        )

    name = Path(raw[len(prefix):] if prefix else raw).name
    if not name:
        raise HTTPException(status_code=400, detail="Nombre de archivo no válido")

    folder_resolved = folder.resolve()
    direct = folder / name
    path = None
    if direct.is_file():
        path = direct
    else:
        low = name.lower()
        for f in folder.iterdir():
            if f.is_file() and f.name.lower() == low:
                path = f
                break
    if path is None or not path.is_file():
        raise HTTPException(status_code=400, detail=f"No se encuentra el archivo: {name}")
    resolved = path.resolve()
    try:
        resolved.relative_to(folder_resolved)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"No se encuentra el archivo: {name}")
    return resolved


@app.post("/api/create/chat")
async def create_chat(req: CreateChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Escribe un mensaje")
    current = None
    if req.plan:
        current = {
            "title": req.plan.title,
            "summary": req.plan.summary,
            "scenes": [s.model_dump() for s in req.plan.scenes],
        }
    try:
        data = await asyncio.to_thread(
            chat_edit_plan,
            req.message.strip(),
            [t.model_dump() for t in req.history],
            current,
            req.clips,
            req.format,
            req.language,
            req.idea,
            req.captions,
        )
        return data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sequence")
async def get_sequence():
    seq, source = load_sequence()
    if not seq or not (seq.get("scenes") or source):
        return {"ok": False, "sequence": empty_sequence(), "source": None}
    if source and str(source).startswith("migrated:"):
        seq, source = save_sequence(seq, force=True)
    return {"ok": True, "sequence": seq, "source": source}


@app.get("/api/sequence/vo")
@app.head("/api/sequence/vo")
async def get_sequence_vo():
    path = sequence_vo_path()
    try:
        if not path.is_file() or path.stat().st_size < 200:
            raise HTTPException(status_code=404, detail="Sin VO")
    except HTTPException:
        raise
    except OSError:
        raise HTTPException(status_code=404, detail="Sin VO")
    return FileResponse(
        str(path),
        media_type="audio/mpeg",
        filename="vo.mp3",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/sequence/music")
@app.head("/api/sequence/music")
async def get_sequence_music():
    path = sequence_music_path()
    try:
        if not path.is_file() or path.stat().st_size < 200:
            raise HTTPException(status_code=404, detail="Sin canción")
    except HTTPException:
        raise
    except OSError:
        raise HTTPException(status_code=404, detail="Sin canción")
    return FileResponse(
        str(path),
        media_type="audio/mpeg",
        filename="music.mp3",
        headers={"Cache-Control": "no-store"},
    )


@app.put("/api/sequence/music")
async def put_sequence_music(file: UploadFile = File(...)):
    original = file.filename or "cancion.mp3"
    ctype = file.content_type or ""
    kind = classify_upload_name(original, ctype)
    ext = Path(original).suffix.lower()
    if kind != "audio" and ext not in AUDIO_UPLOAD_EXT:
        raise HTTPException(status_code=400, detail="Usa mp3, wav, m4a o aac")
    tmp = TEMP_DIR / f"mus_{uuid.uuid4().hex[:10]}{ext or '.bin'}"
    try:
        with tmp.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"No se pudo guardar: {e}")
    finally:
        await file.close()
    stored = persist_sequence_music(tmp)
    tmp.unlink(missing_ok=True)
    if not stored:
        raise HTTPException(status_code=500, detail="No se pudo guardar la canción")
    duration = probe_duration(stored)
    return {
        "ok": True,
        "url": SEQUENCE_MUSIC_URL,
        "name": Path(original).name[:120],
        "duration": round(duration, 2) if duration else None,
        "kind": "audio",
    }


@app.delete("/api/sequence/music")
async def delete_sequence_music():
    clear_sequence_music()
    return {"ok": True, "url": SEQUENCE_MUSIC_URL, "name": "", "kind": "audio"}


@app.post("/api/sequence")
async def post_sequence(payload: dict):
    force = bool(payload.get("force"))
    raw = payload.get("sequence") if isinstance(payload.get("sequence"), dict) else payload
    seq, source = save_sequence(raw, force=force)
    body = {
        "ok": source not in ("stale", "protected"),
        "source": source,
        "scenes": len(seq.get("scenes") or []),
        "sequence": seq,
        "rev": seq.get("rev") or 0,
    }
    if source == "stale":
        body["stale"] = True
    if source == "protected":
        body["skipped"] = "empty"
    return body


@app.get("/api/create/autosave")
async def get_create_autosave():
    """Shared sequence, wrapped as the old Assemble payload."""
    seq, source = load_sequence()
    if seq and (seq.get("scenes") or source):
        if source and str(source).startswith("migrated:"):
            seq, source = save_sequence(seq, force=True)
        return {"ok": True, "saved": sequence_as_assemble(seq), "source": source}
    data, source = load_create_autosave()
    if data:
        return {"ok": True, "saved": data, "source": source}
    return {"ok": False, "saved": None}


@app.post("/api/create/autosave")
async def post_create_autosave(payload: dict):
    seq, source = save_sequence(payload, force=False)
    n = len(seq.get("scenes") or [])
    if source == "protected":
        return {"ok": False, "skipped": "empty", "source": source}
    if source == "stale":
        return {"ok": False, "stale": True, "source": source, "scenes": n, "sequence": seq, "rev": seq.get("rev") or 0}
    return {"ok": True, "scenes": n, "source": source, "sequence": seq, "rev": seq.get("rev") or 0}


@app.post("/api/create/plan")
async def create_plan(req: CreatePlanRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Describe el video que quieres crear")
    try:
        plan = await asyncio.to_thread(
            plan_video,
            req.prompt,
            req.format,
            req.target_seconds,
            req.clips,
            req.language,
        )
        return plan
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_EXPORT_JOBS: dict = {}


def _prepare_export(req: ExportRequest):
    if not check_ffmpeg():
        raise HTTPException(
            status_code=500,
            detail="FFmpeg no está disponible. En PowerShell (como administrador si hace falta): winget install ffmpeg",
        )
    if not req.clips:
        raise HTTPException(status_code=400, detail="No hay clips en el timeline")
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Ponle un nombre al archivo exportado")
    if len(req.clips) > 40:
        raise HTTPException(status_code=400, detail="Máximo 40 clips por export")

    resolved = []
    for c in req.clips:
        if c.outPoint <= c.inPoint:
            raise HTTPException(status_code=400, detail="Cada clip debe tener fin mayor que inicio")
        path = _resolve_media_src(c.src)
        resolved.append({
            "path": path,
            "in_point": c.inPoint,
            "out_point": c.outPoint,
            "duration": c.duration,
            "speed": c.speed,
            "volume": c.volume,
            "muted": c.muted,
            "rotation": c.rotation,
            "filter": c.filter,
            "freeze": c.freeze,
            "fadeIn": c.fadeIn,
            "fadeOut": c.fadeOut,
            "crop": c.crop.model_dump() if c.crop else None,
            "voiceover": c.voiceover,
            "narration": c.narration or "",
            "voiceStyle": c.voiceStyle or "documentary",
            "duckOriginal": c.duckOriginal,
            "voiceEngine": c.voiceEngine or "edge",
            "voiceName": c.voiceName or "",
            "voFx": c.voFx or "none",
            "voRate": sanitize_vo_rate(c.voRate),
            "grade": c.grade if isinstance(c.grade, dict) else None,
        })
    texts = [t.model_dump() for t in req.texts if t.content.strip()]
    return resolved, texts, sanitize_script(req.script)


def _sequence_music_for_export() -> Optional[dict]:
    if not sequence_music_exists():
        return None
    seq, _ = load_sequence()
    ui = seq.get("ui") if isinstance(seq, dict) and isinstance(seq.get("ui"), dict) else {}
    music = ui.get("music") if isinstance(ui.get("music"), dict) else {}
    if music.get("mute"):
        return None
    try:
        vol = float(music.get("volume") if music.get("volume") is not None else 0.25)
    except (TypeError, ValueError):
        vol = 0.25
    vol = max(0.0, min(1.0, vol))
    if vol <= 0.001:
        return None
    return {"path": sequence_music_path(), "volume": vol, "mute": False}


def _run_export_job(job_id: str, resolved, texts, name: str, resolution: str, script: str = "") -> None:
    job = _EXPORT_JOBS.get(job_id)
    if not job:
        return

    def progress(pct, label):
        cur = _EXPORT_JOBS.get(job_id)
        if not cur or cur.get("status") != "running":
            return
        cur["pct"] = int(pct)
        cur["label"] = label

    try:
        success, message, path = render_editor_export(
            resolved, texts, name, resolution, progress=progress, script=script,
            music=_sequence_music_for_export(),
        )
        cur = _EXPORT_JOBS.get(job_id)
        if not cur:
            return
        if success and path:
            duration = probe_duration(path)
            cur.update({
                "status": "done",
                "pct": 100,
                "label": "Listo",
                "message": message,
                "filename": path.name,
                "path": f"/clips/{path.name}",
                "duration": round(duration, 2) if duration else None,
            })
        else:
            cur.update({
                "status": "error",
                "pct": cur.get("pct") or 0,
                "label": message or "Error al exportar",
                "message": message or "Error al exportar",
            })
    except Exception as e:
        cur = _EXPORT_JOBS.get(job_id)
        if cur:
            cur.update({
                "status": "error",
                "label": str(e),
                "message": str(e),
            })


@app.post("/api/export/start")
async def export_start(req: ExportRequest):
    resolved, texts, script = _prepare_export(req)
    job_id = uuid.uuid4().hex[:12]
    _EXPORT_JOBS[job_id] = {
        "id": job_id,
        "status": "running",
        "pct": 1,
        "label": "En cola…",
        "message": "",
        "filename": None,
        "path": None,
        "duration": None,
    }
    if len(_EXPORT_JOBS) > 30:
        stale = [k for k, v in _EXPORT_JOBS.items() if v.get("status") != "running"]
        for k in stale[:12]:
            _EXPORT_JOBS.pop(k, None)
    asyncio.create_task(asyncio.to_thread(
        _run_export_job, job_id, resolved, texts, req.name, req.resolution, script
    ))
    return {"ok": True, "job_id": job_id, "scenes": len(resolved)}


@app.get("/api/export/jobs/{job_id}")
async def export_job_status(job_id: str):
    job = _EXPORT_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="No hay ese trabajo de export")
    return job


@app.post("/api/export")
async def export_timeline(req: ExportRequest):
    """
    Recibe la EDL del editor (clips + textos) y renderiza un MP4 con FFmpeg.
    Los clips deben apuntar a /clips/... o /uploads/... (no blob:).
    """
    resolved, texts, script = _prepare_export(req)
    success, message, path = await asyncio.to_thread(
        render_editor_export,
        resolved,
        texts,
        req.name,
        req.resolution,
        None,
        script,
        _sequence_music_for_export(),
    )

    if not success:
        raise HTTPException(status_code=500, detail=message)

    duration = probe_duration(path) if path else None
    return {
        "success": True,
        "message": message,
        "filename": path.name if path else None,
        "path": f"/clips/{path.name}" if path else None,
        "duration": round(duration, 2) if duration else None,
    }


class CaptionClipIn(BaseModel):
    src: str
    inPoint: float = 0
    outPoint: float = 4
    timelineStart: float = 0
    narration: str = ""
    muted: bool = False


class CaptionGenerateRequest(BaseModel):
    language: str = "es"
    source: str = "original"  # original | narration | transcript
    punctuation: bool = True
    title_case: bool = False
    show_profanity: bool = False
    clips: List[CaptionClipIn] = []
    transcript: str = ""
    idea: str = ""
    duration: float = 0
    script: str = ""
    voUrl: str = ""
    voDuration: float = 0


class TtsRequest(BaseModel):
    text: str
    style: str = "documentary"
    voiceName: str = ""
    engine: str = ""
    voice_id: Optional[str] = None
    voRate: float = 1.0


@app.get("/api/eleven/status")
@app.get("/api/tts/status")
async def eleven_status():
    data = tts_status()
    data["eleven_voices"] = 0
    if eleven_key():
        voices = await asyncio.to_thread(list_voices)
        data["eleven_voices"] = len(voices)
    return data


@app.post("/api/eleven/tts")
@app.post("/api/tts")
async def eleven_tts(req: TtsRequest):
    dest = TEMP_DIR / f"vo_{uuid.uuid4().hex[:8]}.mp3"
    ok, msg, *rest = await asyncio.to_thread(
        synthesize_voiceover,
        req.text,
        dest,
        req.style or "documentary",
        req.voice_id,
        req.voiceName or None,
        req.engine or "",
        sanitize_vo_rate(req.voRate),
    )
    if not ok:
        raise HTTPException(status_code=500, detail=msg)
    words = rest[0] if rest else []
    public = UPLOADS_DIR / dest.name
    try:
        shutil.copy2(dest, public)
    except Exception:
        public = dest
    stable = persist_sequence_vo(public if public.is_file() else dest)
    used_engine = resolve_engine(req.engine or "")
    used = pick_edge_voice(req.style or "documentary", req.voiceName)
    if used_engine == "eleven":
        used = pick_voice(req.style) or used
    dur = 0.0
    try:
        dur = float(probe_duration(stable or public) or 0)
    except Exception:
        dur = 0.0
    cues = []
    if words:
        cues = words_to_cues(words, 32)
    elif dur > 0.4:
        cues = cues_from_untimed_script(req.text, dur)
    return {
        "ok": True,
        "url": SEQUENCE_VO_URL if stable else f"/uploads/{public.name}",
        "voice": used,
        "engine": used_engine,
        "words": words,
        "cues": cues,
        "duration": round(dur, 3) if dur else 0,
    }


@app.post("/api/create/captions")
async def create_captions(req: CaptionGenerateRequest):
    if not req.clips and not (req.transcript or "").strip():
        raise HTTPException(status_code=400, detail="No hay clips ni transcripción")

    def pack(cues: List[dict], source: str, note: str) -> dict:
        cleaned = []
        for c in cues or []:
            body = apply_transcript_controls(
                str(c.get("text") or ""), req.punctuation, req.title_case, req.show_profanity
            )
            if not body:
                continue
            try:
                a = float(c.get("start") or 0)
                b = float(c.get("end") or (a + 1.5))
            except (TypeError, ValueError):
                continue
            if b <= a:
                b = a + 1.2
            cleaned.append({"start": round(a, 2), "end": round(b, 2), "text": body})
        return {
            "ok": True,
            "source": source,
            "cues": cleaned,
            "text": format_cues_transcript(cleaned),
            "note": note,
        }

    def timeline_duration() -> float:
        if req.duration and req.duration > 0.4:
            return float(req.duration)
        total = 0.0
        for c in req.clips:
            total = max(total, float(c.timelineStart or 0) + max(0.4, float(c.outPoint) - float(c.inPoint)))
        return total or 30.0

    source = (req.source or "original").strip().lower()
    blob = (req.transcript or "").strip()
    script_text = (req.script or "").strip()
    if not blob:
        blob = script_text or "\n".join(
            (c.narration or "").strip() for c in req.clips if (c.narration or "").strip()
        ).strip()
    if not script_text:
        script_text = blob

    if source in ("narration", "vo"):
        vo_dur = float(req.voDuration or 0)
        if vo_dur <= 0.4:
            vo_path = resolve_vo_source(req.voUrl)
            if vo_path is not None:
                try:
                    vo_dur = float(probe_duration(vo_path) or 0)
                except Exception:
                    vo_dur = 0.0
        if script_text and vo_dur > 0.4:
            cues = cues_from_untimed_script(
                script_text, vo_dur, req.punctuation, req.title_case, req.show_profanity
            )
            if cues:
                return pack(cues, "vo", "Captions alineados a la voz, no al recorte.")

    if blob and (source in ("transcript", "narration") or looks_like_timestamped(blob) or len(blob) > 40):
        if looks_like_timestamped(blob):
            cues = parse_timestamped_transcript(
                blob, req.punctuation, req.title_case, req.show_profanity, timeline_duration()
            )
            if cues:
                return pack(cues, "transcript", "Captions con tus timestamps.")
        if source not in ("narration", "vo"):
            cues = cues_from_untimed_script(
                blob, timeline_duration(), req.punctuation, req.title_case, req.show_profanity
            )
            if cues:
                return pack(cues, "script", "Captions del texto pegado, repartidos en el timeline.")

    if source in ("original", "auto", "scribe") and eleven_key() and check_ffmpeg() and req.clips:
        job = TEMP_DIR / f"cap_{uuid.uuid4().hex[:8]}"
        job.mkdir(parents=True, exist_ok=True)
        all_words: List[dict] = []
        try:
            ff = ffmpeg_bin()
            transcribed = 0
            for i, c in enumerate(req.clips):
                if c.muted:
                    continue
                try:
                    path = _resolve_media_src(c.src)
                except Exception:
                    continue
                piece = job / f"a{i:02d}.wav"
                ok, _msg = await asyncio.to_thread(extract_audio, path, piece)
                if not ok:
                    continue
                trimmed = job / f"t{i:02d}.wav"
                dur = max(0.2, float(c.outPoint) - float(c.inPoint))
                cmd = [
                    ff, "-y",
                    "-ss", f"{max(0.0, c.inPoint):.3f}",
                    "-t", f"{dur:.3f}",
                    "-i", str(piece),
                    str(trimmed),
                ]
                await asyncio.to_thread(_run_ff, cmd)
                if not trimmed.exists() or trimmed.stat().st_size < 200:
                    continue
                ok, msg, words = await asyncio.to_thread(
                    transcribe_audio, trimmed, req.language or "es"
                )
                if not ok:
                    continue
                transcribed += 1
                off = float(c.timelineStart or 0)
                for w in words:
                    txt = apply_transcript_controls(
                        w.get("text") or "", req.punctuation, req.title_case, req.show_profanity
                    )
                    if not txt:
                        continue
                    all_words.append({
                        "text": txt,
                        "start": round(float(w["start"]) + off, 3),
                        "end": round(float(w["end"]) + off, 3),
                    })
            if all_words:
                cues = words_to_cues(all_words, 28)
                return pack(cues, "scribe", "Captions transcritos del audio.")
        finally:
            shutil.rmtree(job, ignore_errors=True)

    if source not in ("narration", "vo"):
        cues = cues_from_clip_scripts(
            req.clips, req.punctuation, req.title_case, req.show_profanity
        )
        if cues:
            return pack(cues, "narration", "Captions de la narración de cada escena.")

    scene_payload = []
    for c in req.clips:
        scene_payload.append({
            "clip": Path(c.src).name,
            "src": c.src,
            "inPoint": c.inPoint,
            "outPoint": c.outPoint,
            "timelineStart": c.timelineStart,
            "narration": c.narration or "",
        })
    ai_cues = await asyncio.to_thread(
        grok_caption_cues,
        req.idea or "",
        req.language or "es",
        scene_payload,
        timeline_duration(),
        blob,
    )
    if ai_cues:
        return pack(ai_cues, "ai", "Captions escritos por el director (no son el título del clip).")

    if source in ("narration", "vo"):
        raise HTTPException(
            status_code=400,
            detail="Genera la voz (Play) primero para alinear captions a la narración.",
        )
    raise HTTPException(
        status_code=400,
        detail="Pega una transcripción con tiempos (0:00 texto) o escribe narración en las escenas.",
    )


def _run_ff(cmd):
    from video_tools import _run
    return _run(cmd)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "20th-century",
        "xai": bool(_xai_key()),
        "elevenlabs": bool(eleven_key()),
        "ffmpeg": check_ffmpeg(),
        "ffmpeg_path": ffmpeg_bin(),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
