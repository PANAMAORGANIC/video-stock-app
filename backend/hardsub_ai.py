"""
Quita hardsubs (texto quemado) con Grok Imagine Video Edit.
Parte el clip en trozos de <= 8.5s, edita cada uno, concatena y recupera el audio original.
"""

from __future__ import annotations

import base64
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import httpx

from video_create import _xai_key
from video_tools import (
    CLIPS_DIR,
    TEMP_DIR,
    ffmpeg_bin,
    probe_duration,
    unique_path,
    _run,
    check_ffmpeg,
)

API = "https://api.x.ai/v1"
EDIT_MODEL = "grok-imagine-video"
CHUNK = 8.5
MAX_TOTAL = 26.0  # ~3 lotes
PROMPTS = {
    "hardsubs": (
        "Remove all burned-in subtitles, captions, on-screen text, and lower-third "
        "lettering from this video. Keep the same camera, people, objects, lighting, "
        "and motion. Fill those areas so they look like the original scene with no writing."
    ),
    "watermark": (
        "Remove all watermarks, logos, channel bugs, stamps, translucent overlays, "
        "and branding marks from this video. Keep the same camera, people, objects, "
        "lighting, and motion. Reconstruct the covered pixels so the scene looks clean "
        "and natural, with no logo left in any corner."
    ),
    "both": (
        "Remove all burned-in subtitles, captions, watermarks, logos, channel bugs, "
        "and on-screen branding from this video. Keep the same camera, people, objects, "
        "lighting, and motion. Fill those areas so the scene looks original and clean."
    ),
}
SUFFIX = {"hardsubs": "_clean", "watermark": "_nowm", "both": "_clean"}


def _headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _cut(src: Path, start: float, duration: float, dest: Path) -> None:
    ff = ffmpeg_bin()
    cmd = [
        ff, "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-sn", "-dn",
        "-movflags", "+faststart",
        str(dest),
    ]
    r = _run(cmd)
    if r.returncode != 0 or not dest.exists():
        raise RuntimeError((r.stderr or "FFmpeg cut failed")[-400:])


def _poll_video(client: httpx.Client, key: str, request_id: str, timeout: float = 480) -> str:
    url = f"{API}/videos/{request_id}"
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=60.0)
        if r.status_code >= 400:
            raise RuntimeError(f"Poll Imagine {r.status_code}: {r.text[:300]}")
        data = r.json()
        status = (data.get("status") or "").lower()
        if status in {"done", "completed", "succeeded", "success"}:
            video = data.get("video") or data
            out = video.get("url") if isinstance(video, dict) else None
            if not out:
                out = data.get("url")
            if not out:
                raise RuntimeError(f"Imagine terminó sin URL: {str(data)[:400]}")
            return out
        if status in {"failed", "expired", "error", "cancelled"}:
            raise RuntimeError(f"Imagine {status}: {str(data)[:400]}")
        time.sleep(4)
    raise RuntimeError("Imagine tardó demasiado (timeout).")


def _edit_chunk(client: httpx.Client, key: str, chunk: Path, dest: Path, prompt: str) -> None:
    raw = chunk.read_bytes()
    if len(raw) > 24 * 1024 * 1024:
        raise RuntimeError("El trozo de video es demasiado grande para enviarlo a Imagine.")
    b64 = base64.b64encode(raw).decode("ascii")
    payload = {
        "model": EDIT_MODEL,
        "prompt": prompt,
        "video": {"url": f"data:video/mp4;base64,{b64}", "type": "video_url"},
    }
    r = client.post(f"{API}/videos/edits", headers=_headers(key), json=payload, timeout=120.0)
    if r.status_code >= 400:
        # fallback shape without type
        payload["video"] = {"url": f"data:video/mp4;base64,{b64}"}
        r = client.post(f"{API}/videos/edits", headers=_headers(key), json=payload, timeout=120.0)
    if r.status_code >= 400:
        raise RuntimeError(f"Imagine edit {r.status_code}: {r.text[:400]}")
    data = r.json()
    req_id = data.get("request_id") or data.get("id")
    if not req_id:
        url = (data.get("video") or {}).get("url") or data.get("url")
        if url:
            _download(client, url, dest)
            return
        raise RuntimeError(f"Respuesta Imagine sin request_id: {str(data)[:400]}")
    url = _poll_video(client, key, req_id)
    _download(client, url, dest)


def _download(client: httpx.Client, url: str, dest: Path) -> None:
    r = client.get(url, timeout=180.0, follow_redirects=True)
    r.raise_for_status()
    dest.write_bytes(r.content)
    if dest.stat().st_size < 1000:
        raise RuntimeError("El video limpio descargado está vacío.")


def _concat_and_audio(chunks: list[Path], audio_src: Path, dest: Path, job: Path) -> None:
    ff = ffmpeg_bin()
    lst = job / "list.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in chunks), encoding="utf-8")
    joined = job / "joined.mp4"
    r = _run([
        ff, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-c", "copy", str(joined),
    ], cwd=job)
    if r.returncode != 0:
        r = _run([
            ff, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-an", str(joined),
        ], cwd=job)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or "concat failed")[-400:])
    # audio original
    r = _run([
        ff, "-y",
        "-i", str(joined),
        "-i", str(audio_src),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        "-sn",
        str(dest),
    ])
    if r.returncode != 0:
        shutil.copy2(joined, dest)


def remove_overlays(
    src: Path,
    start: float,
    end: float,
    mode: str = "hardsubs",
    region: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Optional[Path]]:
    mode = (mode or "hardsubs").lower().strip()
    if mode not in PROMPTS:
        mode = "hardsubs"
    prompt = PROMPTS[mode]
    if region and mode in {"watermark", "both"}:
        try:
            nx = float(region.get("x", 0))
            ny = float(region.get("y", 0))
            nw = float(region.get("w", 0))
            nh = float(region.get("h", 0))
            if nw > 0 and nh > 0:
                prompt += (
                    " The watermark or logo is inside this rectangle, given as fractions "
                    f"of the frame: x={nx:.3f}, y={ny:.3f}, width={nw:.3f}, height={nh:.3f} "
                    "(origin top-left). Remove that mark completely and reconstruct the scene."
                )
        except (TypeError, ValueError):
            pass
    key = _xai_key()
    if not key:
        return False, "Falta XAI_API_KEY en el archivo .env (misma carpeta que start.bat).", None
    if not check_ffmpeg():
        return False, "FFmpeg no está disponible.", None
    if not src.exists():
        return False, f"No se encuentra {src.name}", None

    start = max(0.0, float(start))
    src_dur = probe_duration(src) or 0
    if end <= start:
        end = start + min(CHUNK, src_dur or CHUNK)
    end = min(end, src_dur) if src_dur else end
    total = end - start
    if total < 0.4:
        return False, "El tramo es demasiado corto.", None
    if total > MAX_TOTAL + 0.05:
        return False, (
            f"Máximo {MAX_TOTAL:.0f}s por pasada (límite de Imagine ~8.7s por lote). "
            f"Recorta el clip a {MAX_TOTAL:.0f}s o menos y vuelve a intentar."
        ), None

    job = TEMP_DIR / f"overlay_{uuid.uuid4().hex[:10]}"
    job.mkdir(parents=True, exist_ok=True)
    try:
        original = job / "orig.mp4"
        _cut(src, start, total, original)
        chunks_in = []
        t = 0.0
        i = 0
        while t < total - 0.05:
            dur = min(CHUNK, total - t)
            piece = job / f"in_{i:02d}.mp4"
            _cut(original, t, dur, piece)
            chunks_in.append(piece)
            t += dur
            i += 1

        chunks_out = []
        with httpx.Client(timeout=180.0) as client:
            for n, piece in enumerate(chunks_in):
                outp = job / f"out_{n:02d}.mp4"
                _edit_chunk(client, key, piece, outp, prompt)
                chunks_out.append(outp)

        dest = unique_path(CLIPS_DIR, src.stem + SUFFIX[mode], ".mp4")
        _concat_and_audio(chunks_out, original, dest, job)
        labels = {
            "hardsubs": "Hardsubs quitados",
            "watermark": "Watermarks / logos quitados",
            "both": "Hardsubs y watermarks quitados",
        }
        return True, f"{labels[mode]}. Guardado como {dest.name}", dest
    except Exception as e:
        return False, str(e), None
    finally:
        shutil.rmtree(job, ignore_errors=True)


def remove_hardsubs(src: Path, start: float, end: float) -> Tuple[bool, str, Optional[Path]]:
    return remove_overlays(src, start, end, "hardsubs")
