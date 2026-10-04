"""
Herramientas de video: descarga, recorte, eliminación de subtítulos y export del editor.
Usa yt-dlp + FFmpeg.
"""

import os
import re
import subprocess
import shutil
import time
import array
import threading
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import yt_dlp
import uuid
import json


BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
CLIPS_DIR = STORAGE_DIR / "clips"
TEMP_DIR = STORAGE_DIR / "temp"
UPLOADS_DIR = STORAGE_DIR / "uploads"
PROXIES_DIR = STORAGE_DIR / "proxies"
EXPORTS_TMP = STORAGE_DIR / "exports" / "tmp"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PROXIES_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_TMP.mkdir(parents=True, exist_ok=True)

MIC_AUDIO_EXT = {".caf", ".wav", ".mp3", ".m4a"}

CREATE_NO_WINDOW = 0x08000000

_FFMPEG_CACHE: Optional[str] = None
_FFPROBE_CACHE: Optional[str] = None

RESOLUTIONS = {
    "1080": (1920, 1080),
    "720": (1280, 720),
    "480": (854, 480),
}


def _win_kwargs() -> dict:
    kwargs: dict = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":
        kwargs["creationflags"] = CREATE_NO_WINDOW
    return kwargs


def _candidate_bins(name: str) -> List[Path]:
    exe = f"{name}.exe" if os.name == "nt" else name
    env_key = "FFMPEG_PATH" if name == "ffmpeg" else "FFPROBE_PATH"
    env_val = os.environ.get(env_key)
    out: List[Path] = []
    if env_val:
        out.append(Path(env_val))
    which = shutil.which(name)
    if which:
        out.append(Path(which))
    local = os.environ.get("LOCALAPPDATA", "")
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    home = Path.home()
    out.extend([
        Path(local) / "Microsoft" / "WinGet" / "Links" / exe,
        Path(pf) / "ffmpeg" / "bin" / exe,
        Path(r"C:\ffmpeg\bin") / exe,
        home / "scoop" / "shims" / exe,
        Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "chocolatey" / "bin" / exe,
    ])
    packages = Path(local) / "Microsoft" / "WinGet" / "Packages"
    if packages.exists():
        out.extend(packages.glob(f"Gyan.FFmpeg*/ffmpeg-*/bin/{exe}"))
        out.extend(packages.glob(f"Gyan.FFmpeg*/bin/{exe}"))
    return out


def ffmpeg_bin() -> Optional[str]:
    global _FFMPEG_CACHE
    if _FFMPEG_CACHE and Path(_FFMPEG_CACHE).exists():
        return _FFMPEG_CACHE
    for p in _candidate_bins("ffmpeg"):
        if p.exists() and p.is_file():
            _FFMPEG_CACHE = str(p)
            return _FFMPEG_CACHE
    _FFMPEG_CACHE = None
    return None


def ffprobe_bin() -> Optional[str]:
    global _FFPROBE_CACHE
    if _FFPROBE_CACHE and Path(_FFPROBE_CACHE).exists():
        return _FFPROBE_CACHE
    for p in _candidate_bins("ffprobe"):
        if p.exists() and p.is_file():
            _FFPROBE_CACHE = str(p)
            return _FFPROBE_CACHE
    ff = ffmpeg_bin()
    if ff:
        sibling = Path(ff).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if sibling.exists():
            _FFPROBE_CACHE = str(sibling)
            return _FFPROBE_CACHE
    _FFPROBE_CACHE = None
    return None


def check_ffmpeg() -> bool:
    return ffmpeg_bin() is not None


def _run(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None, **_win_kwargs())
    except OSError as e:
        winerr = getattr(e, "winerror", None) or e.errno
        if winerr == 206:
            return subprocess.CompletedProcess(
                cmd,
                1,
                "",
                "[WinError 206] FFmpeg command line too long for Windows.",
            )
        raise


def safe_filename(name: str, max_len: int = 80) -> str:
    raw = (name or "").strip() or "clip"
    cleaned = "".join(c if c.isalnum() or c in "-_ " else "_" for c in raw).strip()
    cleaned = cleaned.replace(" ", "_")[:max_len].strip("._") or "clip"
    return cleaned


def make_still_clip(
    src: Path,
    seconds: float = 2.0,
    at: float = 0.4,
    stem: Optional[str] = None,
) -> Tuple[bool, str, Optional[Path]]:
    """2s still (stock-image style) from one frame of a clip."""
    ff = ffmpeg_bin()
    if not ff or not src.exists():
        return False, f"No se encuentra {src.name if src else 'clip'}", None
    seconds = max(0.6, min(8.0, float(seconds or 2)))
    at = max(0.0, float(at or 0))
    stem = safe_filename(stem or f"still_{src.stem}")[:50]
    dest = unique_path(CLIPS_DIR, stem, ".mp4")
    jpg = TEMP_DIR / f"still_{uuid.uuid4().hex[:8]}.jpg"
    try:
        grab = [
            ff, "-y", "-ss", f"{at:.3f}", "-i", str(src),
            "-frames:v", "1", "-q:v", "2", str(jpg),
        ]
        r = _run(grab)
        if r.returncode != 0 or not jpg.exists() or jpg.stat().st_size < 200:
            grab = [
                ff, "-y", "-i", str(src),
                "-frames:v", "1", "-q:v", "2", str(jpg),
            ]
            r = _run(grab)
        if not jpg.exists() or jpg.stat().st_size < 200:
            return False, (r.stderr or "No pude extraer el fotograma")[-300:], None
        cmd = [
            ff, "-y",
            "-loop", "1", "-t", f"{seconds:.3f}", "-i", str(jpg),
            "-f", "lavfi", "-t", f"{seconds:.3f}", "-i", "anullsrc=r=44100:cl=stereo",
            "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
            "-r", "25", "-c:a", "aac", "-shortest",
            "-movflags", "+faststart",
            str(dest),
        ]
        r = _run(cmd)
        if r.returncode != 0 or not dest.exists() or dest.stat().st_size < 1000:
            cmd = [
                ff, "-y",
                "-loop", "1", "-t", f"{seconds:.3f}", "-i", str(jpg),
                "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                "-r", "25", "-an",
                "-movflags", "+faststart",
                str(dest),
            ]
            r = _run(cmd)
        if r.returncode != 0 or not dest.exists() or dest.stat().st_size < 1000:
            return False, (r.stderr or "FFmpeg still falló")[-300:], None
        return True, dest.name, dest
    finally:
        try:
            jpg.unlink(missing_ok=True)
        except Exception:
            pass


def still_from_image(
    src: Path,
    seconds: float = 4.0,
    stem: Optional[str] = None,
) -> Tuple[bool, str, Optional[Path]]:
    """Turn a still image into a short MP4 so it can sit on the Assemble timeline."""
    ff = ffmpeg_bin()
    if not ff:
        return False, "FFmpeg no está disponible", None
    if not src.exists() or src.stat().st_size < 80:
        return False, f"No se encuentra la imagen {src.name if src else ''}", None
    seconds = max(1.0, min(30.0, float(seconds or 4)))
    stem = safe_filename(stem or f"img_{src.stem}")[:50]
    if not stem.lower().startswith("img_"):
        stem = "img_" + stem
    dest = unique_path(CLIPS_DIR, stem, ".mp4")
    vf = (
        "scale=1280:720:force_original_aspect_ratio=decrease,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,format=yuv420p"
    )
    cmd = [
        ff, "-y",
        "-loop", "1", "-t", f"{seconds:.3f}", "-i", str(src),
        "-f", "lavfi", "-t", f"{seconds:.3f}", "-i", "anullsrc=r=44100:cl=stereo",
        "-vf", vf,
        "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
        "-r", "25", "-c:a", "aac", "-shortest",
        "-movflags", "+faststart",
        str(dest),
    ]
    r = _run(cmd)
    if r.returncode != 0 or not dest.exists() or dest.stat().st_size < 1000:
        cmd = [
            ff, "-y",
            "-loop", "1", "-t", f"{seconds:.3f}", "-i", str(src),
            "-vf", vf,
            "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
            "-r", "25", "-an",
            "-movflags", "+faststart",
            str(dest),
        ]
        r = _run(cmd)
    if r.returncode != 0 or not dest.exists() or dest.stat().st_size < 1000:
        return False, (r.stderr or "FFmpeg still falló")[-300:], None
    return True, dest.name, dest


def unique_path(directory: Path, stem: str, suffix: str = ".mp4") -> Path:
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    for i in range(2, 200):
        candidate = directory / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
    return directory / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"


def _ff_escape_path(path: Path) -> str:
    """Escapa una ruta para usarla dentro de un filtro de FFmpeg."""
    s = path.resolve().as_posix()
    return s.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


class _YtdlpLogger:
    def __init__(self) -> None:
        self.errors: List[str] = []

    def debug(self, msg):  # noqa: ANN001
        return

    def info(self, msg):  # noqa: ANN001
        return

    def warning(self, msg):  # noqa: ANN001
        return

    def error(self, msg):  # noqa: ANN001
        self.errors.append(str(msg))


def _normalize_download_url(url: str) -> str:
    u = (url or "").strip()
    m = re.search(r"dai\.ly/([A-Za-z0-9]+)", u, re.I)
    if m:
        return f"https://www.dailymotion.com/video/{m.group(1)}"
    m = re.search(r"dailymotion\.com/(?:embed/)?video/([A-Za-z0-9]+)", u, re.I)
    if m:
        return f"https://www.dailymotion.com/video/{m.group(1)}"
    return u


def _find_downloaded(stem: str) -> Optional[Path]:
    video_ext = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".ts", ".m4a"}
    skip_ext = {".part", ".ytdl", ".json"}
    matches: List[Path] = []
    for p in TEMP_DIR.glob(f"{stem}*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in skip_ext or p.name.endswith(".part"):
            continue
        matches.append(p)
    if not matches:
        return None
    ranked = [p for p in matches if p.suffix.lower() in video_ext]
    pool = ranked or matches
    return max(pool, key=lambda p: p.stat().st_size if p.exists() else 0)


def download_video_segment(
    video_url: str,
    start: float,
    end: float,
    output_name: Optional[str] = None,
    remove_soft_subs: bool = True,
) -> Tuple[bool, str, Optional[Path]]:
    """
    Descarga el segmento y lo recorta.
    Retorna: (success, message, path_del_archivo)
    """
    if not check_ffmpeg():
        return False, "FFmpeg no está instalado", None

    video_url = _normalize_download_url(video_url)
    if not video_url.startswith("http"):
        return False, "URL de video no válida", None

    if output_name is None:
        output_name = f"clip_{uuid.uuid4().hex[:10]}"

    safe_name = safe_filename(output_name)
    final_path = unique_path(CLIPS_DIR, safe_name, ".mp4")
    stem = f"full_{uuid.uuid4().hex[:8]}"
    temp_cut = TEMP_DIR / f"cut_{uuid.uuid4().hex[:8]}.mp4"
    logger = _YtdlpLogger()
    ff = ffmpeg_bin()

    ydl_opts: Dict[str, Any] = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/best",
        "outtmpl": str(TEMP_DIR / f"{stem}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "merge_output_format": "mp4",
        "retries": 5,
        "fragment_retries": 8,
        "logger": logger,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        },
    }
    host = video_url.lower()
    if "dailymotion.com" in host or "dai.ly" in host:
        ydl_opts["http_headers"]["Referer"] = "https://www.dailymotion.com/"
    if ff:
        ydl_opts["ffmpeg_location"] = str(Path(ff).parent)

    ig_cookies = STORAGE_DIR / "instagram_cookies.txt"
    if "instagram.com" in host and ig_cookies.exists():
        try:
            from search_youtube import is_netscape_cookiefile
            if is_netscape_cookiefile(ig_cookies):
                ydl_opts["cookiefile"] = str(ig_cookies)
        except Exception:
            pass

    ranged = False
    try:
        from yt_dlp.utils import download_range_func

        pad_start = max(0.0, float(start) - 0.2)
        pad_end = float(end) + 0.2
        if pad_end > pad_start:
            ydl_opts["download_ranges"] = download_range_func(None, [(pad_start, pad_end)])
            ydl_opts["force_keyframes_at_cuts"] = True
            ranged = True
    except Exception:
        ranged = False

    def _run_ydl(opts: Dict[str, Any]) -> None:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([video_url])

    try:
        try:
            _run_ydl(ydl_opts)
        except Exception:
            if not ranged:
                raise
            ydl_opts.pop("download_ranges", None)
            ydl_opts.pop("force_keyframes_at_cuts", None)
            ranged = False
            _run_ydl(ydl_opts)

        downloaded = _find_downloaded(stem)
        if not downloaded or not downloaded.exists() or downloaded.stat().st_size < 1000:
            extra = "; ".join(logger.errors[-3:]) if logger.errors else ""
            hint = extra or "yt-dlp no dejó un archivo de video"
            return False, f"No se pudo descargar el video: {hint}", None

        if not downloaded.suffix:
            renamed = downloaded.with_suffix(".mp4")
            try:
                downloaded.rename(renamed)
                downloaded = renamed
            except Exception:
                pass

        duration = max(0.2, float(end) - float(start))
        ss = 0.0 if ranged else max(0.0, float(start))
        cmd = [
            ff, "-y",
            "-ss", f"{ss:.3f}",
            "-i", str(downloaded),
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
        ]
        if remove_soft_subs:
            cmd.extend(["-sn"])
        cmd.append(str(temp_cut))

        result = _run(cmd)
        if result.returncode != 0 or not temp_cut.exists() or temp_cut.stat().st_size < 1000:
            temp_cut.unlink(missing_ok=True)
            err = (result.stderr or "")[-300:]
            return False, f"Error en FFmpeg (recorte): {err or 'archivo inválido'}", None

        shutil.move(str(temp_cut), str(final_path))
        try:
            downloaded.unlink(missing_ok=True)
        except Exception:
            pass

        return True, f"Clip guardado como {final_path.name}", final_path

    except Exception as e:
        extra = "; ".join(logger.errors[-2:]) if logger.errors else ""
        detail = str(e).strip() or extra or "error desconocido"
        if extra and extra not in detail:
            detail = f"{detail} | {extra}"
        return False, f"No se pudo descargar el video: {detail[:400]}", None
    finally:
        _cleanup_old_temp()


def publish_edited_clip(
    src: Path,
    start: float,
    end: float,
    output_name: str,
    extras: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Optional[Path]]:
    """Recorta el tramo editado (crop/giro incluidos) y lo deja en storage/clips."""
    if not check_ffmpeg():
        return False, "FFmpeg no está disponible", None
    if not src.exists():
        return False, f"No se encuentra {src.name}", None
    extras = extras or {}
    start = max(0.0, float(start))
    dur_src = probe_duration(src) or 0.0
    if end <= start:
        end = (dur_src or start + 3)
    if dur_src:
        end = min(end, dur_src)
    duration = end - start
    if duration < 0.2:
        return False, "El tramo es demasiado corto", None

    vw, vh = probe_video_size(src)
    vf_parts = []
    crop_vf = _crop_filter(extras.get("crop"), vw, vh)
    if crop_vf:
        vf_parts.append(crop_vf)
    rot = _rotate_filter(extras.get("rotation") or 0)
    if rot:
        vf_parts.append(rot)
    col = _color_filter(str(extras.get("filter") or "none"))
    if col:
        vf_parts.append(col)
    grade = _grade_filter(extras.get("grade") if isinstance(extras.get("grade"), dict) else None)
    if grade:
        vf_parts.append(grade)

    almost_full = start <= 0.05 and dur_src and abs(end - dur_src) < 0.25
    if almost_full and not vf_parts:
        if src.parent.resolve() == CLIPS_DIR.resolve():
            return True, src.name, src
        stem = safe_filename(output_name or src.stem)
        dest = unique_path(CLIPS_DIR, stem, src.suffix if src.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv", ".m4v"} else ".mp4")
        try:
            shutil.copy2(src, dest)
            return True, dest.name, dest
        except Exception:
            pass

    ff = ffmpeg_bin()
    stem = safe_filename(output_name or src.stem)
    dest = unique_path(CLIPS_DIR, stem, ".mp4")
    cmd = [
        ff, "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
    ]
    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])
    cmd.extend([
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-sn", "-dn",
        "-movflags", "+faststart",
        str(dest),
    ])
    result = _run(cmd)
    if result.returncode != 0 or not dest.exists():
        dest.unlink(missing_ok=True)
        return False, (result.stderr or "Error FFmpeg")[-400:], None
    return True, f"Listo para Video Creation: {dest.name}", dest


def replace_clip_audio(
    video: Path,
    audio: Path,
    offset_ms: int,
    output_name: str = "",
    dest_dir: Optional[Path] = None,
) -> Tuple[bool, str, Optional[Path]]:
    """Copy the picture. Replace camera audio with an external mic track.

    offset_ms is applied as ffmpeg -itsoffset (seconds = ms/1000). Positive delays
    the mic; negative starts it earlier. Output duration matches the video.
    video and audio are only read. The new file is a different path.
    """
    if not check_ffmpeg():
        return False, "FFmpeg no está disponible", None
    video = Path(video)
    audio = Path(audio)
    if not video.is_file():
        return False, f"No se encuentra {video.name}", None
    if not audio.is_file() or audio.stat().st_size < 32:
        return False, "El audio de micrófono está vacío", None
    if audio.suffix.lower() not in MIC_AUDIO_EXT:
        return False, "Usa CAF, WAV, MP3 o M4A", None
    try:
        offset_ms = int(offset_ms)
    except (TypeError, ValueError):
        return False, "Desfase (ms) tiene que ser un entero", None
    duration = probe_duration(video)
    if not duration or duration < 0.05:
        return False, "No pude leer la duración del video", None
    info = get_video_info(video)
    streams = info.get("streams") or []
    if not any(s.get("codec_type") == "video" for s in streams):
        return False, "El clip no tiene video", None
    ainfo = get_video_info(audio)
    if not any(s.get("codec_type") == "audio" for s in (ainfo.get("streams") or [])):
        return False, "El archivo no tiene audio", None

    folder = Path(dest_dir) if dest_dir is not None else CLIPS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(output_name or f"{video.stem}_mic")
    dest = unique_path(folder, stem, ".mp4")
    try:
        if dest.resolve() == video.resolve() or dest.resolve() == audio.resolve():
            dest = unique_path(folder, stem + "_mic", ".mp4")
    except OSError:
        pass

    offset_sec = f"{offset_ms / 1000.0:.6f}"
    dur = f"{duration:.6f}"
    ff = ffmpeg_bin()
    cmd = [
        ff, "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-itsoffset", offset_sec,
        "-i", str(audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-af", f"aresample=async=1:first_pts=0,apad=whole_dur={dur},atrim=0:{dur}",
        "-c:a", "aac", "-b:a", "160k",
        "-sn", "-dn",
        "-t", dur,
        "-movflags", "+faststart",
        str(dest),
    ]
    result = _run(cmd)
    if result.returncode != 0 or not dest.is_file() or dest.stat().st_size < 500:
        dest.unlink(missing_ok=True)
        err = (result.stderr or "Error FFmpeg")[-400:]
        return False, err or "No se pudo reemplazar el audio", None
    out_info = get_video_info(dest)
    out_streams = out_info.get("streams") or []
    n_video = sum(1 for s in out_streams if s.get("codec_type") == "video")
    n_audio = sum(1 for s in out_streams if s.get("codec_type") == "audio")
    out_dur = probe_duration(dest) or 0.0
    if n_video != 1 or n_audio != 1 or abs(out_dur - duration) > 0.35:
        dest.unlink(missing_ok=True)
        return False, "El clip nuevo no coincide con la duración del video", None
    try:
        if dest.resolve() == video.resolve():
            dest.unlink(missing_ok=True)
            return False, "No se puede sobrescribir el original", None
    except OSError:
        pass
    return True, dest.name, dest


def remove_soft_subtitles(input_path: Path, output_path: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Elimina solo subtítulos suaves (pistas) de un video ya descargado.
    """
    ok, msg, path = strip_captions(input_path, output_path=output_path, replace=False)
    if not ok:
        return False, msg
    return True, str(path)


def strip_captions(
    input_path: Path,
    output_path: Optional[Path] = None,
    replace: bool = True,
) -> Tuple[bool, str, Optional[Path]]:
    """
    Quita pistas de subtítulos (soft) y closed captions del contenedor.
    No borra texto quemado en los píxeles (hardsubs).
    """
    ff = ffmpeg_bin()
    if not ff:
        return False, "FFmpeg no disponible", None
    if not input_path.exists():
        return False, f"No se encuentra {input_path.name}", None

    dest = output_path or (TEMP_DIR / f"nosubs_{uuid.uuid4().hex[:8]}.mp4")
    copy_cmd = [
        ff, "-y", "-i", str(input_path),
        "-map", "0", "-map", "-0:s?",
        "-c", "copy",
        "-sn", "-dn",
        "-movflags", "+faststart",
        str(dest),
    ]
    result = _run(copy_cmd)
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size < 1000:
        dest.unlink(missing_ok=True)
        re_cmd = [
            ff, "-y", "-i", str(input_path),
            "-sn", "-dn",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(dest),
        ]
        result = _run(re_cmd)
        if result.returncode != 0:
            dest.unlink(missing_ok=True)
            return False, (result.stderr or "Error FFmpeg")[-400:], None

    final = input_path if replace else (
        output_path or input_path.with_name(input_path.stem + "_nosubs" + ".mp4")
    )
    if replace:
        tmp_swap = input_path.with_name(input_path.stem + ".__tmp__.mp4")
        try:
            if tmp_swap.exists():
                tmp_swap.unlink()
            shutil.move(str(dest), str(tmp_swap))
            input_path.unlink(missing_ok=True)
            tmp_swap.rename(input_path)
            final = input_path
        except Exception:
            shutil.move(str(dest), str(input_path))
            final = input_path
    elif dest != final:
        shutil.move(str(dest), str(final))

    return True, f"Subtítulos de pista quitados: {final.name}", final


def get_video_info(path: Path) -> dict:
    """Obtiene duración y streams con ffprobe."""
    probe = ffprobe_bin()
    if not probe or not path.exists():
        return {}
    cmd = [
        probe, "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path)
    ]
    try:
        kwargs = _win_kwargs()
        kwargs["timeout"] = 8
        result = subprocess.run(cmd, **kwargs)
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        return json.loads(result.stdout)
    except Exception:
        return {}


def probe_duration(path: Path) -> Optional[float]:
    info = get_video_info(path)
    try:
        dur = float(info.get("format", {}).get("duration") or 0)
        return dur if dur > 0 else None
    except (TypeError, ValueError):
        return None


PEAKS_RATE = 2000
PEAKS_MIN_BINS = 64
PEAKS_MAX_BINS = 600


def peaks_sidecar(path: Path) -> Path:
    """Cache file next to the media: foo.mp4.peaks.json (not Sequence JSON)."""
    p = Path(path)
    return p.with_name(p.name + ".peaks.json")


def empty_peaks(duration: float = 0.0) -> Dict[str, Any]:
    return {
        "v": 1,
        "peaks": [],
        "hasAudio": False,
        "duration": float(duration or 0),
    }


def _media_stamp(path: Path) -> Dict[str, int]:
    st = path.stat()
    return {"mtime": int(st.st_mtime * 1000), "size": int(st.st_size)}


def _run_bytes(cmd: List[str], timeout: float = 40) -> subprocess.CompletedProcess:
    kwargs: dict = {"capture_output": True, "timeout": timeout}
    if os.name == "nt":
        kwargs["creationflags"] = CREATE_NO_WINDOW
    return subprocess.run(cmd, **kwargs)


def media_has_audio(path: Path) -> Optional[bool]:
    """True/False if probe lists streams; None if probe failed."""
    info = get_video_info(path)
    streams = info.get("streams") or []
    if not streams:
        return None
    return any(s.get("codec_type") == "audio" for s in streams)


def _pack_peaks(pcm: bytes, bins: int) -> List[float]:
    if not pcm:
        return []
    if len(pcm) % 2:
        pcm = pcm[:-1]
    samples = array.array("h")
    samples.frombytes(pcm)
    n = len(samples)
    if n == 0:
        return []
    bins = max(1, min(int(bins), n, PEAKS_MAX_BINS))
    out: List[float] = []
    for i in range(bins):
        a = (i * n) // bins
        b = ((i + 1) * n) // bins
        if b <= a:
            b = a + 1
        peak = 0
        chunk = samples[a:b]
        for s in chunk:
            v = -s if s < 0 else s
            if v > peak:
                peak = v
        out.append(round(min(1.0, peak / 32767.0), 4))
    return out


def compute_clip_peaks(path: Path, bins: Optional[int] = None) -> Dict[str, Any]:
    """Sample per-bin peak amplitude for a file in clips/ or uploads/. Never raises."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return empty_peaks(0)
    dur = probe_duration(path) or 0.0
    has = media_has_audio(path)
    if has is False:
        return empty_peaks(dur)
    if not check_ffmpeg():
        return empty_peaks(dur)
    n_bins = int(bins) if bins else int(max(PEAKS_MIN_BINS, min(PEAKS_MAX_BINS, (dur or 1.0) * 20)))
    ff = ffmpeg_bin()
    if not ff:
        return empty_peaks(dur)
    cmd = [
        ff, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(PEAKS_RATE),
        "-f", "s16le", "-acodec", "pcm_s16le",
        "pipe:1",
    ]
    try:
        result = _run_bytes(cmd, timeout=40)
    except Exception:
        return {"v": 1, "peaks": [], "hasAudio": bool(has), "duration": dur}
    pcm = result.stdout or b""
    if result.returncode != 0 or not pcm:
        return {"v": 1, "peaks": [], "hasAudio": bool(has), "duration": dur}
    peaks = _pack_peaks(pcm, n_bins)
    return {"v": 1, "peaks": peaks, "hasAudio": True, "duration": dur}


def load_clip_peaks(path: Path, bins: Optional[int] = None) -> Dict[str, Any]:
    """Return cached peaks, computing with FFmpeg on miss. Sidecar next to media."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return empty_peaks(0)
    try:
        stamp = _media_stamp(path)
    except OSError:
        return empty_peaks(0)
    cache = peaks_sidecar(path)
    if cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if (
                isinstance(data, dict)
                and int(data.get("mtime") or 0) == stamp["mtime"]
                and int(data.get("size") or 0) == stamp["size"]
                and isinstance(data.get("peaks"), list)
            ):
                peaks = []
                for x in data["peaks"][: PEAKS_MAX_BINS]:
                    try:
                        peaks.append(max(0.0, min(1.0, float(x))))
                    except (TypeError, ValueError):
                        peaks.append(0.0)
                return {
                    "v": 1,
                    "peaks": peaks,
                    "hasAudio": bool(data.get("hasAudio")),
                    "duration": float(data.get("duration") or 0),
                    "mtime": stamp["mtime"],
                    "size": stamp["size"],
                }
        except Exception:
            pass
    data = compute_clip_peaks(path, bins=bins)
    data["mtime"] = stamp["mtime"]
    data["size"] = stamp["size"]
    try:
        cache.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    except Exception:
        pass
    return data


def move_peaks_sidecar(src: Path, dest: Path) -> None:
    old = peaks_sidecar(src)
    new = peaks_sidecar(dest)
    if not old.is_file():
        return
    try:
        if new.exists() and new.resolve() != old.resolve():
            new.unlink()
        old.rename(new)
    except Exception:
        pass


def delete_peaks_sidecar(path: Path) -> None:
    side = peaks_sidecar(path)
    try:
        side.unlink(missing_ok=True)
    except Exception:
        pass


POSTER_WIDTH = 320
PROXY_MIN_BYTES = 15 * 1024 * 1024
PROXY_HEIGHT = 540
_proxy_gen_lock = threading.Lock()


def _poster_t(t: Any) -> float:
    try:
        v = float(t or 0)
    except (TypeError, ValueError):
        v = 0.0
    if v != v:
        v = 0.0
    return max(0.0, round(v, 1))


def poster_sidecar(path: Path, t: float = 0.0) -> Path:
    """JPEG next to the media: foo.mp4.poster.1.5.jpg (not Sequence JSON)."""
    p = Path(path)
    key = f"{_poster_t(t):.1f}"
    return p.with_name(p.name + f".poster.{key}.jpg")


def list_poster_sidecars(path: Path) -> List[Path]:
    p = Path(path)
    parent = p.parent
    if not parent.is_dir():
        return []
    return sorted(parent.glob(p.name + ".poster.*.jpg"))


def poster_is_fresh(src: Path, dest: Path) -> bool:
    try:
        return (
            dest.is_file()
            and dest.stat().st_size > 200
            and dest.stat().st_mtime >= src.stat().st_mtime
        )
    except OSError:
        return False


def ensure_clip_poster(path: Path, t: float = 0.0) -> Optional[Path]:
    """One ~320px JPEG at t. Cached next to the media. Never raises."""
    path = Path(path)
    if not path.is_file():
        return None
    at = _poster_t(t)
    dest = poster_sidecar(path, at)
    if poster_is_fresh(path, dest):
        return dest
    if not check_ffmpeg():
        return dest if dest.is_file() else None
    ff = ffmpeg_bin()
    if not ff:
        return None
    tmp = dest.with_name(dest.name + ".tmp.jpg")
    vf = f"scale='min({POSTER_WIDTH},iw)':-2"
    tries = [
        [ff, "-y", "-ss", f"{at:.3f}", "-i", str(path), "-frames:v", "1", "-vf", vf, "-q:v", "4", str(tmp)],
        [ff, "-y", "-i", str(path), "-frames:v", "1", "-vf", vf, "-q:v", "4", str(tmp)],
        [ff, "-y", "-i", str(path), "-frames:v", "1", "-q:v", "4", str(tmp)],
    ]
    try:
        for cmd in tries:
            try:
                r = _run(cmd)
            except Exception:
                continue
            if tmp.is_file() and tmp.stat().st_size > 200:
                try:
                    if dest.exists():
                        dest.unlink()
                    tmp.replace(dest)
                except Exception:
                    try:
                        shutil.copyfile(tmp, dest)
                    except Exception:
                        return None
                return dest if dest.is_file() else None
        return dest if dest.is_file() else None
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def move_poster_sidecars(src: Path, dest: Path) -> None:
    src, dest = Path(src), Path(dest)
    suffix_from = src.name
    for old in list_poster_sidecars(src):
        rest = old.name[len(suffix_from):]
        new = dest.with_name(dest.name + rest)
        try:
            if new.exists() and new.resolve() != old.resolve():
                new.unlink()
            old.rename(new)
        except Exception:
            pass


def delete_poster_sidecars(path: Path) -> None:
    for old in list_poster_sidecars(path):
        try:
            old.unlink(missing_ok=True)
        except Exception:
            pass


def proxy_path(name: str) -> Path:
    safe = Path(name or "").name
    return PROXIES_DIR / safe


def proxy_needed(path: Path) -> bool:
    try:
        return Path(path).is_file() and Path(path).stat().st_size >= PROXY_MIN_BYTES
    except OSError:
        return False


def proxy_is_fresh(src: Path, dest: Path) -> bool:
    try:
        return (
            dest.is_file()
            and dest.stat().st_size > 2000
            and dest.stat().st_mtime >= src.stat().st_mtime
        )
    except OSError:
        return False


def ensure_clip_proxy(path: Path) -> Tuple[str, Optional[Path]]:
    """540p H.264 in storage/proxies/, cached by filename. Skips files under PROXY_MIN_BYTES."""
    path = Path(path)
    if not path.is_file():
        return "failed", None
    if not proxy_needed(path):
        return "skipped", None
    PROXIES_DIR.mkdir(parents=True, exist_ok=True)
    dest = proxy_path(path.name)
    if proxy_is_fresh(path, dest):
        return "ready", dest
    if not check_ffmpeg():
        return "failed", None
    ff = ffmpeg_bin()
    if not ff:
        return "failed", None
    with _proxy_gen_lock:
        if proxy_is_fresh(path, dest):
            return "ready", dest
        tmp = dest.with_name(dest.name + ".tmp.mp4")
        tries = [
            [
                ff, "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", str(path),
                "-vf", f"scale=-2:{PROXY_HEIGHT}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                "-c:a", "aac", "-b:a", "96k",
                "-movflags", "+faststart",
                str(tmp),
            ],
            [
                ff, "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", str(path),
                "-vf", "scale=-2:480",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                "-an",
                "-movflags", "+faststart",
                str(tmp),
            ],
        ]
        def _drop_tmp() -> None:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

        try:
            for cmd in tries:
                try:
                    # No short fuse: a ~45 min HEVC proxy can run well past 180s.
                    r = subprocess.run(cmd, **_win_kwargs())
                except (OSError, subprocess.SubprocessError):
                    _drop_tmp()
                    continue
                if r.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 2000:
                    try:
                        if dest.exists():
                            dest.unlink()
                        tmp.replace(dest)
                    except Exception:
                        try:
                            shutil.copyfile(tmp, dest)
                        except Exception:
                            _drop_tmp()
                            return "failed", None
                    return ("ready", dest) if dest.is_file() else ("failed", None)
                _drop_tmp()
            return "failed", None
        finally:
            _drop_tmp()


def move_clip_proxy(src: Path, dest: Path) -> None:
    old = proxy_path(Path(src).name)
    new = proxy_path(Path(dest).name)
    if not old.is_file():
        return
    try:
        PROXIES_DIR.mkdir(parents=True, exist_ok=True)
        if new.exists() and new.resolve() != old.resolve():
            new.unlink()
        old.rename(new)
    except Exception:
        pass


def delete_clip_proxy(path: Path) -> None:
    dest = proxy_path(Path(path).name)
    try:
        dest.unlink(missing_ok=True)
        tmp = dest.with_name(dest.name + ".tmp.mp4")
        tmp.unlink(missing_ok=True)
    except Exception:
        pass


def probe_video_size(path: Path) -> Tuple[int, int]:
    info = get_video_info(path)
    for stream in info.get("streams") or []:
        if stream.get("codec_type") != "video":
            continue
        try:
            w = int(stream.get("width") or 0)
            h = int(stream.get("height") or 0)
        except (TypeError, ValueError):
            continue
        if w > 0 and h > 0:
            return w, h
    return 0, 0


def _fit_delogo_rect(x: int, y: int, w: int, h: int, vw: int, vh: int) -> Tuple[int, int, int, int]:
    """delogo exige el recuadro al menos 1 px separado de los bordes."""
    if vw < 16 or vh < 16:
        raise ValueError("El video es demasiado pequeño para quitar un watermark")
    inset = 2
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    if w < 0:
        x += w
        w = -w
    if h < 0:
        y += h
        h = -h
    max_x = vw - inset - 1
    max_y = vh - inset - 1
    x = max(inset, min(x, max_x))
    y = max(inset, min(y, max_y))
    w = max(8, min(w, vw - x - inset))
    h = max(8, min(h, vh - y - inset))
    if x + w > vw - inset:
        w = vw - inset - x
    if y + h > vh - inset:
        h = vh - inset - y
    if w < 8 or h < 8:
        raise ValueError("La zona es demasiado pequeña o está pegada al borde del video")
    return x, y, w, h


def delogo_region(
    src: Path,
    start: float,
    end: float,
    x: float,
    y: float,
    w: float,
    h: float,
    normalized: bool = True,
) -> Tuple[bool, str, Optional[Path]]:
    """Recorta el tramo y tapa el recuadro con FFmpeg delogo (sin IA)."""
    if not check_ffmpeg():
        return False, "FFmpeg no está disponible", None
    if not src.exists():
        return False, f"No se encuentra {src.name}", None

    vw, vh = probe_video_size(src)
    if vw < 16 or vh < 16:
        return False, "No se pudo leer el tamaño del video", None

    try:
        if normalized:
            px = float(x) * vw
            py = float(y) * vh
            pw = float(w) * vw
            ph = float(h) * vh
        else:
            px, py, pw, ph = float(x), float(y), float(w), float(h)
        rx, ry, rw, rh = _fit_delogo_rect(round(px), round(py), round(pw), round(ph), vw, vh)
    except ValueError as e:
        return False, str(e), None

    start = max(0.0, float(start))
    dur_src = probe_duration(src) or 0.0
    if end is None or float(end) <= start:
        end = dur_src or (start + 3)
    if dur_src:
        end = min(float(end), dur_src)
    duration = end - start
    if duration < 0.2:
        return False, "El tramo es demasiado corto", None

    ff = ffmpeg_bin()
    dest = unique_path(CLIPS_DIR, src.stem + "_nowm", ".mp4")
    vf = f"delogo=x={rx}:y={ry}:w={rw}:h={rh}:show=0:band=4"
    cmd = [
        ff, "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-sn", "-dn",
        "-movflags", "+faststart",
        str(dest),
    ]
    result = _run(cmd)
    if result.returncode != 0:
        dest.unlink(missing_ok=True)
        cmd[cmd.index("-vf") + 1] = f"delogo=x={rx}:y={ry}:w={rw}:h={rh}:show=0"
        result = _run(cmd)
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size < 1000:
        dest.unlink(missing_ok=True)
        err = (result.stderr or "Error FFmpeg")[-400:]
        return False, f"No se pudo tapar la zona: {err}", None
    return True, f"Watermark tapado en {rx},{ry} {rw}x{rh}. Guardado como {dest.name}", dest


def _has_audio(info: dict) -> bool:
    return any(s.get("codec_type") == "audio" for s in info.get("streams", []))


def _has_video(info: dict) -> bool:
    return any(s.get("codec_type") == "video" for s in info.get("streams", []))


def _find_font(kind: str = "sans") -> Optional[Path]:
    kind = (kind or "sans").lower()
    packs = {
        "serif": [
            Path(r"C:\Windows\Fonts\georgia.ttf"),
            Path(r"C:\Windows\Fonts\times.ttf"),
            Path(r"C:\Windows\Fonts\timesi.ttf"),
        ],
        "rounded": [
            Path(r"C:\Windows\Fonts\comic.ttf"),
            Path(r"C:\Windows\Fonts\segoeui.ttf"),
        ],
        "bold": [
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
            Path(r"C:\Windows\Fonts\impact.ttf"),
            Path(r"C:\Windows\Fonts\segoeuib.ttf"),
        ],
    }
    candidates = packs.get(kind, []) + [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _short_job_dir() -> Path:
    """Windows CreateProcess + MAX_PATH: keep export scratch off the long Downloads tree."""
    if os.name == "nt":
        root = Path(os.environ.get("TEMP") or os.environ.get("TMP") or r"C:\Temp") / "cvexp"
    else:
        root = TEMP_DIR
    root.mkdir(parents=True, exist_ok=True)
    d = root / uuid.uuid4().hex[:10]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cleanup_old_temp(max_age_seconds: int = 3600) -> None:
    now = time.time()
    roots = [TEMP_DIR]
    if os.name == "nt":
        roots.append(Path(os.environ.get("TEMP") or os.environ.get("TMP") or r"C:\Temp") / "cvexp")
    for root in roots:
        if not root.exists():
            continue
        for p in root.glob("*"):
            try:
                age = now - p.stat().st_mtime
                if age <= max_age_seconds:
                    continue
                if p.is_file():
                    p.unlink(missing_ok=True)
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass


def _color_filter(name: str) -> str:
    return {
        "bw": "hue=s=0",
        "sepia": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131",
        "vivid": "eq=contrast=1.2:saturation=1.35",
        "cool": "eq=gamma_b=1.12:gamma_r=0.94:saturation=1.05",
        "warm": "eq=gamma_r=1.12:gamma_b=0.92:saturation=1.08",
    }.get((name or "none").lower(), "")


def _grade_filter(grade: Optional[Dict[str, Any]]) -> str:
    """Resolve-style lift/gamma/gain/sat/temp → FFmpeg eq/colorbalance."""
    if not isinstance(grade, dict):
        return ""
    def num(key, default):
        try:
            return float(grade.get(key) if grade.get(key) is not None else default)
        except (TypeError, ValueError):
            return default
    lift = max(-0.4, min(0.4, num("lift", 0)))
    gamma = max(0.4, min(2.2, num("gamma", 1)))
    gain = max(0.4, min(2.2, num("gain", 1)))
    sat = max(0.0, min(3.0, num("sat", 1)))
    temp = max(-1.0, min(1.0, num("temp", 0)))
    contrast = max(0.4, min(2.2, num("contrast", 1)))
    identity = (
        abs(lift) < 0.01 and abs(gamma - 1) < 0.01 and abs(gain - 1) < 0.01
        and abs(sat - 1) < 0.01 and abs(temp) < 0.02 and abs(contrast - 1) < 0.01
    )
    if identity:
        return ""
    brightness = lift * 0.35 + (gain - 1) * 0.15
    parts = [
        f"eq=contrast={contrast:.3f}:saturation={sat:.3f}:gamma={gamma:.3f}:brightness={brightness:.3f}"
    ]
    if abs(temp) >= 0.02:
        parts.append(f"colorbalance=rs={temp * 0.22:.3f}:gs=0:bs={-temp * 0.22:.3f}")
    return ",".join(parts)


def _rotate_filter(deg: int) -> str:
    deg = int(deg or 0) % 360
    if deg == 90:
        return "transpose=1"
    if deg == 180:
        return "transpose=1,transpose=1"
    if deg == 270:
        return "transpose=2"
    return ""


def _crop_filter(crop: Optional[Dict[str, Any]], vw: int, vh: int) -> str:
    """crop normalizado 0–1 (origen arriba-izquierda) sobre el fotograma original."""
    if not crop or vw < 16 or vh < 16:
        return ""
    try:
        nx = float(crop.get("x", 0))
        ny = float(crop.get("y", 0))
        nw = float(crop.get("w", 1))
        nh = float(crop.get("h", 1))
    except (TypeError, ValueError):
        return ""
    if nw <= 0.04 or nh <= 0.04:
        return ""
    if nw >= 0.995 and nh >= 0.995 and nx <= 0.005 and ny <= 0.005:
        return ""
    x = int(round(nx * vw))
    y = int(round(ny * vh))
    w = int(round(nw * vw))
    h = int(round(nh * vh))
    x = max(0, min(x, vw - 16))
    y = max(0, min(y, vh - 16))
    w = max(16, min(w, vw - x))
    h = max(16, min(h, vh - y))
    x -= x % 2
    y -= y % 2
    w -= w % 2
    h -= h % 2
    if x + w > vw:
        w = vw - x - ((vw - x) % 2)
    if y + h > vh:
        h = vh - y - ((vh - y) % 2)
    if w < 16 or h < 16:
        return ""
    if w >= vw - 2 and h >= vh - 2:
        return ""
    return f"crop={w}:{h}:{x}:{y}"


def _atempo_chain(speed: float) -> str:
    filters = []
    s = max(0.25, min(4.0, float(speed)))
    while s > 2.0001:
        filters.append("atempo=2.0")
        s /= 2.0
    while s < 0.499:
        filters.append("atempo=0.5")
        s *= 2.0
    filters.append(f"atempo={s:.4f}")
    return ",".join(filters)


def _normalize_segment(
    src: Path,
    in_point: float,
    duration: float,
    dest: Path,
    width: int,
    height: int,
    extras: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Recorta, escala y deja video+audio homogéneos para concatenar."""
    ff = ffmpeg_bin()
    if not ff:
        return False, "FFmpeg no disponible"

    extras = extras or {}
    info = get_video_info(src)
    if not _has_video(info):
        return False, f"El archivo no tiene pista de video: {src.name}"

    speed = float(extras.get("speed") or 1) or 1.0
    speed = max(0.25, min(4.0, speed))
    volume = float(extras.get("volume") if extras.get("volume") is not None else 1)
    muted = bool(extras.get("muted"))
    freeze = bool(extras.get("freeze"))
    fade_in = bool(extras.get("fadeIn"))
    fade_out = bool(extras.get("fadeOut"))
    out_dur = max(0.2, float(duration))

    vf_parts = []
    vw, vh = 0, 0
    for stream in info.get("streams") or []:
        if stream.get("codec_type") == "video":
            try:
                vw = int(stream.get("width") or 0)
                vh = int(stream.get("height") or 0)
            except (TypeError, ValueError):
                vw, vh = 0, 0
            break
    crop_vf = _crop_filter(extras.get("crop"), vw, vh)
    if crop_vf:
        vf_parts.append(crop_vf)
    rot = _rotate_filter(extras.get("rotation") or 0)
    if rot:
        vf_parts.append(rot)
    col = _color_filter(str(extras.get("filter") or "none"))
    if col:
        vf_parts.append(col)
    grade = _grade_filter(extras.get("grade") if isinstance(extras.get("grade"), dict) else None)
    if grade:
        vf_parts.append(grade)
    fit = str(extras.get("fit") or "contain").strip().lower()
    if fit == "cover":
        vf_parts.append(
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            "setsar=1,fps=30,format=yuv420p"
        )
    else:
        vf_parts.append(
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            "setsar=1,fps=30,format=yuv420p"
        )
    if freeze:
        vf_parts.append(f"tpad=stop_mode=clone:stop_duration={out_dur:.4f}")
    elif abs(speed - 1.0) > 0.01:
        vf_parts.append(f"setpts=PTS/{speed:.4f}")
    if fade_in:
        vf_parts.append("fade=t=in:st=0:d=0.6")
    if fade_out:
        fade_start = max(0.0, out_dur - 0.6)
        vf_parts.append(f"fade=t=out:st={fade_start:.3f}:d=0.6")
    vf = ",".join(vf_parts)

    af_parts = []
    if freeze or muted or volume <= 0:
        af_parts.append("volume=0")
    else:
        if abs(speed - 1.0) > 0.01:
            af_parts.append(_atempo_chain(speed))
        if abs(volume - 1.0) > 0.01:
            af_parts.append(f"volume={max(0.0, min(2.0, volume)):.3f}")
    af = ",".join(af_parts) if af_parts else None

    src_t = 0.05 if freeze else max(out_dur * speed if not freeze else 0.05, 0.2)

    if freeze:
        cmd = [
            ff, "-y",
            "-ss", f"{in_point:.4f}",
            "-i", str(src),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", f"{out_dur:.4f}",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "128k",
            "-sn",
            "-shortest",
            "-map", "0:v:0", "-map", "1:a:0",
            "-movflags", "+faststart",
            str(dest),
        ]
    elif _has_audio(info):
        cmd = [
            ff, "-y",
            "-ss", f"{in_point:.4f}",
            "-t", f"{src_t:.4f}",
            "-i", str(src),
            "-t", f"{out_dur:.4f}",
            "-vf", vf,
        ]
        if af:
            cmd.extend(["-af", af])
        cmd.extend([
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "128k",
            "-sn",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
            str(dest),
        ])
    else:
        cmd = [
            ff, "-y",
            "-ss", f"{in_point:.4f}",
            "-t", f"{src_t:.4f}",
            "-i", str(src),
            "-f", "lavfi", "-t", f"{out_dur:.4f}",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", f"{out_dur:.4f}",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "128k",
            "-sn",
            "-shortest",
            "-map", "0:v:0", "-map", "1:a:0",
            "-movflags", "+faststart",
            str(dest),
        ]

    result = _run(cmd)
    if result.returncode != 0:
        return False, result.stderr[-500:] if result.stderr else "Error desconocido de FFmpeg"
    if not dest.exists() or dest.stat().st_size < 1000:
        return False, f"El segmento recortado está vacío: {src.name}"
    vo = extras.get("voiceover_path")
    if vo:
        vo_path = Path(vo)
        if vo_path.exists():
            duck = 0.0 if muted else float(extras.get("duck") if extras.get("duck") is not None else 0.18)
            ok_mix, mix_err = _mix_voiceover(
                dest, vo_path, duck, extras.get("voFx") or "none",
            )
            if not ok_mix:
                return False, mix_err
    return True, "ok"


def _vo_fx_filter(kind: str) -> str:
    k = (kind or "none").strip().lower()
    if k == "booth":
        return "highpass=f=80,acompressor=threshold=-18dB:ratio=3:attack=5:release=80,aecho=0.8:0.88:40:0.18"
    if k == "radio":
        return "highpass=f=300,lowpass=f=3400,acompressor=threshold=-16dB:ratio=4:attack=8:release=80"
    if k == "telephone":
        return "highpass=f=400,lowpass=f=2800,acompressor=threshold=-14dB:ratio=6:attack=3:release=50,volume=1.15"
    return ""


def _mix_voiceover(
    video: Path,
    vo: Path,
    duck: float,
    vo_fx: str = "none",
    shortest: bool = True,
) -> Tuple[bool, str]:
    ff = ffmpeg_bin()
    if not ff:
        return False, "FFmpeg no disponible"
    duck = max(0.0, min(1.0, float(duck)))
    tmp = video.with_name(video.stem + ".__vo__.mp4")
    vo_chain = "aformat=sample_rates=44100:channel_layouts=stereo"
    fx = _vo_fx_filter(vo_fx)
    if fx:
        vo_chain += "," + fx
    vo_chain += ",volume=1"
    cmd = [
        ff, "-y",
        "-i", str(video),
        "-i", str(vo),
        "-filter_complex",
        f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo,volume={duck:.3f}[bg];"
        f"[1:a]{vo_chain}[vo];"
        f"[bg][vo]amix=inputs=2:duration=first:dropout_transition=0.4[a]",
        "-map", "0:v:0", "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "128k",
    ]
    if shortest:
        cmd.append("-shortest")
    cmd.extend(["-movflags", "+faststart", str(tmp)])
    result = _run(cmd)
    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1000:
        tmp.unlink(missing_ok=True)
        # Video without usable audio: replace with VO only (keep FX)
        af = vo_chain if fx else "aformat=sample_rates=44100:channel_layouts=stereo"
        cmd2 = [
            ff, "-y",
            "-i", str(video),
            "-i", str(vo),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-af", af,
            "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "128k",
        ]
        if shortest:
            cmd2.append("-shortest")
        cmd2.extend(["-movflags", "+faststart", str(tmp)])
        result = _run(cmd2)
        if result.returncode != 0 or not tmp.exists():
            tmp.unlink(missing_ok=True)
            return False, (result.stderr or "No se pudo mezclar la voiceover")[-400:]
    try:
        video.unlink(missing_ok=True)
        tmp.rename(video)
    except Exception:
        shutil.move(str(tmp), str(video))
    return True, "ok"


def _mix_music(video: Path, music: Path, volume: float) -> Tuple[bool, str]:
    """Mix a looped/trimmed song under the picture. Call after VO mix. VO stays louder."""
    ff = ffmpeg_bin()
    if not ff:
        return False, "FFmpeg no disponible"
    vol = max(0.0, min(1.0, float(volume)))
    tmp = video.with_name(video.stem + ".__mus__.mp4")
    cmd = [
        ff, "-y",
        "-i", str(video),
        "-stream_loop", "-1",
        "-i", str(music),
        "-filter_complex",
        f"[0:a]aformat=sample_rates=44100:channel_layouts=stereo[bg];"
        f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo,volume={vol:.3f}[mus];"
        f"[bg][mus]amix=inputs=2:duration=first:dropout_transition=0.3[a]",
        "-map", "0:v:0", "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        str(tmp),
    ]
    result = _run(cmd)
    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1000:
        tmp.unlink(missing_ok=True)
        cmd2 = [
            ff, "-y",
            "-i", str(video),
            "-stream_loop", "-1",
            "-i", str(music),
            "-filter_complex",
            f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo,volume={vol:.3f}[mus]",
            "-map", "0:v:0", "-map", "[mus]",
            "-c:v", "copy",
            "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            str(tmp),
        ]
        result = _run(cmd2)
        if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1000:
            tmp.unlink(missing_ok=True)
            return False, (result.stderr or "No se pudo mezclar la canción")[-400:]
    try:
        video.unlink(missing_ok=True)
        tmp.rename(video)
    except Exception:
        shutil.move(str(tmp), str(video))
    return True, "ok"


def _concat_segments(segments: List[Path], dest: Path, job_dir: Path) -> Tuple[bool, str]:
    ff = ffmpeg_bin()
    if not ff:
        return False, "FFmpeg no disponible"

    list_file = job_dir / "concat.txt"
    lines = [f"file '{p.name}'" for p in segments]
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = [
        ff, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        "-movflags", "+faststart",
        str(dest),
    ]
    result = _run(cmd, cwd=job_dir)
    if result.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
        return True, "ok"

    # Fallback: re-encode if stream copy concat fails
    cmd = [
        ff, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(dest),
    ]
    result = _run(cmd, cwd=job_dir)
    if result.returncode != 0:
        return False, result.stderr[-500:] if result.stderr else "Error al concatenar"
    return True, "ok"


def _export_vo_signature(script: str, lead: Optional[Dict[str, Any]] = None) -> str:
    lead = lead or {}
    rate = lead.get("voRate")
    parts = [
        "full",
        str(script or ""),
        str(lead.get("voiceStyle") or ""),
        str(lead.get("voiceName") or ""),
        str(lead.get("voiceEngine") or ""),
        str(lead.get("voFx") or ""),
        "" if rate is None else str(rate),
    ]
    return "\x1f".join(parts)[:8000]


def _export_vo_lead(clips: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    for clip in clips or []:
        if not isinstance(clip, dict):
            continue
        if clip.get("voiceover") and str(clip.get("narration") or "").strip():
            return clip
    return None


def _maybe_use_stored_vo(script: str, lead: Optional[Dict[str, Any]], dest: Path) -> bool:
    """Copy storage/autosave/vo.mp3 when its sequence signature still matches."""
    folder = STORAGE_DIR / "autosave"
    vo = folder / "vo.mp3"
    seq_path = folder / "sequence.json"
    try:
        if not vo.is_file() or vo.stat().st_size < 200 or not seq_path.is_file():
            return False
        data = json.loads(seq_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    ui = data.get("ui") if isinstance(data, dict) else None
    stored = ""
    if isinstance(ui, dict) and isinstance(ui.get("vo"), dict):
        stored = str(ui["vo"].get("sig") or "")
    if not stored or stored != _export_vo_signature(script, lead):
        return False
    try:
        shutil.copy2(vo, dest)
    except Exception:
        return False
    try:
        return dest.is_file() and dest.stat().st_size >= 200
    except OSError:
        return False


def _ass_time(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _ass_bgr(hex_color: str) -> str:
    h = (hex_color or "ffffff").lstrip("#")
    if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
        h = "ffffff"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{b}{g}{r}&"


def _ass_alpha(opacity: float) -> str:
    try:
        op = float(opacity)
    except (TypeError, ValueError):
        op = 1.0
    aa = int(round((1.0 - max(0.0, min(1.0, op))) * 255))
    return f"&H{aa:02X}&"


def _ass_color(hex_color: str, opacity: float = 1.0) -> str:
    h = (hex_color or "ffffff").lstrip("#")
    if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
        h = "ffffff"
    r, g, b = h[0:2], h[2:4], h[4:6]
    aa = _ass_alpha(opacity)[2:4]
    return f"&H{aa}{b}{g}{r}&"


def _ass_font_name(kind: str) -> str:
    k = (kind or "sans").lower()
    if k == "serif":
        return "Georgia"
    if k == "rounded":
        return "Comic Sans MS"
    if k == "bold":
        return "Arial Bold"
    return "Arial"


def _ass_escape_text(s: str) -> str:
    return (
        (s or "")
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\r\n", r"\N")
        .replace("\n", r"\N")
        .replace("\r", r"\N")
    )


def _texts_to_ass(texts: List[Dict[str, Any]], vw: int = 1280, vh: int = 720) -> str:
    """Burn-in captions as ASS so FFmpeg gets a tiny command (no WinError 206)."""
    vw = max(16, int(vw or 1280))
    vh = max(16, int(vh or 720))
    align = {"top": 8, "bottom": 2, "lower-third": 2, "center": 5}
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {vw}",
        f"PlayResY: {vh}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,42,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,60,60,80,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for t in texts or []:
        content = (t.get("content") or "").strip()
        if not content:
            continue
        start = float(t.get("start") or 0)
        end = float(t.get("end") or (start + 4))
        if end <= start:
            end = start + 1.0
        pos = (t.get("position") or "lower-third").lower()
        an = align.get(pos, 2)
        try:
            size = max(12, min(120, int(t.get("size") or 42)))
        except (TypeError, ValueError):
            size = 42
        color = (t.get("color") or "ffffff").lstrip("#")
        try:
            opacity = max(0.15, min(1.0, float(t.get("opacity") or 1)))
        except (TypeError, ValueError):
            opacity = 1.0
        font = _ass_font_name(str(t.get("font") or "sans"))
        if pos == "lower-third":
            mv = max(40, int(vh * 0.18))
        elif pos == "bottom":
            mv = max(24, int(vh * 0.08))
        elif pos == "top":
            mv = max(24, int(vh * 0.10))
        else:
            mv = 0
        tags = [
            f"\\an{an}",
            f"\\fn{font}",
            f"\\fs{size}",
            f"\\1c{_ass_bgr(color)}",
            f"\\1a{_ass_alpha(opacity)}",
            "\\bord2",
            "\\3c&H000000&",
        ]
        highlight = (t.get("highlight") or "").lstrip("#")
        if highlight and len(highlight) == 6:
            try:
                bop = max(0.0, min(1.0, float(t.get("boxOpacity") if t.get("boxOpacity") is not None else 0.85)))
            except (TypeError, ValueError):
                bop = 0.85
            tags.append(f"\\3c{_ass_bgr(highlight)}")
            tags.append(f"\\3a{_ass_alpha(bop)}")
            tags.append("\\bord8")
        override = "{" + "".join(tags) + "}"
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,{mv},,{override}{_ass_escape_text(content)}"
        )
    return "\n".join(lines) + "\n"


def _apply_texts(src: Path, dest: Path, texts: List[Dict[str, Any]], job_dir: Path) -> Tuple[bool, str]:
    ff = ffmpeg_bin()
    if not ff:
        return False, "FFmpeg no disponible"

    cues = [t for t in (texts or []) if (t.get("content") or "").strip()]
    if not cues:
        shutil.copy2(src, dest)
        return True, "ok"

    vw, vh = probe_video_size(src)
    if vw < 16 or vh < 16:
        vw, vh = 1280, 720
    ass_path = job_dir / "captions.ass"
    ass_path.write_text(_texts_to_ass(cues, vw, vh), encoding="utf-8-sig")
    # Relative name + cwd=job_dir keeps the FFmpeg argv tiny on Windows.
    vf = "subtitles=captions.ass"
    cmd = [
        ff, "-y",
        "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(dest),
    ]
    result = _run(cmd, cwd=job_dir)
    if result.returncode != 0:
        cmd = [
            ff, "-y",
            "-i", str(src),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(dest),
        ]
        result = _run(cmd, cwd=job_dir)
        if result.returncode != 0:
            return False, result.stderr[-500:] if result.stderr else "Error al aplicar textos"
    return True, "ok"



def render_editor_export(
    clips: List[Dict[str, Any]],
    texts: List[Dict[str, Any]],
    output_name: str,
    resolution: str = "720",
    progress: Optional[Any] = None,
    script: str = "",
    music: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Optional[Path]]:
    """
    Recorta cada clip, concatena, aplica textos overlay y guarda un MP4 en storage/clips.
    clips: [{path: Path, in_point: float, out_point: float}]
    """
    def report(pct: float, label: str) -> None:
        if not progress:
            return
        try:
            progress(max(0, min(100, int(pct))), str(label or ""))
        except Exception:
            pass

    if not check_ffmpeg():
        return False, "FFmpeg no está instalado. En PowerShell: winget install ffmpeg", None
    if not clips:
        return False, "No hay clips para exportar", None

    key = str(resolution).replace("p", "").strip()
    width, height = RESOLUTIONS.get(key, RESOLUTIONS["720"])

    stem = safe_filename(output_name)
    final_path = unique_path(CLIPS_DIR, stem, ".mp4")
    job_dir = _short_job_dir()

    try:
        report(3, "Preparando render…")
        segments: List[Path] = []
        n = max(1, len(clips))
        for i, clip in enumerate(clips):
            src = Path(clip["path"])
            in_point = max(0.0, float(clip["in_point"]))
            out_point = float(clip["out_point"])
            extras = {
                "speed": clip.get("speed", 1),
                "volume": clip.get("volume", 1),
                "muted": clip.get("muted", False),
                "rotation": clip.get("rotation", 0),
                "filter": clip.get("filter", "none"),
                "grade": clip.get("grade") if isinstance(clip.get("grade"), dict) else None,
                "freeze": clip.get("freeze", False),
                "fadeIn": clip.get("fadeIn", False),
                "fadeOut": clip.get("fadeOut", False),
                "crop": clip.get("crop"),
            }
            base_pct = 6 + (72 * i / n)
            pretty = src.name
            report(base_pct, f"Escena {i + 1} de {n}: {pretty}")
            if extras["freeze"]:
                duration = float(clip.get("duration") or 2)
            else:
                src_len = max(0.2, out_point - in_point)
                speed = float(extras["speed"] or 1) or 1
                duration = float(clip.get("duration") or (src_len / speed))
            if duration < 0.2:
                return False, f"El clip {src.name} es demasiado corto (mínimo 0.2s)", None
            if not src.exists():
                return False, f"No se encuentra el archivo: {src.name}", None

            dest = job_dir / f"seg_{i:03d}.mp4"
            report(base_pct + 2, f"Escena {i + 1} de {n}: recortando con FFmpeg…")
            ok, msg = _normalize_segment(src, in_point, duration, dest, width, height, extras)
            if not ok:
                return False, f"Error recortando {src.name}: {msg}", None
            segments.append(dest)
            report(6 + (72 * (i + 1) / n), f"Escena {i + 1} de {n}: lista")

        report(80, "Uniendo escenas…")
        concat_path = job_dir / "concat.mp4"
        ok, msg = _concat_segments(segments, concat_path, job_dir)
        if not ok:
            return False, f"Error al unir clips: {msg}", None

        full_script = (script or "").strip()
        if not full_script:
            full_script = "\n\n".join(
                (c.get("narration") or "").strip()
                for c in clips
                if (c.get("narration") or "").strip()
            )
        want_vo = bool(full_script) or any(c.get("voiceover") for c in clips)
        if want_vo and full_script:
            report(84, "Generando voiceover…")
            lead = next(
                (
                    c for c in clips
                    if c.get("voiceover") or (c.get("narration") or "").strip()
                ),
                clips[0],
            )
            try:
                from tts_audio import synthesize_voiceover
                vo_path = job_dir / "vo_full.mp3"
                stored = STORAGE_DIR / "autosave" / "vo.mp3"
                ok_vo, vo_msg = False, ""
                if stored.is_file() and stored.stat().st_size > 200:
                    shutil.copy2(stored, vo_path)
                    ok_vo = vo_path.is_file() and vo_path.stat().st_size > 200
                    vo_msg = "stored"
                if not ok_vo:
                    ok_vo, vo_msg, *_vo_rest = synthesize_voiceover(
                        full_script,
                        vo_path,
                        str(lead.get("voiceStyle") or "documentary"),
                        None,
                        lead.get("voiceName"),
                        str(lead.get("voiceEngine") or ""),
                        lead.get("voRate") if lead.get("voRate") is not None else 1.0,
                    )
                if ok_vo and vo_path.exists():
                    duck = 0.0 if lead.get("muted") else (
                        0.16 if lead.get("duckOriginal") or any(c.get("duckOriginal") for c in clips)
                        else 0.28
                    )
                    ok_mix, mix_err = _mix_voiceover(
                        concat_path,
                        vo_path,
                        duck,
                        lead.get("voFx") or "none",
                        shortest=False,
                    )
                    if not ok_mix:
                        return False, mix_err, None
                else:
                    print(f"VO full: {vo_msg}")
            except Exception as e:
                print(f"VO full: {e}")

        mus = music if isinstance(music, dict) else None
        mus_path = Path(mus["path"]) if mus and mus.get("path") else None
        mus_vol = 0.25
        mus_mute = False
        if mus:
            try:
                mus_vol = float(mus.get("volume") if mus.get("volume") is not None else 0.25)
            except (TypeError, ValueError):
                mus_vol = 0.25
            mus_mute = bool(mus.get("mute"))
        if mus_path is None:
            stored_mus = STORAGE_DIR / "autosave" / "music.mp3"
            try:
                if stored_mus.is_file() and stored_mus.stat().st_size >= 200:
                    mus_path = stored_mus
            except OSError:
                mus_path = None
        if mus_path is not None and not mus_mute and mus_vol > 0.001:
            try:
                if mus_path.is_file() and mus_path.stat().st_size >= 200:
                    report(86, "Mezclando canción…")
                    ok_mus, mus_err = _mix_music(concat_path, mus_path, mus_vol)
                    if not ok_mus:
                        return False, mus_err, None
            except OSError:
                pass

        if texts:
            report(88, f"Aplicando {len(texts)} captions…")
            ok, msg = _apply_texts(concat_path, final_path, texts, job_dir)
            if not ok:
                return False, f"Error al aplicar textos: {msg}", None
        else:
            report(92, "Guardando MP4…")
            shutil.move(str(concat_path), str(final_path))

        report(100, f"Listo: {final_path.name}")
        return True, f"Video exportado como {final_path.name}", final_path

    except OSError as e:
        winerr = getattr(e, "winerror", None) or e.errno
        if winerr == 206:
            return False, "Windows no pudo lanzar FFmpeg (comando demasiado largo). Captions van por archivo .ass; reintenta el export.", None
        return False, f"Error: {e}", None
    except Exception as e:
        return False, f"Error: {e}", None
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
        _cleanup_old_temp()
