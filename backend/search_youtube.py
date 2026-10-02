"""
YouTube search + transcript extraction
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"
YDL_CACHE_DIR = STORAGE_DIR / "ydl-cache"
_WIN_BAD = re.compile(r'[\\/:*?"<>|]+')


def sanitize_search_query(query: str) -> str:
    """Strip Windows-illegal filename chars so ytsearchN:query is a safe cache key."""
    q = (query or "").strip()
    q = _WIN_BAD.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q[:200] or "video"


def ydl_cache_dir() -> str:
    YDL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return str(YDL_CACHE_DIR)


def is_netscape_cookiefile(path: Path) -> bool:
    """True only for Netscape cookie text. Skip Chrome SQLite DBs (errno 22 on Windows)."""
    try:
        raw = Path(path).read_bytes()[:800]
    except OSError:
        return False
    if not raw or raw.startswith(b"SQLite") or b"\x00" in raw[:80]:
        return False
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return False
    head = text.lstrip()
    if head.startswith("# Netscape") or "HTTP Cookie File" in head[:80]:
        return True
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return line.count("\t") >= 6
    return False


def safe_ydl_extract(opts: Dict[str, Any], url: str) -> Optional[dict]:
    """extract_info without leaking OSError from YoutubeDL.__exit__ / cookiefile / cache."""
    ydl = None
    try:
        ydl = yt_dlp.YoutubeDL(opts)
        return ydl.extract_info(url, download=False)
    except OSError as e:
        print(f"yt-dlp OSError: {e}")
        return None
    except Exception as e:
        print(f"yt-dlp: {e}")
        return None
    finally:
        if ydl is not None:
            try:
                ydl.__exit__(None, None, None)
            except Exception:
                pass


def search_youtube_videos(query: str, max_results: int = 8) -> List[Dict]:
    """
    Busca videos en YouTube usando yt-dlp (sin API key).
    Solo el extractor de YouTube (ytsearchN). No activa TikTok ni el resto.
    """
    from concurrent.futures import ThreadPoolExecutor

    from search_rank import format_duration, llm_query_terms, understand_query

    q = sanitize_search_query(query)
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "force_generic_extractor": False,
        "default_search": "ytsearch",
        "noplaylist": True,
        "ignoreerrors": True,
        "skip_download": True,
        "cachedir": ydl_cache_dir(),
        "socket_timeout": 15,
        "retries": 0,
        "extractor_retries": 0,
        "allowed_extractors": ["youtube", "youtube:search", "youtube:tab", "youtube:search_url"],
    }

    n = max(1, min(20, int(max_results or 8)))
    seen = set()
    results: List[Dict] = []

    def _absorb(info) -> None:
        if not info or "entries" not in info:
            return
        for entry in info.get("entries") or []:
            if not entry:
                continue
            video_id = entry.get("id")
            if not video_id or video_id in seen:
                continue
            seen.add(video_id)
            desc = entry.get("description") or entry.get("alt_title") or ""
            results.append({
                "video_id": video_id,
                "title": entry.get("title") or "Sin título",
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail": entry.get("thumbnail") or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                "duration": format_duration(entry.get("duration_string") or entry.get("duration")),
                "duration_sec": entry.get("duration"),
                "channel": entry.get("channel") or entry.get("uploader"),
                "description": str(desc)[:500],
                "platform": "youtube",
            })

    # Original words first. EN topic expansion is search 2 only if needed.
    llm_fut = None
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        llm_fut = pool.submit(llm_query_terms, query)
        _absorb(safe_ydl_extract(ydl_opts, f"ytsearch{n}:{q}"))
        llm = None
        if llm_fut.done():
            try:
                llm = llm_fut.result()
            except Exception:
                llm = None
        en = ""
        if llm and isinstance(llm, dict):
            en_terms = [str(x) for x in (llm.get("terms_en") or []) if str(x).strip()]
            vis = [str(x) for x in (llm.get("visuals") or []) if str(x).strip()]
            en = " ".join((en_terms + vis)[:8])
        if not en:
            u = understand_query(query)
            strs = u.get("search_strings") or []
            en = strs[0] if strs else ""
        en = sanitize_search_query(en)
        if en and en != q and len(results) < n:
            extra_n = max(4, n - len(results))
            _absorb(safe_ydl_extract(ydl_opts, f"ytsearch{extra_n}:{en}"))
    finally:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            pool.shutdown(wait=False)
    return results[:n]


def get_transcript_with_timestamps(video_id: str, languages: List[str] = None) -> Optional[List[Dict]]:
    """
    Obtiene el transcript con timestamps.
    Compatible con la API actual de youtube-transcript-api.
    """
    if languages is None:
        languages = ["es", "es-419", "es-ES", "en", "en-US", "en-GB"]

    try:
        api = YouTubeTranscriptApi()

        data = None
        try:
            data = api.fetch(video_id, languages=languages)
        except Exception:
            # Fallback: listar transcripts disponibles
            try:
                transcript_list = api.list(video_id)
                chosen = None
                for t in transcript_list:
                    lang = getattr(t, "language_code", None) or ""
                    if lang in languages:
                        chosen = t
                        break
                if chosen is None:
                    for t in transcript_list:
                        chosen = t
                        break
                if chosen is not None:
                    data = chosen.fetch()
            except Exception as e2:
                print(f"No transcript for {video_id}: {e2}")
                return None

        if not data:
            return None

        segments = []
        for item in data:
            if hasattr(item, "start"):
                segments.append({
                    "start": float(item.start),
                    "duration": float(getattr(item, "duration", 0) or 0),
                    "text": (getattr(item, "text", "") or "").strip()
                })
            elif isinstance(item, dict):
                segments.append({
                    "start": float(item.get("start", 0)),
                    "duration": float(item.get("duration", 0)),
                    "text": (item.get("text") or "").strip()
                })

        return segments if segments else None

    except Exception as e:
        print(f"Error transcript {video_id}: {e}")
        return None
