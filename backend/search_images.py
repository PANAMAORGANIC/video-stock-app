"""Search the public web for still images.

Wikimedia Commons + Openverse only. No Google, no Bing, no API keys.
Import downloads the file and FFmpeg loops it into an MP4 still for Assemble.
"""
from __future__ import annotations

import ipaddress
import socket
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urljoin, urlparse

import httpx

from video_tools import TEMP_DIR, still_from_image

HEADERS = {
    "User-Agent": "Clearview/1.0 (personal video editor; localhost)",
    "Accept": "application/json",
    "Accept-Language": "es,en;q=0.8",
}

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def _ip_blocked(ip: ipaddress._BaseAddress) -> bool:
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return bool(
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
    )


def _host_blocked(host: str) -> bool:
    host = (host or "").strip().strip("[]").lower()
    if not host:
        return True
    parsed = None
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        if host.isdigit():
            try:
                parsed = ipaddress.ip_address(int(host))
            except ValueError:
                parsed = None
    if parsed is not None:
        return _ip_blocked(parsed)
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except (ValueError, IndexError, TypeError):
            continue
        if _ip_blocked(ip):
            return True
    return False


def _ok_image_url(url: str) -> bool:
    u = (url or "").strip()
    if not u.startswith("http://") and not u.startswith("https://"):
        return False
    try:
        p = urlparse(u)
    except Exception:
        return False
    host = (p.hostname or "").lower()
    if not host or _host_blocked(host):
        return False
    path = (p.path or "").lower()
    if path.endswith((".svg", ".pdf", ".mp4", ".webm", ".mov", ".mp3", ".html", ".htm")):
        return False
    return True


def _guess_ext(url: str, content_type: str = "") -> str:
    path = urlparse(url).path.lower()
    for ext in IMAGE_EXT:
        if path.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    ct = (content_type or "").split(";")[0].strip().lower()
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tif",
    }.get(ct, ".jpg")


def _item(image_id: str, title: str, url: str, thumb: str = "", source: str = "web") -> Optional[dict]:
    if not _ok_image_url(url):
        return None
    title = unescape((title or "").strip()) or "Imagen"
    thumb = thumb if _ok_image_url(thumb) else url
    return {
        "video_id": (image_id or url)[:80],
        "title": title[:120],
        "url": url,
        "thumbnail": thumb,
        "duration": "4s",
        "duration_sec": 4.0,
        "channel": source,
        "source": source,
        "platform": "image",
        "kind": "image",
    }


def parse_commons_query(data: Any, max_results: int = 8) -> List[dict]:
    """Wikimedia action=query generator=search payload → image hits."""
    out: List[dict] = []
    if not isinstance(data, dict):
        return out
    pages = ((data.get("query") or {}).get("pages") or {})
    if not isinstance(pages, dict):
        return out
    rows = list(pages.values())
    rows.sort(key=lambda p: int(p.get("index") or 0) if isinstance(p, dict) else 0)
    for page in rows:
        if not isinstance(page, dict):
            continue
        info = page.get("imageinfo") or []
        if not isinstance(info, list) or not info:
            continue
        ii = info[0] if isinstance(info[0], dict) else {}
        mime = str(ii.get("mime") or "").lower()
        if mime and not mime.startswith("image/"):
            continue
        if "svg" in mime:
            continue
        url = str(ii.get("url") or ii.get("thumburl") or "").strip()
        thumb = str(ii.get("thumburl") or url).strip()
        title = str(page.get("title") or "").replace("File:", "").replace("Archivo:", "")
        item = _item(str(page.get("pageid") or title), title, url, thumb, "commons")
        if item:
            out.append(item)
        if len(out) >= max_results:
            break
    return out


def parse_openverse(data: Any, max_results: int = 8) -> List[dict]:
    out: List[dict] = []
    if not isinstance(data, dict):
        return out
    results = data.get("results") or []
    if not isinstance(results, list):
        return out
    for row in results:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        thumb = str(row.get("thumbnail") or url).strip()
        title = str(row.get("title") or row.get("id") or "Imagen")
        item = _item(str(row.get("id") or url), title, url, thumb, "openverse")
        if item:
            out.append(item)
        if len(out) >= max_results:
            break
    return out


def _merge(buckets: List[List[dict]], max_results: int) -> List[dict]:
    seen = set()
    out: List[dict] = []
    for bucket in buckets:
        for it in bucket:
            url = (it.get("url") or "").split("?")[0]
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(it)
            if len(out) >= max_results:
                return out
    return out


def _fold_query(q: str) -> str:
    nk = unicodedata.normalize("NFKD", q or "")
    return "".join(c for c in nk if not unicodedata.combining(c))


def query_passes(q: str) -> List[str]:
    """Original query, then the same words without accents. No new topic."""
    q = " ".join((q or "").split())
    out: List[str] = []
    seen = set()
    for cand in (q, _fold_query(q)):
        cand = " ".join(cand.split())
        key = cand.lower()
        if cand and key not in seen:
            seen.add(key)
            out.append(cand)
    return out


def pack_image_search(
    commons_hits: Optional[List[dict]] = None,
    openverse_hits: Optional[List[dict]] = None,
    commons_error: Optional[str] = None,
    openverse_error: Optional[str] = None,
) -> Dict[str, Any]:
    commons_hits = commons_hits or []
    openverse_hits = openverse_hits or []
    merged = _merge([commons_hits, openverse_hits], 12)
    used: List[str] = []
    if commons_hits:
        used.append("commons")
    if openverse_hits:
        used.append("openverse")
    source = "+".join(used) if used else None
    extra_bits = []
    if commons_error:
        extra_bits.append("Commons: " + commons_error)
    if openverse_error:
        extra_bits.append("Openverse: " + openverse_error)
    extra = "; ".join(extra_bits) if extra_bits else None
    if merged:
        return {
            "results": merged[:12],
            "source": source,
            "error": extra,
            "hard_fail": False,
        }
    if commons_error and openverse_error:
        err = extra or "Commons y Openverse fallaron."
        return {"results": [], "source": None, "error": err, "hard_fail": True}
    parts: List[str] = []
    if commons_error:
        parts.append("Commons: " + commons_error)
    else:
        parts.append("Wikimedia Commons no devolvió imágenes")
    if openverse_error:
        parts.append("Openverse: " + openverse_error)
    else:
        parts.append("Openverse no devolvió imágenes")
    return {
        "results": [],
        "source": None,
        "error": ". ".join(parts) + ".",
        "hard_fail": False,
    }


def _fetch_json(url: str, timeout: float = 12.0) -> Tuple[Any, Optional[str]]:
    try:
        r = httpx.get(url, headers=HEADERS, timeout=timeout, follow_redirects=True)
    except Exception as e:
        return None, str(e)[:240]
    if r.status_code >= 400:
        return None, f"HTTP {r.status_code}"
    try:
        return r.json(), None
    except Exception as e:
        return None, ("parse: " + str(e))[:240]


def _commons(query: str, n: int) -> Tuple[List[dict], Optional[str]]:
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&format=json"
        "&generator=search&gsrsearch=" + quote(query) +
        "&gsrnamespace=6&gsrlimit=" + str(n) +
        "&prop=imageinfo&iiprop=url|mime|size&iiurlwidth=640"
    )
    data, err = _fetch_json(url)
    if err:
        return [], "Commons " + err
    hits = parse_commons_query(data, n)
    return hits, None


def _openverse(query: str, n: int) -> Tuple[List[dict], Optional[str]]:
    url = "https://api.openverse.org/v1/images/?q=" + quote(query) + "&page_size=" + str(n)
    data, err = _fetch_json(url)
    if err:
        return [], "Openverse " + err
    hits = parse_openverse(data, n)
    return hits, None


def _search_pass(query: str, n: int) -> Dict[str, Any]:
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_c = pool.submit(_commons, query, n)
        f_o = pool.submit(_openverse, query, n)
        commons_hits, commons_error = f_c.result()
        openverse_hits, openverse_error = f_o.result()
    return pack_image_search(
        commons_hits=commons_hits,
        openverse_hits=openverse_hits,
        commons_error=commons_error,
        openverse_error=openverse_error,
    )


def search_web_images(query: str, max_results: int = 12) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        return {
            "results": [],
            "source": None,
            "error": "Escribe qué imagen necesitas.",
            "hard_fail": False,
        }
    n = max(4, min(12, int(max_results or 12)))
    last: Optional[Dict[str, Any]] = None
    for qq in query_passes(q):
        packed = _search_pass(qq, n)
        last = packed
        if packed.get("results"):
            return packed
    return last or pack_image_search()


IMAGE_UPLOAD_EXT = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif",
    ".bmp", ".jfif", ".tif", ".tiff",
}
VIDEO_UPLOAD_EXT = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}
AUDIO_UPLOAD_EXT = {".mp3", ".wav", ".m4a", ".aac"}
_IMAGE_MIME_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/pjpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/x-bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/tif": ".tif",
}


def classify_upload_name(name: str, mime: str = "") -> Optional[str]:
    ext = Path(name or "").suffix.lower()
    if ext in IMAGE_UPLOAD_EXT:
        return "image"
    if ext in VIDEO_UPLOAD_EXT:
        return "video"
    if ext in AUDIO_UPLOAD_EXT:
        return "audio"
    m = (mime or "").split(";")[0].strip().lower()
    if m.startswith("image/"):
        return "image"
    if m.startswith("video/"):
        return "video"
    if m.startswith("audio/"):
        return "audio"
    return None


def image_ext_for_upload(name: str, mime: str = "") -> str:
    ext = Path(name or "").suffix.lower()
    if ext in IMAGE_UPLOAD_EXT:
        return ext
    m = (mime or "").split(";")[0].strip().lower()
    return _IMAGE_MIME_EXT.get(m, ".jpg")


def import_local_image(
    src: Path,
    title: str = "",
    seconds: float = 4.0,
) -> Tuple[bool, str, Optional[Path]]:
    """Turn a local still into a 4s MP4 in storage/clips (same path as web import)."""
    from video_tools import safe_filename
    if not src or not Path(src).is_file():
        return False, "No se encuentra la imagen", None
    seconds = max(1.0, min(30.0, float(seconds or 4)))
    stem = "img_" + safe_filename(title or Path(src).stem or "drop")[:40]
    return still_from_image(Path(src), seconds, stem)


def import_web_image(
    url: str,
    title: str = "",
    seconds: float = 4.0,
) -> Tuple[bool, str, Optional[Path]]:
    url = (url or "").strip()
    if not _ok_image_url(url):
        return False, "URL de imagen no válida", None
    seconds = max(1.0, min(30.0, float(seconds or 4)))
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    current = url
    r = None
    try:
        for _ in range(4):
            if not _ok_image_url(current):
                return False, "URL de imagen no válida", None
            r = httpx.get(current, headers=HEADERS, timeout=25.0, follow_redirects=False)
            if r.status_code in {301, 302, 303, 307, 308}:
                loc = (r.headers.get("location") or "").strip()
                if not loc:
                    return False, "No se pudo descargar la imagen: redirect sin Location", None
                current = urljoin(current, loc)
                continue
            r.raise_for_status()
            break
        else:
            return False, "No se pudo descargar la imagen: demasiados redirects", None
    except Exception as e:
        return False, f"No se pudo descargar la imagen: {e}", None
    if r is None:
        return False, "No se pudo descargar la imagen", None
    ctype = (r.headers.get("content-type") or "").lower()
    if ctype and not ctype.startswith("image/") and "octet-stream" not in ctype:
        return False, f"Eso no es una imagen ({ctype.split(';')[0]})", None
    if len(r.content) < 80:
        return False, "La imagen está vacía", None
    if len(r.content) > 12 * 1024 * 1024:
        return False, "La imagen pesa más de 12 MB", None
    ext = _guess_ext(str(r.url), ctype)
    tmp = TEMP_DIR / f"imgdl_{abs(hash(url)) % 10**8}{ext}"
    try:
        tmp.write_bytes(r.content)
        return import_local_image(tmp, title or Path(urlparse(url).path).stem or "web", seconds)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
