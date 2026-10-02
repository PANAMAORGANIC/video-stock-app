"""
Búsqueda de footage en YouTube, Wikimedia Commons, Internet Archive y Dailymotion.
YouTube: yt-dlp ytsearch (solo extractor YouTube). Commons/Archive: HTTP público, sin API key.
TikTok/Instagram: solo si el usuario elige esa plataforma o pega una URL.
"""

from __future__ import annotations

import html as htmlmod
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse

import httpx

from search_youtube import (
    search_youtube_videos,
    safe_ydl_extract,
    ydl_cache_dir,
    is_netscape_cookiefile,
)
from search_rank import (
    filter_and_rank,
    format_duration,
    parse_duration_sec,
)

STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"
IG_URL_RE = re.compile(
    r"https?://(?:www\.)?instagram\.com/(reels?|p|tv)/([A-Za-z0-9_-]+)",
    re.I,
)
IG_SHORT_RE = re.compile(
    r"instagram\.com/(reels?|p|tv)/([A-Za-z0-9_-]+)",
    re.I,
)

DDG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

YDL_BASE = {
    "quiet": True,
    "no_warnings": True,
    "ignoreerrors": True,
    "skip_download": True,
    "noplaylist": False,
    "socket_timeout": 15,
    "retries": 0,
    "extractor_retries": 0,
}


def _ydl_opts(extra: Optional[dict] = None, cookiefile: Optional[str] = None) -> dict:
    opts = {**YDL_BASE, "cachedir": ydl_cache_dir()}
    if extra:
        opts.update(extra)
    if cookiefile and is_netscape_cookiefile(Path(cookiefile)):
        opts["cookiefile"] = cookiefile
    return opts

YT_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?[^#]*v=|shorts/|embed/|live/)|youtu\.be/)([A-Za-z0-9_-]{11})",
    re.I,
)
DM_ID_RE = re.compile(
    r"(?:dailymotion\.com/(?:embed/)?video/|dai\.ly/)([a-zA-Z0-9]+)",
    re.I,
)
TT_ID_RE = re.compile(
    r"tiktok\.com/@[^/]+/video/(\d+)|tiktok\.com/t/([A-Za-z0-9]+)|vm\.tiktok\.com/([A-Za-z0-9]+)",
    re.I,
)
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.I)


def _fmt_duration(entry: dict) -> Optional[str]:
    ds = entry.get("duration_string")
    if ds:
        return str(ds)
    d = entry.get("duration")
    if d is None:
        return None
    try:
        total = int(float(d))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"
    except Exception:
        return None


def _thumb(entry: dict) -> str:
    t = entry.get("thumbnail")
    if t:
        return t
    thumbs = entry.get("thumbnails") or []
    if thumbs:
        return thumbs[-1].get("url") or ""
    return ""


def _hashtag(query: str) -> str:
    words = re.findall(r"[a-zA-Z0-9áéíóúüñ]+", query.lower())
    tag = "".join(words[:3]) or "video"
    return tag[:40]


def _normalize(entry: dict, platform: str, fallback_url: Optional[str] = None) -> Optional[dict]:
    if not entry:
        return None
    video_id = str(entry.get("id") or "").strip()
    url = entry.get("webpage_url") or entry.get("url") or fallback_url or ""
    url = str(url).strip()
    if url and not url.startswith("http"):
        if platform == "dailymotion" and video_id:
            url = f"https://www.dailymotion.com/video/{video_id}"
        elif platform == "youtube" and video_id:
            url = f"https://www.youtube.com/watch?v={video_id}"
        else:
            url = fallback_url or url
    if not url or not str(url).startswith("http"):
        return None
    if not video_id:
        video_id = url.rstrip("/").split("/")[-1].split("?")[0]
    title = (entry.get("title") or entry.get("fulltitle") or "Sin título").strip()
    if title in {"", "NA"}:
        title = "Sin título"
    desc = entry.get("description") or entry.get("alt_title") or ""
    dur = _fmt_duration(entry) or format_duration(entry.get("duration"))
    return {
        "video_id": video_id,
        "title": title,
        "url": url,
        "thumbnail": _thumb(entry),
        "duration": dur,
        "duration_sec": parse_duration_sec(entry.get("duration") or dur),
        "channel": entry.get("channel") or entry.get("uploader") or entry.get("creator"),
        "description": str(desc)[:500],
        "platform": platform,
    }


def _ydl_playlist(url: str, max_results: int, platform: str) -> List[dict]:
    opts = _ydl_opts({
        "extract_flat": "in_playlist",
        "playlistend": max_results,
    })
    results: List[dict] = []
    try:
        info = safe_ydl_extract(opts, url)
        entries = []
        if not info:
            return []
        if "entries" in info:
            entries = [e for e in (info.get("entries") or []) if e]
        else:
            entries = [info]
        for entry in entries[:max_results]:
            item = _normalize(entry, platform)
            if item:
                results.append(item)
    except Exception as e:
        print(f"Error búsqueda {platform}: {e}")
    return results


def _ydl_one(url: str, platform: str) -> Optional[dict]:
    cookies = None
    if platform == "instagram" or "instagram.com" in (url or "").lower():
        cookies = _instagram_cookiefile()
    opts = _ydl_opts({"extract_flat": False, "noplaylist": True}, cookiefile=cookies)
    try:
        info = safe_ydl_extract(opts, url)
        if info and info.get("_type") == "playlist":
            entries = [e for e in (info.get("entries") or []) if e]
            info = entries[0] if entries else info
        return _normalize(info or {}, platform, fallback_url=url)
    except Exception as e:
        print(f"Error metadata {platform} {url}: {e}")
        return None


def _extract_urls(text: str) -> List[str]:
    found = []
    for raw in URL_RE.findall(text or ""):
        url = raw.rstrip(").,;]")
        if url not in found:
            found.append(url)
    return found


def detect_platform_from_url(url: str) -> str:
    u = (url or "").lower()
    if "youtube.com" in u or "youtu.be" in u:
        return "youtube"
    if "dailymotion.com" in u or "dai.ly" in u:
        return "dailymotion"
    if "tiktok.com" in u:
        return "tiktok"
    if "instagram.com" in u:
        return "instagram"
    return "web"


def _fallback_from_url(url: str) -> Optional[dict]:
    plat = detect_platform_from_url(url)
    if plat == "youtube":
        m = YT_ID_RE.search(url)
        if not m:
            return None
        vid = m.group(1)
        return {
            "video_id": vid,
            "title": f"YouTube · {vid}",
            "url": f"https://www.youtube.com/watch?v={vid}",
            "thumbnail": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
            "duration": None,
            "channel": None,
            "platform": "youtube",
        }
    if plat == "dailymotion":
        m = DM_ID_RE.search(url)
        if not m:
            return None
        vid = m.group(1)
        return {
            "video_id": vid,
            "title": f"Dailymotion · {vid}",
            "url": f"https://www.dailymotion.com/video/{vid}",
            "thumbnail": f"https://www.dailymotion.com/thumbnail/video/{vid}",
            "duration": None,
            "channel": None,
            "platform": "dailymotion",
        }
    if plat == "tiktok":
        m = TT_ID_RE.search(url)
        vid = next((g for g in (m.groups() if m else ()) if g), None)
        if not vid:
            vid = url.rstrip("/").split("/")[-1].split("?")[0]
        return {
            "video_id": vid,
            "title": f"TikTok · {vid}",
            "url": url.split("?")[0],
            "thumbnail": "",
            "duration": None,
            "channel": None,
            "platform": "tiktok",
        }
    if plat == "instagram":
        m = IG_URL_RE.search(url) or IG_SHORT_RE.search(url)
        if not m:
            return None
        return _ig_item(m.group(1), m.group(2))
    return {
        "video_id": url.rstrip("/").split("/")[-1].split("?")[0] or "video",
        "title": url,
        "url": url,
        "thumbnail": "",
        "duration": None,
        "channel": None,
        "platform": plat,
    }


def resolve_video_url(url: str) -> Optional[dict]:
    """Abre un enlace concreto (YouTube, Dailymotion, TikTok, Instagram, etc.)."""
    url = (url or "").strip()
    if not url.startswith("http"):
        return None
    plat = detect_platform_from_url(url)
    item = _ydl_one(url, plat if plat != "web" else "youtube")
    if item:
        item["platform"] = plat if plat != "web" else (item.get("platform") or "youtube")
        return item
    return _fallback_from_url(url)


def _ddg_links(query: str, site: str, max_results: int) -> List[str]:
    links: List[str] = []
    host = (site or "").split("/")[0]
    try:
        r = httpx.post(
            "https://html.duckduckgo.com/html/",
            data={"q": f"site:{host} {query}"},
            headers=DDG_HEADERS,
            timeout=20.0,
            follow_redirects=True,
        )
        hrefs = re.findall(r'href="([^"]+)"', r.text)
        for raw in hrefs:
            h = raw
            if "uddg=" in h:
                qs = parse_qs(urlparse(h).query)
                h = unquote(qs.get("uddg", [h])[0])
            h = unquote(h).split("&")[0].split("#")[0]
            if not h.startswith("http"):
                continue
            if site not in h:
                continue
            if h not in links:
                links.append(h)
            if len(links) >= max_results * 4:
                break
    except Exception as e:
        print(f"DuckDuckGo error ({site}): {e}")
    return links


def _ig_kind(raw: str) -> str:
    k = (raw or "p").lower().rstrip("s")
    if k == "reel":
        return "reel"
    if k == "tv":
        return "tv"
    return "p"


def _ig_item(kind: str, code: str, title: Optional[str] = None) -> dict:
    kind = _ig_kind(kind)
    code = code.strip()
    url = f"https://www.instagram.com/{kind}/{code}/"
    return {
        "video_id": code,
        "title": title or f"Instagram · {code}",
        "url": url,
        "thumbnail": f"https://www.instagram.com/p/{code}/media/?size=l",
        "duration": None,
        "channel": None,
        "platform": "instagram",
    }


def _ig_title(code: str) -> str:
    try:
        r = httpx.get(
            f"https://www.instagram.com/p/{code}/embed/",
            headers=DDG_HEADERS,
            timeout=12.0,
            follow_redirects=True,
        )
        m = re.search(r"shared by (?:&#064;|@)([A-Za-z0-9._]+)", r.text, re.I)
        if m:
            return f"@{m.group(1)} · Instagram"
        m = re.search(r'alt="([^"]{8,80})"', r.text)
        if m:
            return htmlmod.unescape(m.group(1))
    except Exception:
        pass
    return f"Instagram · {code}"


def _bing_instagram_posts(query: str, max_results: int) -> List[Tuple[str, str]]:
    """Instagram no indexa bien; Bing Videos a veces incluye /p/SHORTCODE."""
    variants = [
        f"instagram reel {query}",
        f"{query} instagram reels",
        f"{query} instagram.com/p/",
    ]
    found: List[Tuple[str, str]] = []
    seen = set()
    try:
        with httpx.Client(headers=DDG_HEADERS, follow_redirects=True, timeout=20.0) as client:
            for q in variants:
                if len(found) >= max_results:
                    break
                for url, params in (
                    ("https://www.bing.com/videos/search", {"q": q}),
                    ("https://www.bing.com/search", {"q": q, "count": "20"}),
                ):
                    try:
                        r = client.get(url, params=params)
                    except Exception:
                        continue
                    for kind, code in IG_SHORT_RE.findall(r.text or ""):
                        if code in seen:
                            continue
                        seen.add(code)
                        found.append((_ig_kind(kind), code))
                        if len(found) >= max_results:
                            break
                    if len(found) >= max_results:
                        break
    except Exception as e:
        print(f"Bing Instagram error: {e}")
    return found


def _instagram_cookiefile() -> Optional[str]:
    path = STORAGE_DIR / "instagram_cookies.txt"
    if path.exists() and path.stat().st_size > 80 and is_netscape_cookiefile(path):
        return str(path)
    return None


def search_dailymotion(
    query: str, max_results: int = 6, warnings: Optional[List[str]] = None
) -> List[dict]:
    q = quote(query.strip(), safe="")
    found = _ydl_playlist(
        f"https://www.dailymotion.com/search/{q}/videos",
        max_results,
        "dailymotion",
    )
    if len(found) >= max_results:
        return found[:max_results]
    if not found and warnings is not None:
        warnings.append(
            "dailymotion: búsqueda directa sin resultados, usando búsqueda web"
        )

    dm_re = re.compile(r"dailymotion\.com/video/([a-zA-Z0-9]+)", re.I)
    extra_urls = []
    for link in _ddg_links(query, "dailymotion.com/video", max_results):
        m = dm_re.search(link)
        if not m:
            continue
        url = f"https://www.dailymotion.com/video/{m.group(1)}"
        if url not in extra_urls and all(url != x.get("url") for x in found):
            extra_urls.append(url)
        if len(found) + len(extra_urls) >= max_results:
            break

    for url in extra_urls:
        item = _ydl_one(url, "dailymotion")
        if not item:
            continue
        found.append(item)
        if len(found) >= max_results:
            break
    return found[:max_results]


def search_tiktok(
    query: str, max_results: int = 6, warnings: Optional[List[str]] = None
) -> List[dict]:
    tag = _hashtag(query)
    found = _ydl_playlist(f"https://www.tiktok.com/tag/{quote(tag)}", max_results, "tiktok")
    if len(found) >= max_results:
        return found[:max_results]
    if not found and warnings is not None:
        warnings.append(
            "tiktok: búsqueda directa sin resultados, usando búsqueda web"
        )

    video_re = re.compile(r"tiktok\.com/@[^/]+/video/\d+", re.I)
    extra_urls = []
    for link in _ddg_links(query + " video", "tiktok.com", max_results):
        m = video_re.search(link)
        if m:
            url = "https://www." + m.group(0).lstrip("www.")
            if url not in extra_urls and all(url != x.get("url") for x in found):
                extra_urls.append(url)
        if len(found) + len(extra_urls) >= max_results:
            break

    for url in extra_urls:
        item = _ydl_one(url, "tiktok")
        if not item:
            continue
        found.append(item)
        if len(found) >= max_results:
            break
    return found[:max_results]


def search_instagram(query: str, max_results: int = 6) -> List[dict]:
    """
    Orden:
    1) URL pegada
    2) Cuenta del usuario (sessionid/cookies)
    3) hashtag yt-dlp con cookies
    4) Bing / DuckDuckGo (suelen fallar)
    """
    from instagram_account import search_with_account

    found: List[dict] = []
    seen = set()

    def add(kind: str, code: str, title: Optional[str] = None) -> None:
        code = (code or "").strip()
        if not code or code in seen:
            return
        seen.add(code)
        found.append(_ig_item(kind, code, title))

    for kind, code in IG_URL_RE.findall(query or ""):
        add(kind, code)

    if len(found) < max_results:
        try:
            acc_items, _err = search_with_account(query, max_results)
        except OSError as e:
            print(f"Instagram account search OSError: {e}")
            acc_items = []
        except Exception as e:
            print(f"Instagram account search: {e}")
            acc_items = []
        for it in acc_items:
            vid = it.get("video_id")
            if vid and vid not in seen:
                seen.add(vid)
                found.append(it)
            if len(found) >= max_results:
                break

    cookiefile = _instagram_cookiefile()
    if cookiefile and len(found) < max_results:
        tag = _hashtag(query)
        first = (re.findall(r"[a-zA-Z0-9]+", query.lower()) or ["video"])[0]
        for tag_try in dict.fromkeys([first, tag]):
            opts = _ydl_opts({
                "extract_flat": "in_playlist",
                "playlistend": max_results,
            }, cookiefile=cookiefile)
            try:
                info = safe_ydl_extract(
                    opts,
                    f"https://www.instagram.com/explore/tags/{quote(tag_try)}/",
                )
                for entry in (info or {}).get("entries") or []:
                    item = _normalize(entry, "instagram")
                    if item and item["video_id"] not in seen:
                        seen.add(item["video_id"])
                        found.append(item)
                    if len(found) >= max_results:
                        break
            except OSError as e:
                print(f"Instagram cookies/tag {tag_try} OSError: {e}")
            except Exception as e:
                print(f"Instagram cookies/tag {tag_try}: {e}")
            if len(found) >= max_results:
                break

    if len(found) < max_results:
        for kind, code in _bing_instagram_posts(query, max_results):
            add(kind, code)
            if len(found) >= max_results:
                break

    if len(found) < max_results:
        for link in _ddg_links(query + " reel", "instagram.com", max_results):
            m = IG_SHORT_RE.search(link)
            if m:
                add(m.group(1), m.group(2))
            if len(found) >= max_results:
                break

    for item in found:
        if item["title"].startswith("Instagram ·"):
            item["title"] = _ig_title(item["video_id"])
            if item["title"].startswith("@"):
                item["channel"] = item["title"].split(" · ")[0]

    return found[:max_results]


def _run_platform(name: str, fn, timeout: float = 22.0) -> Tuple[List[dict], Optional[str]]:
    """Run one extractor with a hard timeout. OSError (errno 22) never bubbles."""
    from concurrent.futures import TimeoutError as FuturesTimeout

    def _call():
        items = fn() or []
        for it in items:
            if isinstance(it, dict) and not it.get("platform"):
                it["platform"] = name
        return items

    ex = ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(_call)
        items = fut.result(timeout=timeout) or []
        return items, None
    except FuturesTimeout:
        print(f"Error búsqueda {name}: timeout {timeout}s")
        return [], f"{name}: tiempo agotado"
    except OSError as e:
        print(f"Error búsqueda {name}: {e}")
        return [], f"{name}: {e}"
    except Exception as e:
        print(f"Error búsqueda {name}: {e}")
        return [], f"{name}: {e}"
    finally:
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            ex.shutdown(wait=False)


_COMMONS_HEADERS = {
    "User-Agent": "Clearview/1.0 (personal video editor; localhost)",
    "Accept": "application/json",
}


def _http_json(url: str, timeout: float = 12.0) -> Tuple[Optional[dict], Optional[str]]:
    try:
        r = httpx.get(url, headers=_COMMONS_HEADERS, timeout=timeout, follow_redirects=True)
    except Exception as e:
        return None, str(e)[:240]
    if r.status_code >= 400:
        return None, f"HTTP {r.status_code}"
    try:
        data = r.json()
    except Exception as e:
        return None, ("parse: " + str(e))[:240]
    return data if isinstance(data, dict) else None, None


def search_commons_video(query: str, max_results: int = 6) -> List[dict]:
    """Wikimedia Commons file namespace, videos only. No API key."""
    n = max(2, min(10, int(max_results or 6)))
    q = quote("filetype:video " + (query or "").strip())
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&format=json"
        "&generator=search&gsrsearch=" + q +
        "&gsrnamespace=6&gsrlimit=" + str(n) +
        "&prop=imageinfo&iiprop=url|mime|size|extmetadata&iiurlwidth=640"
    )
    data, err = _http_json(url)
    if err or not data:
        if err:
            print(f"Commons video: {err}")
        return []
    pages = ((data.get("query") or {}).get("pages") or {})
    if not isinstance(pages, dict):
        return []
    rows = list(pages.values())
    rows.sort(key=lambda p: int(p.get("index") or 0) if isinstance(p, dict) else 0)
    out: List[dict] = []
    for page in rows:
        if not isinstance(page, dict):
            continue
        info = page.get("imageinfo") or []
        ii = info[0] if isinstance(info, list) and info and isinstance(info[0], dict) else {}
        mime = str(ii.get("mime") or "").lower()
        if mime and not mime.startswith("video/"):
            continue
        file_url = str(ii.get("url") or "").strip()
        if not file_url.startswith("http"):
            continue
        title = str(page.get("title") or "").replace("File:", "").replace("Archivo:", "")
        thumb = str(ii.get("thumburl") or "").strip()
        desc = ""
        meta = ii.get("extmetadata") or {}
        if isinstance(meta, dict):
            for key in ("ImageDescription", "ObjectName"):
                node = meta.get(key) or {}
                if isinstance(node, dict):
                    desc = str(node.get("value") or "")
                    break
        dur = None
        if isinstance(meta, dict) and isinstance(meta.get("Duration"), dict):
            dur = meta["Duration"].get("value")
        vid = str(page.get("pageid") or title)
        out.append({
            "video_id": vid,
            "title": htmlmod.unescape(title)[:160] or "Commons",
            "url": file_url,
            "thumbnail": thumb,
            "duration": format_duration(dur),
            "duration_sec": parse_duration_sec(dur),
            "channel": "Wikimedia Commons",
            "description": htmlmod.unescape(re.sub(r"<[^>]+>", " ", desc or ""))[:500],
            "platform": "commons",
        })
        if len(out) >= n:
            break
    return out


def search_archive_video(query: str, max_results: int = 6) -> List[dict]:
    """Internet Archive movies. No API key."""
    n = max(2, min(10, int(max_results or 6)))
    q = (query or "").strip()
    url = (
        "https://archive.org/advancedsearch.php?q="
        + quote(f"({q}) AND mediatype:(movies)")
        + "&fl[]=identifier&fl[]=title&fl[]=description&fl[]=runtime"
        + "&rows=" + str(n) + "&page=1&output=json"
    )
    data, err = _http_json(url, timeout=14.0)
    if err or not data:
        if err:
            print(f"Archive.org: {err}")
        return []
    docs = ((data.get("response") or {}).get("docs") or [])
    if not isinstance(docs, list):
        return []
    out: List[dict] = []
    for row in docs:
        if not isinstance(row, dict):
            continue
        ident = str(row.get("identifier") or "").strip()
        if not ident:
            continue
        title = str(row.get("title") or ident)
        desc = row.get("description") or ""
        if isinstance(desc, list):
            desc = " ".join(str(x) for x in desc[:2])
        runtime = row.get("runtime")
        if isinstance(runtime, list):
            runtime = runtime[0] if runtime else None
        out.append({
            "video_id": ident,
            "title": title[:160],
            "url": f"https://archive.org/details/{ident}",
            "thumbnail": f"https://archive.org/services/img/{ident}",
            "duration": format_duration(runtime),
            "duration_sec": parse_duration_sec(runtime),
            "channel": "Internet Archive",
            "description": str(desc)[:500],
            "platform": "archive",
        })
        if len(out) >= n:
            break
    return out


def search_footage_pack(
    query: str, platform: str = "all", max_results: int = 8
) -> Tuple[List[dict], List[str]]:
    """Search one or all platforms. Never raises. Empty list + warnings on failure."""
    platform = (platform or "all").lower().strip()
    q = query.strip()
    warnings: List[str] = []
    n = max(1, int(max_results) or 8)
    urls = _extract_urls(q)
    if urls:
        found: List[dict] = []
        seen = set()
        for url in urls[:n]:
            try:
                item = resolve_video_url(url)
            except Exception as e:
                warnings.append(str(e)[:200])
                continue
            if not item:
                continue
            key = (item.get("platform"), item.get("video_id") or item.get("url"))
            if key in seen:
                continue
            seen.add(key)
            found.append(item)
        if found:
            return found, warnings
    if IG_URL_RE.search(q) and platform not in {"instagram", "all"}:
        platform = "instagram"
    noisy = ("tiktok.com" in q.lower()) or ("instagram.com" in q.lower())
    jobs = {
        "youtube": lambda: search_youtube_videos(q, n),
        "dailymotion": lambda: search_dailymotion(q, n, warnings),
        "tiktok": lambda: search_tiktok(q, n, warnings),
        "instagram": lambda: search_instagram(q, n),
        "commons": lambda: search_commons_video(q, n),
        "archive": lambda: search_archive_video(q, n),
    }
    if platform in jobs:
        items, err = _run_platform(platform, jobs[platform])
        if err:
            warnings.append(err)
        kept, dropped, raw = filter_and_rank(items, q, n)
        if dropped and raw:
            warnings.append(f"{len(kept)} de {raw} coinciden")
        return kept, warnings
    if platform != "all":
        return [], [f"Plataforma no soportada: {platform}"]
    yt_n = max(8, min(12, n + 4))
    side_n = max(4, min(6, n))
    # Default all: free documentary sources. TikTok/Instagram only on explicit pick or URL.
    scaled = {
        "youtube": lambda: search_youtube_videos(q, yt_n),
        "commons": lambda: search_commons_video(q, side_n),
        "archive": lambda: search_archive_video(q, side_n),
        "dailymotion": lambda: search_dailymotion(q, side_n, warnings),
    }
    if noisy:
        scaled["tiktok"] = lambda: search_tiktok(q, side_n, warnings)
        scaled["instagram"] = lambda: search_instagram(q, side_n)
    bucket: Dict[str, List[dict]] = {k: [] for k in scaled}
    workers = min(4, max(1, len(scaled)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_run_platform, name, fn): name for name, fn in scaled.items()}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                items, err = fut.result()
            except Exception as e:
                items, err = [], f"{name}: {e}"
            bucket[name] = items or []
            if err:
                warnings.append(err)
    ordered: List[dict] = []
    for name in ("youtube", "commons", "archive", "dailymotion", "tiktok", "instagram"):
        ordered.extend(bucket.get(name) or [])
    kept, dropped, raw = filter_and_rank(ordered, q, n)
    if dropped and raw:
        warnings.append(f"{len(kept)} de {raw} coinciden")
    return kept, warnings


def search_footage_videos(query: str, platform: str = "youtube", max_results: int = 8) -> List[dict]:
    items, _warn = search_footage_pack(query, platform, max_results)
    return items
