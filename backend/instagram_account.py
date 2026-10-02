"""
Sesión de Instagram del usuario (cookies / sessionid) para buscar con su cuenta.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"
COOKIE_PATH = STORAGE_DIR / "instagram_cookies.txt"
IG_APP_ID = "936619743392459"
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def cookie_path() -> Path:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return COOKIE_PATH


def parse_netscape_cookies(text: str) -> Dict[str, str]:
    cookies: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 7:
            domain = parts[0].lstrip(".")
            if "instagram.com" in domain or domain in {"instagram.com", "www.instagram.com"}:
                cookies[parts[5]] = parts[6]
        elif line.startswith("sessionid="):
            cookies["sessionid"] = line.split("=", 1)[1].strip()
    return cookies


def load_cookie_map() -> Dict[str, str]:
    path = cookie_path()
    if not path.exists():
        return {}
    try:
        return parse_netscape_cookies(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def write_netscape(cookies: Dict[str, str]) -> Path:
    path = cookie_path()
    lines = [
        "# Netscape HTTP Cookie File",
        "# Personal Video Stock — Instagram",
    ]
    for name, value in cookies.items():
        if not name or value is None:
            continue
        secure = "TRUE" if name in {"sessionid", "csrftoken", "ds_user_id"} else "FALSE"
        lines.append(f".instagram.com\tTRUE\t/\t{secure}\t1999999999\t{name}\t{value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_cookies_file(raw: str) -> Path:
    parsed = parse_netscape_cookies(raw)
    if "sessionid" not in parsed:
        # Puede ser un dump crudo; guardar tal cual si parece netscape
        if "sessionid" not in raw:
            raise ValueError("El archivo no contiene la cookie sessionid de Instagram.")
        path = cookie_path()
        path.write_text(raw, encoding="utf-8")
        return path
    return write_netscape(parsed)


def save_sessionid(sessionid: str, csrftoken: Optional[str] = None) -> Path:
    sid = (sessionid or "").strip().strip('"')
    if not sid or len(sid) < 10:
        raise ValueError("sessionid vacío o demasiado corto")
    cookies = {"sessionid": sid}
    if csrftoken:
        cookies["csrftoken"] = csrftoken.strip()
    return write_netscape(cookies)


def clear_session() -> None:
    path = cookie_path()
    path.unlink(missing_ok=True)


def import_from_browser(browser: str = "chrome") -> Tuple[bool, str]:
    """Intenta leer cookies de Chrome/Edge/Firefox. Cierra el navegador si falla."""
    browser = (browser or "chrome").lower().strip()
    if browser not in {"chrome", "edge", "firefox", "brave", "opera"}:
        return False, f"Navegador no soportado: {browser}"
    try:
        from yt_dlp.cookies import extract_cookies_from_browser
        jar = extract_cookies_from_browser(browser)
    except Exception as e:
        return False, (
            f"No pude leer cookies de {browser}. Cierra {browser} por completo "
            "(todas las ventanas) y reintenta, o pega el sessionid / sube cookies.txt. "
            f"Detalle: {e}"
        )

    cookies: Dict[str, str] = {}
    for c in jar:
        domain = getattr(c, "domain", "") or ""
        if "instagram.com" in domain:
            cookies[c.name] = c.value

    if "sessionid" not in cookies:
        return False, (
            f"En {browser} no hay sesión de Instagram. Entra en instagram.com, "
            "inicia sesión, cierra el navegador y reintenta."
        )
    write_netscape(cookies)
    return True, "Cookies de Instagram importadas."


def _client(cookies: Dict[str, str]) -> httpx.Client:
    csrf = cookies.get("csrftoken") or ""
    headers = {
        "User-Agent": BROWSER_UA,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
        "X-IG-App-ID": IG_APP_ID,
        "X-ASBD-ID": "129477",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://www.instagram.com/",
        "X-CSRFToken": csrf,
    }
    return httpx.Client(
        headers=headers,
        cookies=cookies,
        follow_redirects=True,
        timeout=25.0,
    )


def _json(client: httpx.Client, url: str, params: Optional[dict] = None, method: str = "GET", data=None) -> Optional[dict]:
    try:
        if method == "POST":
            r = client.post(url, params=params, data=data)
        else:
            r = client.get(url, params=params)
        if r.status_code >= 400:
            print(f"IG {r.status_code} {url} {r.text[:180]}")
            return None
        ctype = r.headers.get("content-type", "")
        if "json" not in ctype and not r.text[:1] in "{[":
            return None
        return r.json()
    except Exception as e:
        print(f"IG request error {url}: {e}")
        return None


def whoami() -> Dict[str, Any]:
    cookies = load_cookie_map()
    if "sessionid" not in cookies:
        return {"connected": False, "username": None, "message": "No hay sesión de Instagram."}

    with _client(cookies) as client:
        # refresca csrftoken
        try:
            client.get("https://www.instagram.com/")
        except Exception:
            pass
        data = _json(client, "https://www.instagram.com/api/v1/accounts/current_user/", {"edit": "true"})
        if not data:
            data = _json(client, "https://www.instagram.com/api/v1/web/fxcal/ig_sso_users/")
        username = None
        user = (data or {}).get("user") or {}
        username = user.get("username")
        if not username and isinstance(data, dict):
            users = data.get("users") or data.get("ig_sso_users") or []
            if users and isinstance(users[0], dict):
                username = users[0].get("username") or users[0].get("user", {}).get("username")
        if not username:
            # último recurso: ds_user_id presente
            if cookies.get("ds_user_id") or cookies.get("sessionid"):
                return {
                    "connected": True,
                    "username": None,
                    "message": "Sesión guardada, pero Instagram no devolvió el usuario. Prueba a buscar.",
                }
            return {
                "connected": False,
                "username": None,
                "message": "La sesión caducó. Vuelve a conectar Instagram.",
            }
        return {"connected": True, "username": username, "message": f"Conectado como @{username}"}


def _walk_media(obj: Any, bag: List[dict], limit: int) -> None:
    if len(bag) >= limit:
        return
    if isinstance(obj, dict):
        media = obj.get("media") if isinstance(obj.get("media"), dict) else obj
        code = media.get("code") if isinstance(media, dict) else None
        mtype = media.get("media_type") if isinstance(media, dict) else None
        product = media.get("product_type") if isinstance(media, dict) else None
        is_video = mtype in (2, 8) or product in {"clips", "igtv", "feed"}
        if code and (is_video or (mtype == 2) or product == "clips"):
            if mtype == 1 and product != "clips":
                pass
            else:
                if not any(x.get("code") == code for x in bag):
                    if mtype != 1 or product == "clips":
                        bag.append(media)
        for v in obj.values():
            _walk_media(v, bag, limit)
            if len(bag) >= limit:
                return
    elif isinstance(obj, list):
        for v in obj:
            _walk_media(v, bag, limit)
            if len(bag) >= limit:
                return


def _media_to_item(media: dict) -> Optional[dict]:
    code = media.get("code")
    if not code:
        return None
    product = media.get("product_type") or ""
    kind = "reel" if product == "clips" else "p"
    user = media.get("user") or media.get("owner") or {}
    username = user.get("username")
    caption = media.get("caption")
    text = ""
    if isinstance(caption, dict):
        text = caption.get("text") or ""
    elif isinstance(caption, str):
        text = caption
    title = (text.split("\n")[0][:90] if text else None) or (f"@{username} · Instagram" if username else f"Instagram · {code}")
    thumb = ""
    cands = ((media.get("image_versions2") or {}).get("candidates") or [])
    if cands:
        thumb = cands[0].get("url") or ""
    if not thumb:
        thumb = f"https://www.instagram.com/p/{code}/media/?size=l"
    duration = media.get("video_duration")
    dur_s = None
    if duration:
        try:
            d = float(duration)
            m, s = divmod(int(d), 60)
            dur_s = f"{m}:{s:02d}"
        except Exception:
            dur_s = None
    return {
        "video_id": code,
        "title": title,
        "url": f"https://www.instagram.com/{kind}/{code}/",
        "thumbnail": thumb,
        "duration": dur_s,
        "channel": f"@{username}" if username else None,
        "platform": "instagram",
    }


def _collect_from_hashtag(client: httpx.Client, tag: str, limit: int) -> List[dict]:
    bag: List[dict] = []
    tag = tag.strip().lstrip("#")
    if not tag:
        return []
    payloads = [
        ("GET", f"https://www.instagram.com/api/v1/tags/web_info/?tag_name={tag}", None),
        ("GET", f"https://www.instagram.com/explore/tags/{tag}/?__a=1&__d=dis", None),
        ("POST", f"https://i.instagram.com/api/v1/tags/{tag}/sections/", {"tab": "recent", "include_persistent": "true"}),
    ]
    for method, url, data in payloads:
        data_json = _json(client, url, method=method, data=data)
        if data_json:
            _walk_media(data_json, bag, limit)
        if len(bag) >= limit:
            break
    return bag


def _collect_from_user(client: httpx.Client, username: str, limit: int) -> List[dict]:
    username = username.strip().lstrip("@")
    bag: List[dict] = []
    data = _json(
        client,
        "https://www.instagram.com/api/v1/users/web_profile_info/",
        params={"username": username},
    )
    if data:
        _walk_media(data, bag, limit)
    return bag


def search_with_account(query: str, max_results: int = 8) -> Tuple[List[dict], Optional[str]]:
    """
    Busca reels/videos con la cuenta del usuario.
    Devuelve (items, error_si_no_hay_sesion).
    """
    cookies = load_cookie_map()
    if "sessionid" not in cookies:
        return [], "no_session"

    q = (query or "").strip()
    items: List[dict] = []
    seen = set()

    def add_media_list(medias: List[dict]) -> None:
        for media in medias:
            it = _media_to_item(media)
            if not it or it["video_id"] in seen:
                continue
            # Preferir videos/reels; descartar fotos puras
            mtype = media.get("media_type")
            product = media.get("product_type")
            if mtype == 1 and product != "clips":
                continue
            seen.add(it["video_id"])
            items.append(it)

    with _client(cookies) as client:
        try:
            client.get("https://www.instagram.com/")
        except Exception:
            pass

        handle = None
        m = re.match(r"^@?([A-Za-z0-9._]{2,30})$", q)
        if m and " " not in q:
            handle = m.group(1)
            add_media_list(_collect_from_user(client, handle, max_results))

        # topsearch → hashtags y usuarios
        top = _json(
            client,
            "https://www.instagram.com/web/search/topsearch/",
            params={"context": "blended", "query": q, "include_reel": "true"},
        )
        if not top:
            top = _json(
                client,
                "https://www.instagram.com/api/v1/web/search/topsearch/",
                params={"query": q},
            )
        if not top:
            top = _json(
                client,
                "https://www.instagram.com/api/v1/fbsearch/web/top_search/",
                params={"query": q, "count": 20},
            )

        hashtags = []
        users = []
        if isinstance(top, dict):
            for h in top.get("hashtags") or []:
                tag = (h.get("hashtag") or h).get("name") if isinstance(h, dict) else None
                if tag:
                    hashtags.append(tag)
            for u in top.get("users") or []:
                un = (u.get("user") or u).get("username") if isinstance(u, dict) else None
                if un:
                    users.append(un)
            _walk_media(top, [], 1)  # no-op safety
            tmp: List[dict] = []
            _walk_media(top, tmp, max_results)
            add_media_list(tmp)

        words = re.findall(r"[a-zA-Z0-9_]{2,}", q.lower())
        if words:
            hashtags.append(words[0])
            hashtags.append("".join(words[:2]))

        for tag in dict.fromkeys(hashtags):
            if len(items) >= max_results:
                break
            add_media_list(_collect_from_hashtag(client, tag, max_results))

        for un in users[:3]:
            if len(items) >= max_results:
                break
            add_media_list(_collect_from_user(client, un, max_results))

    return items[:max_results], None
