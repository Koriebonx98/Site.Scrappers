#!/usr/bin/env python3
"""
PC Games Cover Downloader (JSON-driven, script-dir JSON)

This script reads All.Games.json located in the same directory as this script
(or a JSON file you pass with --json-file), downloads cover images (SteamGridDB
preferred, Steam Store fallback), and saves images under an image directory
with multiple-location fallbacks. It only reads the JSON file (no drive scanning).

Usage:
    python pc_games_cover_downloader_from_json.py
    python pc_games_cover_downloader_from_json.py --json-file other.json

Defaults:
- JSON file default: <script dir>/All.Games.json
- Status file default: <script dir>/PC.Games.txt
- Image dir preferred default: <script dir>/PC (Windows)

Options:
- --api-key to provide SteamGridDB API key (or set STEAMGRIDDB_API_KEY env var)
- --no-pause to skip waiting for Enter on exit
- --verbose for extra logging
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# === Defaults & config ===
STEAMGRIDDB_API_URL = "https://www.steamgriddb.com/api/v2"
REQUEST_TIMEOUT = 12
PAUSE_BETWEEN_REQUESTS = 1.0
USER_AGENT = "PCGamesCoverDownloader/1.0 (+https://example.invalid/)"

def script_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()

DEFAULT_JSON_FILE = script_dir() / "All.Games.json"
DEFAULT_STATUS_FILE = script_dir() / "PC.Games.txt"
DEFAULT_IMAGE_DIR = script_dir() / Path("PC (Windows)")

def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def make_session(retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST", "HEAD", "OPTIONS"])
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": USER_AGENT})
    return s

# === Status file helpers (script-dir aware) ===
def candidate_status_paths(preferred: Path) -> Tuple[Path, ...]:
    # Prioritize script-dir location, then given preferred, then cwd, home, temp
    sdir = script_dir()
    candidates = [
        sdir / preferred.name,
        preferred,
        Path.cwd() / preferred.name,
        Path.home() / preferred.name,
        Path(tempfile.gettempdir()) / preferred.name,
    ]
    seen = set()
    final = []
    for p in candidates:
        try:
            p = p.resolve()
        except Exception:
            pass
        if p in seen:
            continue
        seen.add(p)
        final.append(p)
    return tuple(final)

def load_status(preferred: Path) -> Tuple[Dict[str, str], Path]:
    candidates = candidate_status_paths(preferred)
    status: Dict[str, str] = {}
    for p in candidates:
        try:
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line:
                            continue
                        if "|" in line:
                            name, st = line.split("|", 1)
                            status[name.strip()] = st.strip()
                        else:
                            status[line.strip()] = ""
                log(f"[STATUS] Loaded {len(status)} entries from: {p}")
                return status, p
        except PermissionError:
            log(f"[STATUS] Permission denied reading: {p} (trying next)")
        except Exception as ex:
            log(f"[STATUS] Error reading {p}: {ex} (trying next)")
    # no readable file found; use first candidate for saving
    log(f"[STATUS] No readable status file found; will attempt to save to: {candidates[0]}")
    return status, candidates[0]

def save_status(status: Dict[str, str], preferred: Path) -> Path:
    candidates = candidate_status_paths(preferred)
    last_exc = None
    for p in candidates:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as f:
                for name, st in status.items():
                    if st:
                        f.write(f"{name} | {st}\n")
                    else:
                        f.write(f"{name}\n")
            log(f"[STATUS] Saved {len(status)} entries to: {p}")
            return p
        except PermissionError:
            log(f"[STATUS] Permission denied writing to: {p} (trying next)")
        except Exception as ex:
            log(f"[STATUS] Error writing to {p}: {ex} (trying next)")
            last_exc = ex
    raise last_exc if last_exc is not None else RuntimeError("Failed to save status to any candidate path")

# === Helpers ===
def safe_folder_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name)
    cleaned = cleaned.strip()
    if not cleaned:
        cleaned = "Unknown Game"
    return cleaned

def guess_ext_from_content_type(ct: Optional[str]) -> str:
    if not ct:
        return ".jpg"
    ct = ct.split(";", 1)[0].strip().lower()
    if ct == "image/png":
        return ".png"
    if ct in ("image/jpeg", "image/jpg"):
        return ".jpg"
    return ".jpg"

def download_image(session: requests.Session, image_url: str, dest_base: Path) -> bool:
    try:
        dest_base.parent.mkdir(parents=True, exist_ok=True)
    except Exception as ex:
        log(f"[IMG] Could not create directory {dest_base.parent}: {ex}")
        return False

    ext = Path(image_url.split("?", 1)[0]).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = None

    try:
        with session.get(image_url, stream=True, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status_code != 200:
                log(f"[IMG] HTTP {resp.status_code} when downloading {image_url}")
                return False
            if not ext:
                ext = guess_ext_from_content_type(resp.headers.get("Content-Type"))
            if ext == ".jpeg":
                ext = ".jpg"
            final = dest_base.with_suffix(ext)
            with final.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            log(f"[IMG] Saved {final}")
            return True
    except Exception as ex:
        log(f"[IMG] Exception downloading {image_url}: {ex}")
        return False

def choose_image_dir(preferred: Path, status_read_path: Optional[Path]) -> Optional[Path]:
    # Prefer script-dir image folder first, then the provided preferred, then cwd, status file parent, home, Pictures, temp
    sdir = script_dir()
    candidates = [
        sdir / preferred.name,
        preferred,
        Path.cwd() / preferred.name,
    ]
    if status_read_path:
        candidates.append(status_read_path.parent / preferred.name)
    candidates += [
        Path.home() / preferred.name,
        Path.home() / "Pictures" / preferred.name,
        Path(tempfile.gettempdir()) / preferred.name,
    ]
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        try:
            p.mkdir(parents=True, exist_ok=True)
            test_file = p / ".write_test"
            try:
                with test_file.open("w", encoding="utf-8") as f:
                    f.write("ok")
                test_file.unlink(missing_ok=True)
            except Exception:
                log(f"[IMG] Directory created but not writable: {p} (skipping)")
                continue
            log(f"[IMG] Using image directory: {p}")
            return p
        except PermissionError:
            log(f"[IMG] Permission denied creating image dir: {p} (trying next)")
        except Exception as ex:
            log(f"[IMG] Error creating image dir {p}: {ex} (trying next)")
    log("[IMG] Could not create any candidate image directory. Images will not be saved.")
    return None

# === SteamGridDB & Steam Store helpers ===
def steamgriddb_search(session: requests.Session, game_name: str, api_key: Optional[str]) -> Optional[int]:
    if not api_key:
        return None
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{STEAMGRIDDB_API_URL}/search/autocomplete/{requests.utils.quote(game_name)}"
    try:
        r = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            log(f"[SteamGridDB] Search error {r.status_code} for '{game_name}'")
            return None
        data = r.json().get("data", [])
        if not data:
            return None
        return data[0].get("id")
    except Exception as ex:
        log(f"[SteamGridDB] Exception searching '{game_name}': {ex}")
        return None

def steamgriddb_get_grid(session: requests.Session, game_id: int, api_key: str, dimensions: str = "600x900") -> Optional[str]:
    if not api_key or not game_id:
        return None
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{STEAMGRIDDB_API_URL}/grids/game/{game_id}"
    try:
        r = session.get(url, headers=headers, params={"dimensions": dimensions}, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            log(f"[SteamGridDB] Grid error {r.status_code} for id {game_id}")
            return None
        data = r.json().get("data", [])
        if not data:
            return None
        item = data[0]
        return item.get("url") or item.get("thumb")
    except Exception as ex:
        log(f"[SteamGridDB] Exception getting grid for id {game_id}: {ex}")
        return None

def steam_store_cover(session: requests.Session, game_name: str) -> Optional[str]:
    try:
        url = f"https://store.steampowered.com/api/storesearch/?term={requests.utils.quote(game_name)}&cc=us&l=en"
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            log(f"[SteamStore] Search error {r.status_code} for '{game_name}'")
            return None
        items = r.json().get("items", [])
        if not items:
            return None
        appid = items[0].get("id")
        if not appid:
            return None
        return f"https://cdn.cloudflare.steamstatic.com/steam/apps/{appid}/library_600x900.jpg"
    except Exception as ex:
        log(f"[SteamStore] Exception searching store for '{game_name}': {ex}")
        return None

# === JSON loader (handles Url as string or list) ===
def normalize_entry(entry: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if not isinstance(entry, dict):
        return None
    raw_name = entry.get("Name") or entry.get("name") or ""
    if not raw_name:
        return None
    raw_url = entry.get("Url") or entry.get("url") or ""
    url = ""
    if isinstance(raw_url, str):
        url = raw_url.strip()
    elif isinstance(raw_url, list):
        for u in raw_url:
            if isinstance(u, str) and "steamrip.com" in u:
                url = u.strip()
                break
        if not url:
            for u in raw_url:
                if isinstance(u, str) and u.strip():
                    url = u.strip()
                    break
    else:
        url = ""
    return (str(raw_name).strip(), url)

def load_games_from_json(json_path: Path) -> List[Dict[str, str]]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        log(f"[JSON] File not found: {json_path}")
        return []
    except Exception as ex:
        log(f"[JSON] Error loading {json_path}: {ex}")
        return []

    if not isinstance(data, list):
        log(f"[JSON] Unexpected format: top-level is not a list in {json_path}")
        return []

    results: List[Dict[str, str]] = []
    seen_names = set()
    for item in data:
        norm = normalize_entry(item)
        if not norm:
            continue
        name, url = norm
        if name in seen_names:
            continue
        seen_names.add(name)
        results.append({"Name": name, "Url": url})
    log(f"[JSON] Loaded {len(results)} entries from {json_path}")
    return results

# === Main run ===
def run(json_file: Path, status_file: Path, preferred_image_dir: Path, api_key: Optional[str],
        no_pause: bool, verbose: bool) -> int:
    session = make_session()
    if verbose:
        log("[INFO] Verbose mode on")
        log(f"[INFO] Using JSON file: {json_file}")

    status_map, status_path = load_status(status_file)

    games = load_games_from_json(json_file)
    if not games:
        log("[INFO] No games to process. Exiting.")
        if not no_pause:
            try:
                input("Press Enter to exit...")
            except Exception:
                pass
        return 0

    chosen_image_dir = choose_image_dir(preferred_image_dir, status_path)
    if chosen_image_dir is None:
        log("[IMG] No usable image directory; downloads will be attempted but cannot be saved.")

    for entry in games:
        name = entry["Name"]
        if status_map.get(name) == "Downloaded Image":
            if verbose:
                log(f"[SKIP] {name} already downloaded")
            continue

        log(f"[PROCESS] {name}")
        success = False

        # SteamGridDB
        grid_url = None
        if api_key:
            gid = steamgriddb_search(session, name, api_key)
            if gid:
                grid_url = steamgriddb_get_grid(session, gid, api_key)
                if verbose and grid_url:
                    log(f"[SteamGridDB] Found grid URL for '{name}': {grid_url}")

        if grid_url and chosen_image_dir:
            dest_base = chosen_image_dir / safe_folder_name(name) / "Cover"
            success = download_image(session, grid_url, dest_base)
        elif grid_url and not chosen_image_dir:
            log(f"[IMG] No image directory; skipping SteamGridDB result for '{name}'")

        # Steam Store fallback
        if not success:
            if verbose:
                log(f"[FALLBACK] Trying Steam Store for '{name}'")
            store_url = steam_store_cover(session, name)
            if store_url and chosen_image_dir:
                dest_base = chosen_image_dir / safe_folder_name(name) / "Cover"
                success = download_image(session, store_url, dest_base)
            elif store_url and not chosen_image_dir:
                log(f"[IMG] No image directory; skipping Steam store result for '{name}'")

        # Page scraping fallback (if JSON provided a page URL)
        if not success and entry.get("Url"):
            page_url = entry["Url"]
            try:
                if verbose:
                    log(f"[PAGE] Trying to extract image from page {page_url}")
                r = session.get(page_url, timeout=REQUEST_TIMEOUT)
                if r.status_code == 200:
                    html = r.text
                    m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, flags=re.I)
                    if not m:
                        m = re.search(r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']', html, flags=re.I)
                    img_url = m.group(1) if m else None
                    if not img_url:
                        m2 = re.search(r'<img[^>]+src=["\']([^"\']+\.(?:jpg|jpeg|png))["\'][^>]*>', html, flags=re.I)
                        img_url = m2.group(1) if m2 else None
                    if img_url:
                        if img_url.startswith("/"):
                            base = re.match(r'^(https?://[^/]+)', page_url)
                            if base:
                                img_url = base.group(1) + img_url
                        if chosen_image_dir:
                            dest_base = chosen_image_dir / safe_folder_name(name) / "Cover"
                            success = download_image(session, img_url, dest_base)
                        else:
                            log(f"[IMG] No image directory; skipping image found on page for '{name}'")
            except Exception as ex:
                if verbose:
                    log(f"[PAGE] Exception fetching page {page_url}: {ex}")

        if success:
            status_map[name] = "Downloaded Image"
            try:
                save_status(status_map, status_path)
            except Exception as ex:
                log(f"[STATUS] Warning: incremental save failed: {ex}")
        else:
            log(f"[MISS] No cover saved for {name}")

        time.sleep(PAUSE_BETWEEN_REQUESTS)

    try:
        final_path = save_status(status_map, status_path)
        if final_path != status_path:
            log(f"[STATUS] Note: status file saved to different location: {final_path}")
    except Exception as ex:
        log(f"[STATUS] Final save failed: {ex}")

    log("All done.")
    if not no_pause:
        try:
            input("Press Enter to exit...")
        except Exception:
            pass
    return 0

# === CLI ===
def parse_args():
    p = argparse.ArgumentParser(description="Download cover images for games listed in a JSON file (script-dir JSON default).")
    p.add_argument("--json-file", type=Path, default=DEFAULT_JSON_FILE, help="JSON file with games (array of {Name,Url}); default is script-dir All.Games.json")
    p.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_FILE, help="Path (or filename) for the status file (script-dir default).")
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Preferred directory to store downloaded images (script-dir default).")
    p.add_argument("--api-key", type=str, default=os.environ.get("STEAMGRIDDB_API_KEY", ""), help="SteamGridDB API key (or set STEAMGRIDDB_API_KEY env var).")
    p.add_argument("--no-pause", action="store_true", help="Do not wait for Enter before exiting.")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        rc = run(args.json_file, args.status_file, args.image_dir, args.api_key or None, args.no_pause, args.verbose)
        sys.exit(rc)
    except KeyboardInterrupt:
        log("\nInterrupted by user (Ctrl+C).")
        try:
            input("Press Enter to exit...")
        except Exception:
            pass
        sys.exit(1)
    except Exception:
        log("\nAn unexpected error occurred:")
        traceback.print_exc()
        try:
            input("\nScript encountered an error. Press Enter to exit...")
        except Exception:
            pass
        sys.exit(1)