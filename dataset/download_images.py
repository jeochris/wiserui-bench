#!/usr/bin/env python3
"""
download_images.py

Downloads win/lose images for WiserUI-Bench from their original source URLs.
Images are saved to images/{index}/win.png and images/{index}/lose.png.

Usage:
    python dataset/download_images.py
    python dataset/download_images.py --data path/to/WiserUI_Bench.json --out images/
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

DEFAULT_DATA = "WiserUI_Bench.json"
DEFAULT_OUT  = "images"
REQUEST_DELAY = 0.5  # seconds between requests to avoid rate limiting

REFERER_MAP = {
    "goodui.org":            "https://goodui.org/",
    "static.wingify.com":    "https://vwo.com/",
    "framerusercontent.com": "https://abtest.design/",
}


def make_headers(url: str) -> dict:
    host = urlparse(url).netloc
    ref  = next((v for k, v in REFERER_MAP.items() if k in host), None)
    h = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    if ref:
        h["Referer"] = ref
    return h


def clean_vwo_url(url: str) -> str:
    """Remove VWO image transform prefix (e.g. /tr:w-400,h-300/) for original resolution."""
    return re.sub(r"/tr:[^/]+/", "/", url)


def download_image(url: str, out_path: Path, session: requests.Session) -> bool:
    url = clean_vwo_url(url)
    try:
        r = session.get(url, headers=make_headers(url), timeout=20)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path))
        return True
    except Exception as e:
        print(f"    ✗ Failed ({e})")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Path to WiserUI_Bench.json (default: WiserUI_Bench.json)")
    parser.add_argument("--out",  default=DEFAULT_OUT,
                        help="Output directory for images (default: images/)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip items where both win.png and lose.png already exist")
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    out_dir = Path(args.out)
    session = requests.Session()

    skipped = failed = success = 0

    for item in tqdm(data, desc="Downloading"):
        idx      = item["index"]
        win_url  = item.get("win_url")
        lose_url = item.get("lose_url")

        win_path  = out_dir / str(idx) / "win.png"
        lose_path = out_dir / str(idx) / "lose.png"

        # Skip items with no valid image URL (e.g. page no longer exists)
        if not win_url or not lose_url or win_url == item.get("source"):
            print(f"[{idx}] Skipping — no image URL available")
            skipped += 1
            continue

        # Skip if already downloaded
        if args.skip_existing and win_path.exists() and lose_path.exists():
            skipped += 1
            continue

        item_ok = True
        for label, url, path in [("win", win_url, win_path), ("lose", lose_url, lose_path)]:
            if args.skip_existing and path.exists():
                continue
            ok = download_image(url, path, session)
            if not ok:
                item_ok = False
                failed += 1
            else:
                success += 1
            time.sleep(REQUEST_DELAY)

        if not item_ok:
            print(f"  [idx={idx}] One or more images failed to download.")

    print(f"\nDone. success={success}, skipped={skipped}, failed={failed}")
    if failed:
        print("Re-run the script to retry failed items.")


if __name__ == "__main__":
    main()
