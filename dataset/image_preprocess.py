#!/usr/bin/env python3
"""
image_preprocess.py

Removes purple annotation markers (circles and lines) from WiserUI-Bench images
using inpainting. These markers were added by GoodUI to highlight UI changes.

IMPORTANT — Limitations of automated preprocessing:
    This script handles most cases automatically, but some images may require
    manual intervention, for example:
      - Images where UI elements overlap with the annotation area
      - Images where the annotation region is unusually large or irregularly shaped
      - Images that need cropping to remove watermarks or borders
    For such cases, inspect the output images manually and edit them as needed
    before running inference.

Usage:
    python dataset/image_preprocess.py
    python dataset/image_preprocess.py --images path/to/images/
"""

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_IMAGES = "images"

LIGHT_PURPLE_LOWER = np.array([170, 130, 180])
LIGHT_PURPLE_UPPER = np.array([210, 170, 230])
PURPLE_LINE_BGR    = (121, 84, 118)
PURPLE_TOLERANCE   = 15


def process_image(image_path: str, output_path: str) -> bool:
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Failed to load: {image_path}")
        return False

    # ── Detect light-purple annotation circles ──
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=30, minRadius=5, maxRadius=100
    )

    # ── Build inpainting mask from purple lines + circles ──
    lower = np.array([max(0,   c - PURPLE_TOLERANCE) for c in PURPLE_LINE_BGR])
    upper = np.array([min(255, c + PURPLE_TOLERANCE) for c in PURPLE_LINE_BGR])
    mask  = cv2.inRange(image, lower, upper)
    mask  = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            roi       = image[max(0,y-r):min(image.shape[0],y+r),
                              max(0,x-r):min(image.shape[1],x+r)]
            avg_color = np.mean(roi, axis=(0, 1))
            if (np.all(avg_color >= LIGHT_PURPLE_LOWER) and
                    np.all(avg_color <= LIGHT_PURPLE_UPPER)):
                br = int(r * 1.5)
                x1, y1 = max(0, x-br), max(0, y-br)
                x2, y2 = min(image.shape[1], x+br), min(image.shape[0], y+br)
                mask[y1:y2, x1:x2] = 255

    inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    cv2.imwrite(output_path, inpainted)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=DEFAULT_IMAGES,
                        help="Root directory containing images/{index}/win.png … (default: images/)")
    args = parser.parse_args()

    pairs = []
    for root, _, files in os.walk(args.images):
        for fname in files:
            if fname in ("win.png", "lose.png"):
                pairs.append(os.path.join(root, fname))
    pairs.sort()

    ok = fail = 0
    for path in tqdm(pairs, desc="Preprocessing"):
        if process_image(path, path):
            ok += 1
        else:
            fail += 1

    print(f"\nDone. processed={ok}, failed={fail}")
    if fail:
        print("Check the failed paths above and process them manually if needed.")


if __name__ == "__main__":
    main()
