"""Emoji sprites for the reactions.

Noto Color Emoji PNGs (Apache-2.0) from googlefonts/noto-emoji, vendored in
gestcam/assets/ at 128px (~120 KB total). A missing or unreadable file degrades that
one effect to its procedural primitive; it never breaks the pipeline.

Vendored rather than downloaded: the old loader made serial, timeout-less HTTP calls
inside run() before the camera opened, which is an indefinite startup hang for fifteen
files on a slow network.
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np

ASSETS = pathlib.Path(__file__).parent / "assets"

# One flat namespace. All eight REACTIONS names must appear here, because the hold-ring
# preview looks a sprite up by reaction name; the extra names are per-effect variety.
# These keys must match the string literals in effects.py — a typo degrades silently to
# the procedural fallback rather than erroring.
SPRITES = {
    "hearts": "2764", "heart_sparkle": "1f496", "heart_purple": "1f49c",
    "thumbs_up": "1f44d", "thumbs_down": "1f44e",
    "balloons": "1f388",
    "confetti": "1f389", "confetti_ball": "1f38a", "sparkles": "2728", "star": "1f31f",
    "rain": "1f327",
    "lasers": "1f918",
    "fireworks": "1f386", "sparkler": "1f387", "firecracker": "1f9e8",
}


def load_sprites(log=print) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, cp in SPRITES.items():
        img = cv2.imread(str(ASSETS / f"emoji_u{cp}.png"), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            log(f"[gestcam] sprite {name} unreadable; using fallback")
            continue
        out[name] = img  # BGRA, straight alpha
    return out
