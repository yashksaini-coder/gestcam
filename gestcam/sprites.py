"""Emoji sprites for the thumbs reactions.

Noto Color Emoji PNGs (Apache-2.0) from googlefonts/noto-emoji, fetched once into
~/.cache/gestcam. If the download fails (offline first run) the Thumb effect falls
back to a procedural bubble, so a missing asset never breaks the pipeline.
"""

from __future__ import annotations

import urllib.request

import cv2
import numpy as np

from .recognizer import cache_dir

NOTO_PNG = "https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/128/emoji_u{cp}.png"
SPRITES = {"thumbs_up": "1f44d", "thumbs_down": "1f44e"}


def load_sprites(log=print) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, cp in SPRITES.items():
        p = cache_dir() / f"emoji_u{cp}.png"
        if not p.exists():
            try:
                urllib.request.urlretrieve(NOTO_PNG.format(cp=cp), p)
            except Exception as e:  # offline: procedural fallback
                log(f"[gestcam] sprite {name} unavailable ({e}); using fallback")
                continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            log(f"[gestcam] sprite {name} unreadable; using fallback")
            continue
        out[name] = img  # BGRA
    return out
