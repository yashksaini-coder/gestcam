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


def _premultiply(bgra: np.ndarray) -> np.ndarray:
    """Multiply RGB by alpha once, at load.

    Noto PNGs are straight alpha with RGB=0 in their transparent pixels (emoji_u1f44d
    has 9165 such pixels plus 346 semi-transparent ones). cv2.INTER_AREA averages RGB
    and A independently, so a 128→48 downscale pulls that black into every edge pixel:
    measured edge RGB mean 81.8 straight vs 124.3 premultiplied. The halo gets worse as
    sprites shrink — i.e. exactly in the particle effects. Rounding, not floor: floor
    costs a systematic half-LSB darkening.
    """
    a = bgra[:, :, 3:4].astype(np.uint16)
    out = bgra.copy()
    out[:, :, :3] = ((bgra[:, :, :3].astype(np.uint16) * a + 127) // 255).astype(np.uint8)
    return out


class SpriteSet:
    """Premultiplied Noto emoji, cached per (name, 8px size bucket).

    Per-call cv2.resize dominates sprite cost: 200 sprites @20px measured 19.8 ms at
    720p, 5.5 ms prescaled. Particle size is near-constant frame to frame, so the
    resize is pure waste.

    No rotation dimension, deliberately: its only would-be consumer is Confetti, and
    warping 24 sprites per frame costs ~1.4 ms of a 33 ms budget. Paying ~384 cache
    entries and 22.5° quantisation to save that is complexity with no need.
    """

    CACHE_MAX_PX = 128  # above this, resize per call (0.12 ms at 260px) so the Thumb's
                        # ease_out_back size sweep cannot balloon the cache

    def __init__(self, raw: dict[str, np.ndarray]):
        self.raw = {k: _premultiply(v) for k, v in raw.items()}
        # ponytail: plain dict, no eviction. ~130 entries / ~1.2 MB in practice.
        # LRU only if a future effect sweeps far more sizes than the Thumb pop does.
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def get(self, name: str, size_px: float) -> np.ndarray | None:
        """Prescaled premultiplied BGRA, or None so the caller draws its primitive.

        The returned array is SHARED and must be treated as read-only: an effect that
        mutates one corrupts the cache for every other effect.
        """
        base = self.raw.get(name)
        if base is None or size_px < 6:
            return None
        s = max(8, (int(size_px) // 8) * 8)
        img = self._cache.get((name, s))
        if img is not None:
            return img
        interp = cv2.INTER_AREA if s < base.shape[0] else cv2.INTER_LINEAR
        img = cv2.resize(base, (s, s), interpolation=interp)
        if s <= self.CACHE_MAX_PX:
            self._cache[(name, s)] = img
        return img
