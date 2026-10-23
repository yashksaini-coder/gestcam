"""Reaction effects. Time-based particle systems composited onto BGR frames in layers.

Design constraints that shaped this:

* Everything is driven by time, never by frame count. Particle motion is closed-form in
  particle age (position = f(t), see `_drag_path`), so a 15 fps frame and a 60 fps frame
  at the same t are the same picture, and trails need no position history.
* Three passes per effect, so effects can sit behind the person (Apple: fireworks
  "bursting in the background"; balloons and confetti both in front and behind):
  `draw_back` paints the background, `draw_front` paints over the person, `draw_light`
  adds to a half-resolution buffer that is blurred into glow and added behind the
  person. With a person mask the back layer goes on a copy that is blended under the
  person; without one, back then front are painted straight onto the frame.
* Full-frame compositing is fine as long as it is a few native uint8 cv2 calls, never
  numpy float. Measured at 1280x720 (i7-13620H): numpy float mask blend 18.9 ms (why
  an older version of this file banned compositing outright); cv2.blendLinear mask blend
  0.3 ms in a tight loop but the whole mask path (copy + weights + blend) costs ~2.3 ms
  inside a real frame, where OpenCV's thread pool wakes cold; addWeighted 0.5-0.7 ms;
  half-res blur + upscale + add 0.8-1.8 ms; 3-channel mask tint via merge 0.55 ms (via
  cv2.transform 1->3 ch: 8.6 ms).
* Effect-wide opacity (fade in, fade out, retire) is applied by the engine, by drawing
  that effect onto a copy and addWeighted-ing it back (~0.9 ms, only while fading).
  Effects therefore fade by alpha, never by darkening their colours toward black, and
  each effect's last live frame is near-transparent by construction. Light-pass
  contributions are additive, so effects scale those by `self.opacity` themselves.
* Sprites arrive from SpriteSet already premultiplied and already scaled to a cached
  size bucket; `blend_sprite` only composites. Every effect falls back to a procedural
  primitive when a sprite is missing, so gestcam works with no assets.
* Each effect owns its RNG so `demo` renders are reproducible.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np

from .segment import DILATE_FRAC, FEATHER_FRAC
from .sprites import BALLOON_HUES, SpriteSet
from .state import SUBSET

BGR = tuple[int, int, int]

PALETTE: list[BGR] = [  # BGR
    (60, 76, 231),    # red
    (0, 152, 255),    # orange
    (0, 213, 255),    # yellow
    (113, 204, 46),   # green
    (219, 152, 52),   # blue
    (182, 89, 155),   # purple
    (203, 192, 255),  # pink
]
# Additive colours: added onto a darkened background, so they must be bright, and none
# is pure primary (pure 255/0/0 reads as a debug overlay, measured E8).
SPARK: list[BGR] = [(90, 200, 255), (110, 90, 255), (140, 255, 150), (255, 200, 90),
                    (255, 130, 210), (220, 235, 255)]
WARM: list[BGR] = [(40, 150, 255), (70, 70, 255), (190, 90, 255), (60, 205, 255)]

# Two-hand reaction → its one-hand subset (state.py owns the pairing). A live subset
# effect is retired when its superset fires, so 👍 then 👍👍 doesn't leave a bubble under
# fireworks.
SUBSET_OF = SUBSET
RETIRE_S = 0.25


def _heart_unit() -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, 40, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    pts = np.stack([x, -y], axis=1) / 17.0  # unit-ish, y down
    return pts.astype(np.float32)


HEART = _heart_unit()


def _ease_out_back(x: float, s: float = 1.7) -> float:
    x = max(0.0, min(1.0, x))
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def _ema(cur: float, target: float, dt: float, tau: float) -> float:
    """Time-based smoothing: same lag at 15 fps and 60 fps (a per-frame factor isn't)."""
    return cur + (target - cur) * (1.0 - math.exp(-dt / tau))


def _drag_path(vx, vy, g: float, k: float, t):
    """Displacement after t s from launch velocity (vx, vy), downward accel g, linear
    drag k. Closed form of v' = g - k*v, so it is exact at any frame rate."""
    e = (1.0 - np.exp(-k * t)) / k
    return vx * e, vy * e + (g / k) * (t - e)


def _rot_rect(cx: float, cy: float, w: float, h: float, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    hw, hh = w / 2, h / 2
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (corners @ rot.T + (cx, cy)).astype(np.int32)


def _darken(img: np.ndarray, k: float, opacity: float = 1.0) -> None:
    """Scale brightness by k, faded in by opacity (k=0.5 at opacity 0.5 is 0.75)."""
    k = 1.0 - (1.0 - k) * opacity
    if k < 0.999:
        cv2.convertScaleAbs(img, dst=img, alpha=k)  # 0.49 ms at 720p


def blend_sprite(frame: np.ndarray, spr: np.ndarray, cx: float, cy: float,
                 alpha: float = 1.0) -> None:
    """Blend a PREMULTIPLIED, already-scaled BGRA sprite centred on (cx, cy), in place.

    Only SpriteSet (or `_premul`) produces valid input: a straight-alpha sprite blended
    here shows a dark halo. Premultiplied means the composite is `frame*(1-a) + rgb`, so
    `alpha` scales both terms rather than only the source.
    """
    if alpha <= 0:
        return
    sh, sw = spr.shape[:2]
    H, W = frame.shape[:2]
    x0, y0 = int(cx - sw / 2), int(cy - sh / 2)
    x1, y1 = x0 + sw, y0 + sh
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    spr = spr[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    a, rgb = spr[:, :, 3], spr[:, :, :3]
    if alpha < 1.0:
        a, rgb = cv2.convertScaleAbs(a, alpha=alpha), cv2.convertScaleAbs(rgb, alpha=alpha)
    # uint8 cv2 ops: 0.14 ms for a 190 px sprite vs 0.38 ms as numpy float (12 balloons
    # were ~3 ms of the frame that way). Rounds where the float path truncated: <=1 LSB.
    inv = cv2.cvtColor(cv2.bitwise_not(a), cv2.COLOR_GRAY2BGR)
    frame[y0:y1, x0:x1] = cv2.add(cv2.multiply(frame[y0:y1, x0:x1], inv, scale=1 / 255), rgb)


def _premul(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Straight BGR + uint8 alpha -> premultiplied BGRA, for sprites built at runtime."""
    a = alpha.astype(np.float32)[..., None] / 255.0
    return np.dstack([(bgr.astype(np.float32) * a + 0.5).astype(np.uint8), alpha])


def head_box(mask: np.ndarray) -> tuple[float, float, float] | None:
    """(centre x, top y, half width) of the head, from a person mask.

    The top of the mask's bounding box is the head proxy, so no face model. Sampled on
    an 8 px grid (0.05 ms); a row needs 3 cells (24 px) so a stray speck of hair or a
    background blob above the head doesn't become the "top".

    ponytail: assumes one person. With two in frame the band spans both heads, so the
    badge lands outside both (seen on a street photo) — connected components if needed.
    """
    m = mask[::8, ::8] > 127
    rows = np.flatnonzero(m.sum(1) >= 3)
    if rows.size == 0:
        return None
    top = rows[0]
    band = m[top: top + max(2, m.shape[0] // 6)]  # head band: top 1/6 of frame height
    cols = np.flatnonzero(band.sum(0) >= 2)
    if cols.size == 0:
        return None
    return (cols[0] + cols[-1] + 1) / 2 * 8, top * 8.0, (cols[-1] - cols[0] + 1) / 2 * 8


class Effect:
    """Base. Subclasses implement any of draw_back/draw_front/draw_light and set the
    matching class flags so the engine can skip work (notably the mask composite).

    Lifetime is `end` (in effect time): `extend` pushes it out (Apple's coalescing),
    `retire` pulls it in to a quick fade. `opacity` derives from t, fade_in, end and
    fade_out; the engine applies it to back/front, effects apply it to light.
    """

    back = front = light = False
    z = 1                 # front-pass order; 0 = full-frame grading drawn first
    # True when every back/front op can take opacity directly (darkening factor, sprite
    # alpha, tint gain): skips the engine's copy + addWeighted, ~2 ms on fade frames.
    fades_itself = False
    # Background brightness under a person mask (1 = none). The ENGINE applies it, through
    # an undilated weight (see `_TIGHT`); draw_back darkens only when there is no mask.
    dim = 1.0
    duration = 3.0
    fade_in = 0.0
    fade_out = 0.4

    def __init__(self, w: int, h: int, sprites=None, anchors=None, seed: int | None = None):
        self.w, self.h = w, h
        self.sprites = sprites
        self.t = 0.0
        self.end = self.duration
        self.alive = True
        self.retiring = False
        self.kind = ""
        self.rng = random.Random(seed)
        self.anchors = list(anchors) if anchors else None

    @property
    def opacity(self) -> float:
        a = min(1.0, (self.end - self.t) / self.fade_out)
        if self.fade_in:
            a = min(a, self.t / self.fade_in)
        return max(0.0, a)

    def update(self, dt: float) -> None:
        self.t += dt
        self.alive = self.t < self.end

    def follow(self, anchors, head, dt: float) -> None:
        if anchors:
            self.anchors = list(anchors)

    def extend(self, anchors=None) -> None:
        if anchors:
            self.anchors = list(anchors)
        self.end = max(self.end, self.t + self.duration)

    def retire(self) -> None:
        self.retiring = True
        if self.end - self.t > RETIRE_S:
            self.end, self.fade_out = self.t + RETIRE_S, RETIRE_S

    def draw_back(self, img: np.ndarray, mask) -> None: ...
    def draw_front(self, img: np.ndarray, mask) -> None: ...
    def draw_light(self, buf: np.ndarray) -> None: ...

    def _centre(self, default: tuple[float, float]) -> tuple[float, float]:
        if not self.anchors:
            return default
        return (sum(a[0] for a in self.anchors) / len(self.anchors),
                sum(a[1] for a in self.anchors) / len(self.anchors))


# --------------------------------------------------------------------------- hearts
class Hearts(Effect):
    """Red hearts float up and outward from between the hands, scaling in, growing a
    little as they "fly toward the camera", fading by alpha. Front layer."""

    front = True
    duration = 1.4        # emission window; each heart then lives its own life
    LIFE = 2.0

    def __init__(self, w, h, sprites=None, anchors=None, seed=None):
        super().__init__(w, h, sprites, anchors, seed)
        self.src = self._centre((w * 0.5, h * 0.6))
        self.head = None
        self.parts: list[dict] = []
        self._spawn(18)

    def _spawn(self, n: int) -> None:
        for _ in range(n):
            side = self.rng.choice((-1, 1))
            # Up and OUT, never straight up: straight up is where the face is.
            a = math.radians(-90 + side * self.rng.uniform(25, 75))
            v = self.rng.uniform(0.5, 0.9) * self.h
            self.parts.append(dict(
                born=self.t + self.rng.uniform(0, self.duration), x0=None, y0=None,
                jx=self.rng.uniform(-0.03, 0.03) * self.w,
                vx=math.cos(a) * v, vy=math.sin(a) * v,
                life=self.LIFE * self.rng.uniform(0.85, 1.15),
                size=self.rng.uniform(0.035, 0.06) * self.h, phase=self.rng.uniform(0, 6.28),
                emoji="heart_sparkle" if self.rng.random() < 0.25 else "hearts",
                col=(60, 60, 235) if self.rng.random() < 0.75 else (150, 110, 255),
            ))
        self.end = max(p["born"] + p["life"] for p in self.parts)

    def extend(self, anchors=None) -> None:
        super().extend(anchors)
        self.parts = [p for p in self.parts if self.t < p["born"] + p["life"]]
        self._spawn(10)

    def follow(self, anchors, head, dt):
        super().follow(anchors, head, dt)
        self.head = head or self.head
        tx, ty = self._centre(self.src)
        self.src = (_ema(self.src[0], tx, dt, 0.15), _ema(self.src[1], ty, dt, 0.15)) if dt else (tx, ty)
        # Emitted here, not in update: the engine calls follow after update, so this is
        # where the hands and head of THIS frame are known.
        for p in self.parts:
            if p["x0"] is None and self.t >= p["born"]:
                p["x0"], p["y0"] = self.src[0] + p["jx"], self.src[1]
                if self.head:
                    self._steer(p)

    _AGES = np.linspace(0.0, LIFE * 1.15, 24)

    def _steer(self, p: dict) -> None:
        """Keep a heart's path out of the face box (head top to mouth, head width).

        Hands beside the face: go away from the head rather than half across it (seen on
        a close-up still). Hands centred under the chin, the usual heart pose: a 25-75°
        launch put ~40% of hearts through the mouth and eyes, so lower the launch, down
        to 20° below horizontal (buoyancy lifts it later), until the closed-form path
        clears the box, else drop it. Hands held over the face itself are already
        covering it: those hearts only get sent outward."""
        cx, top, hw = self.head
        side = 1.0 if p["x0"] >= cx else -1.0
        if abs(p["x0"] - cx) < hw and p["y0"] < top + 2.0 * hw:
            p["vx"] = side * abs(p["vx"])
            return
        v = math.hypot(p["vx"], p["vy"])
        a0 = math.degrees(math.atan2(-p["vy"], abs(p["vx"])))   # elevation above horizontal
        # Head half-width + wobble (0.01 w) + half the drawn heart (size * 1.8 / 2).
        half = hw + 0.01 * self.w + p["size"] * 0.9
        for a in np.arange(a0, -21.0, -8.0):
            vx, vy = side * v * math.cos(math.radians(a)), -v * math.sin(math.radians(a))
            dx, dy = _drag_path(vx, vy, -0.08 * self.h, 1.8, self._AGES)
            x, y = p["x0"] + dx, p["y0"] + dy
            if not ((np.abs(x - cx) < half) & (y > top - p["size"]) & (y < top + 2.0 * hw)).any():
                p["vx"], p["vy"] = vx, vy
                return
        p["life"] = 0.0   # never shown: better one heart fewer than one on the face

    def draw_front(self, img, mask):
        for p in self.parts:
            age = self.t - p["born"]
            if p["x0"] is None or not 0 <= age < p["life"]:
                continue
            u = age / p["life"]
            # drag 1.8, buoyancy -0.08h: they decelerate outward, then drift up.
            dx, dy = _drag_path(p["vx"], p["vy"], -0.08 * self.h, 1.8, age)
            x = p["x0"] + dx + math.sin(age * 3 + p["phase"]) * 0.01 * self.w
            y = p["y0"] + dy
            k = _ease_out_back(age / 0.3) * (1.0 + 0.35 * u)
            alpha = min(1.0, (1.0 - u) / 0.4)
            size = p["size"] * k
            spr = self.sprites.get(p["emoji"], size * 1.8) if self.sprites else None
            if spr is not None:
                blend_sprite(img, spr, x, y, alpha)
            elif alpha > 0.3:  # opaque primitive: drop it rather than fade it
                pts = (HEART * size + (x, y)).astype(np.int32)
                cv2.fillPoly(img, [pts], p["col"], lineType=cv2.LINE_AA)


# --------------------------------------------------------------------------- balloons
class Balloons(Effect):
    """Multi-coloured balloons rise from the bottom edge fully off the top. Half go behind
    the person (smaller, slower: further away), half in front (bigger, faster, kept to
    the outer thirds so they don't drift across the face)."""

    back = front = True
    fade_out = 0.3

    def __init__(self, w, h, sprites=None, anchors=None, seed=None):
        super().__init__(w, h, sprites, anchors, seed)
        self.parts: list[dict] = []
        self._spawn(12)

    def _spawn(self, n: int) -> None:
        for i in range(n):
            front = i % 2 == 1
            if front:
                x = self.rng.choice((self.rng.uniform(0.03, 0.28), self.rng.uniform(0.72, 0.97)))
                size, v = self.rng.uniform(0.2, 0.26) * self.h, self.rng.uniform(0.42, 0.52) * self.h
            else:
                x = self.rng.uniform(0.05, 0.95)
                size, v = self.rng.uniform(0.13, 0.18) * self.h, self.rng.uniform(0.32, 0.42) * self.h
            # The first two are born at once, already peeking in: visible within ~0.1 s.
            # The rest start just below the edge and slide in; at 0.35 they popped in a
            # 15-35 px slice in one frame (~2-3 frames of rise). A re-trigger (extend)
            # has balloons on screen already, so none of its batch needs to peek.
            first = i < 2 and self.t == 0.0
            born = self.t + (0.0 if first else self.rng.uniform(0.05, 1.1))
            self.parts.append(dict(
                x=x * self.w, born=born, v=v, size=size, front=front,
                y0=self.h + size * (0.35 if first else 0.5) + (0 if first else 2),
                phase=self.rng.uniform(0, 6.28),
                hue=self.rng.randrange(len(BALLOON_HUES)), col=self.rng.choice(PALETTE),
            ))
        # Alive until the last one has fully cleared the top edge.
        self.end = max(p["born"] + (p["y0"] + p["size"]) / p["v"] for p in self.parts)

    def extend(self, anchors=None) -> None:
        self.parts = [p for p in self.parts if self.t < p["born"] + (p["y0"] + p["size"]) / p["v"]]
        self._spawn(8)

    def _draw(self, img, front: bool):
        for p in self.parts:
            age = self.t - p["born"]
            if p["front"] != front or age < 0:
                continue
            y = p["y0"] - p["v"] * age
            if y < -p["size"]:
                continue
            x = p["x"] + math.sin(self.t * 1.3 + p["phase"]) * 0.025 * self.w
            spr = self.sprites.get(f"balloon{p['hue']}", p["size"]) if self.sprites else None
            if spr is not None:  # the emoji carries its own string; don't draw a second
                blend_sprite(img, spr, x, y)
                continue
            r = int(p["size"] * 0.38)
            x, y = int(x), int(y - p["size"] * 0.1)
            sway = math.sin(self.t * 1.6 + p["phase"])
            pts = np.array([[x, y + int(r * 1.15)], [x + int(sway * r * 0.4), y + int(r * 2.1)],
                            [x - int(sway * r * 0.3), y + int(r * 3.0)]], np.int32)
            cv2.polylines(img, [pts], False, (70, 70, 70), 1, cv2.LINE_AA)
            cv2.fillConvexPoly(img, np.array([[x - r // 6, y + int(r * 1.2)], [x + r // 6, y + int(r * 1.2)],
                                              [x, y + int(r * 1.0)]], np.int32), p["col"], cv2.LINE_AA)
            cv2.ellipse(img, (x, y), (int(r * 0.85), r), 0, 0, 360, p["col"], -1, cv2.LINE_AA)
            cv2.ellipse(img, (x - int(r * 0.3), y - int(r * 0.35)), (int(r * 0.18), int(r * 0.3)), -25, 0, 360,
                        (255, 255, 255), -1, cv2.LINE_AA)

    def draw_back(self, img, mask):
        self._draw(img, False)

    def draw_front(self, img, mask):
        self._draw(img, True)


# --------------------------------------------------------------------------- confetti
class Confetti(Effect):
    """Two cannons at the bottom corners kick paper up and inward while a curtain of it
    flutters down from the top edge across the full width; drag kills the launch speed
    and the pieces flutter down. Width follows |cos(flip)| for the 3D tumble, and the
    back face is shaded darker. Half the pieces are behind the person."""

    back = front = True
    fade_out = 0.6
    G, K = 0.5, 2.6       # x h: terminal fall = G/K = 0.19 h/s, a slow paper flutter
    FALL = 0.4            # share of pieces dropped from the top edge
    MAX_AGE = 8.0         # s: a cannon piece is off the bottom by ~5.5 s, a falling one by ~6

    def __init__(self, w, h, sprites=None, anchors=None, seed=None):
        super().__init__(w, h, sprites, anchors, seed)
        self.n = 0
        self.cols = np.array(PALETTE, np.float32)
        self.p: dict[str, np.ndarray] = {}
        self._spawn(260)

    def _spawn(self, n: int) -> None:
        r = np.random.default_rng(self.rng.randrange(1 << 30))
        nf = int(n * self.FALL)
        nc = n - nf
        # Cannons alone cap the apex at ~0.2 h (K=2.6 drag) and never reach the top
        # centre (measured: 0 of 220 pieces, seeds 1-8), so they read as two corner
        # puffs. The falling share is what fills the scene, as Apple's "falling through".
        side = np.where(np.arange(nc) % 4 < 2, -1.0, 1.0)            # left / right cannon
        ang = np.radians(r.uniform(28, 82, nc))                       # from horizontal
        spd = r.uniform(1.1, 2.7, nc) * self.h
        born = np.concatenate([r.uniform(0, 0.3, nc), r.uniform(0, 1.2, nf)])
        new = dict(
            born=self.t + born,
            x0=np.concatenate([np.where(side < 0, -0.01, 1.01), r.uniform(0, 1, nf)]) * self.w,
            y0=np.concatenate([np.full(nc, 1.02), -r.uniform(0.03, 0.08, nf)]) * self.h,
            vx=np.concatenate([-side * np.cos(ang) * spd, r.uniform(-0.1, 0.1, nf) * self.h]),
            vy=np.concatenate([-np.sin(ang) * spd, r.uniform(0.1, 0.4, nf) * self.h]),
            # 1.5x the first cut: 10-17 px pieces were specks at 720p.
            w=r.uniform(0.021, 0.036, n) * self.h, hh=r.uniform(0.012, 0.02, n) * self.h,
            ang=r.uniform(0, 3.14, n), spin=r.uniform(-5, 5, n),
            flip=r.uniform(5, 11, n), phase=r.uniform(0, 6.28, n),
            col=r.integers(0, len(PALETTE), n), front=(np.arange(n) % 2 == 1),
        )
        self.p = new if not self.p else {k: np.concatenate([self.p[k], new[k]]) for k in new}
        self.end = self.t + 4.4

    def extend(self, anchors=None) -> None:
        # Drop pieces that are gone (off the bottom, or past any piece's life) before
        # topping up: re-fired every 3.2 s the arrays grew by 140 each time (2.6k at
        # 60 s) while ~80 were drawn, and the vector math runs over all of them.
        age = self.t - self.p["born"]
        _, dy = _drag_path(self.p["vx"], self.p["vy"], self.G * self.h, self.K, np.maximum(age, 0))
        keep = (age < 0) | ((self.p["y0"] + dy < self.h + 20) & (age < self.MAX_AGE))
        self.p = {k: v[keep] for k, v in self.p.items()}
        self._spawn(140)

    def _draw(self, img, front: bool):
        p = self.p
        age = self.t - p["born"]
        sel = (age > 0) & (p["front"] == front)
        if not sel.any():
            return
        age = age[sel]
        dx, dy = _drag_path(p["vx"][sel], p["vy"][sel], self.G * self.h, self.K, age)
        settle = 1.0 - np.exp(-1.5 * age)   # flutter grows as the launch speed dies
        x = p["x0"][sel] + dx + np.sin(age * 3.0 + p["phase"][sel]) * 0.02 * self.w * settle
        y = p["y0"][sel] + dy
        flip = np.cos(age * p["flip"][sel] + p["phase"][sel])
        ww = p["w"][sel] * np.maximum(0.15, np.abs(flip))
        ang = p["ang"][sel] + p["spin"][sel] * age
        cols = self.cols[p["col"][sel]] * np.where(flip > 0, 1.0, 0.7)[:, None]
        hh = p["hh"][sel]
        for i in np.flatnonzero((y > -20) & (y < self.h + 20) & (x > -20) & (x < self.w + 20)):
            cv2.fillConvexPoly(img, _rot_rect(x[i], y[i], ww[i], hh[i], ang[i]),
                               tuple(int(c) for c in cols[i]), cv2.LINE_AA)

    def draw_back(self, img, mask):
        self._draw(img, False)

    def draw_front(self, img, mask):
        self._draw(img, True)


# --------------------------------------------------------------------------- rain
CLOUD_TOP = 0.3   # fraction of the cloud canvas above which there is never any cloud


def _cloud_sprite(size: int, rng: random.Random) -> np.ndarray:
    """A soft dark storm cloud, premultiplied BGRA, built once per cloud. Procedural on
    purpose: the 🌧 emoji is a cheerful white cloud with blue drops, not a storm."""
    s = size
    alpha = np.zeros((s, s), np.uint8)
    # Puffs sit on a flat base at 0.62 s and never reach above 0.3 s (CLOUD_TOP), so
    # the caller can place the body fully in frame.
    for i in range(6):
        cx = int(s * (0.2 + 0.12 * i + rng.uniform(-0.02, 0.02)))
        r = int(s * rng.uniform(0.1, 0.14) * (1.45 if i in (2, 3) else 1.0))
        cv2.circle(alpha, (cx, int(s * 0.62) - r), r, 255, -1, cv2.LINE_AA)
    cv2.rectangle(alpha, (int(s * 0.2), int(s * 0.5)), (int(s * 0.8), int(s * 0.62)), 255, -1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), s * 0.01)
    # Moonlit top, dark belly: contrast is what makes it read over a darkened room.
    shade = np.clip(1.0 - (np.arange(s, dtype=np.float32) / s - 0.32) * 2.2, 0.38, 1.0)[:, None, None]
    bgr = (np.array([205, 198, 192], np.float32) * shade).repeat(s, 1)
    # Crop to the body rows: the square canvas is ~60% empty and blend cost is per pixel.
    y0, y1 = int(s * (CLOUD_TOP - 0.02)), int(s * 0.68)
    return _premul(bgr[y0:y1], (alpha[y0:y1].astype(np.float32) * 0.92).astype(np.uint8))


class Rain(Effect):
    """A dark and stormy night: the background darkens (more than the person, so the face
    stays readable), storm clouds slide in at the top, and drops with fading tails fall
    with a slight wind, all behind the person."""

    back = front = light = True
    fades_itself = True
    dim = 0.5
    z = 0
    duration = 3.6
    fade_in = 0.4
    fade_out = 0.6
    WIND = 0.12           # dx per dy

    def __init__(self, w, h, sprites=None, anchors=None, seed=None):
        super().__init__(w, h, sprites, anchors, seed)
        r = np.random.default_rng(self.rng.randrange(1 << 30))
        n = 170
        self.dx0 = r.uniform(-0.15, 1.05, n) * w
        self.dph = r.uniform(0, 1, n)
        self.dv = r.uniform(1.1, 1.5, n) * h
        self.dln = r.uniform(0.05, 0.09, n) * h
        self.dthick = np.where(r.uniform(0, 1, n) < 0.15, 2, 1)
        slots = [0.14, 0.38, 0.62, 0.86]
        self.rng.shuffle(slots)
        self.clouds = []
        for fx in slots[:3 + (self.rng.random() < 0.5)]:
            s = int(self.rng.uniform(0.3, 0.42) * h)
            self.clouds.append(dict(x=(fx + self.rng.uniform(-0.04, 0.04)) * w, s=s,
                                    y=s * 0.2 + self.rng.uniform(0.0, 0.04) * h,   # body fully in frame (crop is 0.38 s tall)
                                    spr=_cloud_sprite(s, self.rng)))

    def draw_back(self, img, mask):
        op = self.opacity
        if mask is None:
            _darken(img, 0.68, op)
        ease = 1.0 - (1.0 - min(1.0, self.t / 0.6)) ** 3
        for c in self.clouds:
            x = c["x"] + 0.015 * self.w * self.t           # drift with the wind
            blend_sprite(img, c["spr"], x, c["y"] - (1.0 - ease) * 0.35 * self.h, op)

    def draw_front(self, img, mask):
        if mask is not None:
            _darken(img, 0.85, self.opacity)   # the person a little; the background got 0.5 already

    def draw_light(self, buf):
        if self.t < 0.15:
            return
        s = buf.shape[1] / self.w
        span = self.h + self.dln
        y = (self.dph * span + self.dv * self.t) % span - self.dln       # tail end
        x = self.dx0 + self.WIND * y
        base = np.array([235, 220, 200], np.float32) * self.opacity
        # Three segments, dim tail -> bright head: a streak that fades instead of
        # ending in a hard cut (E6). Additive, so on the light buffer fading is free.
        # One polylines call per (segment, thickness): 6 calls, not 510 cv2.line (2.5 ms).
        for k, lvl in ((0, 0.15), (1, 0.35), (2, 0.7)):
            ya, yb = y + self.dln * k / 3, y + self.dln * (k + 1) / 3
            seg = np.stack([np.stack([x + self.WIND * (ya - y), ya], 1),
                            np.stack([x + self.WIND * (yb - y), yb], 1)], 1) * s
            for th in (1, 2):
                cv2.polylines(buf, list(seg[self.dthick == th].astype(np.int32)), False,
                              tuple(int(c) for c in base * lvl), th, cv2.LINE_AA)


# --------------------------------------------------------------------------- fireworks
class Fireworks(Effect):
    """Staggered rockets rise and burst into ring-shaped shells behind the person, over a
    darkened background. All light: no emoji tiles (🎆/🎇 are 88% opaque squares, E3)."""

    back = light = True
    fades_itself = True
    fade_in = 0.3
    fade_out = 0.5
    dim = 0.55
    N_SPARKS = 56
    RISE = 0.45           # s from launch to burst
    G, K = 0.12, 2.0      # x h: low gravity so shells droop late, not rain down (E4);
                          # ring radius tends to 0.8/2.0 = 0.4 h

    def __init__(self, w, h, sprites=None, anchors=None, seed=None):
        super().__init__(w, h, sprites, anchors, seed)
        self.bursts: list[dict] = []
        self._spawn(6, 0.0)

    def _spawn(self, n: int, t0: float) -> None:
        r = np.random.default_rng(self.rng.randrange(1 << 30))
        for i in range(n):
            m = self.N_SPARKS
            a = np.linspace(0, 2 * np.pi, m, endpoint=False) + r.uniform(0, 0.1, m)
            v = 0.8 * self.h * r.uniform(0.95, 1.05, m)    # near-uniform speed -> a ring
            if r.uniform() < 0.4:                           # some shells get an inner ring
                v[::3] *= 0.55
            bx = r.uniform(0.1, 0.9) * self.w
            self.bursts.append(dict(
                t0=t0 + i * 0.42 + r.uniform(0, 0.15),
                x=bx, y=r.uniform(0.12, 0.45) * self.h, lx=bx + r.uniform(-0.04, 0.04) * self.w,
                vx=np.cos(a) * v, vy=np.sin(a) * v, life=r.uniform(1.2, 1.6, m),
                phase=r.uniform(0, 6.28, m), col=np.array(SPARK[int(r.integers(len(SPARK)))], np.float32),
            ))
        self.end = max(b["t0"] + self.RISE + b["life"].max() for b in self.bursts)

    def extend(self, anchors=None) -> None:
        self.bursts = [b for b in self.bursts if self.t < b["t0"] + self.RISE + b["life"].max()]
        self._spawn(4, self.t)

    def draw_back(self, img, mask):
        if mask is None:
            _darken(img, 0.7, self.opacity)

    def draw_light(self, buf):
        s = buf.shape[1] / self.w
        op = self.opacity
        for b in self.bursts:
            age = self.t - b["t0"]
            if age < 0:
                continue
            if age < self.RISE:   # rocket: ease-out climb with a short fading tail
                for lag, lvl in ((0.0, 0.9), (0.06, 0.45), (0.12, 0.2)):
                    u = max(0.0, age - lag) / self.RISE
                    y = self.h + (b["y"] - self.h) * (1 - (1 - u) ** 2)
                    x = b["lx"] + (b["x"] - b["lx"]) * u
                    cv2.circle(buf, (int(x * s), int(y * s)), max(1, int(3 - lag * 15)),
                               tuple(int(c * lvl * op) for c in (210, 240, 255)), -1, cv2.LINE_AA)
                continue
            age -= self.RISE
            if age < 0.1:         # ignition flash: a small graded core, brightest inside
                # Was one 0.05 h disc at level 230: the 2x glow boost clipped all of it
                # to a flat white plateau ~100 px across (~9k px at 720p) every burst.
                f = 1 - age / 0.1
                for rr, lvl in ((0.02, 0.12), (0.012, 0.25), (0.006, 0.45)):
                    cv2.circle(buf, (int(b["x"] * s), int(b["y"] * s)), max(1, int(rr * self.h * s * (0.6 + 0.4 * f))),
                               tuple(int(c * lvl * f * op) for c in (230, 240, 255)), -1, cv2.LINE_AA)
            live = age < b["life"]
            if not live.any():
                continue
            u = np.clip(age / b["life"], 0, 1)
            inten = (1 - u) ** 1.6 * np.where(u > 0.5, 0.65 + 0.35 * np.sin(age * 35 + b["phase"]), 1.0)
            # Ramp in (quadratic, 0.1 s) as the ring opens: at age < 0.05 s all 56 hot
            # sparks overlap in one ~50 px blob that clips flat, worst on double-ring shells.
            inten = inten * min(1.0, age / 0.1) ** 2
            hx, hy = _drag_path(b["vx"], b["vy"], self.G * self.h, self.K, age)
            # Tail at least halfway out: from the centre, 56 white-hot trails converge into
            # a filled disc that the glow boost clipped flat for frames +1..+3 (~6k px).
            tx, ty = _drag_path(b["vx"], b["vy"], self.G * self.h, self.K, max(age * 0.5, age - 0.2))
            # White-hot for the first 0.2 s, then the shell's colour.
            hot = max(0.0, 1 - age / 0.2)
            col = b["col"] * (1 - hot) + np.float32(250) * hot
            p0 = np.stack([b["x"] + tx, b["y"] + ty], 1) * s
            p1 = np.stack([b["x"] + hx, b["y"] + hy], 1) * s
            trails = np.stack([p0, p1], 1).astype(np.int32)
            dots = np.stack([p1, p1], 1).astype(np.int32)
            # Intensity quantised to 8 levels so each level is one polylines call
            # (a zero-length segment with thickness is a dot): ~16 calls per shell, not 112.
            lvl = np.ceil(inten * op * 8)
            for q in np.unique(lvl[live & (lvl > 0)]):
                sel = live & (lvl == q)
                c = col * (q / 8)
                # Trails grow in as the white-hot phase ends: while the ring is small, 56
                # trails pack into a solid annulus that the glow boost clips flat.
                cv2.polylines(buf, list(trails[sel]), False, tuple(int(v) for v in c * 0.5 * (1 - hot)), 1, cv2.LINE_AA)
                cv2.polylines(buf, list(dots[sel]), False, tuple(int(v) for v in c), 3 if hot > 0 else 2, cv2.LINE_AA)


# --------------------------------------------------------------------------- lasers
class Lasers(Effect):
    """A laser show from behind the person: a few wide, tapered beams in a warm palette
    sweep slowly from emitters behind the shoulders, glowing via the light buffer, and
    cast a faint tint of their colour onto the person."""

    back = front = light = True
    fades_itself = True
    # Beams are added light: at 0.8 a lit room (grey 200) clipped them to pale yellow
    # (median beam saturation 54, vs 208 on black). Added colour c on grey g saturates at
    # best (cmax-cmin)/(g+cmax), so the room has to go darker and the beams a little
    # softer. Measured at 720p, grey 0/128/200: 0.8 & 1.0 -> 186/77/40 (4-hue palette),
    # 0.55 & 1.0 -> 186/101/72, 0.45 & 0.6 -> 189/121/100.
    dim = 0.45
    GAIN = 0.6
    duration = 3.0
    fade_in = 0.3
    fade_out = 0.5

    def __init__(self, w, h, sprites=None, anchors=None, seed=None):
        super().__init__(w, h, sprites, anchors, seed)
        cx, cy = self._centre((w * 0.5, h * 0.62))
        self.origins = [[cx - 0.16 * w, cy], [cx + 0.16 * w, cy]]
        # Colours without replacement, cycled: drawn with replacement, 8% of seeds gave
        # a two-hue show (seed 8: orange/yellow only, both clip to the same cream).
        cols = WARM[:]
        self.rng.shuffle(cols)
        self.beams = []
        for side in (-1, 1):
            for i in range(3):
                self.beams.append(dict(
                    side=side, base=side * (15 + 32 * i + self.rng.uniform(-6, 6)),   # deg from up
                    amp=self.rng.uniform(14, 24), f=self.rng.uniform(0.25, 0.45),
                    ph=self.rng.uniform(0, 6.28), col=cols[len(self.beams) % len(cols)]))
        self.tint = np.mean([b["col"] for b in self.beams], axis=0)

    def follow(self, anchors, head, dt):
        super().follow(anchors, head, dt)
        if head is not None:   # behind the shoulders: head width out, a bit below the chin
            cx, top, hw = head
            y = min(top + hw * 2.6, self.h * 0.9)
            tgt = [(cx - max(hw * 1.1, 0.08 * self.w), y), (cx + max(hw * 1.1, 0.08 * self.w), y)]
        elif self.anchors:
            ax = sorted(a[0] for a in self.anchors)
            ay = sum(a[1] for a in self.anchors) / len(self.anchors)
            tgt = [(ax[0], ay), (ax[-1], ay)] if len(ax) > 1 else [(ax[0] - 0.16 * self.w, ay), (ax[0] + 0.16 * self.w, ay)]
        else:
            return
        for o, (tx, ty) in zip(self.origins, tgt, strict=False):
            o[0] = _ema(o[0], tx, dt, 0.25) if dt else tx
            o[1] = _ema(o[1], ty, dt, 0.25) if dt else ty

    def draw_back(self, img, mask):
        if mask is None:   # no mask: the face darkens too, so only a little
            _darken(img, 0.8, self.opacity)

    def draw_front(self, img, mask):
        if mask is None:
            return
        k = 0.16 * (0.8 + 0.2 * math.sin(self.t * 2.0)) * self.opacity
        # Built at half resolution and upscaled: a tint has no detail to lose (0.3 ms vs 0.55).
        # Through the tight weight like the grade: on the dilated mask the tint lit a
        # ~20 px band of background round the person.
        small = cv2.LUT(cv2.resize(mask, (self.w // 2, self.h // 2), interpolation=cv2.INTER_NEAREST), _TIGHT)
        tint = cv2.merge([cv2.convertScaleAbs(small, alpha=c * k / 255) for c in self.tint])
        cv2.add(img, cv2.resize(tint, (img.shape[1], img.shape[0])), dst=img)

    def draw_light(self, buf):
        s = buf.shape[1] / self.w
        op = self.opacity
        L = 1.5 * max(self.w, self.h) * s
        for b in self.beams:
            ox, oy = self.origins[0 if b["side"] < 0 else 1]
            ox, oy = ox * s, oy * s
            a = math.radians(b["base"] + b["amp"] * math.sin(2 * math.pi * b["f"] * self.t + b["ph"]))
            dx, dy = math.sin(a), -math.cos(a)
            px, py = -dy, dx
            w0, w1 = 1.0, 0.045 * self.h * s     # tapered: a point at the source, wide far away
            quad = np.array([[ox + px * w0, oy + py * w0], [ox + dx * L + px * w1, oy + dy * L + py * w1],
                             [ox + dx * L - px * w1, oy + dy * L - py * w1], [ox - px * w0, oy - py * w0]], np.int32)
            col = np.array(b["col"], np.float32) * (op * self.GAIN)
            cv2.fillConvexPoly(buf, quad, tuple(int(c * 0.35) for c in col), cv2.LINE_AA)
            cv2.line(buf, (int(ox), int(oy)), (int(ox + dx * L), int(oy + dy * L)),
                     tuple(int(c * 0.9) for c in col), 1, cv2.LINE_AA)
        for ox, oy in self.origins:
            cv2.circle(buf, (int(ox * s), int(oy * s)), max(2, int(0.012 * self.h * s)),
                       tuple(int(240 * op) for _ in range(3)), -1, cv2.LINE_AA)


# --------------------------------------------------------------------------- thumbs
def _bubble(glyph: np.ndarray | None, up: bool, size: int = 200) -> np.ndarray:
    """Premultiplied BGRA speech bubble with the glyph inside; tail at the bottom LEFT.

    Built once per Thumb (~1 ms) and only resized per frame. Light grey, slightly
    translucent, with a soft shadow so it reads on both bright and dark rooms.
    """
    s = size
    body = np.zeros((s, s), np.uint8)
    r, x0, y0, x1, y1 = int(s * 0.3), int(s * 0.06), int(s * 0.04), int(s * 0.94), int(s * 0.8)
    cv2.rectangle(body, (x0 + r, y0), (x1 - r, y1), 255, -1)
    cv2.rectangle(body, (x0, y0 + r), (x1, y1 - r), 255, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(body, (cx, cy), r, 255, -1, cv2.LINE_AA)
    tail = np.array([[int(s * 0.2), int(s * 0.7)], [int(s * 0.08), int(s * 0.97)], [int(s * 0.42), int(s * 0.78)]], np.int32)
    cv2.fillConvexPoly(body, tail, 255, cv2.LINE_AA)
    shadow = cv2.GaussianBlur(np.roll(body, int(s * 0.03), axis=0), (0, 0), s * 0.03)
    a_body = body.astype(np.float32) / 255 * 0.94
    a_sh = shadow.astype(np.float32) / 255 * 0.35
    a = a_body + a_sh * (1 - a_body)
    rgb = np.full((s, s, 3), 242, np.float32) * a_body[..., None]      # premultiplied
    gs, gc = int(s * 0.56), (int(s * 0.5), int(s * 0.42))
    if glyph is not None:
        g = cv2.resize(glyph, (gs, gs), interpolation=cv2.INTER_AREA).astype(np.float32)
        ys, xs = gc[1] - gs // 2, gc[0] - gs // 2
        ga = g[..., 3] / 255
        roi = rgb[ys:ys + gs, xs:xs + gs]
        roi[:] = g[..., :3] + roi * (1 - ga[..., None])
        # "over" for alpha too: premultiplied RGB must never exceed alpha, or the
        # uint8 cast in blend_sprite wraps (an opaque glyph on a 0.94 bubble did: R=255+).
        a[ys:ys + gs, xs:xs + gs] = ga + a[ys:ys + gs, xs:xs + gs] * (1 - ga)
    else:  # no asset: coloured disc with a white arrow
        disc = np.zeros((s, s, 3), np.uint8)
        col = (113, 204, 46) if up else (60, 76, 231)
        cv2.circle(disc, gc, gs // 2, col, -1, cv2.LINE_AA)
        d = gs // 4 if up else -gs // 4
        cv2.fillConvexPoly(disc, np.array([[gc[0], gc[1] - d], [gc[0] - gs // 5, gc[1] + d // 2],
                                           [gc[0] + gs // 5, gc[1] + d // 2]], np.int32), (255, 255, 255), cv2.LINE_AA)
        m = disc.any(2)
        rgb[m] = disc[m] * a_body[m][:, None]
    return np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), (a * 255).astype(np.uint8)])


class Thumb(Effect):
    """Apple-style badge: the glyph in a rounded speech bubble beside the head, popping in
    with ease-out-back and following the head/hand smoothly while alive. Front layer."""

    front = True
    duration = 2.4
    fade_out = 0.35
    SIZE = 0.24           # x h

    def __init__(self, w, h, sprites=None, up: bool = True, anchors=None, seed=None):
        super().__init__(w, h, sprites, anchors, seed)
        self.up = up
        glyph = sprites.get("thumbs_up" if up else "thumbs_down", 128) if sprites else None
        self.spr = {-1: None, 1: None}
        base = _bubble(glyph, up)
        self.spr[1] = base                  # bubble right of head: tail points down-left
        self.spr[-1] = base[:, ::-1].copy()
        self.pos: list[float] | None = None
        self.side = 1

    def _target(self, head) -> tuple[float, float, int]:
        bs = self.SIZE * self.h
        hand = self._centre((self.w * 0.5, self.h * 0.7)) if self.anchors else None
        if head is not None:
            cx, top, hw = head
            side = (1 if hand[0] >= cx else -1) if hand else (1 if cx < self.w / 2 else -1)
            for sd in (side, -side):   # the hand's side first; the other if it won't fit
                x = cx + sd * (hw + bs * 0.55)
                if bs * 0.5 <= x <= self.w - bs * 0.5:
                    return x, top + bs * 0.45, sd
            sd = 1 if cx < self.w / 2 else -1    # no room: the roomier top corner
            return (self.w - bs * 0.55 if sd > 0 else bs * 0.55), bs * 0.55, sd
        if hand:   # no mask: up and outward from the hand; the face is usually mid-frame
            sd = 1 if hand[0] >= self.w / 2 else -1
            return hand[0] + sd * 0.1 * self.w, hand[1] - 0.25 * self.h, sd
        return self.w * 0.8, self.h * 0.3, 1

    def follow(self, anchors, head, dt):
        super().follow(anchors, head, dt)
        x, y, self.side = self._target(head)
        bs = self.SIZE * self.h
        x = min(max(x, bs * 0.5), self.w - bs * 0.5)
        y = min(max(y, bs * 0.5), self.h - bs * 0.5)
        if self.pos is None or not dt:
            self.pos = [x, y]
        else:
            self.pos = [_ema(self.pos[0], x, dt, 0.15), _ema(self.pos[1], y, dt, 0.15)]

    def draw_front(self, img, mask):
        if self.pos is None:
            self.follow(None, None, 0.0)
        size = int(self.SIZE * self.h * _ease_out_back(self.t / 0.35))
        if size < 4:
            return
        spr = cv2.resize(self.spr[self.side], (size, size), interpolation=cv2.INTER_AREA)
        blend_sprite(img, spr, self.pos[0], self.pos[1] + 0.006 * self.h * math.sin(self.t * 2.5))


# --------------------------------------------------------------------------- engine
@dataclass
class HoldRing:
    """Small progress arc showing 'keep holding' feedback near the hands.

    x/y are a hint: the ring is clamped into frame at draw time, because raised hands
    routinely put it off the top edge.
    """
    x: float
    y: float
    progress: float
    reaction: str | None = None   # which reaction is pending, for the preview glyph


def _tight_lut() -> np.ndarray:
    """uint8 LUT: compositing mask -> weight that hugs the model's own silhouette.

    refine_mask dilates by r = DILATE_FRAC/FEATHER_FRAC feather sigmas, then feathers, so
    the raw edge sits where the mask reads Phi(r) (0.88) and the dilated edge at 0.5.
    Remapping Phi(r-1)..Phi(r+1) to 0..1 puts a ~2 sigma feather back on the raw edge:
    the dilation is right for hiding particles behind a lagging mask, wrong for a
    full-frame grade (a ~20 px undarkened band round the person, 30-60% brighter than
    the room). A hard 0/255 mask maps to itself, so the person stays bit-exact."""
    r = DILATE_FRAC / FEATHER_FRAC
    phi = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))  # noqa: E731
    lo, hi = phi(r - 1.0), phi(r + 1.0)
    return np.round(np.clip((np.arange(256) / 255.0 - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


_TIGHT = _tight_lut()


EFFECTS = {"hearts": Hearts, "balloons": Balloons, "confetti": Confetti,
           "rain": Rain, "fireworks": Fireworks, "lasers": Lasers}


class EffectEngine:
    def __init__(self, w: int, h: int, sprites=None, seed: int | None = None):
        self.w, self.h = w, h
        # Accept a raw dict too: demo and the tests build one by hand.
        self.sprites = sprites if isinstance(sprites, SpriteSet) else SpriteSet(sprites or {})
        self.active: list[Effect] = []
        self.seed = seed
        self._n = 0

    def resize(self, w: int, h: int) -> None:
        self.w, self.h = w, h

    def trigger(self, reaction: str, anchors: list[tuple[float, float]] | None = None) -> None:
        if reaction not in EFFECTS and reaction not in ("thumbs_up", "thumbs_down"):
            raise ValueError(f"unknown reaction {reaction!r}")
        # Apple's documented coalescing: the same reaction while one is playing extends
        # it (prolong + top up) rather than stacking a second copy.
        for e in self.active:
            if e.kind == reaction and not e.retiring:
                e.extend(anchors)
                return
        sub = SUBSET_OF.get(reaction)
        for e in self.active:
            if e.kind == sub:
                e.retire()
        self._n += 1
        seed = None if self.seed is None else self.seed + self._n
        if reaction in EFFECTS:
            e = EFFECTS[reaction](self.w, self.h, self.sprites, anchors, seed)
        else:
            e = Thumb(self.w, self.h, self.sprites, reaction == "thumbs_up", anchors, seed)
        e.kind = reaction
        self.active.append(e)

    @staticmethod
    def _pass(effects: list[Effect], method: str, img: np.ndarray, mask) -> None:
        for e in effects:
            a = e.opacity
            if a <= 0.0:
                continue
            if a >= 0.995 or e.fades_itself:
                getattr(e, method)(img, mask)
                continue
            tmp = img.copy()   # fade by alpha, not by darkening: ~0.9 ms, only while fading
            getattr(e, method)(tmp, mask)
            cv2.addWeighted(tmp, a, img, 1.0 - a, 0, dst=img)

    def _light(self, effects: list[Effect], img: np.ndarray) -> None:
        h, w = img.shape[:2]
        buf = np.zeros((h // 2, w // 2, 3), np.uint8)
        for e in effects:
            e.draw_light(buf)
        glow = cv2.GaussianBlur(buf, (0, 0), 3)
        cv2.addWeighted(buf, 1.0, glow, 2.0, 0, dst=buf)   # sharp core + boosted halo
        cv2.add(img, cv2.resize(buf, (w, h), interpolation=cv2.INTER_LINEAR), dst=img)

    def step(self, frame: np.ndarray, dt: float, hold: HoldRing | None = None,
             mask: np.ndarray | None = None, anchors: list[tuple[float, float]] | None = None) -> None:
        """Advance and draw every live effect onto `frame` in place.

        mask: uint8 HxW, 255 = person (None: no layering, everything over the frame).
        anchors: current palm centres of the hands behind the latest decision; effects
        that follow hands smooth them over time. None keeps the last ones.
        """
        dt = max(0.0, min(dt, 0.1))  # clamp: a 2 s stall shouldn't teleport particles
        if mask is not None and mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        head = head_box(mask) if mask is not None and self.active else None
        for e in self.active:
            e.update(dt)
            e.follow(anchors, head, dt)
        self.active = [e for e in self.active if e.alive]
        backs = [e for e in self.active if e.back]
        lights = [e for e in self.active if e.light]
        if mask is not None and (backs or lights):
            k = math.prod(1.0 - (1.0 - e.dim) * max(0.0, e.opacity) for e in backs)
            if k < 0.999:
                # Grade first, then let the person keep the original through the TIGHT
                # weight; particles still go behind the dilated one below. Measured 2.6 ms
                # at 720p (scale + LUT + weights + blend) in place of copy + _darken
                # (~0.8 ms), only while a dimming effect is live; fireworks step p50 12 ms.
                bg = cv2.convertScaleAbs(frame, alpha=k)
                wt = cv2.LUT(mask, _TIGHT).astype(np.float32) * np.float32(1 / 255)
                keep = cv2.blendLinear(frame, bg, wt, 1.0 - wt)
            else:
                bg, keep = frame.copy(), frame
            self._pass(backs, "draw_back", bg, mask)
            if lights:
                self._light(lights, bg)
            w = mask.astype(np.float32) * np.float32(1 / 255)
            cv2.blendLinear(keep, bg, w, 1.0 - w, dst=frame)   # person over the back layer
        else:
            self._pass(backs, "draw_back", frame, mask)
            if lights:
                self._light(lights, frame)
        self._pass(sorted((e for e in self.active if e.front), key=lambda e: e.z), "draw_front", frame, mask)
        if hold and 0.05 < hold.progress < 1.0:
            r = int(self.h * 0.03)
            # Clamp here, not in the caller: raised hands put the ring above y=0 and
            # cv2.ellipse clips it silently — no feedback exactly while holding.
            cx = int(min(max(hold.x, r + 4), self.w - r - 4))
            cy = int(min(max(hold.y, r + 4), self.h - r - 4))
            # Dark track first so the white arc reads against a bright background.
            cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, 360, (40, 40, 40), 5, cv2.LINE_AA)
            cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, int(360 * hold.progress), (255, 255, 255), 3, cv2.LINE_AA)
            # The glyph fades up with the hold, so it reads as "this is what's coming"
            # rather than as a static badge — and a misread can be abandoned early.
            spr = self.sprites.get(hold.reaction, r * 1.4) if hold.reaction else None
            if spr is not None:
                blend_sprite(frame, spr, cx, cy, 0.35 + 0.65 * hold.progress)

    @property
    def busy(self) -> bool:
        return bool(self.active)
