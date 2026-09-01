"""Reaction effects. Time-based particle systems drawn straight onto BGR frames.

Design constraints that shaped this:

* Everything is driven by `dt` (seconds), never by frame count — effects must look
  identical at 15 fps on a struggling laptop and 60 fps on a desktop.
* No full-frame alpha compositing. Opaque OpenCV primitives are drawn directly on the
  frame; only emoji sprites alpha-blend, and only inside their ROI. A full-frame float
  blend at 720p costs ~10-15 ms/frame in numpy — too much.
* Sprites arrive from SpriteSet already premultiplied and already scaled to a cached
  size bucket; `blend_sprite` only composites. Every effect falls back to its
  procedural primitive when a sprite is missing, so gestcam works with no assets.
* Each effect owns its RNG so `demo` renders are reproducible.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np

from .sprites import SpriteSet

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
RAINBOW: list[BGR] = [(0, 0, 255), (0, 127, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0), (255, 0, 127)]


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


def _rot_rect(cx: float, cy: float, w: float, h: float, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    hw, hh = w / 2, h / 2
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (corners @ rot.T + (cx, cy)).astype(np.int32)


def blend_sprite(frame: np.ndarray, spr: np.ndarray, cx: float, cy: float,
                 alpha: float = 1.0) -> None:
    """Blend a PREMULTIPLIED, already-scaled BGRA sprite centred on (cx, cy), in place.

    Only SpriteSet produces valid input: a straight-alpha sprite blended here shows a
    dark halo. Premultiplied means the composite is `frame*(1-a) + rgb`, so `alpha`
    scales both terms rather than only the source.
    """
    if alpha <= 0:
        return
    s = spr.shape[0]
    H, W = frame.shape[:2]
    x0, y0 = int(cx - s / 2), int(cy - s / 2)
    x1, y1 = x0 + s, y0 + s
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    spr = spr[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    a = spr[:, :, 3:4].astype(np.float32) * (np.float32(alpha) / 255.0)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (roi * (1 - a) + spr[:, :, :3].astype(np.float32) * np.float32(alpha)).astype(np.uint8)


class Effect:
    """Base: subclasses implement update(dt) and draw(frame); set self.alive = False when done."""

    def __init__(self, w: int, h: int, sprites=None, seed: int | None = None):
        self.w, self.h = w, h
        self.sprites = sprites
        self.t = 0.0
        self.alive = True
        self.rng = random.Random(seed)

    def update(self, dt: float) -> None:
        self.t += dt

    def draw(self, frame: np.ndarray) -> None:
        raise NotImplementedError


# --------------------------------------------------------------------------- hearts
class Hearts(Effect):
    duration = 3.2

    def __init__(self, w, h, sprites=None, anchors=None, seed=None):
        super().__init__(w, h, sprites, seed)
        cx = self.w * 0.5
        cy = self.h * 0.55
        if anchors:
            cx = sum(a[0] for a in anchors) / len(anchors)
            cy = sum(a[1] for a in anchors) / len(anchors)
        self.parts = []
        for i in range(16):
            self.parts.append(dict(
                x=cx + self.rng.uniform(-0.18, 0.18) * self.w,
                y=cy + self.rng.uniform(-0.05, 0.05) * self.h,
                vy=-self.rng.uniform(0.12, 0.22) * self.h,
                size=self.rng.uniform(0.045, 0.09) * self.h,
                delay=self.rng.uniform(0, 1.2),
                phase=self.rng.uniform(0, 6.28),
                col=(60, 60, 235) if self.rng.random() < 0.6 else (150, 120, 255),
                emoji=self.rng.choice(("hearts", "heart_sparkle", "heart_purple")),
            ))

    def update(self, dt):
        super().update(dt)
        for p in self.parts:
            if self.t > p["delay"]:
                p["y"] += p["vy"] * dt
                p["x"] += math.sin(self.t * 2.5 + p["phase"]) * 0.03 * self.w * dt
        self.alive = self.t < self.duration

    def draw(self, frame):
        for p in self.parts:
            age = self.t - p["delay"]
            if age < 0:
                continue
            life = self.duration - p["delay"]
            k = 1.0 - max(0.0, (age - life * 0.7) / (life * 0.3 + 1e-6))  # shrink at the end
            k = max(0.0, k) * (1.0 + 0.08 * math.sin(age * 8 + p["phase"]))
            # 1.8: HEART's unit half-extent is ~0.94, so its full width is ~1.88*size —
            # this matches the emoji to the polygon it replaces. The min() is a cost cap,
            # tunable in `gestcam demo --preview`.
            spr = self.sprites.get(p["emoji"], min(p["size"] * k * 1.8, self.h * 0.13)) if self.sprites else None
            if spr is not None:
                blend_sprite(frame, spr, p["x"], p["y"], 1.0)
                continue
            pts = (HEART * (p["size"] * k) + (p["x"], p["y"])).astype(np.int32)
            cv2.fillPoly(frame, [pts], p["col"], lineType=cv2.LINE_AA)


# --------------------------------------------------------------------------- balloons
class Balloons(Effect):
    duration = 4.5

    def __init__(self, w, h, sprites=None, anchors=None, seed=None):
        super().__init__(w, h, sprites, seed)
        self.parts = []
        for i in range(10):
            r = self.rng.uniform(0.045, 0.07) * self.h
            self.parts.append(dict(
                x=self.rng.uniform(0.08, 0.92) * self.w,
                y=self.h + r * 3 + self.rng.uniform(0, 0.3) * self.h,
                vy=-self.rng.uniform(0.16, 0.26) * self.h,
                r=r, phase=self.rng.uniform(0, 6.28),
                col=self.rng.choice(PALETTE),
            ))

    def update(self, dt):
        super().update(dt)
        for p in self.parts:
            p["y"] += p["vy"] * dt
            p["x"] += math.sin(self.t * 1.6 + p["phase"]) * 0.06 * self.w * dt
        self.alive = self.t < self.duration and any(p["y"] > -p["r"] * 3 for p in self.parts)

    def draw(self, frame):
        for p in self.parts:
            x, y, r = int(p["x"]), int(p["y"]), int(p["r"])
            if y < -r * 3 or y > self.h + r * 3:
                continue
            sway = math.sin(self.t * 1.6 + p["phase"])
            # string
            pts = np.array([[x, y + int(r * 1.15)], [x + int(sway * r * 0.4), y + int(r * 2.1)], [x - int(sway * r * 0.3), y + int(r * 3.0)]], np.int32)
            cv2.polylines(frame, [pts], False, (70, 70, 70), 1, cv2.LINE_AA)
            # 2.2r matches the emoji to the ellipse it replaces; tunable in demo. The
            # string stays procedural either way — no emoji carries a swaying string.
            spr = self.sprites.get("balloons", r * 2.2) if self.sprites else None
            if spr is not None:
                blend_sprite(frame, spr, x, y, 1.0)
                continue
            # knot
            cv2.fillConvexPoly(frame, np.array([[x - r // 6, y + int(r * 1.2)], [x + r // 6, y + int(r * 1.2)], [x, y + int(r * 1.0)]], np.int32), p["col"], cv2.LINE_AA)
            # body + highlight
            cv2.ellipse(frame, (x, y), (int(r * 0.85), r), 0, 0, 360, p["col"], -1, cv2.LINE_AA)
            cv2.ellipse(frame, (x - int(r * 0.3), y - int(r * 0.35)), (int(r * 0.18), int(r * 0.3)), -25, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)


# --------------------------------------------------------------------------- confetti
class Confetti(Effect):
    duration = 4.0

    def __init__(self, w, h, sprites=None, anchors=None, seed=None):
        super().__init__(w, h, sprites, seed)
        self.parts = []
        # 200 rects are the texture and the density; they cost almost nothing. The last
        # 24 also carry an emoji as readable "hero" pieces — an emoji silhouette needs
        # ~40px to read at 720p/30fps, and 200 of those would be unreadable mush and
        # ~5.5 ms besides.
        for i in range(224):
            emoji = self.rng.choice(("confetti", "confetti_ball", "sparkles", "star")) if i >= 200 else None
            self.parts.append(dict(
                emoji=emoji,
                x=self.rng.uniform(0, 1) * self.w,
                y=-self.rng.uniform(0, 0.25) * self.h,
                vy=self.rng.uniform(0.22, 0.38) * self.h,
                delay=self.rng.uniform(0, 0.8),
                phase=self.rng.uniform(0, 6.28),
                w=self.rng.uniform(0.016, 0.026) * self.h,
                h=self.rng.uniform(0.008, 0.014) * self.h,
                ang=self.rng.uniform(0, 3.14), spin=self.rng.uniform(-6, 6),
                col=self.rng.choice(PALETTE),
            ))

    def update(self, dt):
        super().update(dt)
        for p in self.parts:
            if self.t < p["delay"]:
                continue
            p["y"] += p["vy"] * dt
            p["x"] += math.sin(self.t * 4 + p["phase"]) * 0.12 * self.w * dt
            p["ang"] += p["spin"] * dt
        self.alive = self.t < self.duration

    def draw(self, frame):
        fade = 1.0 if self.t < self.duration - 0.6 else max(0.0, (self.duration - self.t) / 0.6)
        for p in self.parts:
            if self.t < p["delay"] or p["y"] > self.h + 20:
                continue
            k = max(0.15, fade)
            if p["emoji"] and self.sprites:
                spr = self.sprites.get(p["emoji"], p["w"] * 3.5 * k)
                if spr is not None:
                    # ponytail: warp per call, ~1.4 ms for 24 sprites. A pre-rotation
                    # LUT costs ~384 cache entries and 22.5° quantised spin to save
                    # that; only worth it if a future effect spins hundreds at once.
                    # warpAffine is correct here only because the sprite is
                    # premultiplied — linear filtering of straight alpha is not.
                    sz = spr.shape[0]
                    m = cv2.getRotationMatrix2D((sz / 2, sz / 2), math.degrees(p["ang"]), 1.0)
                    blend_sprite(frame, cv2.warpAffine(spr, m, (sz, sz), flags=cv2.INTER_LINEAR,
                                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)),
                                 p["x"], p["y"], 1.0)
                    continue
            cv2.fillConvexPoly(frame, _rot_rect(p["x"], p["y"], p["w"] * k, p["h"] * k, p["ang"]), p["col"], cv2.LINE_AA)


# --------------------------------------------------------------------------- rain
class Rain(Effect):
    duration = 3.5

    def __init__(self, w, h, sprites=None, anchors=None, seed=None):
        super().__init__(w, h, sprites, seed)
        self.drops = [dict(
            x=self.rng.uniform(0, 1) * self.w,
            y=self.rng.uniform(-1, 0) * self.h,
            v=self.rng.uniform(0.9, 1.3) * self.h,
            ln=self.rng.uniform(0.02, 0.04) * self.h,
        ) for _ in range(140)]
        cw = self.w
        self.cloud = [(int(cw * (0.08 + i * 0.12) + self.rng.uniform(-0.03, 0.03) * cw),
                       int(self.rng.uniform(0.0, 0.05) * self.h),
                       int(self.rng.uniform(0.09, 0.14) * self.h)) for i in range(8)]

    def update(self, dt):
        super().update(dt)
        for d in self.drops:
            d["y"] += d["v"] * dt
            if d["y"] > self.h:
                d["y"] = -d["ln"]
                d["x"] = self.rng.uniform(0, 1) * self.w
        self.alive = self.t < self.duration

    def draw(self, frame):
        # Stormy: dim the frame a little (single native op, cheap).
        fade = min(1.0, self.t / 0.4, max(0.0, (self.duration - self.t) / 0.6))
        if fade > 0:
            cv2.addWeighted(frame, 1 - 0.3 * fade, frame, 0, 0, dst=frame)
        cy_off = int(-0.12 * self.h * (1 - fade))
        # 0.24*h and the +0.06*h nudge are the two knobs here. self.cloud (8 puffs) is
        # kept solely for the no-sprite path — don't delete it.
        spr = self.sprites.get("rain", self.h * 0.24) if self.sprites else None
        if spr is not None:
            for fx in (0.2, 0.5, 0.8):
                blend_sprite(frame, spr, fx * self.w, cy_off + self.h * 0.06, 1.0)
        else:
            for (x, y, r) in self.cloud:
                cv2.circle(frame, (x, y + cy_off), r, (120, 120, 125), -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y + cy_off - r // 4), int(r * 0.85), (150, 150, 155), -1, cv2.LINE_AA)
        if self.t > 0.3:
            for d in self.drops:
                x, y = int(d["x"]), int(d["y"])
                cv2.line(frame, (x, y), (x - 2, y + int(d["ln"])), (235, 210, 170), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- fireworks
class Fireworks(Effect):
    duration = 3.2

    def __init__(self, w, h, sprites=None, anchors=None, seed=None):
        super().__init__(w, h, sprites, seed)
        self.bursts = []
        for i in range(5):
            self.bursts.append(dict(
                emoji=self.rng.choice(("fireworks", "sparkler", "firecracker")),
                t0=i * 0.45 + self.rng.uniform(0, 0.2),
                x=self.rng.uniform(0.15, 0.85) * self.w,
                y=self.rng.uniform(0.12, 0.5) * self.h,
                col=self.rng.choice(PALETTE),
                parts=[dict(a=self.rng.uniform(0, 6.283), v=self.rng.uniform(0.25, 0.55) * self.h,
                            x=0.0, y=0.0, hist=[]) for _ in range(70)],
            ))
        self.g = 0.55 * self.h

    def update(self, dt):
        super().update(dt)
        for b in self.bursts:
            age = self.t - b["t0"]
            if age < 0:
                continue
            for p in b["parts"]:
                p["hist"].append((p["x"], p["y"]))
                if len(p["hist"]) > 5:
                    p["hist"].pop(0)
                drag = math.exp(-1.6 * age)
                vx, vy = math.cos(p["a"]) * p["v"] * drag, math.sin(p["a"]) * p["v"] * drag + self.g * age
                p["x"] += vx * dt
                p["y"] += vy * dt
        self.alive = self.t < self.duration

    def draw(self, frame):
        for b in self.bursts:
            age = self.t - b["t0"]
            if age < 0 or age > 1.9:
                continue
            k = max(0.0, 1.0 - age / 1.9)
            col = tuple(int(c * (0.35 + 0.65 * k)) for c in b["col"])
            rad = max(1, int(3 * k + 1))
            for p in b["parts"]:
                x, y = int(b["x"] + p["x"]), int(b["y"] + p["y"])
                tx, ty = p["hist"][0] if p["hist"] else (p["x"], p["y"])
                cv2.line(frame, (int(b["x"] + tx), int(b["y"] + ty)), (x, y), col, rad, cv2.LINE_AA)
                cv2.circle(frame, (x, y), rad, (255, 255, 255) if age < 0.5 else col, -1, cv2.LINE_AA)
            # 0.16*h is the knob. The elif preserves the old behaviour exactly on the
            # no-sprite path: it is only reachable when spr is None and age < 0.5.
            if age < 0.5:
                pop = min(1.0, age / 0.12)
                spr = self.sprites.get(b["emoji"], self.h * 0.16 * pop) if self.sprites else None
                if spr is not None:
                    blend_sprite(frame, spr, b["x"], b["y"], max(0.0, 1.0 - age / 0.5))
                elif age < 0.12:
                    cv2.circle(frame, (int(b["x"]), int(b["y"])), int(self.h * 0.06 * (1 - age / 0.12)), (255, 255, 255), -1, cv2.LINE_AA)


# --------------------------------------------------------------------------- lasers
class Lasers(Effect):
    duration = 2.6

    def __init__(self, w, h, sprites=None, anchors=None, seed=None):
        super().__init__(w, h, sprites, seed)
        self.origins = list(anchors) if anchors else [(w * 0.3, h * 0.55), (w * 0.7, h * 0.55)]
        self.beams = [self.rng.uniform(0, 6.283) for _ in range(9)]
        self.spin = self.rng.choice([-1, 1]) * 0.9

    def draw(self, frame):
        fade = min(1.0, self.t / 0.25, max(0.0, (self.duration - self.t) / 0.4))
        L = int(math.hypot(self.w, self.h))
        for oi, (ox, oy) in enumerate(self.origins):
            for bi, a0 in enumerate(self.beams):
                a = a0 + self.t * self.spin * (1 if oi % 2 == 0 else -1)
                col = RAINBOW[(bi + int(self.t * 6)) % len(RAINBOW)]
                col = tuple(int(c * fade) for c in col)
                ex, ey = int(ox + math.cos(a) * L), int(oy + math.sin(a) * L)
                cv2.line(frame, (int(ox), int(oy)), (ex, ey), tuple(c // 2 for c in col), 7, cv2.LINE_AA)
                cv2.line(frame, (int(ox), int(oy)), (ex, ey), col, 2, cv2.LINE_AA)
            # 0.10*h sits the horns emoji over the palm, where the beams originate.
            pulse = 1 + 0.3 * math.sin(self.t * 20)
            spr = self.sprites.get("lasers", self.h * 0.10 * pulse) if self.sprites else None
            if spr is not None:
                blend_sprite(frame, spr, ox, oy, fade)
            else:
                cv2.circle(frame, (int(ox), int(oy)), int(self.h * 0.02 * pulse), (255, 255, 255), -1, cv2.LINE_AA)

    def update(self, dt):
        super().update(dt)
        self.alive = self.t < self.duration


# --------------------------------------------------------------------------- thumbs
class Thumb(Effect):
    duration = 2.4

    def __init__(self, w, h, sprites=None, up: bool = True, anchors=None, seed=None):
        super().__init__(w, h, sprites, seed)
        self.up = up
        if anchors:
            ax, ay = anchors[0]
            self.x = min(max(ax, w * 0.15), w * 0.85)
            self.y = min(max(ay - h * 0.12, h * 0.2), h * 0.85)
        else:
            self.x, self.y = w * 0.78, h * 0.5

    def update(self, dt):
        super().update(dt)
        self.alive = self.t < self.duration

    def draw(self, frame):
        pop = _ease_out_back(self.t / 0.35)
        fade = 1.0 if self.t < self.duration - 0.45 else max(0.0, (self.duration - self.t) / 0.45)
        size = self.h * 0.36 * pop
        y = self.y - 0.05 * self.h * self.t
        spr = self.sprites.get("thumbs_up" if self.up else "thumbs_down", size) if self.sprites else None
        if spr is not None:
            blend_sprite(frame, spr, self.x, y, fade)
        else:  # procedural fallback: white bubble with a colour disc
            r = int(size / 2)
            if r > 1 and fade > 0:
                cv2.circle(frame, (int(self.x), int(y)), r, (245, 245, 245), -1, cv2.LINE_AA)
                cv2.circle(frame, (int(self.x), int(y)), int(r * 0.55), (113, 204, 46) if self.up else (60, 76, 231), -1, cv2.LINE_AA)


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
        self._n += 1
        seed = None if self.seed is None else self.seed + self._n
        w, h = self.w, self.h
        mk = {"hearts": Hearts, "balloons": Balloons, "confetti": Confetti,
              "rain": Rain, "fireworks": Fireworks, "lasers": Lasers}
        cls = mk.get(reaction)
        if cls is not None:
            self.active.append(cls(w, h, self.sprites, anchors, seed))
        elif reaction in ("thumbs_up", "thumbs_down"):
            self.active.append(Thumb(w, h, self.sprites, reaction == "thumbs_up", anchors, seed))
        else:
            raise ValueError(f"unknown reaction {reaction!r}")

    def step(self, frame: np.ndarray, dt: float, hold: HoldRing | None = None) -> None:
        dt = max(0.0, min(dt, 0.1))  # clamp: a 2 s stall shouldn't teleport particles
        for e in self.active:
            e.update(dt)
        self.active = [e for e in self.active if e.alive]
        for e in self.active:
            e.draw(frame)
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
