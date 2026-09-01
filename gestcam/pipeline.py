"""The main loop.

    capture thread ──newest frame──▶ main loop ──rgb (downscaled)──▶ MediaPipe thread
                                        │                                   │
                                        │◀──────── latest Observation ──────┘
                                        ▼
                             reaction_for() → ReactionState → EffectEngine.step()
                                        ▼
                               sinks: v4l2loopback / preview / file

Only the main loop touches the frame buffer that goes out. Inference is fire-and-forget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .effects import EffectEngine, HoldRing
from .recognizer import GestureRecognizer, ensure_model, pick
from .state import ReactionState


@dataclass
class RunConfig:
    device: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mjpg: bool = True
    mirror: bool = False
    infer_width: int = 640      # downscale for the model; landmarks are normalized so this is free
    infer_every: int = 1        # submit every Nth frame to the model
    num_hands: int = 2
    min_score: float = 0.5
    hold_s: float = 0.7
    cooldown_s: float = 2.5
    hold_ring: bool = True
    stale_s: float = 0.4        # ignore observations older than this (model hiccup)
    model: str | None = None
    stats_every_s: float = 5.0


class Stats:
    def __init__(self):
        self.frames = 0
        self.infer_submitted = 0
        self.infer_ms: list[float] = []
        self.loop_ms: list[float] = []
        self.t0 = time.monotonic()
        self.last_print = self.t0

    def line(self) -> str:
        now = time.monotonic()
        fps = self.frames / max(1e-6, now - self.t0)
        im = np.percentile(self.infer_ms, [50, 95]) if self.infer_ms else (0, 0)
        lm = np.percentile(self.loop_ms, [50, 95]) if self.loop_ms else (0, 0)
        return (f"[gestcam] out {fps:5.1f} fps | loop p50/p95 {lm[0]:.1f}/{lm[1]:.1f} ms | "
                f"infer p50/p95 {im[0]:.1f}/{im[1]:.1f} ms | inferred {self.infer_submitted}")

    def reset_window(self):
        self.infer_ms.clear()
        self.loop_ms.clear()


def anchors_px(hands, w: int, h: int) -> list[tuple[float, float]]:
    """Palm centres of the hands the rule actually used, in pixels."""
    return [(x * w, y * h) for x, y in (hd.palm_center() for hd in hands)]


def run(cfg: RunConfig, make_sink, log=print, should_stop=lambda: False) -> None:
    """`make_sink(width, height, fps)` is called once the camera has negotiated its
    real format, so sinks never guess the frame size."""
    from .sprites import load_sprites
    from .streams import Capture

    model = ensure_model(cfg.model, log=log)
    cap = Capture(cfg.device, cfg.width, cfg.height, cfg.fps, mjpg=cfg.mjpg, log=log)
    log(f"[gestcam] camera: {cap.width}x{cap.height} @ {cap.fps:.0f} fps ({cap.fourcc})")
    sink = make_sink(cap.width, cap.height, cap.fps)
    rec = GestureRecognizer(model, num_hands=cfg.num_hands, log=log)
    engine = EffectEngine(cap.width, cap.height, sprites=load_sprites(log=log))
    state = ReactionState(hold_s=cfg.hold_s, cooldown_s=cfg.cooldown_s)
    stats = Stats()

    scale = min(1.0, cfg.infer_width / cap.width)
    infer_size = (int(cap.width * scale), int(cap.height * scale))
    seq = -1
    last_t = time.monotonic()
    last_obs_ts = -1.0
    last_anchors: list[tuple[float, float]] = []
    try:
        while not should_stop():
            got = cap.wait_frame(seq)
            if got is None:
                log("[gestcam] camera stalled (no frame for 1 s)")
                continue
            frame, ts, seq = got
            t_loop = time.monotonic()
            frame = frame.copy()  # capture thread keeps writing into its own buffers
            if cfg.mirror:
                frame = cv2.flip(frame, 1)

            if stats.frames % cfg.infer_every == 0:
                small = cv2.resize(frame, infer_size, interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                if rec.submit(rgb, ts):
                    stats.infer_submitted += 1

            obs = rec.latest()
            now = time.monotonic()
            candidate = None
            anchors: list[tuple[float, float]] = []
            if obs is not None and now - obs.ts <= cfg.stale_s:
                candidate, used = pick(obs.hands, cfg.min_score)
                anchors = anchors_px(used, cap.width, cap.height)
                if obs.ts != last_obs_ts and obs.infer_ms:
                    stats.infer_ms.append(obs.infer_ms)
                    last_obs_ts = obs.ts
            if anchors:
                last_anchors = anchors
            fired = state.update(candidate, now)
            if fired:
                log(f"[gestcam] ▶ {fired}  (hands={[h.label for h in obs.hands] if obs else []})")
                engine.trigger(fired, anchors)

            hold = None
            # Ring follows the state machine, not the per-frame candidate. The hold
            # survives sub-grace dropouts, so the ring must too — a ring that blinks
            # out reads as "it lost me" and the user drops the gesture one frame
            # before it would have fired.
            if cfg.hold_ring and state.current and last_anchors:
                ax = sum(a[0] for a in last_anchors) / len(last_anchors)
                ay = sum(a[1] for a in last_anchors) / len(last_anchors)
                hold = HoldRing(ax, ay - cap.height * 0.12, state.progress(now))
            engine.step(frame, now - last_t, hold)
            last_t = now

            if not sink.send(frame):
                break
            stats.frames += 1
            stats.loop_ms.append((time.monotonic() - t_loop) * 1000)
            if now - stats.last_print >= cfg.stats_every_s:
                log(stats.line())
                stats.reset_window()
                stats.last_print = now
    finally:
        rec.close()
        cap.close()
        sink.close()


# ------------------------------------------------------------------ demo (no camera / no model)
def demo(sink, width: int = 1280, height: int = 720, fps: int = 30, seconds: float = 24.0,
         seed: int = 7, log=print, source=None) -> None:
    """Fire every reaction in sequence on a synthetic background (or a real camera if
    `source` is a Capture). Lets you tune the visuals without a model or gestures."""
    from .recognizer import REACTIONS
    from .sprites import load_sprites

    engine = EffectEngine(width, height, sprites=load_sprites(log=log), seed=seed)
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    per = seconds / len(REACTIONS)
    schedule = {int(i * per * fps) + 5: r for i, r in enumerate(REACTIONS)}
    seq = -1
    t = 0.0
    dt = 1.0 / fps
    for i in range(n):
        if source is not None:
            got = source.wait_frame(seq)
            frame = got[0].copy() if got else np.zeros((height, width, 3), np.uint8)
            if got:
                seq = got[2]
        else:
            frame = synthetic_frame(width, height, t, rng)
        if i in schedule:
            r = schedule[i]
            anchors = [(width * 0.42, height * 0.6), (width * 0.58, height * 0.6)]
            log(f"[demo] {r}")
            engine.trigger(r, anchors)
        engine.step(frame, dt)
        cv2.putText(frame, f"gestcam demo  t={t:5.1f}s", (16, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if not sink.send(frame):
            break
        t += dt
        if source is None:
            time.sleep(0)  # file/preview sinks pace themselves; real-time not needed
    sink.close()


def synthetic_frame(w: int, h: int, t: float, rng) -> np.ndarray:
    """A moving gradient with a 'person' blob, so effects are judged against something."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = 40 + 60 * (0.5 + 0.5 * np.sin(xx / w * 3 + t * 0.4)) + 20 * (yy / h)
    frame = np.stack([g * 1.1, g * 0.9, g * 0.8], axis=2).astype(np.uint8)
    cx, cy = int(w * 0.5 + 20 * np.sin(t * 0.7)), int(h * 0.62)
    cv2.ellipse(frame, (cx, cy + int(h * 0.3)), (int(w * 0.18), int(h * 0.35)), 0, 0, 360, (90, 70, 60), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy - int(h * 0.12)), int(h * 0.13), (150, 170, 200), -1, cv2.LINE_AA)
    return frame
