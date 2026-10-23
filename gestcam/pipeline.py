"""The main loop, and the offline evaluator that shares its decision step.

    capture thread ──newest frame──▶ main loop ──rgb (downscaled)──▶ MediaPipe threads
                                        │                             (gestures, person mask)
                                        │◀──── latest Observation / mask ───┘
                                        ▼
                  decide(): pick() → ReactionState (once per NEW observation)
                                        ▼
                    EffectEngine.step(frame, dt, hold, mask, anchors)
                                        ▼
                               sinks: v4l2loopback / preview / file

Only the main loop touches the frame buffer that goes out. Inference is fire-and-forget.
`evaluate` runs the same decide() over a clip in VIDEO mode on the clip's own clock, so
what eval reports is what the live loop would have fired.
"""

from __future__ import annotations

import json
import math
import pathlib
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .effects import EffectEngine, HoldRing, _ema
from .recognizer import GestureRecognizer, Hand, Observation, ensure_model, pick
from .state import ReactionState


@dataclass
class RunConfig:
    device: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    mjpg: bool = True
    mirror: bool = False
    infer_width: int = (
        640  # downscale for the model; landmarks are normalized so this is free
    )
    infer_every: int = 1  # submit every Nth frame to the model
    num_hands: int = 2
    detect_conf: float = 0.5  # palm detector floor; hearts are detection-limited
    min_score: float = 0.5
    hold_s: float = 0.7
    cooldown_s: float = 2.5
    defer_s: float = 0.15  # extra hold for a one-hand pose while a 2nd hand is in frame
    hold_ring: bool = True
    stale_s: float = 0.4  # ignore observations older than this (model hiccup)
    model: str | None = None
    segment: bool = True  # person mask so back-layer effects go behind you
    debug: bool = False  # landmarks, poses, decision, hold accumulators on the output
    record_raw: str | None = None  # .mp4 of the frames the model saw, pre-effects
    stats_every_s: float = 5.0


# Segmentation auto-off (measured, track D): with both models on, gesture infer p95 was
# 28-29 ms and segmenter 8.6 ms on an i7-13620H, so 33 ms (one frame) leaves ~4 ms of
# margin. Two consecutive 5 s windows ignore single spikes (p99 hit 63 ms once). Never
# re-enabled: flapping on/off is worse than staying off.
SEG_OFF_MS = 33.0
SEG_OFF_WINDOWS = 2
RING_TAU = 0.09  # s; the old 0.3-per-frame EMA at 30 fps, now the same lag at any fps


class Stats:
    def __init__(self):
        self.frames = 0
        self.infer_submitted = 0
        self.infer_ms: list[float] = []
        self.seg_ms: list[float] = []
        self.loop_ms: list[float] = []
        self.mask_loop_ms: list[float] = []  # loop time of frames that composited a mask
        self.t0 = time.monotonic()
        self.last_print = self.t0
        # fps is per window like every other number on the line: a session average
        # carried startup forever and smoothed a 15 fps window into "27".
        self.win_t0, self.win_frames0 = self.t0, 0

    def line(self) -> str:
        now = time.monotonic()
        fps = (self.frames - self.win_frames0) / max(1e-6, now - self.win_t0)
        im = np.percentile(self.infer_ms, [50, 95]) if self.infer_ms else (0, 0)
        lm = np.percentile(self.loop_ms, [50, 95]) if self.loop_ms else (0, 0)
        seg = ""
        if self.seg_ms:
            sm = np.percentile(self.seg_ms, [50, 95])
            seg = f"seg p50/p95 {sm[0]:.1f}/{sm[1]:.1f} ms | "
        return (
            f"[gestcam] out {fps:5.1f} fps | loop p50/p95 {lm[0]:.1f}/{lm[1]:.1f} ms | "
            f"infer p50/p95 {im[0]:.1f}/{im[1]:.1f} ms | {seg}inferred {self.infer_submitted}"
        )

    def reset_window(self):
        self.infer_ms.clear()
        self.seg_ms.clear()
        self.loop_ms.clear()
        self.mask_loop_ms.clear()
        self.win_t0, self.win_frames0 = time.monotonic(), self.frames


def overload_streak(streak: int, *windows: list[float]) -> int:
    """Consecutive stats windows in which any p95 (gesture model, segmenter, loop time of
    masked frames) missed the frame budget. Loop time counts because the mask's main
    cost is compositing on the main thread, not the model: under load ~9-15 masked
    step() means were 17-25 ms vs 4-10 unmasked, and all 8 at once 36 / 49 p95 ms
    (scratchpad build_effects/perf.py), so models can be on budget while the loop isn't.
    Only masked frames count, so a slow unmasked frame never costs the mask."""
    p95 = max((np.percentile(xs, 95) for xs in windows if xs), default=0.0)
    return streak + 1 if p95 > SEG_OFF_MS else 0


def anchors_px(hands, w: int, h: int) -> list[tuple[float, float]]:
    """Palm centres of the hands the rule actually used, in pixels."""
    return [(x * w, y * h) for x, y in (hd.palm_center() for hd in hands)]


def model_rgb(frame: np.ndarray, infer_width: int) -> np.ndarray:
    """BGR frame → RGB at most `infer_width` wide, the input both models get."""
    h, w = frame.shape[:2]
    if w > infer_width:
        size = (infer_width, round(h * infer_width / w))
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def make_state(cfg: RunConfig) -> ReactionState:
    return ReactionState(hold_s=cfg.hold_s, cooldown_s=cfg.cooldown_s, defer_s=cfg.defer_s)


@dataclass
class Decision:
    candidate: str | None
    hands: list[Hand] = field(default_factory=list)  # the hands that decided it
    anchors: list[tuple[float, float]] = field(default_factory=list)  # their palms, px
    fired: str | None = None


def decide(obs: Observation, state: ReactionState, w: int, h: int, min_score: float) -> Decision:
    """One NEW observation → the state machine, on the observation's own clock.

    Call it once per observation, never per output frame: re-feeding the same
    observation counted one piece of evidence many times, so how long a hold took
    depended on model latency and --infer-every (T6). n_hands lets the state defer a
    one-hand reaction while a second hand is in frame and may be about to join.
    """
    candidate, used = pick(obs.hands, min_score)
    fired = state.update(candidate, obs.ts, n_hands=len(obs.hands))
    return Decision(candidate, used, anchors_px(used, w, h), fired)


def run(cfg: RunConfig, make_sink, log=print, should_stop=lambda: False) -> None:
    """`make_sink(width, height, fps)` is called once the camera has negotiated its
    real format, so sinks never guess the frame size."""
    from .segment import PersonSegmenter, ensure_segmenter, refine_mask
    from .sprites import load_sprites
    from .streams import Capture, FileOutput

    model = ensure_model(cfg.model, log=log)
    cap = Capture(cfg.device, cfg.width, cfg.height, cfg.fps, mjpg=cfg.mjpg, log=log)
    log(
        f"[gestcam] camera: {cap.width}x{cap.height} @ {cap.fps:.0f} fps ({cap.fourcc})"
    )
    W, H = cap.width, cap.height
    sink = make_sink(W, H, cap.fps)
    raw = FileOutput(cfg.record_raw, W, H, cap.fps) if cfg.record_raw else None
    raw_ts: list[float] = []  # capture ts per raw frame → ts_sidecar, eval's clock
    rec = GestureRecognizer(
        model, num_hands=cfg.num_hands, log=log, detect_conf=cfg.detect_conf
    )
    seg = None
    if cfg.segment:
        try:  # optional: without it every effect is simply drawn over the person
            seg = PersonSegmenter(ensure_segmenter(log=log), log=log)
        except Exception as e:
            log(f"[gestcam] segmentation off: {e}")
    engine = EffectEngine(W, H, sprites=load_sprites(log=log))
    state = make_state(cfg)
    stats = Stats()

    seq = -1
    last_t = time.monotonic()
    obs: Observation | None = None
    dec = Decision(None)
    last_obs_ts = -1.0
    last_seg_ts = -1.0
    mask: np.ndarray | None = None
    overload = 0
    last_anchors: list[tuple[float, float]] = []
    ring: tuple[float, float] | None = None
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
            # The mask is read only while a hold or an effect is on screen (head_box,
            # back/light composite). Idle it cost ~0.5 core and 3x loop p50 with an
            # effect up in 12% of frames (review/perf/idle_cost.py), and starved the
            # gesture model. A hold takes >= hold_s (0.7 s) and a mask arrives ~10 ms
            # after submit, so it is warm long before anything fires.
            want_mask = seg is not None and (state.current is not None or engine.busy)

            if stats.frames % cfg.infer_every == 0:
                rgb = model_rgb(frame, cfg.infer_width)
                if rec.submit(rgb, ts):
                    stats.infer_submitted += 1
                if want_mask:
                    seg.submit(rgb, ts)  # same pixels, same timestamp as the gestures
            if raw is not None:
                # After submit, so the 6-10 ms mp4v encode doesn't delay the models.
                raw.send(frame)  # what the models see: mirrored if --mirror, no effects
                raw_ts.append(ts)

            now = time.monotonic()
            latest = rec.latest()
            if latest is not None and latest.ts != last_obs_ts:
                obs, last_obs_ts = latest, latest.ts
                if obs.infer_ms:
                    stats.infer_ms.append(obs.infer_ms)
                dec = decide(obs, state, W, H, cfg.min_score)
                if dec.anchors:
                    last_anchors = dec.anchors
                if dec.fired:
                    poses = [f"{h.label}/{h.pose(cfg.min_score)}" for h in obs.hands]
                    log(f"[gestcam] ▶ {dec.fired}  (hands={poses})")
                    engine.trigger(dec.fired, dec.anchors)
            fresh = obs is not None and now - obs.ts <= cfg.stale_s

            if not want_mask:
                mask = None  # also resets refine_mask's smoothing for the next hold
            else:
                got_mask = seg.latest()
                if got_mask is not None and got_mask[0] != last_seg_ts:
                    last_seg_ts = got_mask[0]
                    stats.seg_ms.append(seg.latency_ms)
                    # refined per new mask; prev smoothing hides the mask's frame of lag
                    mask = refine_mask(got_mask[1], W, H, mask)
                if now - last_seg_ts > cfg.stale_s:
                    mask = None  # an old silhouette misplaces effects worse than none

            hold = None
            # Ring follows the state machine, not the per-frame candidate. The hold
            # survives sub-grace dropouts, so the ring must too — a ring that blinks
            # out reads as "it lost me" and the user drops the gesture one frame
            # before it would have fired.
            if not state.current:
                last_anchors = []  # hold over: the next ring starts at the new hands
            dt = now - last_t
            if cfg.hold_ring and state.current and last_anchors and fresh:
                ax = sum(a[0] for a in last_anchors) / len(last_anchors)
                ay = (
                    sum(a[1] for a in last_anchors) / len(last_anchors)
                    - cap.height * 0.12
                )
                # EMA so landmark jitter doesn't read as "it lost me". Time-based, so
                # the lag is ~90 ms at any fps; the hand is stationary while held.
                ring = (
                    (ax, ay)
                    if ring is None
                    else (_ema(ring[0], ax, dt, RING_TAU), _ema(ring[1], ay, dt, RING_TAU))
                )
                # The glyph previews what will fire, so a misread can be abandoned
                # before the 2.5 s cooldown locks it in.
                hold = HoldRing(ring[0], ring[1], state.progress(now), state.current)
            else:
                ring = None
            # Effects follow the hands that are making a gesture right now; None (no
            # gesture) keeps the last ones, so a bubble doesn't chase a lowered hand.
            follow = dec.anchors if fresh and dec.candidate else None
            engine.step(frame, dt, hold, mask=mask, anchors=follow)
            last_t = now
            if cfg.debug:
                draw_debug(frame, obs.hands if fresh else [], dec, state, cfg.min_score)

            if not sink.send(frame):
                break
            stats.frames += 1
            stats.loop_ms.append((time.monotonic() - t_loop) * 1000)
            if mask is not None:
                stats.mask_loop_ms.append(stats.loop_ms[-1])
            if now - stats.last_print >= cfg.stats_every_s:
                log(stats.line())
                if seg is not None:
                    overload = overload_streak(
                        overload, stats.infer_ms, stats.seg_ms, stats.mask_loop_ms
                    )
                    if overload >= SEG_OFF_WINDOWS:
                        log(
                            f"[gestcam] segmentation off for this session: model or masked "
                            f"loop p95 over "
                            f"{SEG_OFF_MS:.0f} ms for {SEG_OFF_WINDOWS} windows (CPU starved)"
                        )
                        seg.close()
                        seg, mask = None, None
                stats.reset_window()
                stats.last_print = now
    finally:
        rec.close()
        if seg is not None:
            seg.close()
        cap.close()
        sink.close()
        if raw is not None:
            raw.close()
            if raw_ts:
                ts_sidecar(cfg.record_raw).write_text(
                    json.dumps([round((t - raw_ts[0]) * 1000, 1) for t in raw_ts])
                )


# ------------------------------------------------------------------ debug overlay
_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
    (17, 18), (18, 19), (19, 20),
)  # fmt: skip


def _text(img, s: str, org, scale: float = 0.5, col=(255, 255, 255)) -> None:
    # A backing box, not a thick black outline: Hershey glyphs advance wider with
    # thickness, so a thickness-3 outline drifts off the thickness-1 text.
    x, y = int(org[0]), int(org[1])
    (tw, th), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (x - 3, y - th - 3), (x + tw + 3, y + base), (20, 20, 20), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def draw_debug(frame, hands: list[Hand], dec: Decision, state: ReactionState, min_score: float) -> None:
    """Landmarks (green = the hands that decided), per-hand pose and canned label,
    the decision, and every hold accumulator against its threshold."""
    h, w = frame.shape[:2]
    used = {id(x) for x in dec.hands}
    for hd in hands:
        pts = np.round(hd.lm * (w, h)).astype(np.int32)
        col = (80, 230, 80) if id(hd) in used else (210, 210, 210)
        cv2.polylines(frame, [pts[list(b)] for b in _BONES], False, col, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, col, -1, cv2.LINE_AA)
        _text(
            frame,
            f"{hd.pose(min_score) or '-'} | {hd.label}:{hd.score:.2f} | "
            f"{hd.handedness[:1]}:{hd.handed_score:.2f}",
            (pts[0][0] - 80, min(pts[0][1] + 24, h - 8)),
        )
    _text(frame, f"candidate {dec.candidate}   last fire {state._last_fired}", (12, 24), 0.6)
    # Reading the state's private accumulators is deliberate: this is its debug view.
    y = 48
    for r, acc in sorted(state._acc.items()):
        if acc <= 0:
            continue
        frac = min(1.0, acc / state._threshold(r))
        cv2.rectangle(frame, (12, y - 12), (212, y), (60, 60, 60), -1)
        cv2.rectangle(frame, (12, y - 12), (12 + int(200 * frac), y), (80, 230, 80), -1)
        _text(frame, f"{r} {acc:.2f}s", (220, y))
        y += 20


# ------------------------------------------------------------------ offline eval
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
STILL_DT_MS = 33  # a still is fed as a 30 fps clip ...
STILL_S = 1.2  # ... this long, so hold 0.7 s + defer 0.15 s + tracking warm-up fits
STILL_WARMUP_S = STILL_S - 0.85  # ... and longer holds get the same warm-up margin


def ts_sidecar(clip) -> pathlib.Path:
    """`me.mp4` → `me.mp4.ts.json`: --record-raw's capture times, ms from the first frame.

    The mp4 is constant-rate at the camera's claimed fps, but the loop writes one frame
    per iteration and Capture drops frames when it falls behind. Below the claimed fps
    (24 fps under load; dim webcams deliver 15 while reporting 30) replaying at the
    container fps ran eval's clock fast: the same 20 frames fired at 15 fps and not at 30.
    """
    return pathlib.Path(f"{clip}.ts.json")


def _clips(path: pathlib.Path, still_s: float = STILL_S):
    """(name, iterator of (ts_ms, bgr)). A video is one clip on its own clock (its
    ts_sidecar if there is one, else the container fps); a directory is one clip per
    image, each image repeated for `still_s`."""

    def video(p):
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        side = ts_sidecar(p)
        stamps = json.loads(side.read_text()) if side.exists() else []
        i, last = 0, -1
        while True:
            ok, f = cap.read()
            if not ok:
                break
            # ponytail: a sidecar shorter than the clip falls back to fps per frame, clamped
            # to increase (VIDEO mode needs it); only a hand-edited sidecar gets there.
            last = max(int(stamps[i] if i < len(stamps) else i * 1000 / fps), last + 1)
            yield last, f
            i += 1
        cap.release()

    def still(p):
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"cannot read {p}")
        for i in range(math.ceil(still_s * 1000 / STILL_DT_MS)):
            yield i * STILL_DT_MS, img

    if path.is_dir():
        for p in sorted(q for q in path.iterdir() if q.suffix.lower() in IMAGE_EXTS):
            yield p.name, still(p)
    elif path.suffix.lower() in IMAGE_EXTS:
        yield path.name, still(path)
    else:
        yield path.name, video(path)


def _hand_rec(hd: Hand, min_score: float) -> dict:
    return {
        "label": hd.label,
        "score": round(hd.score, 3),
        "pose": hd.pose(min_score),
        "hand": hd.handedness,
        "hand_score": round(hd.handed_score, 3),
    }


def evaluate(path: str, cfg: RunConfig, expect: str | None = None, emit=print, as_json: bool = False) -> int:
    """Run recognizer (VIDEO mode) → decide() over a clip or a folder of stills.

    Each clip gets a fresh tracker and state, so one image's hold can't leak into the
    next. Prints a line whenever a frame's reading changes (repeats are suppressed: a
    still repeats 37 times), every fire, and a summary. With `expect`, returns 1 unless
    every clip fired it and nothing else did. Raises ValueError on input that would make
    that check vacuous: an unknown `expect`, no clips, or a clip with no frames.
    """
    from .recognizer import REACTIONS

    if expect is not None and expect not in REACTIONS:
        raise ValueError(f"unknown reaction {expect!r}; one of {', '.join(REACTIONS)}")
    model = ensure_model(cfg.model, log=emit)
    quiet = (lambda *a: None) if as_json else emit
    clips_out = []
    # A still must outlast the hold being tuned, or --hold >= ~1.15 can never fire.
    still_s = max(STILL_S, cfg.hold_s + cfg.defer_s + STILL_WARMUP_S)
    for name, frames in _clips(pathlib.Path(path), still_s):
        rec = GestureRecognizer(
            model, num_hands=cfg.num_hands, log=quiet, detect_conf=cfg.detect_conf, mode="video"
        )
        state = make_state(cfg)
        recs, fires, prev = [], [], None
        quiet(f"== {name}")
        try:
            for ts_ms, bgr in frames:
                h, w = bgr.shape[:2]
                obs = rec.recognize(model_rgb(bgr, cfg.infer_width), ts_ms)
                dec = decide(obs, state, w, h, cfg.min_score)
                hands = [_hand_rec(hd, cfg.min_score) for hd in obs.hands]
                recs.append({"t": obs.ts, "hands": hands, "candidate": dec.candidate, "fired": dec.fired})
                txt = " | ".join(
                    f"{x['hand'][:1]} {x['label']}:{x['score']:.2f} {x['pose'] or '-'}" for x in hands
                ) or "no hands"
                txt += f"  -> {dec.candidate}"
                if txt != prev:
                    quiet(f"  {obs.ts:6.3f}s  {txt}")
                    prev = txt
                if dec.fired:
                    fires.append({"t": obs.ts, "reaction": dec.fired})
                    quiet(f"  {obs.ts:6.3f}s  ▶ FIRE {dec.fired}")
        finally:
            rec.close()
        if not recs:
            raise ValueError(f"{name}: no frames (missing or unreadable file?)")
        clips_out.append({"name": name, "frames": recs, "fires": fires})

    if not clips_out:
        raise ValueError(f"no clips in {path}")
    hit = sum(any(f["reaction"] == expect for f in c["fires"]) for c in clips_out)
    wrong = sum(f["reaction"] != expect for c in clips_out for f in c["fires"]) if expect else 0
    summary = {
        "clips": len(clips_out),
        "clips_fired": sum(bool(c["fires"]) for c in clips_out),
        "fires": {c["name"]: [f["reaction"] for f in c["fires"]] for c in clips_out},
        "expect": expect,
        "expect_hit": hit if expect else None,
        "wrong_fires": wrong if expect else None,
    }
    if as_json:
        emit(json.dumps({"clips": clips_out, "summary": summary}, indent=1))
    else:
        for c in clips_out:
            emit(f"{c['name']:<28} {', '.join(f['reaction'] for f in c['fires']) or '-'}")
        line = f"[eval] {summary['clips']} clips, {summary['clips_fired']} fired"
        if expect:
            line += f" | expect {expect}: {hit}/{len(clips_out)} hit, {wrong} wrong fires"
        emit(line)
    return 1 if expect and (hit < len(clips_out) or wrong) else 0


# ------------------------------------------------------------------ demo (no camera / no model)
# Seconds after each trigger that best show it, for `demo --sheet`.
SHEET_AT = {
    "hearts": 1.0,
    "thumbs_up": 0.8,
    "thumbs_down": 0.8,
    "balloons": 1.8,
    "rain": 1.2,
    "confetti": 0.9,
    "lasers": 1.4,  # earlier still shows the confetti before it
    "fireworks": 1.0,  # two open bursts, before the next launch flash
}


def demo(
    sink,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    seconds: float = 24.0,
    seed: int = 7,
    log=print,
    source=None,
    sheet: str | None = None,
) -> None:
    """Fire every reaction in sequence on a synthetic background (or a real camera if
    `source` is a Capture). Lets you tune the visuals without a model or gestures.
    The synthetic scene carries its own person mask, so back-layer effects go behind.
    `sheet`: also write a 4x2 contact sheet of each reaction at SHEET_AT."""
    from .recognizer import REACTIONS
    from .sprites import load_sprites

    engine = EffectEngine(width, height, sprites=load_sprites(log=log), seed=seed)
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    per = seconds / len(REACTIONS)
    schedule = {int(i * per * fps) + 5: r for i, r in enumerate(REACTIONS)}
    snaps: dict[int, list[str]] = {}  # a list: two snapshots can land on one frame
    for s, r in schedule.items():
        snaps.setdefault(s + round(SHEET_AT[r] * fps), []).append(r)
    tiles: dict[str, np.ndarray] = {}
    seq = -1
    t = 0.0
    dt = 1.0 / fps
    for i in range(n):
        mask = None
        if source is not None:
            got = source.wait_frame(seq)
            frame = got[0].copy() if got else np.zeros((height, width, 3), np.uint8)
            if got:
                seq = got[2]
        else:
            frame, mask = synthetic_frame(width, height, t, rng)
        if i in schedule:
            r = schedule[i]
            anchors = [(width * 0.42, height * 0.6), (width * 0.58, height * 0.6)]
            log(f"[demo] {r}")
            engine.trigger(r, anchors)
        engine.step(frame, dt, mask=mask)
        for r in snaps.get(i, ()):
            tiles[r] = frame.copy()
        cv2.putText(
            frame,
            f"gestcam demo  t={t:5.1f}s",
            (16, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if not sink.send(frame):
            break
        t += dt
        if source is None:
            time.sleep(0)  # file/preview sinks pace themselves; real-time not needed
    sink.close()
    if sheet:
        tw, th = 480, 270
        # A short --seconds or quitting the preview leaves snapshots untaken: blank them.
        missing = [r for r in REACTIONS if r not in tiles]
        if missing:
            log(f"[demo] sheet: not captured (run too short or quit early): {', '.join(missing)}")
        cells = []
        for r in REACTIONS:
            tile = tiles.get(r, np.zeros((th, tw, 3), np.uint8))
            cell = cv2.resize(tile, (tw, th), interpolation=cv2.INTER_AREA)
            _text(cell, r, (10, th - 12), 0.6)
            cells.append(cell)
        grid = np.vstack([np.hstack(cells[:4]), np.hstack(cells[4:])])
        cv2.imwrite(sheet, grid)
        log(f"[demo] sheet → {sheet}")


def synthetic_frame(w: int, h: int, t: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """A moving gradient with a 'person' blob, so effects are judged against something.
    Returns (frame, mask): the blob drawn again as the person mask (255 = person)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = 40 + 60 * (0.5 + 0.5 * np.sin(xx / w * 3 + t * 0.4)) + 20 * (yy / h)
    frame = np.stack([g * 1.1, g * 0.9, g * 0.8], axis=2).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cx, cy = int(w * 0.5 + 20 * np.sin(t * 0.7)), int(h * 0.62)
    for img, body, head in ((frame, (90, 70, 60), (150, 170, 200)), (mask, 255, 255)):
        cv2.ellipse(
            img,
            (cx, cy + int(h * 0.3)),
            (int(w * 0.18), int(h * 0.35)),
            0,
            0,
            360,
            body,
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            img, (cx, cy - int(h * 0.12)), int(h * 0.13), head, -1, cv2.LINE_AA
        )
    return frame, mask
