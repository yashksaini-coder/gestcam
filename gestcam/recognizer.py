"""Hand gesture recognition → reaction candidates.

Two layers, deliberately separated so the second is unit-testable without a model:

1. `GestureRecognizer`: thin wrapper around MediaPipe Tasks GestureRecognizer in
   LIVE_STREAM mode. Inference runs on MediaPipe's own thread; we only ever read the
   *latest* result. The capture/compose loop never blocks on the model.
2. `reaction_for(hands)`: pure function. Takes per-hand canned labels + landmarks,
   returns one of REACTIONS or None. Two-hand reactions win over single-hand ones.
"""

from __future__ import annotations

import math
import os
import pathlib
import threading
import time
import urllib.request
from dataclasses import dataclass, field

import numpy as np

# Apple's eight reactions, same gestures where the canned MediaPipe model has them.
REACTIONS = (
    "hearts",       # two-hand heart
    "thumbs_up",    # 👍
    "thumbs_down",  # 👎
    "balloons",     # ✌️
    "rain",         # 👎👎
    "confetti",     # ✌️✌️
    "lasers",       # 🤘🤘
    "fireworks",    # 👍👍
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20


def cache_dir() -> pathlib.Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    p = pathlib.Path(base) / "gestcam"
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_model(path: str | None = None, log=print) -> pathlib.Path:
    """Return the path to gesture_recognizer.task, downloading it on first run."""
    p = pathlib.Path(path) if path else cache_dir() / "gesture_recognizer.task"
    if p.exists() and p.stat().st_size > 1_000_000:
        return p
    log(f"[gestcam] downloading gesture model → {p}")
    tmp = p.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.replace(p)
    return p


@dataclass
class Hand:
    label: str                 # canned label: Thumb_Up, Thumb_Down, Victory, ILoveYou, Open_Palm, Closed_Fist, Pointing_Up, None
    score: float
    handedness: str            # "Left" / "Right" (as reported for the frame we fed, mirroring included)
    lm: np.ndarray             # (21, 2) normalized x,y in [0,1]

    @property
    def size(self) -> float:
        """Palm size proxy: wrist → middle MCP distance (normalized units)."""
        return float(np.linalg.norm(self.lm[MIDDLE_MCP] - self.lm[WRIST])) + 1e-6

    def palm_center(self) -> tuple[float, float]:
        c = self.lm[[WRIST, INDEX_MCP, PINKY_MCP]].mean(axis=0)
        return float(c[0]), float(c[1])

    def finger_extended(self, mcp: int, pip: int, tip: int, ratio: float = 2.0) -> bool:
        """A finger is extended when its tip is much farther from its MCP than its PIP is.

        Straight finger: |tip-mcp| ≈ 3 segments, |pip-mcp| = 1 segment → ratio ≈ 3.
        Curled finger: the tip folds back toward the palm → ratio < ~1.5.
        Rotation/scale invariant, no angles needed.
        """
        d_tip = np.linalg.norm(self.lm[tip] - self.lm[mcp])
        d_pip = np.linalg.norm(self.lm[pip] - self.lm[mcp]) + 1e-6
        return bool(d_tip / d_pip > ratio)

    def is_horns(self) -> bool:
        """🤘 (or 🤟): index + pinky out, middle + ring folded. Thumb ignored."""
        return (
            self.finger_extended(INDEX_MCP, INDEX_PIP, INDEX_TIP)
            and self.finger_extended(PINKY_MCP, PINKY_PIP, PINKY_TIP)
            and not self.finger_extended(MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP)
            and not self.finger_extended(RING_MCP, RING_PIP, RING_TIP)
        )


@dataclass
class Observation:
    ts: float                          # time.monotonic() when the frame was captured
    hands: list[Hand] = field(default_factory=list)
    infer_ms: float = 0.0              # model latency for this observation


def is_heart(a: Hand, b: Hand, tol: float = 0.55) -> bool:
    """Two-hand heart: index tips touching, thumb tips touching, index tips above thumbs."""
    size = 0.5 * (a.size + b.size)
    idx = np.linalg.norm(a.lm[INDEX_TIP] - b.lm[INDEX_TIP]) / size
    thb = np.linalg.norm(a.lm[THUMB_TIP] - b.lm[THUMB_TIP]) / size
    if idx > tol or thb > tol * 1.3:
        return False
    index_y = 0.5 * (a.lm[INDEX_TIP][1] + b.lm[INDEX_TIP][1])
    thumb_y = 0.5 * (a.lm[THUMB_TIP][1] + b.lm[THUMB_TIP][1])
    return index_y < thumb_y  # image y grows downward → index tips visually above thumbs


def reaction_for(hands: list[Hand], min_score: float = 0.5) -> str | None:
    """Map the hands seen in one frame to a reaction candidate (or None).

    Precedence: two-hand reactions > single-hand. Hands whose canned label is below
    `min_score` are treated as "None" for canned rules but still usable for the
    landmark rules (heart, horns), which don't depend on the classifier.
    """
    if not hands:
        return None

    def lab(h: Hand) -> str:
        return h.label if h.score >= min_score else "None"

    if len(hands) >= 2:
        a, b = hands[0], hands[1]
        if is_heart(a, b):
            return "hearts"
        la, lb = lab(a), lab(b)
        if la == lb == "Thumb_Up":
            return "fireworks"
        if la == lb == "Thumb_Down":
            return "rain"
        if la == lb == "Victory":
            return "confetti"
        if a.is_horns() and b.is_horns():
            return "lasers"

    best = max(hands, key=lambda h: h.score)
    single = {"Thumb_Up": "thumbs_up", "Thumb_Down": "thumbs_down", "Victory": "balloons"}
    return single.get(lab(best))


class GestureRecognizer:
    """MediaPipe GestureRecognizer in LIVE_STREAM mode, exposing only the latest result."""

    def __init__(self, model_path: str | pathlib.Path, num_hands: int = 2, log=print):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._lock = threading.Lock()
        self._latest: Observation | None = None
        self._pending: dict[int, float] = {}    # timestamp_ms → monotonic send time
        self._t0 = time.monotonic()
        self._last_ts_ms = -1
        self._log = log

        opts = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._on_result,
        )
        self._rec = vision.GestureRecognizer.create_from_options(opts)

    # -- MediaPipe callback (runs on MediaPipe's thread) -------------------------------
    def _on_result(self, result, image, timestamp_ms: int) -> None:
        sent = self._pending.pop(timestamp_ms, None)
        now = time.monotonic()
        hands: list[Hand] = []
        for i, lms in enumerate(result.hand_landmarks or []):
            gestures = result.gestures[i] if i < len(result.gestures) and result.gestures[i] else []
            top = gestures[0] if gestures else None
            handed = result.handedness[i][0].category_name if i < len(result.handedness) and result.handedness[i] else "?"
            arr = np.array([[l.x, l.y] for l in lms], dtype=np.float32)
            hands.append(Hand(
                label=(top.category_name if top else "None") or "None",
                score=float(top.score) if top else 0.0,
                handedness=handed,
                lm=arr,
            ))
        obs = Observation(
            ts=self._t0 + timestamp_ms / 1000.0,
            hands=hands,
            infer_ms=(now - sent) * 1000.0 if sent else 0.0,
        )
        with self._lock:
            self._latest = obs

    # -- called from the main loop --------------------------------------------------
    def submit(self, rgb: np.ndarray, capture_ts: float) -> bool:
        """Queue a frame for inference. Returns False if the frame was skipped.

        MediaPipe requires strictly increasing timestamps; if two frames land in the
        same millisecond we skip the second one instead of raising.
        """
        ts_ms = int((capture_ts - self._t0) * 1000)
        if ts_ms <= self._last_ts_ms:
            return False
        self._last_ts_ms = ts_ms
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        self._pending[ts_ms] = time.monotonic()
        # Drop stale bookkeeping if MediaPipe silently dropped frames under load.
        if len(self._pending) > 30:
            for k in sorted(self._pending)[:-10]:
                self._pending.pop(k, None)
        self._rec.recognize_async(img, ts_ms)
        return True

    def latest(self) -> Observation | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._rec.close()
