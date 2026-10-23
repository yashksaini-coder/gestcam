"""Hand gesture recognition → reaction candidates.

Two layers, deliberately separated so the second is unit-testable without a model:

1. `GestureRecognizer`: thin wrapper around MediaPipe Tasks GestureRecognizer. In
   LIVE_STREAM mode inference runs on MediaPipe's own thread; we only ever read the
   *latest* result. The capture/compose loop never blocks on the model. VIDEO mode is
   the same tracker, synchronous, for offline eval.
2. `pick(hands)`: pure function. Each hand gets one pose from `Hand.pose()` (canned
   label fused with landmark geometry); pairs of hands become two-hand reactions.
   Returns one of REACTIONS (or None) *and* the hands that decided it, so effects can
   anchor on the hand that actually fired. `reaction_for` is the reaction-only adapter.

Landmark fixtures from real photos (tests/fixtures/photo_hands.json) pin the rules.
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
THUMB_MCP, THUMB_TIP = 2, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20


def _d(p, q, aspect: float = 1.0) -> float:
    """Distance in units of image height.

    MediaPipe normalizes x by image width and y by image height, so a raw norm over
    landmarks mixes units: at 1280x720 an equal physical span reads 1.78x smaller
    horizontally. Scaling x by aspect (w/h) puts both axes in the same unit.
    """
    return math.hypot((p[0] - q[0]) * aspect, p[1] - q[1])


def cache_dir() -> pathlib.Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    p = pathlib.Path(base) / "gestcam"
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_model(
    path: str | None = None,
    log=print,
    url: str = MODEL_URL,
    filename: str = "gesture_recognizer.task",
    min_bytes: int = 1_000_000,
) -> pathlib.Path:
    """Return the path to a cached model, downloading it on first run.

    min_bytes is the "this is a real model, not a truncated download" floor: the
    gesture bundle is 8.4 MB, the selfie segmenter 250 KB, so each caller sets its own.
    """
    p = pathlib.Path(path) if path else cache_dir() / filename
    if p.exists() and p.stat().st_size > min_bytes:
        return p
    log(f"[gestcam] downloading {filename} → {p}")
    tmp = p.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(p)
    return p


@dataclass
class Hand:
    label: str                 # canned label: Thumb_Up, Thumb_Down, Victory, ILoveYou, Open_Palm, Closed_Fist, Pointing_Up, None
    score: float
    handedness: str            # "Left" / "Right" (as reported for the frame we fed, mirroring included)
    lm: np.ndarray             # (21, 2) normalized x,y in [0,1]
    aspect: float = 1.0        # image width / height; 1.0 keeps square-image tests honest
    handed_score: float = 1.0  # MediaPipe's confidence in `handedness`

    def _d(self, i: int, j: int) -> float:
        return _d(self.lm[i], self.lm[j], self.aspect)

    @property
    def size(self) -> float:
        """Palm size proxy: wrist → middle MCP distance, in units of image height.

        Every threshold in this file is expressed as a multiple of this, so it must
        not depend on how the hand happens to be rotated.
        """
        return self._d(MIDDLE_MCP, WRIST) + 1e-6

    def palm_center(self) -> tuple[float, float]:
        c = self.lm[[WRIST, INDEX_MCP, PINKY_MCP]].mean(axis=0)
        return float(c[0]), float(c[1])

    def finger_extended(self, pip: int, tip: int, margin: float = 0.15) -> bool:
        """Extended when the tip reaches farther from the WRIST than the PIP does.

        Referencing the wrist rather than the MCP gives a long, stable baseline.
        mcp→tip and mcp→pip are short and near-parallel, so any foreshortening (hand
        tilted toward or away from the lens) collapses the long leg and the old ratio
        dived toward 1. Straight: tip is ~+0.7 palm-sizes past the PIP. Curled: ~-0.35
        behind it. margin=0.15 is roughly 62° of PIP flexion — the knob to tune on camera.
        """
        return self._d(tip, WRIST) - self._d(pip, WRIST) > margin * self.size

    def finger_curled(self, pip: int, tip: int, margin: float = 0.05) -> bool:
        """~85° of flexion. The gap to finger_extended is a deliberate dead band:
        a half-flexed finger is neither, so is_horns stays silent instead of guessing."""
        return self._d(tip, WRIST) - self._d(pip, WRIST) < -margin * self.size

    def is_horns(self) -> bool:
        """🤘 (or 🤟): index + pinky out, middle + ring folded. Thumb ignored."""
        return (
            self.finger_extended(INDEX_PIP, INDEX_TIP)
            and self.finger_extended(PINKY_PIP, PINKY_TIP)
            and self.finger_curled(MIDDLE_PIP, MIDDLE_TIP)
            and self.finger_curled(RING_PIP, RING_TIP)
        )

    def is_victory(self) -> bool:
        """✌️: index + middle out, ring + pinky folded, pointing up. Thumb ignored.

        Measured on 22 Commons photos: 6/6 V hands, 0 of the other hands. The canned
        classifier got 2 of those 6 (None:.96, None:.91, Pointing_Up:.64 on textbook V
        signs). "Pointing up" = index MCP→tip within 60° of vertical (real V hands
        measured ≤33° off): one hand of an inverted heart is a V pointing *down*, and
        without this it fired balloons when the palm detector found only that hand.
        """
        vx = (self.lm[INDEX_TIP][0] - self.lm[INDEX_MCP][0]) * self.aspect
        vy = self.lm[INDEX_TIP][1] - self.lm[INDEX_MCP][1]
        return (
            vy < 0 and abs(vx) <= -vy * math.tan(math.radians(60))
            and self.finger_extended(INDEX_PIP, INDEX_TIP)
            and self.finger_extended(MIDDLE_PIP, MIDDLE_TIP)
            and self.finger_curled(RING_PIP, RING_TIP)
            and self.finger_curled(PINKY_PIP, PINKY_TIP)
        )

    def thumb_dir(self, min_len: float = 0.45, beyond: float = 0.25, beyond_down: float = 0.1) -> int:
        """+1 thumbs-up, -1 thumbs-down, 0 neither. Four fingers must be folded.

        - thumb extended: MCP→tip at least min_len palms (real thumbs 0.53-1.86 upright;
          rolled, the real model jitters them to 0.48-0.50 frame to frame).
        - thumb tip is the extreme point of the hand: above every other landmark by
          `beyond` palms (below by `beyond_down`). That is "roughly vertical" without
          an angle, and a resting fist's thumb lies alongside the knuckles, not past them.
          Measured upright: thumbs-up +0.74..+2.43, thumbs-down +0.35..+0.93.
        - Down gets the thinner margin: wrist pronation keeps a thumbs-down short, and
          the curled index hangs right beside it, so roll eats the margin fast. Real
          model, geometry-only td_videocall: +0.35 upright, +0.21 letterboxed 16:9,
          +0.15 rolled 20° — all lost at 0.25. ponytail: 0.1 is not checked against
          real resting fists (none in the photo set; every four-curled hand in 170
          variants was a thumb gesture); Closed_Fist >= conflict_score still vetoes.
          Re-measure on the user's --record-raw clip. Rolled 35° (-0.04) stays lost.
        Lengths are in palm sizes, aspect-corrected, so distance to camera drops out.
        """
        if not all(self.finger_curled(p, t) for p, t in _FINGERS):
            return 0
        if self._d(THUMB_TIP, THUMB_MCP) < min_len * self.size:
            return 0
        y, rest = self.lm[THUMB_TIP][1], self.lm[INDEX_MCP:, 1]
        if rest.min() - y > beyond * self.size:
            return 1
        if y - rest.max() > beyond_down * self.size:
            return -1
        return 0

    def pose(self, min_score: float = 0.5, conflict_score: float = 0.7) -> str | None:
        """This hand's pose, fusing the canned label with landmark geometry.

        The canned top-1 label alone missed 4 of 6 clear V signs on real photos, and
        MediaPipe 0.10 returns only one category (no runner-up to fall back on), so
        geometry gets a vote:
        - horns: geometry only, and it outranks the label. The model has no horns class
          and labelled horns hands Victory; horns (middle folded) and V (middle out)
          are geometrically exclusive, so the label is simply wrong there.
        - canned label mapped to a pose at >= min_score: trusted.
        - otherwise geometry, unless the model is confident (>= conflict_score) the hand
          is some *other* named class (Open_Palm, Closed_Fist, ...). "None" never
          conflicts: the model says None:.96 on V signs. Nor does Pointing_Up against
          V, for the horns reason: pointing has the middle folded, V has it out. The
          model reads real V signs Pointing_Up .64-.72 (v_adolescent rolled 20°: .70-.71
          on every frame, so a .7 veto meant it never fired; mirrored v_girl flickered).
        """
        if self.is_horns():
            return "horns"
        canned = _CANNED_POSE.get(self.label)
        if canned and self.score >= min_score:
            return canned
        geo = "victory" if self.is_victory() else _THUMB_POSE.get(self.thumb_dir())
        if geo is None or (geo == "victory" and self.label == "Pointing_Up"):
            return geo
        if self.label not in ("None", "") and _CANNED_POSE.get(self.label) != geo and self.score >= conflict_score:
            return None
        return geo


_FINGERS = ((INDEX_PIP, INDEX_TIP), (MIDDLE_PIP, MIDDLE_TIP), (RING_PIP, RING_TIP), (PINKY_PIP, PINKY_TIP))
_CANNED_POSE = {"Thumb_Up": "thumb_up", "Thumb_Down": "thumb_down", "Victory": "victory"}
_THUMB_POSE = {1: "thumb_up", -1: "thumb_down"}


@dataclass
class Observation:
    ts: float                          # time.monotonic() when the frame was captured
    hands: list[Hand] = field(default_factory=list)
    infer_ms: float = 0.0              # model latency for this observation


# Handedness below this is a coin flip (MediaPipe is ~0.9+ on a clear hand); trust
# "same hand twice" only above it, else fall back to landmark distance alone.
HANDED_CONF = 0.8


def _opposite_or_unsure(a: Hand, b: Hand) -> bool:
    """False only when MediaPipe is confident both are the same hand (Left+Left).

    --mirror flips both labels, so "they differ" survives mirroring.
    """
    if min(a.handed_score, b.handed_score) < HANDED_CONF:
        return True
    return a.handedness != b.handedness


def _mean_lm_dist(a: Hand, b: Hand) -> float:
    """Mean distance over all 21 landmarks, in palm sizes.

    A duplicate box for one hand lands every landmark on top of its twin; two real
    hands held together (thumbs side by side, wrists 0.5 palm apart) still differ by
    a palm or more somewhere. The old wrist-only floor couldn't tell these apart.
    """
    d = (a.lm - b.lm) * np.array([a.aspect, 1.0], np.float32)
    return float(np.hypot(d[:, 0], d[:, 1]).mean()) / max(a.size, b.size)


def _palm_gap(a: Hand, b: Hand) -> float:
    """Palm-centre distance in units of the larger palm."""
    return _d(a.palm_center(), b.palm_center(), a.aspect) / max(a.size, b.size)


def is_heart(a: Hand, b: Hand, tol: float = 0.55) -> bool:
    """Two-hand heart: index tips touching, thumb tips touching, the two junctions at
    least a third of a palm apart vertically, from opposite hands.

    Both orientations count. Apple's reviewers describe thumbs at the bottom, but both
    real hearts MediaPipe detected in the photo set were the inverted one (thumbs on
    top, fingers pointing down, vy -0.92 / -0.68), and nobody can tell them apart on a
    call. What differs is how to reject the open-palm "frame" pose (probe P3), which
    puts index and thumb tips together just the same:
    - thumbs at the bottom: middle/ring/pinky must not be extended. They curl round
      into the lobes; in the frame pose they stand straight up.
    - thumbs on top: the fingers point down (index tips below both wrists). The
      inverted heart's fingers ARE straight (measured M+.68 R+.56 P+.27), so the curl
      test would reject every real one; pointing down is what the frame pose never does.

    Tip tolerances are in palm sizes, so arm's length and 30 cm normalise identically.
    """
    if not _opposite_or_unsure(a, b):
        return False
    size = 0.5 * (a.size + b.size)
    if _d(a.lm[INDEX_TIP], b.lm[INDEX_TIP], a.aspect) / size > tol:
        return False
    if _d(a.lm[THUMB_TIP], b.lm[THUMB_TIP], a.aspect) / size > tol * 1.3:
        return False
    index_y = 0.5 * (a.lm[INDEX_TIP][1] + b.lm[INDEX_TIP][1])
    thumb_y = 0.5 * (a.lm[THUMB_TIP][1] + b.lm[THUMB_TIP][1])
    if thumb_y - index_y > 0.30 * size:  # classic: image y grows downward
        return not any(
            h.finger_extended(p, t) for h in (a, b) for p, t in _FINGERS[1:]
        )
    if index_y - thumb_y > 0.30 * size:  # inverted
        return index_y > max(a.lm[WRIST][1], b.lm[WRIST][1])
    return False


# Same pose on both hands → the "big" scene effect (Apple's pattern).
PAIR = {"thumb_up": "fireworks", "thumb_down": "rain", "victory": "confetti", "horns": "lasers"}
SINGLE = {"thumb_up": "thumbs_up", "thumb_down": "thumbs_down", "victory": "balloons"}


def pairable(a: Hand, b: Hand, dup: float = 0.3, far: float = 6.0) -> bool:
    """Could these two detections be one person's two hands?

    - dup 0.3: mean landmark distance under 0.3 palm = one hand detected twice.
    - handedness must differ when MediaPipe is confident about both.
    - far 6.0: palm centres more than 6 palms apart = probably two people's single
      hands. Measured on real photos: one person's spread arms ✌️✌️ sit 4.5 palms
      apart, two people's 🤘 side by side 4.4, so no distance separates them;
      6 keeps every one-person pair and only rejects people at opposite edges.
      ponytail: distance-only person split; a person mask/face count is the upgrade.
    """
    return _mean_lm_dist(a, b) >= dup and _opposite_or_unsure(a, b) and _palm_gap(a, b) <= far


def pick(hands: list[Hand], min_score: float = 0.5) -> tuple[str | None, list[Hand]]:
    """Map the hands seen in one frame to (reaction candidate, the hands that decided it).

    Precedence: two-hand reactions > single-hand. Every pair of detected hands is
    tried, closest first, so a third hand (another person) can't hide a real pair.
    Within a pair, heart geometry decides first (the model has no heart class and
    labelled heart hands Thumb_Down), then the same pose() on both hands. pose()
    fuses label and geometry, so one hand at Thumb_Up:.49 no longer downgrades
    fireworks to thumbs_up.

    The second element is what effects anchor on. Only this function knows which hand
    matched; returning every detected hand instead would pin a sprite over whichever
    hand MediaPipe happened to list first.

    ponytail: two-hand recall at the live 16:9 framing is unmeasured. The 22-photo eval
    runs mostly portrait photos; letterboxed into 16:9 (hands smaller in the palm
    detector's square input), tu_two_b finds 1 of 2 hands (→ thumbs_up via the
    one-hand fallback below) and v_double 0 of 2, at infer width 640, 960 and 1280
    alike. A second detection pass on the centre-square crop recovered both (and
    tu_two_a) on those letterboxes, but costs a second 17-24 ms inference whenever
    fewer than 2 hands are tracked, and a letterbox overstates the shrink. Decide on
    the user's --record-raw clip; tests/fixtures/photo_variants.json is the regression set.
    """
    if not hands:
        return None, []
    poses = [h.pose(min_score) for h in hands]
    pairs = sorted(
        ((i, j) for i in range(len(hands)) for j in range(i + 1, len(hands))),
        key=lambda ij: _palm_gap(hands[ij[0]], hands[ij[1]]),
    )
    for i, j in pairs:
        a, b = hands[i], hands[j]
        if not pairable(a, b):
            continue
        if is_heart(a, b):
            return "hearts", [a, b]
        if poses[i] == poses[j] and poses[i] in PAIR:
            return PAIR[poses[i]], [a, b]

    # Rank by "the model agrees" first, then its score. Hand.score is the score of the
    # top canned class, and for a resting hand that's "None" at 0.9+ — ranking on it
    # alone would let a geometry-only hand outrank a deliberate Thumb_Up@0.7.
    cands = [(h, p) for h, p in zip(hands, poses, strict=True) if p in SINGLE]
    if not cands:
        return None, []
    h, p = max(cands, key=lambda hp: (_CANNED_POSE.get(hp[0].label) == hp[1], hp[0].score))
    return SINGLE[p], [h]


def reaction_for(hands: list[Hand], min_score: float = 0.5) -> str | None:
    """Just the reaction. Kept so the pure-rule tests need no plumbing."""
    return pick(hands, min_score)[0]


def hands_from_result(result, width: int, height: int) -> list[Hand]:
    """MediaPipe GestureRecognizerResult → Hand list (top-1 canned label, x,y only)."""
    hands: list[Hand] = []
    for i, lms in enumerate(result.hand_landmarks or []):
        gestures = result.gestures[i] if i < len(result.gestures) and result.gestures[i] else []
        top = gestures[0] if gestures else None
        handed = result.handedness[i][0] if i < len(result.handedness) and result.handedness[i] else None
        hands.append(Hand(
            label=(top.category_name if top else "None") or "None",
            score=float(top.score) if top else 0.0,
            handedness=handed.category_name if handed else "?",
            lm=np.array([[l.x, l.y] for l in lms], dtype=np.float32),
            aspect=width / height,
            handed_score=float(handed.score) if handed else 0.0,
        ))
    return hands


class GestureRecognizer:
    """MediaPipe GestureRecognizer.

    mode="live": LIVE_STREAM, submit()/latest(); the main loop never blocks on it.
    mode="video": VIDEO, synchronous recognize(rgb, ts_ms) for offline eval. Same
    tracker as live (hands found earlier stay tracked), unlike IMAGE mode.
    """

    def __init__(
        self,
        model_path: str | pathlib.Path,
        num_hands: int = 2,
        log=print,
        detect_conf: float = 0.5,
        track_conf: float = 0.5,
        mode: str = "live",
    ):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if mode not in ("live", "video"):
            raise ValueError(f"mode must be 'live' or 'video', not {mode!r}")
        self._mp = mp
        self._lock = threading.Lock()
        self._latest: Observation | None = None
        self._pending: dict[int, float] = {}    # timestamp_ms → monotonic send time
        self._t0 = time.monotonic()
        self._last_ts_ms = -1
        self._log = log

        live = mode == "live"
        opts = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.LIVE_STREAM if live else vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=detect_conf,
            min_hand_presence_confidence=track_conf,
            min_tracking_confidence=track_conf,
            result_callback=self._on_result if live else None,
        )
        self._rec = vision.GestureRecognizer.create_from_options(opts)

    # -- MediaPipe callback (runs on MediaPipe's thread) -------------------------------
    def _on_result(self, result, image, timestamp_ms: int) -> None:
        sent = self._pending.pop(timestamp_ms, None)
        now = time.monotonic()
        obs = Observation(
            ts=self._t0 + timestamp_ms / 1000.0,
            hands=hands_from_result(result, image.width, image.height),
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

    def recognize(self, rgb: np.ndarray, ts_ms: int) -> Observation:
        """mode="video" only: run one frame synchronously. ts_ms must strictly increase.

        Observation.ts is ts_ms in seconds (the clip's clock, not monotonic()).
        """
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        t = time.monotonic()
        result = self._rec.recognize_for_video(img, int(ts_ms))
        return Observation(
            ts=ts_ms / 1000.0,
            hands=hands_from_result(result, img.width, img.height),
            infer_ms=(time.monotonic() - t) * 1000.0,
        )

    def latest(self) -> Observation | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._rec.close()
