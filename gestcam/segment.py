"""Person segmentation, so background effects can be drawn *behind* the person.

Same shape as recognizer.GestureRecognizer: MediaPipe Tasks ImageSegmenter in
LIVE_STREAM mode, inference on MediaPipe's thread, the main loop only reads the latest
mask. VIDEO mode is a synchronous twin for offline eval and tests.

Model: Google's selfie_segmenter.tflite (Apache-2.0, 250 KB float16), one confidence
channel = person. It returns the mask at the size it was fed (640x360 here), so
`refine_mask` does the upsample to output size together with the dilate/feather/smooth
the compositor needs.
"""

from __future__ import annotations

import pathlib
import threading
import time

import cv2
import numpy as np

from gestcam import recognizer

# Pinned like the gesture model: `latest` could swap the model DILATE/FEATHER/SMOOTH and
# the latencies below were measured on. v1 == latest today (same md5, 249537 B).
SEG_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/1/selfie_segmenter.tflite"
)

# Mask shaping, as fractions of frame height so 640x360 and 1280x720 look the same.
# Dilation covers the one-frame mask lag: at 30 fps a head turning at a normal pace
# moves ~1% of frame height per frame, so 1.2% keeps the effect off the person's edge.
# Feather hides the 640→1280 bilinear staircase and the model's own edge noise.
DILATE_FRAC = 0.012
FEATHER_FRAC = 0.01
# EMA weight of the NEW mask where the person shrinks away. Growth is never smoothed
# (see refine_mask), so this only sets how fast a vacated area returns to background:
# 0.5 leaves 25% after 2 frames, 6% after 4. Lower = steadier edges, longer ghosts.
SMOOTH_NEW = 0.5


def ensure_segmenter(path: str | None = None, log=print) -> pathlib.Path:
    """Return the path to selfie_segmenter.tflite, downloading it on first run."""
    return recognizer.ensure_model(
        path, log, url=SEG_URL, filename="selfie_segmenter.tflite", min_bytes=100_000
    )


def _odd(n: float) -> int:
    return max(1, int(round(n)) | 1)


def refine_mask(small: np.ndarray, w: int, h: int, prev: np.ndarray | None = None) -> np.ndarray:
    """Model mask (any size, uint8, 255 = person) → compositing mask (h x w uint8).

    Dilate and feather at the SMALL size, then upsample: the bilinear upsample is itself
    a 2x smooth, and doing the morphology on a quarter of the pixels is what keeps this
    under 1 ms at 1280x720 (measured 0.63 p50 / 0.94 p95 ms, 640x360 in, i7-13620H).

    Temporal smoothing is asymmetric on purpose. Where the person GROWS (moves into new
    pixels) we take the new mask immediately, because a lagging mask there paints the
    effect over the person, which is the artifact we're hiding. Where it SHRINKS we EMA
    toward the new mask, which damps edge flicker on a still person (the model's
    jitter is mostly the edge dipping for one frame) and costs only a brief ghost of
    "person" where an effect is hidden a frame or two too long, which nobody notices.
    Measured on a still photo with sensor noise: frame-to-frame flicker 0.44 → 0.25
    (mean |diff|); after an 80 px jump, ghost area 1.8% → 0.6% → 0.3% of the frame over
    frames 1-3, and 0% of the person's new pixels ever uncovered.
    """
    sh = small.shape[0]
    k = _odd(2 * DILATE_FRAC * sh)
    m = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = cv2.GaussianBlur(m, (0, 0), FEATHER_FRAC * sh)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    if prev is None or prev.shape != m.shape:
        return m
    ema = cv2.addWeighted(m, SMOOTH_NEW, prev, 1.0 - SMOOTH_NEW, 0)
    return cv2.max(m, ema)


def _to_u8(result) -> np.ndarray:
    # Confidence in [0,1] float32 → uint8. convertScaleAbs saturates and copies, so the
    # array no longer aliases MediaPipe's buffer once the callback returns.
    conf = result.confidence_masks[0].numpy_view()
    return cv2.convertScaleAbs(np.squeeze(conf), alpha=255.0)


class PersonSegmenter:
    """ImageSegmenter wrapper. mode="live": submit/latest/close. mode="video": segment()."""

    def __init__(self, model_path: str | pathlib.Path, log=print, mode: str = "live"):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if mode not in ("live", "video"):
            raise ValueError(f"mode must be 'live' or 'video', not {mode!r}")
        self._mp = mp
        self._mode = mode
        self._lock = threading.Lock()
        self._latest: tuple[float, np.ndarray] | None = None
        # submit → result latency of `_latest`, ms: the same span as the gesture model's
        # infer_ms, since both p95s go against one budget (SEG_OFF_MS). Capture → result
        # also charged the segmenter for loop work before submit, e.g. --record-raw's
        # 6-10 ms encode. Timed in the callback: the main loop polls once per frame, so
        # timing it there adds up to 33 ms of wait.
        self.latency_ms = 0.0
        self._pending: dict[int, float] = {}  # timestamp_ms → monotonic submit time
        self._t0 = time.monotonic()
        self._last_ts_ms = -1
        self._log = log

        live = mode == "live"
        opts = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.LIVE_STREAM if live else vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
            result_callback=self._on_result if live else None,
        )
        self._seg = vision.ImageSegmenter.create_from_options(opts)

    def _image(self, rgb: np.ndarray):
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    # -- MediaPipe callback (runs on MediaPipe's thread) -------------------------------
    def _on_result(self, result, image, timestamp_ms: int) -> None:
        mask = _to_u8(result)
        sent = self._pending.pop(timestamp_ms, None)
        now = time.monotonic()
        with self._lock:
            self._latest = (self._t0 + timestamp_ms / 1000.0, mask)
            self.latency_ms = (now - sent) * 1000.0 if sent else 0.0

    # -- live mode, called from the main loop --------------------------------------------
    def submit(self, rgb: np.ndarray, capture_ts: float) -> bool:
        """Queue a frame. False if skipped (same-millisecond duplicate: MediaPipe needs
        strictly increasing timestamps and raises otherwise)."""
        ts_ms = int((capture_ts - self._t0) * 1000)
        if ts_ms <= self._last_ts_ms:
            return False
        self._last_ts_ms = ts_ms
        img = self._image(rgb)
        self._pending[ts_ms] = time.monotonic()
        if len(self._pending) > 30:  # MediaPipe dropped frames under load: forget them
            for k in sorted(self._pending)[:-10]:
                self._pending.pop(k, None)
        self._seg.segment_async(img, ts_ms)
        return True

    def latest(self) -> tuple[float, np.ndarray] | None:
        """(capture ts, mask at the submitted size, uint8 255 = person), or None."""
        with self._lock:
            return self._latest

    # -- video mode: synchronous, for offline eval -----------------------------------
    def segment(self, rgb: np.ndarray, ts_ms: int) -> np.ndarray:
        return _to_u8(self._seg.segment_for_video(self._image(rgb), int(ts_ms)))

    def close(self) -> None:
        self._seg.close()
