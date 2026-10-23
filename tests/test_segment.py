import os
import pathlib
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gestcam import recognizer, segment
from gestcam.segment import refine_mask

W, H = 1280, 720


def disc(r=60, w=640, h=360, cx=320, cy=180):
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def test_refine_mask_shape_dtype_range():
    out = refine_mask(disc(), W, H)
    assert out.shape == (H, W) and out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255
    # Odd sizes and a mismatched prev (e.g. after resize) must not crash or leak in.
    out2 = refine_mask(disc(w=257, h=145, cx=128, cy=72, r=30), 333, 187, prev=np.full((10, 10), 255, np.uint8))
    assert out2.shape == (187, 333) and out2[0, 0] == 0


def test_dilation_grows_person_region():
    small = disc()
    plain = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    grown = refine_mask(small, W, H)
    assert (grown >= 128).sum() > 1.05 * (plain >= 128).sum()
    # Interior stays solid: feathering only touches the edge band.
    assert grown[H // 2, W // 2] == 255


def test_growth_is_immediate_shrink_converges():
    a, b = disc(cx=200), disc(cx=440)  # person jumps sideways
    prev = refine_mask(a, W, H)
    target = refine_mask(b, W, H)
    out = refine_mask(b, W, H, prev)
    # New person pixels covered on the very first frame (no effect painted over them).
    assert out[target >= 128].min() >= 128
    ghost = [(out >= 128).sum() - (target >= 128).sum()]
    for _ in range(4):
        out = refine_mask(b, W, H, out)
        ghost.append((out >= 128).sum() - (target >= 128).sum())
    assert ghost[0] > 0 and ghost[1] < ghost[0] and ghost[-1] == 0
    # Then it settles exactly on the target and stays there: a still person is rock steady.
    for _ in range(8):
        out = refine_mask(b, W, H, out)
    assert cv2.absdiff(out, target).max() <= 1
    assert np.array_equal(refine_mask(b, W, H, out), out)


def test_ensure_segmenter_reuses_ensure_model(monkeypatch):
    seen = {}

    def fake(path=None, log=print, **kw):
        seen.update(kw, path=path)
        return pathlib.Path("/x/selfie_segmenter.tflite")

    monkeypatch.setattr(recognizer, "ensure_model", fake)
    assert segment.ensure_segmenter().name == "selfie_segmenter.tflite"
    assert seen == {"url": segment.SEG_URL, "filename": "selfie_segmenter.tflite", "min_bytes": 100_000, "path": None}


def test_confidence_to_uint8_saturates_and_copies():
    conf = np.array([[[0.0], [0.5]], [[1.0], [1.2]]], np.float32)  # HxWx1, as MediaPipe gives
    res = SimpleNamespace(confidence_masks=[SimpleNamespace(numpy_view=lambda: conf)])
    out = segment._to_u8(res)
    assert out.dtype == np.uint8 and out.shape == (2, 2)
    assert out.tolist() == [[0, 128], [255, 255]]
    assert not np.shares_memory(out, conf)


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        segment.PersonSegmenter("unused.tflite", mode="image")


MODEL = recognizer.cache_dir() / "selfie_segmenter.tflite"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="selfie_segmenter.tflite not cached")


@needs_model
def test_live_mode_latest_and_duplicate_skip():
    seg = segment.PersonSegmenter(MODEL)
    try:
        frame = np.full((360, 640, 3), 128, np.uint8)
        t = time.monotonic()
        assert seg.submit(frame, t)
        assert not seg.submit(frame, t)  # same millisecond: skipped, not raised
        deadline = time.monotonic() + 5
        while seg.latest() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        ts, mask = seg.latest()
        assert abs(ts - t) < 0.002 and mask.shape == (360, 640) and mask.dtype == np.uint8
    finally:
        seg.close()


@needs_model
def test_latency_is_submit_to_result():
    """Same span as the gesture model's infer_ms: loop work between capture and submit
    (e.g. --record-raw's encode) must not count against the segmenter's budget."""
    seg = segment.PersonSegmenter(MODEL)
    try:
        time.sleep(0.6)
        assert seg.submit(np.full((360, 640, 3), 128, np.uint8), seg._t0 + 0.01)  # captured 0.59 s ago
        deadline = time.monotonic() + 5
        while seg.latest() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert 0 < seg.latency_ms < 300
    finally:
        seg.close()


def test_segmenter_url_is_pinned():
    # `latest` would silently swap the model the DILATE/FEATHER/SMOOTH tuning was measured on.
    assert "/float16/1/" in segment.SEG_URL


PHOTO = os.environ.get("GESTCAM_TEST_PHOTO")  # a webcam-style portrait, person centred


@needs_model
@pytest.mark.skipif(not (PHOTO and os.path.exists(PHOTO)), reason="set GESTCAM_TEST_PHOTO to a portrait photo")
def test_real_photo_person_is_255():
    bgr = cv2.resize(cv2.imread(PHOTO), (640, 360), interpolation=cv2.INTER_AREA)
    seg = segment.PersonSegmenter(MODEL, mode="video")
    try:
        m = seg.segment(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 0)
    finally:
        seg.close()
    centre = m[120:240, 240:400]
    corners = np.concatenate([m[:20, :20].ravel(), m[:20, -20:].ravel()])
    assert (centre >= 128).mean() > 0.9 and np.median(centre) == 255
    assert corners.mean() < 32


def test_replace_background_keeps_person_swaps_rest():
    import numpy as np

    from gestcam.segment import replace_background

    frame = np.full((72, 128, 3), 200, np.uint8)
    small = np.zeros((36, 64), np.uint8)
    small[:, :32] = 255  # left half is the person
    bg = np.full_like(frame, 40)
    replace_background(frame, small, bg)
    assert frame[36, 10].tolist() == [200] * 3  # person untouched
    assert frame[36, 120].tolist() == [40] * 3  # room replaced
