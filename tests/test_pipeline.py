import json
import time

import cv2
import numpy as np
import pytest

from gestcam import cli, pipeline, recognizer, segment, streams
from gestcam.pipeline import Decision, RunConfig, draw_debug, model_rgb, overload_streak
from gestcam.recognizer import Hand, Observation
from gestcam.state import ReactionState

W, H = 320, 180


def thumb(handedness="Right", dx=0.0) -> Hand:
    # The canned label carries the pose; the landmarks only need a palm size (0.1,
    # wrist → middle MCP) for pairing, and are geometry-neutral otherwise (no horns).
    lm = np.full((21, 2), 0.55, np.float32)
    lm[0], lm[9] = (0.5, 0.6), (0.5, 0.5)
    return Hand("Thumb_Up", 0.9, handedness, lm + np.float32([dx, 0]))


class FakeCap:
    """30 fps on a virtual clock that starts at now, so obs stay 'fresh'."""

    real_time = False

    def __init__(self, *a, **k):
        self.width, self.height, self.fps, self.fourcc = W, H, 30.0, "MJPG"
        self.t0 = time.monotonic()

    def wait_frame(self, seq, timeout=1.0):
        seq += 1
        if self.real_time:
            time.sleep(max(0.0, self.t0 + seq / 30 - time.monotonic()))
        return np.zeros((H, W, 3), np.uint8), self.t0 + seq / 30, seq

    def close(self):
        pass


class Sink:
    def __init__(self, n):
        self.n, self.frames = n, 0

    def send(self, frame):
        self.frames += 1
        return self.frames < self.n

    def close(self):
        pass


class FakeSeg:
    """Scripted PersonSegmenter: a full-person mask stamped with the last submitted
    ts (or the first one, if `freeze`), at a fixed latency."""

    def __init__(self, latency_ms=5.0, freeze=False):
        self.latency_ms, self.freeze = latency_ms, freeze
        self.ts, self.submits, self.closed = None, 0, 0

    def submit(self, rgb, ts):
        self.submits += 1
        if not (self.freeze and self.ts is not None):
            self.ts = ts
        return True

    def latest(self):
        return None if self.ts is None else (self.ts, np.full((H // 2, W // 2), 255, np.uint8))

    def close(self):
        self.closed += 1


def run_fake(monkeypatch, latest, n=40, real_time=False, seg=None, logs=None, masks=None,
             step_sleep=0.0, **cfg):
    """Drive the real run() loop with a fake camera and a scripted recognizer (and
    segmenter, if `seg`). `masks` collects the mask engine.step got on each frame."""

    class FakeRec:
        def __init__(self, *a, **k):
            self.ts = None

        def submit(self, rgb, ts):
            self.ts = ts
            return True

        def latest(self):
            return latest(self.ts)

        def close(self):
            pass

    class Engine(pipeline.EffectEngine):
        def step(self, frame, dt, hold=None, mask=None, anchors=None):
            if masks is not None:
                masks.append(mask)
            if mask is not None and step_sleep:
                time.sleep(step_sleep)  # a masked composite that blows the frame budget
            return super().step(frame, dt, hold, mask=mask, anchors=anchors)

    monkeypatch.setattr(pipeline, "ensure_model", lambda *a, **k: "model")
    monkeypatch.setattr(pipeline, "GestureRecognizer", FakeRec)
    monkeypatch.setattr(pipeline, "EffectEngine", Engine)
    monkeypatch.setattr(streams, "Capture", FakeCap)
    monkeypatch.setattr(FakeCap, "real_time", real_time)
    if seg is not None:
        monkeypatch.setattr(segment, "PersonSegmenter", lambda *a, **k: seg)
        monkeypatch.setattr(segment, "ensure_segmenter", lambda *a, **k: "seg")
    logs = [] if logs is None else logs
    base = dict(segment=seg is not None, stats_every_s=1e9)
    base.update(cfg)
    pipeline.run(RunConfig(**base), lambda w, h, f: Sink(n), log=logs.append)
    return [line for line in logs if "▶" in line]


def test_run_feeds_state_once_per_observation(monkeypatch):
    """T6: one observation is one piece of evidence. Re-reading the same result every
    frame used to advance the hold on wall-clock time while it was < stale_s old, so
    a 0.3 s hold fired off a single frame. Real time (~0.5 s) so wall time does pass."""
    frozen = {}

    def same(ts):
        frozen.setdefault("obs", Observation(ts, [thumb()]))
        return frozen["obs"]

    assert run_fake(monkeypatch, same, n=15, real_time=True, hold_s=0.3) == []


def test_run_fires_on_observation_clock(monkeypatch):
    fires = run_fake(monkeypatch, lambda ts: Observation(ts, [thumb()]))
    assert len(fires) == 1 and "thumbs_up" in fires[0]


def test_decide_passes_hand_count_and_anchors():
    st = ReactionState()
    a, b = thumb("Right"), thumb("Left", dx=0.15)  # 1.5 palms apart: a real pair
    d = pipeline.decide(Observation(0.0, [a, b]), st, W, H, 0.5)
    assert d.candidate == "fireworks" and len(d.anchors) == 2
    assert st._two_seen == 0.0  # state saw n_hands >= 2 at the observation ts
    cx, cy = a.palm_center()
    assert d.anchors[0] == pytest.approx((cx * W, cy * H))


def test_overload_streak_needs_consecutive_windows():
    ok, slow = [10.0] * 20, [40.0] * 20
    assert overload_streak(0, ok, ok) == 0
    assert overload_streak(0, slow, ok) == 1
    assert overload_streak(1, ok, slow) == 2      # either model counts
    assert overload_streak(1, ok, ok) == 0        # a good window resets the streak
    assert overload_streak(0, [], []) == 0        # no data is not overload


def test_model_rgb_downscales_and_converts():
    f = np.zeros((720, 1280, 3), np.uint8)
    f[..., 0] = 255  # blue in BGR
    out = model_rgb(f, 640)
    assert out.shape == (360, 640, 3) and out[0, 0, 2] == 255
    assert model_rgb(f[:180, :320], 640).shape == (180, 320, 3)  # never upscales


def test_draw_debug_draws_hands_and_bars():
    frame = np.zeros((H, W, 3), np.uint8)
    st = ReactionState()
    hd = thumb()
    hd.lm = np.random.default_rng(0).uniform(0.2, 0.8, (21, 2)).astype(np.float32)
    st.update("thumbs_up", 0.0)
    st.update("thumbs_up", 0.3)
    draw_debug(frame, [hd], Decision("thumbs_up", [hd]), st, 0.5)
    assert (frame[:, :, 1] > 200).sum() > 100   # green landmarks of the deciding hand
    assert frame[40:48, 12:60, 1].max() > 200   # thumbs_up accumulator bar


def test_synthetic_frame_mask_matches_person():
    frame, mask = pipeline.synthetic_frame(W, H, 0.0, None)
    assert mask.shape == (H, W) and mask[int(H * 0.5), W // 2] == 255 and mask[5, 5] == 0


def test_demo_sheet(tmp_path):
    out = tmp_path / "sheet.png"
    pipeline.demo(Sink(10**6), 320, 180, fps=10, sheet=str(out), log=lambda *a: None)
    img = cv2.imread(str(out))
    assert img.shape == (540, 1920, 3)


model_cached = (recognizer.cache_dir() / "gesture_recognizer.task").exists()


@pytest.mark.skipif(not model_cached, reason="gesture model not cached")
def test_evaluate_stills_no_hands(tmp_path, capsys):
    cv2.imwrite(str(tmp_path / "blank.png"), np.full((360, 640, 3), 128, np.uint8))
    rc = pipeline.evaluate(str(tmp_path), RunConfig(), expect="hearts", as_json=True)
    out = json.loads(capsys.readouterr().out)
    clip = out["clips"][0]
    assert rc == 1 and clip["fires"] == []        # expected hearts, got nothing
    assert len(clip["frames"]) == 37 and clip["frames"][-1]["t"] == pytest.approx(1.188)
    assert out["summary"]["expect_hit"] == 0


# ------------------------------------------------------------------ run(): segmentation
held = lambda ts: Observation(ts, [thumb()])  # noqa: E731
idle = lambda ts: Observation(ts, [])  # noqa: E731


def test_seg_idle_runs_no_model_and_passes_no_mask(monkeypatch):
    """Nothing held and nothing on screen: the mask has no reader, so the segmenter
    must not run (it cost ~0.5 core idle and fed the auto-off)."""
    fake, masks = FakeSeg(), []
    run_fake(monkeypatch, idle, n=20, seg=fake, masks=masks)
    assert fake.submits == 0 and all(m is None for m in masks)


def test_seg_runs_while_holding(monkeypatch):
    fake, masks = FakeSeg(), []
    run_fake(monkeypatch, held, n=20, seg=fake, masks=masks, hold_s=10)
    assert fake.submits > 10
    assert masks[-1] is not None and masks[-1].shape == (H, W)


def test_seg_stale_mask_is_dropped(monkeypatch):
    """A mask whose ts stops advancing is passed on until it is stale_s old, then None:
    an old silhouette misplaces effects worse than none."""
    fake, masks = FakeSeg(freeze=True), []
    run_fake(monkeypatch, held, n=15, real_time=True, seg=fake, masks=masks, hold_s=10, stale_s=0.1)
    assert any(m is not None for m in masks[:6])
    assert all(m is None for m in masks[-5:])


def test_seg_auto_off_on_slow_model(monkeypatch):
    fake, masks, logs = FakeSeg(latency_ms=50.0), [], []
    run_fake(monkeypatch, held, n=20, seg=fake, masks=masks, logs=logs, hold_s=10, stats_every_s=0)
    assert sum("segmentation off" in line for line in logs) == 1
    assert fake.closed == 1                      # closed once, not again in finally
    assert masks[-1] is None and fake.submits < 10


def test_seg_auto_off_on_slow_masked_loop(monkeypatch):
    """Models within budget, but the masked composite takes the loop past 33 ms."""
    fake, logs = FakeSeg(latency_ms=5.0), []
    run_fake(monkeypatch, held, n=12, seg=fake, logs=logs, hold_s=10, stats_every_s=0, step_sleep=0.04)
    assert sum("segmentation off" in line for line in logs) == 1 and fake.closed == 1


def test_overload_streak_counts_masked_loop_time():
    ok, slow = [10.0] * 20, [40.0] * 20
    assert overload_streak(0, ok, ok, slow) == 1
    assert overload_streak(1, ok, ok, ok) == 0


def test_stats_fps_is_per_window(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: clock[0])
    s = pipeline.Stats()
    for fps in (10, 30):
        s.frames += 5 * fps
        clock[0] += 5.0
        line = s.line()
        s.reset_window()
    assert " 30.0 fps" in line  # the session average would say 20.0


# ------------------------------------------------------------------ record-raw / eval clock
def write_clip(path, n, fps=30):
    out = streams.FileOutput(str(path), W, H, fps)
    for _ in range(n):
        out.send(np.zeros((H, W, 3), np.uint8))
    out.close()


def test_record_raw_writes_capture_timestamps(monkeypatch, tmp_path):
    raw = tmp_path / "raw.mp4"
    run_fake(monkeypatch, idle, n=10, record_raw=str(raw))
    stamps = json.loads(pipeline.ts_sidecar(raw).read_text())
    assert len(stamps) == 10 and stamps[0] == 0
    assert np.diff(stamps) == pytest.approx(1000 / 30, abs=0.2)  # FakeCap's 30 fps clock


class FakeVideoRec:
    """VIDEO-mode recognizer that sees a held thumbs-up on every frame."""

    def __init__(self, *a, **k):
        pass

    def recognize(self, rgb, ts_ms):
        return Observation(ts_ms / 1000, [thumb()])

    def close(self):
        pass


@pytest.fixture
def fake_eval(monkeypatch):
    monkeypatch.setattr(pipeline, "ensure_model", lambda *a, **k: "model")
    monkeypatch.setattr(pipeline, "GestureRecognizer", FakeVideoRec)


def test_eval_uses_capture_timestamps(fake_eval, tmp_path, capsys):
    """15 frames stamped 30 fps in the container but captured over 1 s (a loop that ran
    at 15 fps): eval must replay the real 1 s, so a 0.7 s hold fires as it did live."""
    clip = tmp_path / "slow.mp4"
    write_clip(clip, 15)
    assert pipeline.evaluate(str(clip), RunConfig(), expect="thumbs_up") == 1  # 0.47 s
    pipeline.ts_sidecar(clip).write_text(json.dumps([i * 1000 / 14 for i in range(15)]))
    assert pipeline.evaluate(str(clip), RunConfig(), expect="thumbs_up") == 0


def test_eval_stills_last_long_enough_for_hold(fake_eval, tmp_path):
    img = tmp_path / "tu.png"
    cv2.imwrite(str(img), np.zeros((H, W, 3), np.uint8))
    assert pipeline.evaluate(str(img), RunConfig(hold_s=1.5), expect="thumbs_up") == 0
    assert pipeline.evaluate(str(img), RunConfig(hold_s=1.5, defer_s=0.5), expect="thumbs_up") == 0


def test_eval_expect_refuses_vacuous_runs(fake_eval, tmp_path):
    with pytest.raises(ValueError, match="no clips"):
        pipeline.evaluate(str(tmp_path), RunConfig(), expect="balloons")
    with pytest.raises(ValueError, match="no frames"):
        pipeline.evaluate(str(tmp_path / "typo.mp4"), RunConfig())
    with pytest.raises(ValueError, match="thumbsup"):
        pipeline.evaluate(str(tmp_path), RunConfig(), expect="thumbsup")
    assert cli.main(["eval", str(tmp_path), "--expect", "balloons"]) == 2


# ------------------------------------------------------------------ demo --sheet
def test_demo_sheet_short_run_blanks_missing_tiles(tmp_path):
    out, logs = tmp_path / "sheet.png", []
    pipeline.demo(Sink(10**6), W, H, fps=10, seconds=4, sheet=str(out), log=logs.append)
    assert cv2.imread(str(out)).shape == (540, 1920, 3)
    assert any("lasers" in line and "not captured" in line for line in logs)


def test_demo_sheet_quit_early(tmp_path):
    out = tmp_path / "sheet.png"
    pipeline.demo(Sink(30), W, H, fps=10, sheet=str(out), log=lambda *a: None)
    assert cv2.imread(str(out)).shape == (540, 1920, 3)
