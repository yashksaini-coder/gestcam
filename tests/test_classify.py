import json
import math
import pathlib

import numpy as np
import pytest

from gestcam import recognizer as R
from gestcam.recognizer import WRIST, Hand, is_heart, pick, reaction_for

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def hand(label, score=0.9, curl=(), offset=(0.0, 0.0), aspect=1.0, left=False, handed_score=1.0):
    """Synthetic 21-landmark hand: fingers straight up unless listed in `curl`.

    left=True mirrors it about the wrist (thumb on the other side) and reports "Left".
    Two-hand poses need one of each: two confident same-side hands are one hand twice.
    """
    lm = np.zeros((21, 2), np.float32)
    wrist = np.array([0.5, 0.9]) + offset
    lm[0] = wrist
    # thumb: to the side
    for i, k in enumerate(range(1, 5)):
        lm[k] = wrist + np.array([-0.05 * (i + 1), -0.03 * (i + 1)])
    fingers = {"index": (5, -0.06), "middle": (9, -0.02), "ring": (13, 0.02), "pinky": (17, 0.06)}
    for name, (base, dx) in fingers.items():
        mcp = wrist + np.array([dx, -0.15])
        for j in range(4):
            if name in curl:
                # fold back toward the palm: mcp → pip up → dip back → tip next to mcp
                lm[base + j] = mcp + np.array([0.0, [0.0, -0.04, -0.02, 0.01][j]])
            else:
                lm[base + j] = mcp + np.array([0.0, -0.06 * j])
    if left:
        lm[:, 0] = 2 * wrist[0] - lm[:, 0]
    return Hand(label=label, score=score, handedness="Left" if left else "Right", lm=lm,
                aspect=aspect, handed_score=handed_score)


S = 0.15  # palm size of mk() hands (wrist → middle MCP)


def mk(label, wrist, thumb_dir=+1, curl=(), score=0.9, side=False, down=False):
    """The readers' probe hand (scratchpad probe_recognizer.py), front view, fingers up.

    thumb_dir=+1 puts the thumb toward +x ("Right"), -1 toward -x ("Left").
    side=True squashes x: a fist seen edge-on, as in a thumbs-up. down=True flips it
    vertically (thumbs-down).
    """
    w = np.array(wrist, float)
    lm = np.zeros((21, 2))
    lm[0] = w
    th = [(0.25, -0.2), (0.45, -0.45), (0.6, -0.7), (0.7, -0.95)]
    for k, (dx, dy) in enumerate(th, 1):
        lm[k] = w + S * np.array([thumb_dir * dx, dy])
    F = {5: (0.3, -1.0, [0.4, .25, .2]), 9: (0.0, -1.0, [.45, .28, .2]),
         13: (-0.25, -0.97, [.42, .26, .2]), 17: (-0.45, -0.9, [.3, .2, .18])}
    for base, (dx, dy, segs) in F.items():
        mcp = w + S * np.array([thumb_dir * dx, dy])
        lm[base] = mcp
        y = 0
        for j, L in enumerate(segs, 1):
            if base in curl:  # pip up a bit, then fold tip back down onto palm
                lm[base + j] = mcp + S * np.array([0, [-0.3, -0.15, 0.15][j - 1]])
            else:
                y -= L
                lm[base + j] = mcp + S * np.array([0, y])
    if side:
        lm[:, 0] = w[0] + (lm[:, 0] - w[0]) * 0.35
    if down:
        lm[:, 1] = 2 * w[1] - lm[:, 1]
    return Hand(label, score, "Right" if thumb_dir > 0 else "Left", lm.astype(np.float32), 1.0)


def fist(label, wrist, d=+1, score=0.9, down=False):
    """Thumbs-up seen from the side, as on a webcam: knuckles stacked toward +x (d=+1,
    "Right") or -x (d=-1, "Left"), fingers folded back toward the palm, thumb straight
    up past the knuckles (real photos: thumb tip 0.74-2.4 palms above everything else).
    down=True flips it into a thumbs-down."""
    w = np.array(wrist, float)
    lm = np.zeros((21, 2))
    lm[0] = w
    for k, (x, y) in enumerate([(0.35, -0.3), (0.55, -0.65), (0.6, -1.1), (0.6, -1.55)], 1):
        lm[k] = w + S * np.array([d * x, y])
    for n, base in enumerate((5, 9, 13, 17)):
        y = -0.35 + 0.25 * n  # index on top
        for j, (x, dy) in enumerate([(0.95, 0), (1.3, 0), (1.2, 0.1), (0.95, 0.12)]):
            lm[base + j] = w + S * np.array([d * x, y + dy])
    if down:
        lm[:, 1] = 2 * w[1] - lm[:, 1]
    return Hand(label, score, "Right" if d > 0 else "Left", lm.astype(np.float32), 1.0)


def test_single_hand_canned():
    assert reaction_for([hand("Thumb_Up")]) == "thumbs_up"
    assert reaction_for([hand("Thumb_Down")]) == "thumbs_down"
    assert reaction_for([hand("Victory")]) == "balloons"
    assert reaction_for([hand("Open_Palm")]) is None
    assert reaction_for([hand("Thumb_Up", score=0.3)]) is None
    # A resting hand scores "None"/Open_Palm at 0.9+ and must not outrank a deliberate
    # gesture and swallow the frame (it did: max() ranked it, then the dict missed).
    assert reaction_for([hand("None", 0.95), hand("Thumb_Up", 0.7, offset=(0.35, 0))]) == "thumbs_up"
    assert reaction_for([hand("Open_Palm", 0.95), hand("Thumb_Up", 0.7, offset=(0.3, 0))]) == "thumbs_up"


def test_two_hand_canned():
    assert reaction_for([hand("Thumb_Up"), hand("Thumb_Up", offset=(0.3, 0), left=True)]) == "fireworks"
    assert reaction_for([hand("Thumb_Down"), hand("Thumb_Down", offset=(0.3, 0), left=True)]) == "rain"
    assert reaction_for([hand("Victory"), hand("Victory", offset=(0.3, 0), left=True)]) == "confetti"
    # mixed → falls back to the best single hand
    assert reaction_for([hand("Thumb_Up", 0.9), hand("Victory", 0.8, offset=(0.3, 0), left=True)]) == "thumbs_up"


def test_left_hands_classify_like_right():
    assert reaction_for([hand("Thumb_Up", left=True)]) == "thumbs_up"
    assert hand("None", curl=("middle", "ring"), left=True).is_horns()
    assert mk("None", (0.5, 0.8), -1, curl=(13, 17)).pose() == "victory"
    assert fist("None", (0.5, 0.8), -1).pose() == "thumb_up"


def test_horns_rule():
    h = hand("None", curl=("middle", "ring"))
    assert h.is_horns()
    assert not hand("None").is_horns()
    assert not hand("None", curl=("middle",)).is_horns()
    assert reaction_for([h, hand("None", curl=("middle", "ring"), offset=(0.3, 0), left=True)]) == "lasers"
    # No one-hand horns reaction (Apple: lasers need both hands).
    assert reaction_for([h]) is None


def test_heart_rule():
    """Classic heart: index tips touch on top, thumb tips at the bottom, the other
    fingers curled round into the lobes."""
    a = hand("None", curl=("middle", "ring", "pinky"))
    b = hand("None", curl=("middle", "ring", "pinky"), offset=(0.3, 0), left=True)
    assert not is_heart(a, b)
    a.lm[8] = b.lm[8] = np.array([0.5, 0.3])
    a.lm[4] = np.array([0.49, 0.5]); b.lm[4] = np.array([0.51, 0.5])
    assert is_heart(a, b)
    assert reaction_for([a, b]) == "hearts"
    # Same shape from two confident right hands is not a heart.
    b.handedness = "Right"
    assert not is_heart(a, b)
    b.handed_score = 0.5  # ...unless MediaPipe isn't sure which hand it is
    assert is_heart(a, b)


def test_inverted_heart():
    """Thumbs on top, fingers pointing down, index tips meeting at the bottom: the
    only heart the palm detector found in the photo set. Its fingers are straight."""
    a = mk("None", (0.42, 0.4), +1, down=True)
    b = mk("None", (0.58, 0.4), -1, down=True)
    a.lm[8] = b.lm[8] = np.array([0.5, 0.75])
    a.lm[4] = np.array([0.49, 0.5]); b.lm[4] = np.array([0.51, 0.5])
    assert is_heart(a, b)
    assert reaction_for([a, b]) == "hearts"


def test_heart_geometry_beats_canned_labels():
    """P4: the model has no heart class and labelled heart hands Thumb_Down → rain."""
    a = hand("Thumb_Down", 0.9, curl=("middle", "ring", "pinky"))
    b = hand("Thumb_Down", 0.9, curl=("middle", "ring", "pinky"), offset=(0.3, 0), left=True)
    a.lm[8] = b.lm[8] = np.array([0.5, 0.3])
    a.lm[4] = np.array([0.49, 0.5]); b.lm[4] = np.array([0.51, 0.5])
    assert reaction_for([a, b]) == "hearts"


def test_adjacent_victory_is_confetti_not_heart():
    """Two V hands with index tips touching satisfy all three of the old is_heart tip
    tests (it used to rely on canned-label precedence). The extended middle fingers
    are what rule it out now."""
    a = hand("Victory", 0.9, curl=("ring", "pinky"))
    b = hand("Victory", 0.9, curl=("ring", "pinky"), offset=(0.25, 0), left=True)
    a.lm[8] = b.lm[8] = np.array([0.5, 0.3])
    a.lm[4] = np.array([0.49, 0.5]); b.lm[4] = np.array([0.51, 0.5])
    size = 0.5 * (a.size + b.size)
    assert R._d(a.lm[4], b.lm[4]) / size < 0.715 and (a.lm[4][1] - a.lm[8][1]) > 0.3 * size
    assert not is_heart(a, b)
    assert reaction_for([a, b]) == "confetti"


def test_open_palm_frame_is_not_heart():
    """P3: open palms side by side, thumbs inward. Index and thumb tips meet just like a
    heart; it used to fire hearts at wrist gaps around 1.1 palms."""
    for sep in (0.14, 0.155, 0.165, 0.17, 0.2, 0.25):
        a = mk("Open_Palm", (0.5 - sep / 2, 0.8), +1)
        b = mk("Open_Palm", (0.5 + sep / 2, 0.8), -1)
        assert reaction_for([a, b]) is None
        assert not is_heart(a, b)


def test_two_thumbs_held_together_are_fireworks():
    """P1: two fists side by side, wrists 0.5-1.5 palms apart. The old wrist floor
    (> 1 palm) called these one hand and fired thumbs_up."""
    for k in (0.5, 0.7, 0.9, 1.0, 1.1, 1.5):
        a = fist("Thumb_Up", (0.5 - k * S / 2, 0.8), -1)
        b = fist("Thumb_Up", (0.5 + k * S / 2, 0.8), +1)
        assert reaction_for([a, b]) == "fireworks", k


def test_two_victory_side_by_side_is_confetti():
    """P2: V hands with wrists 0.8-1.2 palms apart."""
    for k in (0.8, 0.9, 1.0, 1.2):
        a = mk("Victory", (0.5 - k * S / 2, 0.8), +1, curl=(13, 17))
        b = mk("Victory", (0.5 + k * S / 2, 0.8), -1, curl=(13, 17))
        assert reaction_for([a, b]) == "confetti", k


def test_horns_geometry_beats_canned_labels():
    """P5: horns hands labelled Victory gave confetti; ILoveYou was never consulted."""
    for lab in ("Victory", "ILoveYou", "None"):
        h1 = mk(lab, (0.3, 0.8), +1, curl=(9, 13))
        h2 = mk(lab, (0.7, 0.8), -1, curl=(9, 13))
        assert reaction_for([h1, h2]) == "lasers", lab
    assert reaction_for([mk("Victory", (0.5, 0.8), +1, curl=(9, 13))]) is None


def test_horns_foreshortening_limit():
    """P6: horns pointed at the camera. 2D rules hold to 0.6x finger length and fail at
    0.4x; Apple's reviewers report the same failure, so this documents the limit."""
    def shortened(f):
        h = mk("None", (0.5, 0.8), +1, curl=(9, 13))
        for base in (5, 17):
            mcp = h.lm[base].copy()
            for j in (1, 2, 3):
                h.lm[base + j] = mcp + (h.lm[base + j] - mcp) * f
        return h
    assert shortened(1.0).is_horns() and shortened(0.6).is_horns()
    assert not shortened(0.4).is_horns()


def test_pose_geometry_fills_in_for_canned_label():
    """The canned label missed 4/6 clear V signs on real photos (None:.96, Pointing_Up:.64)."""
    v = dict(wrist=(0.5, 0.8), thumb_dir=+1, curl=(13, 17))
    assert mk("None", score=0.96, **v).pose() == "victory"
    assert mk("Pointing_Up", score=0.64, **v).pose() == "victory"
    assert reaction_for([mk("None", score=0.96, **v)]) == "balloons"
    # The model confidently names some other class: geometry must not override it.
    assert mk("Open_Palm", score=0.9, **v).pose() is None
    # ...except Pointing_Up: the model reads real V signs as Pointing_Up up to .72, and a
    # pointing hand has its middle finger folded, which is_victory already rules out.
    assert mk("Pointing_Up", score=0.8, **v).pose() == "victory"
    # Thumbs from geometry, both directions.
    assert fist("None", (0.5, 0.8), score=0.6).pose() == "thumb_up"
    assert fist("None", (0.5, 0.8), score=0.6, down=True).pose() == "thumb_down"
    assert fist("Closed_Fist", (0.5, 0.8), score=0.9).pose() is None
    # Canned label still wins where it's confident and geometry is silent.
    assert hand("Thumb_Up", 0.7).pose() == "thumb_up"
    assert hand("None").pose() is None


def test_thumb_rule_rejects_sideways_thumb():
    """A resting fist with the thumb along the knuckles is not a thumbs-up; the same
    fist rolled ~35° still is."""
    flat = fist("None", (0.5, 0.8))
    # thumb long and straight, but level with the index knuckle instead of past it
    flat.lm[R.THUMB_TIP] = [flat.lm[R.THUMB_MCP][0] + 0.8 * S, flat.lm[R.INDEX_MCP][1]]
    flat.lm[3] = 0.5 * (flat.lm[2] + flat.lm[4])
    assert R._d(flat.lm[R.THUMB_TIP], flat.lm[R.THUMB_MCP]) > 0.5 * flat.size
    assert flat.pose() is None

    for deg in (-35, 35):
        up = fist("None", (0.5, 0.8))
        a = math.radians(deg)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
        up.lm = (up.lm - up.lm[WRIST]) @ rot.T + up.lm[WRIST]
        assert up.pose() == "thumb_up", deg


def test_downward_v_is_not_victory():
    """One hand of an inverted heart is a V pointing down. With only that hand detected
    (heart_sun photo) it fired balloons."""
    h = mk("None", (0.5, 0.4), +1, curl=(13, 17), down=True)
    assert h.pose() is None
    assert reaction_for([h]) is None


def test_weak_second_hand_does_not_downgrade():
    """R2: tu_two_b gave [Thumb_Up:.72, Thumb_Up:.50]; at .49 the pair became thumbs_up."""
    a = fist("Thumb_Up", (0.35, 0.8), -1, score=0.72)
    b = fist("Thumb_Up", (0.65, 0.8), +1, score=0.49)
    assert reaction_for([a, b]) == "fireworks"


def test_size_is_rotation_invariant():
    """MediaPipe normalizes x by width and y by height. Without the aspect correction
    the palm-size divisor — which every other threshold divides by — swings 40% with
    hand rotation alone."""
    W, H = 1280, 720
    base = hand("None").lm

    def rotated(deg, aspect):
        px = np.stack([base[:, 0] * W, base[:, 1] * H], 1)
        o, a = px[WRIST], math.radians(deg)
        r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        px = (px - o) @ r.T + o
        lm = np.stack([px[:, 0] / W, px[:, 1] / H], 1).astype(np.float32)
        return Hand("None", 0.9, "Right", lm, aspect).size

    fixed = [rotated(d, W / H) for d in (0, 30, 60, 90)]
    assert max(fixed) - min(fixed) < 0.01 * fixed[0]
    naive = [rotated(d, 1.0) for d in (0, 30, 60, 90)]
    assert max(naive) - min(naive) > 0.3 * naive[0]  # the bug this guards against


def test_duplicate_detection_is_not_two_hands():
    """MediaPipe reports a second near-duplicate box for one hand. is_heart used to
    accept that and fire "hearts" for one thumbs-up. Mean landmark distance catches it
    even when MediaPipe calls the twin the other hand."""
    c = hand("Thumb_Up", 0.9)
    d = hand("Thumb_Up", 0.9, left=True)
    d.lm = c.lm + 0.003
    assert reaction_for([c, d]) == "thumbs_up"
    assert reaction_for([hand("Thumb_Up"), hand("Thumb_Up", offset=(0.3, 0), left=True)]) == "fireworks"


def test_same_hand_twice_is_not_a_pair():
    """Two confident right hands are one hand seen twice (or two people); when either
    handedness score is low, fall back to the landmark-distance test alone."""
    a, b = hand("Thumb_Up"), hand("Thumb_Up", offset=(0.3, 0))
    assert reaction_for([a, b]) == "thumbs_up"
    b.handed_score = 0.6
    assert reaction_for([a, b]) == "fireworks"


def test_far_apart_hands_do_not_pair():
    """R11: two people's single hands at opposite edges of the frame."""
    a = fist("Thumb_Up", (0.1, 0.8), -1)
    b = fist("Thumb_Up", (0.1 + 7 * S, 0.8), +1)
    assert R._palm_gap(a, b) > 6
    assert reaction_for([a, b]) == "thumbs_up"
    b.lm = b.lm - [3 * S, 0]  # one person's spread arms (measured up to ~4.5 palms)
    assert reaction_for([a, b]) == "fireworks"


def test_every_pair_is_considered():
    """R11: only hands[0]/hands[1] could pair; a third hand listed first hid the pair."""
    other = hand("None", 0.95, offset=(-0.4, 0))
    a = mk("Victory", (0.45, 0.8), +1, curl=(13, 17))
    b = mk("Victory", (0.6, 0.8), -1, curl=(13, 17))
    rx, used = pick([other, a, b])
    assert rx == "confetti" and used == [a, b]


def test_pick_returns_deciding_hands():
    a = hand("None", 0.95)
    b = hand("Thumb_Up", 0.7, offset=(0.35, 0))
    assert pick([a, b]) == ("thumbs_up", [b])
    assert pick([]) == (None, [])


def test_ensure_model_downloads_once(tmp_path, monkeypatch):
    calls = []

    def fake(url, dst):
        calls.append(url)
        pathlib.Path(dst).write_bytes(b"x" * 2000)

    monkeypatch.setattr(R.urllib.request, "urlretrieve", fake)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    p = R.ensure_model(log=lambda *a: None, url="http://example/seg.tflite", filename="seg.tflite", min_bytes=1000)
    assert p == tmp_path / "gestcam" / "seg.tflite" and p.stat().st_size == 2000
    assert not p.with_suffix(".part").exists()
    assert R.ensure_model(log=lambda *a: None, url="http://x", filename="seg.tflite", min_bytes=1000) == p
    assert calls == ["http://example/seg.tflite"]


def _photos():
    return json.loads((FIXTURES / "photo_hands.json").read_text())["photos"]


def _hands(photo):
    return [Hand(h["label"], h["score"], h["handedness"], np.array(h["lm"], np.float32),
                 h["aspect"], h["handed_score"]) for h in photo["hands"]]


@pytest.mark.parametrize("photo", _photos(), ids=lambda p: p["name"])
def test_real_photo_landmarks(photo):
    """Landmarks MediaPipe extracted from openly licensed photos (source URLs in the
    fixture). `expect` differs from `truth` only where the detector failed (note)."""
    assert reaction_for(_hands(photo)) == photo["expect"]


def test_real_photos_never_fire_the_wrong_reaction():
    for photo in _photos():
        assert reaction_for(_hands(photo)) in (photo["truth"], None), photo["name"]


def _variants():
    return json.loads((FIXTURES / "photo_variants.json").read_text())["photos"]


@pytest.mark.parametrize("photo", _variants(), ids=lambda p: p["name"])
def test_real_photo_variants(photo):
    """Rolled and 16:9-letterboxed copies of the fixture photos, through the real model.
    A two-hand truth may degrade only to its one-hand subset (detector lost a hand)."""
    got = reaction_for(_hands(photo))
    assert got == photo["expect"]
    subset = {R.PAIR[p]: R.SINGLE[p] for p in R.SINGLE}
    assert got in (photo["truth"], subset.get(photo["truth"]), None)


def test_v_beats_pointing_up_label():
    """Real v_adolescent V landmarks: the model drifts to Pointing_Up .70-.72 when the
    hand rolls or the frame is mirrored. Geometry V must survive any Pointing_Up score."""
    h = _hands(next(p for p in _photos() if p["name"] == "v_adolescent"))[0]
    assert h.is_victory()
    for s in (0.64, 0.70, 0.75, 0.95):
        h.label, h.score = "Pointing_Up", s
        assert h.pose() == "victory" and reaction_for([h]) == "balloons", s


def test_thumbs_down_survives_roll():
    """td_videocall is geometry-only (None:.59). Its thumb tip sits only 0.35 palms below
    the curled index upright; a -20° roll takes that to ~0.15."""
    p = next(p for p in _photos() if p["name"] == "td_videocall")
    for deg in (-20, 0, 20):
        h = _hands(p)[0]
        a = math.radians(deg)
        pts = h.lm.astype(float) * [h.aspect, 1.0]
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        h.lm = (((pts - pts[WRIST]) @ rot.T + pts[WRIST]) / [h.aspect, 1.0]).astype(np.float32)
        assert h.pose() == "thumb_down", deg


def test_resting_fist_is_not_thumbs_down():
    """The looser thumbs-down margin must not reach a fist whose thumb lies across the
    curled fingers: there the thumb tip is level with, not below, the knuckles."""
    f = fist("None", (0.5, 0.8), down=True)
    f.lm[R.THUMB_TIP] = [f.lm[R.THUMB_MCP][0] + 0.8 * S, f.lm[R.INDEX_TIP:, 1].max()]
    f.lm[3] = 0.5 * (f.lm[2] + f.lm[4])
    assert f.pose() is None


_MODEL = R.cache_dir() / "gesture_recognizer.task"


@pytest.mark.skipif(not _MODEL.exists(), reason="gesture model not cached")
def test_video_mode_recognize_is_synchronous():
    rec = R.GestureRecognizer(_MODEL, mode="video", log=lambda *a: None)
    try:
        obs = rec.recognize(np.zeros((360, 640, 3), np.uint8), 0)
        assert obs.hands == [] and obs.ts == 0.0
        obs = rec.recognize(np.zeros((360, 640, 3), np.uint8), 33)
        assert obs.ts == pytest.approx(0.033)
    finally:
        rec.close()
    with pytest.raises(ValueError):
        R.GestureRecognizer(_MODEL, mode="image")
