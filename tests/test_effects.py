import cv2
import numpy as np

from gestcam.effects import EffectEngine, HoldRing, blend_sprite
from gestcam.recognizer import REACTIONS
from gestcam.sprites import SpriteSet


def test_every_reaction_renders_and_finishes():
    sprite = np.zeros((128, 128, 4), np.uint8); sprite[..., :] = (0, 200, 255, 255)
    eng = EffectEngine(640, 360, sprites={"thumbs_up": sprite}, seed=1)
    for r in REACTIONS:
        eng.trigger(r, [(300.0, 200.0), (340.0, 200.0)])
    assert eng.busy
    frame = np.zeros((360, 640, 3), np.uint8)
    for _ in range(int(6 / (1 / 30))):
        eng.step(frame, 1 / 30, HoldRing(320, 100, 0.5))
    assert not eng.busy
    assert frame.any()


def test_blend_sprite_clips_at_edges():
    frame = np.zeros((100, 100, 3), np.uint8)
    sprite = np.full((16, 16, 4), 255, np.uint8)
    blend_sprite(frame, sprite, -5, -5, 1.0)   # partially off-screen
    blend_sprite(frame, sprite, 200, 200, 1.0)  # fully off-screen: no crash
    assert frame[0, 0].tolist() == [255, 255, 255]


def test_premultiplied_blend_has_no_dark_fringe():
    """Noto PNGs carry RGB=0 in transparent pixels. Downscaling straight alpha averages
    that black into every edge pixel; premultiplying at load is what prevents it."""
    # Straight alpha, exactly like a real Noto PNG: full-intensity RGB wherever the
    # glyph is, RGB=0 in the transparent surround, antialiased alpha between.
    alpha = np.zeros((128, 128), np.uint8)
    cv2.circle(alpha, (64, 64), 50, 255, -1, cv2.LINE_AA)
    src = np.zeros((128, 128, 4), np.uint8)
    src[..., :3][alpha > 0] = 255
    src[..., 3] = alpha
    small = SpriteSet({"disc": src}).get("disc", 48)
    frame = np.full((100, 100, 3), 255, np.uint8)   # white sprite on white ground
    blend_sprite(frame, small, 50, 50, 1.0)
    assert frame.min() > 240, f"dark fringe: min pixel {frame.min()}"


def test_every_reaction_renders_without_sprites():
    """Offline / missing-asset path: every effect must fall back to its primitive."""
    eng = EffectEngine(640, 360, sprites={}, seed=1)
    frame = np.zeros((360, 640, 3), np.uint8)
    for r in REACTIONS:
        eng.trigger(r, [(300.0, 200.0), (340.0, 200.0)])
    for _ in range(int(6 / (1 / 30))):
        eng.step(frame, 1 / 30, HoldRing(320, 100, 0.5))
    assert frame.any()


# ---------------------------------------------------------------- layering, fades, coalescing
from gestcam.effects import Effect, head_box  # noqa: E402
from gestcam.sprites import load_sprites  # noqa: E402

W, H = 640, 360
ANCH = [(260.0, 300.0), (380.0, 300.0)]
SPRITES = load_sprites(log=lambda *_: None)


def _person_mask():
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (320, 120), (60, 75), 0, 0, 360, 255, -1)       # head
    cv2.rectangle(m, (200, 190), (440, H), 255, -1)                # shoulders
    return m


def _alpha(eng_b, eng_w, mask):
    """Drawn alpha per pixel: compositing over black and over white tells alpha exactly,
    whatever the effect draws (sprite, primitive, additive light or darkening)."""
    b = np.zeros((eng_b.h, eng_b.w, 3), np.uint8)
    w = np.full((eng_b.h, eng_b.w, 3), 255, np.uint8)
    anch = [(x / 2, y / 2) for x, y in ANCH]
    eng_b.step(b, 1 / 30, mask=mask, anchors=anch)
    eng_w.step(w, 1 / 30, mask=mask, anchors=anch)
    return 1.0 - (w.astype(np.float32) - b).max(2) / 255.0


def test_every_effect_is_transparent_on_its_last_live_frame():
    """A hard cut on the last frame (balloons vanishing mid-screen, E2) is a visible pop.
    Half resolution: effects scale with w/h, and it keeps this test ~3 s."""
    for mask in (None, _person_mask()[::2, ::2].copy()):
        for r in REACTIONS:
            eb, ew = EffectEngine(W // 2, H // 2, SPRITES, seed=4), EffectEngine(W // 2, H // 2, SPRITES, seed=4)
            eb.trigger(r, [(x / 2, y / 2) for x, y in ANCH])
            ew.trigger(r, [(x / 2, y / 2) for x, y in ANCH])
            peak, last = 0.0, None
            for _ in range(int(12 * 30)):
                a = _alpha(eb, ew, mask)
                if not eb.busy:
                    break
                last = a
                peak = max(peak, float(a.max()))
            assert not eb.busy, r
            assert peak > 0.5, f"{r} never drew anything visible"
            region = last[last > 0.01]
            assert region.size == 0 or region.mean() <= 0.1, f"{r}: last-frame alpha {region.mean():.2f}"


def test_particle_sprites_are_not_opaque_tiles():
    """🎆/🎇 are 88% fully-opaque squares and read as a pasted tile (E3). Every sprite an
    effect draws as a particle must be a silhouette. Measured: balloon 0.34, heart 0.55
    (a heart fills 57% of its square), sparkles 0.21."""
    asked: set[str] = set()

    class Spy(type(SPRITES)):
        def get(self, name, size_px):
            asked.add(name)
            return super().get(name, size_px)

    spy = Spy.__new__(Spy)
    spy.__dict__.update(SPRITES.__dict__)
    eng = EffectEngine(W, H, spy, seed=2)
    for r in ("hearts", "balloons", "confetti", "fireworks", "rain", "lasers"):
        eng.trigger(r, ANCH)
    frame = np.zeros((H, W, 3), np.uint8)
    for _ in range(150):
        eng.step(frame, 1 / 30)
    assert asked, "no sprite was drawn at all"
    for name in asked:
        opaque = (spy.raw[name][:, :, 3] == 255).mean()
        assert opaque <= 0.6, f"{name} is {opaque:.0%} opaque"


def test_fireworks_draw_no_emoji_tiles():
    tile = np.full((128, 128, 4), 255, np.uint8)
    frames = []
    for sprites in ({}, {k: tile for k in ("fireworks", "sparkler", "firecracker", "sparkles", "star")}):
        eng = EffectEngine(W, H, sprites, seed=9)
        eng.trigger("fireworks")
        f = np.zeros((H, W, 3), np.uint8)
        for _ in range(40):
            eng.step(f, 1 / 30)
        frames.append(f)
    assert np.array_equal(*frames)


class _PaintBack(Effect):
    back = True

    def draw_back(self, img, mask):
        img[:] = (0, 255, 0)


def test_back_layer_never_touches_the_person():
    rng = np.random.default_rng(0)
    src = rng.integers(0, 256, (H, W, 3), np.uint8)
    mask = _person_mask()
    eng = EffectEngine(W, H, {}, seed=1)
    eng.active.append(_PaintBack(W, H))
    f = src.copy()
    eng.step(f, 1 / 30, mask=mask)
    assert np.array_equal(f[mask == 255], src[mask == 255])
    assert (f[mask == 0] == (0, 255, 0)).all()

    # Real back-only effects: fireworks (darkening + light) leave the person bit-exact.
    eng = EffectEngine(W, H, SPRITES, seed=1)
    eng.trigger("fireworks")
    for _ in range(40):
        f = src.copy()
        eng.step(f, 1 / 30, mask=mask)
        assert np.array_equal(f[mask == 255], src[mask == 255])
    assert (f[mask == 0].astype(int).mean()) < src[mask == 0].mean() * 0.8   # bg darkened


def test_rain_darkens_background_more_than_person_and_lasers_only_tint():
    src = np.full((H, W, 3), 200, np.uint8)
    mask = _person_mask()
    for r in ("rain", "lasers"):
        eng = EffectEngine(W, H, SPRITES, seed=1)
        eng.trigger(r, ANCH)
        for _ in range(30):   # past the fade-in
            f = src.copy()
            eng.step(f, 1 / 30, mask=mask, anchors=ANCH)
        person = f[mask == 255].astype(int)
        if r == "rain":   # documented: person x0.85, nothing drawn over them
            assert np.abs(person - 170).max() <= 1
            assert np.median(f[mask == 0]) < 0.6 * 200
        else:             # documented: additive tint of at most 0.16 * beam colour
            assert (person >= 200).all() and (person - 200).max() <= 0.16 * 255 + 1


def test_retrigger_extends_instead_of_stacking():
    eng = EffectEngine(W, H, SPRITES, seed=1)
    frame = np.zeros((H, W, 3), np.uint8)
    for r in ("confetti", "hearts", "fireworks", "balloons", "rain", "lasers", "thumbs_up"):
        eng.trigger(r, ANCH)
        for _ in range(30):
            eng.step(frame, 1 / 30)
        e = eng.active[-1]
        end = e.end
        eng.trigger(r, ANCH)
        assert sum(x.kind == r for x in eng.active) == 1, r
        assert eng.active[-1].end > end, f"{r} was not prolonged"
        if r == "confetti":
            assert len(e.p["born"]) > 220   # topped up, not only prolonged


def test_superset_retires_live_subset():
    eng = EffectEngine(W, H, SPRITES, seed=1)
    frame = np.zeros((H, W, 3), np.uint8)
    for sub, sup in (("thumbs_up", "fireworks"), ("thumbs_down", "rain"), ("balloons", "confetti")):
        eng.trigger(sub, ANCH)
        for _ in range(10):
            eng.step(frame, 1 / 30)
        eng.trigger(sup, ANCH)
        for _ in range(9):   # 0.3 s > the 0.25 s retire fade
            eng.step(frame, 1 / 30)
        kinds = [e.kind for e in eng.active]
        assert sub not in kinds and sup in kinds, (sub, kinds)


def test_step_keeps_old_signatures():
    eng = EffectEngine(W, H, SPRITES, seed=1)
    eng.trigger("hearts", ANCH)
    f = np.zeros((H, W, 3), np.uint8)
    eng.step(f, 1 / 30)
    eng.step(f, 1 / 30, HoldRing(10, 10, 0.5, "hearts"))
    eng.step(f, 1 / 30, None, None, None)


def test_effects_are_frame_rate_independent():
    """Same t at 30 fps and at 60 fps must be the same picture (closed-form motion)."""
    for r in ("fireworks", "confetti", "balloons", "rain", "hearts", "lasers"):
        outs = []
        for fps in (30, 60):
            eng = EffectEngine(W, H, SPRITES, seed=6)
            eng.trigger(r, ANCH)
            for _ in range(fps):          # 1.0 s
                f = np.full((H, W, 3), 90, np.uint8)
                eng.step(f, 1 / fps, mask=_person_mask(), anchors=ANCH)
            outs.append(f.astype(np.float32))
        diff = np.abs(outs[0] - outs[1]).mean()
        assert diff < 0.5, f"{r}: mean abs diff {diff:.2f} between 30 and 60 fps"


def test_thumb_badge_sits_beside_head_and_follows():
    mask = _person_mask()
    cx, top, hw = head_box(mask)
    eng = EffectEngine(W, H, SPRITES, seed=1)
    eng.trigger("thumbs_up", [(420.0, 300.0)])
    f = np.zeros((H, W, 3), np.uint8)
    eng.step(f, 1 / 30, mask=mask)
    th = eng.active[0]
    x0 = th.pos[0]
    assert abs(x0 - cx) > hw, "badge overlaps the head"
    assert x0 > cx, "badge should be on the side of the hand that made it"
    moved = np.roll(mask, -60, axis=1)   # person steps left
    eng.step(f, 1 / 30, mask=moved)
    x1 = th.pos[0]
    assert x1 < x0 and x0 - x1 < 60, "should follow, smoothly (time-based EMA)"
    for _ in range(30):
        eng.step(f, 1 / 30, mask=moved)
    assert abs(th.pos[0] - (x0 - 60)) <= 8   # head_box samples an 8 px grid


def test_balloon_hue_variants_and_fine_buckets():
    base = SPRITES.raw["balloons"]
    hues = []
    for i in range(7):
        v = SPRITES.raw[f"balloon{i}"]
        assert v.shape == base.shape and np.array_equal(v[..., 3], base[..., 3])
        hsv = cv2.cvtColor(np.ascontiguousarray(v[..., :3]), cv2.COLOR_BGR2HSV)
        hues.append(int(np.median(hsv[..., 0][v[..., 3] == 255])))
    assert len(set(hues)) == 7, hues
    assert SPRITES.get("hearts", 50).shape[0] == 48
    assert SPRITES.get("hearts", 53).shape[0] == 52   # 4 px buckets below 64 (E10)
    assert SPRITES.get("hearts", 75).shape[0] == 72


# ---------------------------------------------------------------- review fixes (effects group)
from gestcam.effects import WARM, Balloons, Confetti, Hearts, Lasers, _drag_path  # noqa: E402
from gestcam.segment import refine_mask  # noqa: E402


def test_fireworks_flash_is_a_small_core_not_a_white_disc():
    """The burst flash was a flat, clipped white disc ~0.15 h across (2x glow boost on a
    230-level fill), and the white-hot sparks packed into a blob behind it. Clipped
    sparks are fine; a clipped AREA (every px of a 9x9 neighbourhood clipped) is not.
    At 720p, the output size: spark dots are fixed px, so 360p merges them."""
    w, h = 1280, 720
    yy, xx = np.ogrid[:h, :w]
    worst = 0
    for i in range(3):
        eng = EffectEngine(w, h, {}, seed=1)
        eng.trigger("fireworks")
        fw = eng.active[0]
        b = fw.bursts[i]
        near = (xx - b["x"]) ** 2 + (yy - b["y"]) ** 2 <= (0.1 * h) ** 2
        while fw.t < b["t0"] + fw.RISE + 0.15:
            f = np.full((h, w, 3), 110, np.uint8)
            eng.step(f, 1 / 30)
            if fw.t >= b["t0"] + fw.RISE:
                sat = ((f.min(2) >= 245) & near).astype(np.uint8)
                worst = max(worst, int(cv2.erode(sat, np.ones((9, 9), np.uint8)).sum()))
    assert worst < 500, f"flash plateau of {worst} clipped px"   # was ~2.3k


def test_darkening_has_no_bright_halo_around_the_person():
    """refine_mask dilates for the particles' sake; the full-frame grade went through
    that same weight, so a ~20 px band round the person stayed undarkened."""
    raw = np.zeros((H // 2, W // 2), np.uint8)
    cv2.ellipse(raw, (160, 200), (70, 120), 0, 0, 360, 255, -1)
    mask = refine_mask(raw, W, H)
    big = cv2.resize(raw, (W, H), interpolation=cv2.INTER_NEAREST)
    dist = cv2.distanceTransform(255 - big, cv2.DIST_L2, 5)
    for r in ("fireworks", "rain", "lasers"):
        eng = EffectEngine(W, H, {}, seed=3)
        eng.trigger(r, ANCH)
        for _ in range(20):
            f = np.full((H, W, 3), 160, np.uint8)
            eng.step(f, 1 / 30, mask=mask)
        g = f.mean(2)
        band = np.median(g[(dist > 2) & (dist <= 10)])
        far = np.median(g[(dist > 30) & (dist <= 60)])
        assert band <= far * 1.08 + 2, f"{r}: band {band:.0f} vs background {far:.0f}"
        # person kept (rain darkens it to 0.85 on purpose: 136)
        assert np.median(g[cv2.erode(big, np.ones((15, 15))) == 255]) >= 0.85 * 160 - 1


def test_hearts_from_hands_under_the_chin_go_around_the_face():
    """The usual heart pose: hands centred under the chin. A 25-75° launch sent ~40% of
    hearts through the mouth and eyes (E9). Geometry from the review's tu_two_b still."""
    w, h = 1280, 720
    cx, top, hw = 552.0, 56.0, 136.0
    face = np.zeros((h, w), bool)
    face[int(top):int(top + 1.9 * hw), int(cx - hw):int(cx + hw)] = True
    for seed in range(4):
        hs = Hearts(w, h, None, [(470.0, 360.0), (610.0, 360.0)], seed=seed)
        covered = shown = 0
        for _ in range(75):
            hs.update(1 / 30)
            hs.follow(None, (cx, top, hw), 1 / 30)
            img = np.zeros((h, w, 3), np.uint8)
            hs.draw_front(img, None)
            drawn = img.any(2)
            covered, shown = max(covered, int((drawn & face).sum())), max(shown, int(drawn.sum()))
        assert covered < 0.01 * face.sum(), f"seed {seed}: {covered} face px under hearts"
        assert shown > 20000, f"seed {seed}: hearts were steered away entirely ({shown} px)"


def _confetti_xy(c, t):
    p = c.p
    age = t - p["born"]
    dx, dy = _drag_path(p["vx"], p["vy"], c.G * c.h, c.K, np.maximum(age, 0))
    return (p["x0"] + dx)[age > 0], (p["y0"] + dy)[age > 0]


def test_confetti_falls_through_the_whole_scene():
    """Apple: 'festive spots of colour falling through the scene'. The corner cannons
    alone never put a piece above 0.2 h or in the top centre."""
    for seed in (1, 8):
        c = Confetti(W, H, seed=seed)
        for t in (0.6, 1.0, 1.5, 2.0):
            x, y = _confetti_xy(c, t)
            top_centre = ((x > W / 3) & (x < 2 * W / 3) & (y > 0) & (y < H / 3)).sum()
            assert top_centre >= 8, f"seed {seed} t={t}: {top_centre} pieces in top centre"


def test_confetti_arrays_stay_bounded_when_refired():
    eng = EffectEngine(W, H, {}, seed=1)
    f = np.zeros((H, W, 3), np.uint8)
    for i in range(int(60 * 30)):
        if i % 96 == 0:   # every 3.2 s: hold + cooldown, the fastest a user can repeat
            eng.trigger("confetti")
        eng.step(f, 1 / 30)
    assert len(eng.active[0].p["born"]) < 900


def test_laser_beams_use_every_warm_colour_and_keep_their_hue():
    for seed in range(200):
        cols = {b["col"] for b in Lasers(W, H, seed=seed).beams}
        assert len(cols) == len(WARM), (seed, cols)
    mask = np.zeros((H, W), np.uint8)
    eng = EffectEngine(W, H, {}, seed=7)
    eng.trigger("lasers")
    for _ in range(40):
        f = np.full((H, W, 3), 200, np.uint8)
        eng.step(f, 1 / 30, mask=mask)
    bg = np.median(f.reshape(-1, 3), axis=0)
    beam = (f.astype(int) - bg).max(2) > 25
    sat = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 1][beam]
    assert np.median(sat) >= 85, f"beams wash out: median saturation {np.median(sat)}"


def test_later_balloons_enter_from_below_the_frame():
    b = Balloons(W, H, SPRITES, seed=1)
    for p in b.parts:
        top = p["y0"] - p["size"] / 2
        if p["born"] > 0:
            assert top >= H, f"born {p['born']:.2f}: {H - top:.0f} px already on screen"
        else:
            assert top < H   # the first two peek in at once, on purpose


def test_balloon_sizes_are_cached():
    for size in (150, 187):
        assert SPRITES.get("balloon0", size) is SPRITES.get("balloon0", size)
