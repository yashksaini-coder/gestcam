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
