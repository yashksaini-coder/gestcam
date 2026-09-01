import numpy as np
from gestcam.effects import EffectEngine, HoldRing, blend_sprite
from gestcam.recognizer import REACTIONS


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
    blend_sprite(frame, sprite, -5, -5, 20, 1.0)   # partially off-screen
    blend_sprite(frame, sprite, 200, 200, 20, 1.0) # fully off-screen: no crash
    assert frame[0, 0].tolist() == [255, 255, 255]
