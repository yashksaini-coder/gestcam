import numpy as np
from gestcam.recognizer import Hand, reaction_for, is_heart


def hand(label, score=0.9, curl=(), offset=(0.0, 0.0)):
    """Synthetic 21-landmark hand: fingers straight up unless listed in `curl`."""
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
    return Hand(label=label, score=score, handedness="Right", lm=lm)


def test_single_hand_canned():
    assert reaction_for([hand("Thumb_Up")]) == "thumbs_up"
    assert reaction_for([hand("Thumb_Down")]) == "thumbs_down"
    assert reaction_for([hand("Victory")]) == "balloons"
    assert reaction_for([hand("Open_Palm")]) is None
    assert reaction_for([hand("Thumb_Up", score=0.3)]) is None


def test_two_hand_canned():
    assert reaction_for([hand("Thumb_Up"), hand("Thumb_Up", offset=(0.3, 0))]) == "fireworks"
    assert reaction_for([hand("Thumb_Down"), hand("Thumb_Down", offset=(0.3, 0))]) == "rain"
    assert reaction_for([hand("Victory"), hand("Victory", offset=(0.3, 0))]) == "confetti"
    # mixed → falls back to the best single hand
    assert reaction_for([hand("Thumb_Up", 0.9), hand("Victory", 0.8, offset=(0.3, 0))]) == "thumbs_up"


def test_horns_rule():
    h = hand("None", curl=("middle", "ring"))
    assert h.is_horns()
    assert not hand("None").is_horns()
    assert not hand("None", curl=("middle",)).is_horns()
    assert reaction_for([h, hand("None", curl=("middle", "ring"), offset=(0.3, 0))]) == "lasers"


def test_heart_rule():
    a, b = hand("None"), hand("None", offset=(0.3, 0))
    assert not is_heart(a, b)
    # bring index tips together above thumb tips
    a.lm[8] = b.lm[8] = np.array([0.5, 0.3])
    a.lm[4] = np.array([0.49, 0.5]); b.lm[4] = np.array([0.51, 0.5])
    assert is_heart(a, b)
    assert reaction_for([a, b]) == "hearts"
