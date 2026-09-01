from gestcam.state import ReactionState


def feed(st, seq, dt=1 / 30):
    """seq: list of (candidate, seconds) segments. Returns list of (t, fired).
    Time continues across calls on the same state object."""
    t, out = getattr(st, "_test_t", 0.0), []
    for cand, secs in seq:
        n = int(secs / dt)
        for _ in range(n):
            f = st.update(cand, t)
            if f:
                out.append((round(t, 2), f))
            t += dt
    st._test_t = t
    return out


def test_fires_only_after_hold():
    st = ReactionState(hold_s=0.7, cooldown_s=2.5)
    fired = feed(st, [("thumbs_up", 0.5)])
    assert fired == []
    fired = feed(st, [("thumbs_up", 1.0)])
    assert [f for _, f in fired] == ["thumbs_up"]


def test_flicker_does_not_reset_hold():
    st = ReactionState(hold_s=0.7, cooldown_s=2.5, grace_s=0.2)
    # 0.4s held, one 33ms dropout, 0.4s held → should fire (total 0.8s continuous-ish)
    fired = feed(st, [("balloons", 0.4), (None, 1 / 30), ("balloons", 0.45)])
    assert [f for _, f in fired] == ["balloons"]


def test_long_dropout_resets():
    st = ReactionState(hold_s=0.7, cooldown_s=2.5, grace_s=0.2)
    fired = feed(st, [("balloons", 0.5), (None, 0.5), ("balloons", 0.5)])
    assert fired == []


def test_cooldown_and_rearm():
    st = ReactionState(hold_s=0.5, cooldown_s=2.0, release_s=0.3)
    # hold for 4s: fires once, does NOT re-fire while still held
    fired = feed(st, [("confetti", 4.0)])
    assert [f for _, f in fired] == ["confetti"]
    # release briefly then hold again → fires again
    fired = feed(st, [(None, 0.5), ("confetti", 1.0)])
    assert [f for _, f in fired] == ["confetti"]


def test_different_reaction_needs_only_cooldown():
    st = ReactionState(hold_s=0.5, cooldown_s=1.0)
    fired = feed(st, [("thumbs_up", 0.6), ("fireworks", 0.6), ("fireworks", 1.0)])
    names = [f for _, f in fired]
    assert names == ["thumbs_up", "fireworks"]
    # second fired at least cooldown after first
    assert fired[1][0] - fired[0][0] >= 1.0
