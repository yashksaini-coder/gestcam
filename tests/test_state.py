import pytest

from gestcam.state import SUBSET, ReactionState

RATES = (10, 30, 60)


def feed(st, seq, hz=30):
    """seq: (candidate, seconds[, n_hands]) segments. Returns (fires, ring), fires as
    (t, reaction) and ring as (t, current, progress) per update, read after the update
    like the pipeline does. Time continues across calls on the same state object;
    t is k/hz so long runs don't drift."""
    k, fires, ring = getattr(st, "_test_k", 0), [], []
    for cand, secs, *nh in seq:
        for _ in range(round(secs * hz)):
            t = k / hz
            f = st.update(cand, t, *nh)
            if f:
                fires.append((t, f))
            ring.append((t, st.current, st.progress(t)))
            k += 1
    st._test_k = k
    return fires, ring


def check(fires, expected, hz):
    """Same reactions in order, each within one update period of its expected time."""
    assert [f for _, f in fires] == [f for _, f in expected], fires
    for (t, _), (want, _) in zip(fires, expected, strict=True):
        assert want - 1e-6 <= t <= want + 1 / hz + 1e-6, (t, want)


def assert_full_ring_before_every_fire(fires, ring, hz, hold_s=0.7):
    """T3/T5: the ring shows the reaction that is about to fire, climbs from ~0 to
    ~1 for it, and drops to 0 on the fire itself."""
    idx = {t: i for i, (t, *_) in enumerate(ring)}
    for t, f in fires:
        i = idx[t]
        assert ring[i][2] == 0.0
        _, cur, p = ring[i - 1]
        if cur == SUBSET.get(f):  # promotion: the deferred subset's ring was this hold
            assert p >= (hold_s - 1.01 / hz) / (hold_s + 0.15), ring[i - 1]
        else:
            assert cur == f and p >= 1 - 1.01 / (hz * hold_s), ring[i - 1]
        j = i - 1  # walk back over this hold (a promotion starts it on the subset)
        while j > 0 and ring[j - 1][1] in (f, SUBSET.get(f)) and ring[j - 1][2] <= ring[j][2]:
            j -= 1
        assert ring[j][2] <= 1.01 / (hz * hold_s), (t, ring[j])


# --- existing behaviour --------------------------------------------------------------

def test_fires_only_after_hold():
    st = ReactionState(hold_s=0.7, cooldown_s=2.5)
    fired, _ = feed(st, [("thumbs_up", 0.5)])
    assert fired == []
    fired, _ = feed(st, [("thumbs_up", 1.0)])
    assert [f for _, f in fired] == ["thumbs_up"]


def test_flicker_does_not_reset_hold():
    st = ReactionState(hold_s=0.7, cooldown_s=2.5, grace_s=0.2)
    # 0.4s held, one 33ms dropout, 0.45s held → fires
    fired, _ = feed(st, [("balloons", 0.4), (None, 1 / 30), ("balloons", 0.45)])
    assert [f for _, f in fired] == ["balloons"]


def test_long_dropout_resets():
    st = ReactionState(hold_s=0.7, cooldown_s=2.5, grace_s=0.2)
    fired, _ = feed(st, [("balloons", 0.5), (None, 0.5), ("balloons", 0.5)])
    assert fired == []


def test_cooldown_and_rearm():
    st = ReactionState(hold_s=0.5, cooldown_s=2.0, release_s=0.3)
    # hold for 4s: fires once, does NOT re-fire while still held
    fired, _ = feed(st, [("confetti", 4.0)])
    assert [f for _, f in fired] == ["confetti"]
    # release then hold again → fires again
    fired, _ = feed(st, [(None, 0.5), ("confetti", 1.0)])
    assert [f for _, f in fired] == ["confetti"]


def test_different_reaction_needs_only_cooldown():
    # heart is no superset of thumbs_up, so it waits out the cooldown, then holds
    st = ReactionState(hold_s=0.5, cooldown_s=1.0)
    fired, ring = feed(st, [("thumbs_up", 0.6), ("heart", 1.6, 2)])
    check(fired, [(0.5, "thumbs_up"), (1.5 + 0.5, "heart")], 30)
    assert_full_ring_before_every_fire(fired, ring, 30, hold_s=0.5)


# --- sim_state.py (a)-(f), at every update rate --------------------------------------

@pytest.mark.parametrize("hz", RATES)
def test_a_second_thumb_within_hold_promotes(hz):
    # T1: fireworks 0.7 s after the FIRST thumb, no thumbs_up
    fired, ring = feed(ReactionState(), [("thumbs_up", 0.4), ("fireworks", 4.0, 2)], hz)
    check(fired, [(0.7, "fireworks")], hz)
    assert_full_ring_before_every_fire(fired, ring, hz)


@pytest.mark.parametrize("hz", RATES)
def test_b_late_second_thumb_two_hands_in_frame_fires_once(hz):
    # deferral: the second hand is in frame, so thumbs_up waits hold+defer (0.85 s);
    # the second thumb arrives at 0.75 s and inherits the hold → fireworks right away
    st = ReactionState()
    fired, ring = feed(st, [("thumbs_up", 0.75, 2), ("fireworks", 4.0, 2)], hz)
    check(fired, [(round(0.75 * hz) / hz, "fireworks")], hz)
    assert_full_ring_before_every_fire(fired, ring, hz)


@pytest.mark.parametrize("hz", RATES)
def test_b_late_second_thumb_one_hand_upgrades_during_cooldown(hz):
    # one hand in frame: thumbs_up fires at hold_s with no added latency; the second
    # thumb then gets a full hold of its own and upgrades inside the cooldown
    start = round(0.75 * hz) / hz
    fired, ring = feed(ReactionState(), [("thumbs_up", 0.75), ("fireworks", 4.0, 2)], hz)
    check(fired, [(0.7, "thumbs_up"), (start + 0.7, "fireworks")], hz)
    assert fired[1][0] < fired[0][0] + 2.5  # inside thumbs_up's cooldown
    assert_full_ring_before_every_fire(fired, ring, hz)  # T5: glyph is fireworks


@pytest.mark.parametrize("hz", RATES)
def test_c_superset_flicker_fires_superset(hz):
    # T2: second hand's label hovering at min_score → 50/50 thumbs_up/fireworks
    seq = [("thumbs_up" if i % 2 == 0 else "fireworks", 1 / hz, 2) for i in range(4 * hz)]
    fired, ring = feed(ReactionState(), seq, hz)
    check(fired, [(0.7, "fireworks")], hz)
    assert_full_ring_before_every_fire(fired, ring, hz)


@pytest.mark.parametrize("hz", RATES)
def test_d_tracking_dropout_does_not_rearm(hz):
    # T4: a 0.34 s dropout is shorter than release_s; the held pose fires once
    seq = [("thumbs_up", 1.0), (None, 0.34), ("thumbs_up", 3.5)]
    fired, _ = feed(ReactionState(), seq, hz)
    check(fired, [(0.7, "thumbs_up")], hz)


@pytest.mark.parametrize("hz", RATES)
def test_e_rehold_during_cooldown_shows_full_ring(hz):
    # T3: released for 0.6 s (> release_s) and re-held through the cooldown: nothing
    # accrues until 3.2 s, then a full visible hold → fires at 3.9 s
    st = ReactionState()
    fired, ring = feed(st, [("confetti", 0.8, 2), (None, 0.6), ("confetti", 3.0, 2)], hz)
    check(fired, [(0.7, "confetti"), (3.9, "confetti")], hz)
    assert all(p == 0.0 for t, _, p in ring if 0.7 <= t < 3.2)
    assert_full_ring_before_every_fire(fired, ring, hz)


@pytest.mark.parametrize("hz", RATES)
def test_e_short_release_does_not_rearm(hz):
    seq = [("confetti", 0.8, 2), (None, 0.4), ("confetti", 3.0, 2)]
    fired, _ = feed(ReactionState(), seq, hz)
    check(fired, [(0.7, "confetti")], hz)


@pytest.mark.parametrize("hz", RATES)
def test_f_latency_fresh_and_after_other_reaction(hz):
    fired, _ = feed(ReactionState(), [(None, 0.5), ("fireworks", 2.0, 2)], hz)
    check(fired, [(1.2, "fireworks")], hz)
    # balloons is not fireworks' subset: no promotion, plain hold from 0.3 s
    fired, _ = feed(ReactionState(), [("balloons", 0.3), ("fireworks", 2.0, 2)], hz)
    check(fired, [(1.0, "fireworks")], hz)


# --- new rules -----------------------------------------------------------------------

@pytest.mark.parametrize("hz", RATES)
def test_deferral_only_with_two_hands(hz):
    fired, _ = feed(ReactionState(), [("balloons", 2.0, 1)], hz)
    check(fired, [(0.7, "balloons")], hz)
    fired, ring = feed(ReactionState(), [("balloons", 2.0, 2)], hz)
    check(fired, [(0.85, "balloons")], hz)
    assert_full_ring_before_every_fire(fired, ring, hz)


@pytest.mark.parametrize("hz", RATES)
def test_cooldown_blocks_everything_but_the_upgrade(hz):
    # rain is not thumbs_up's superset; it waits for the cooldown (ends 3.2 s)
    seq = [("thumbs_up", 0.8), ("rain", 4.0, 2)]
    fired, ring = feed(ReactionState(), seq, hz)
    check(fired, [(0.7, "thumbs_up"), (3.9, "rain")], hz)
    assert_full_ring_before_every_fire(fired, ring, hz)
    # after an upgrade nothing more fires in the new cooldown, not even a re-upgrade
    seq = [("thumbs_up", 0.8), ("fireworks", 1.0, 2), ("thumbs_up", 1.0), ("heart", 1.0, 2)]
    fired, _ = feed(ReactionState(), seq, hz)
    assert [f for _, f in fired] == ["thumbs_up", "fireworks"]


@pytest.mark.parametrize("hz", RATES)
@pytest.mark.parametrize("n_hands", (1, 2))
def test_held_superset_flicker_never_leaks_subset(hz, n_hands):
    # fireworks fired; the pose stays up with the second hand flickering (and maybe
    # dropping out of detection entirely) for 6 s: no thumbs_up, no stuck ring
    seq = [("fireworks", 1.0, 2)]
    seq += [("thumbs_up", 1 / hz, n_hands) if i % 2 else ("fireworks", 1 / hz, 2) for i in range(6 * hz)]
    st = ReactionState()
    fired, ring = feed(st, seq, hz)
    check(fired, [(0.7, "fireworks")], hz)
    assert all(cur is None for t, cur, _ in ring if t > 3.3), [r for r in ring if r[1]][-1:]


@pytest.mark.parametrize("hz", RATES)
def test_none_flicker_within_grace_keeps_hold(hz):
    # every other observation missing (dropouts of one period <= grace): the hold
    # never resets, it just accrues on the frames that see the pose
    seq = [("heart" if i % 2 == 0 else None, 1 / hz, 2) for i in range(3 * hz)]
    fired, _ = feed(ReactionState(), seq, hz)
    assert [f for _, f in fired] == ["heart"]


def test_stall_credits_at_most_max_dt():
    st = ReactionState()
    st.update("heart", 0.0)
    assert st.update("heart", 1.0) is None  # 1 s gap counts as 0.25 s
    assert st.progress(1.0) == pytest.approx(0.25 / 0.7)
    assert st.current == "heart"


def test_idle_has_no_ring():
    st = ReactionState()
    feed(st, [(None, 1.0)])
    assert st.current is None and st.progress(1.0) == 0.0


@pytest.mark.parametrize("hz", RATES)
def test_flickering_hand_count_keeps_deferral_and_ring(hz):
    # an idle second hand detected on every other frame (n_hands 2,1,2,...): the
    # deferral holds through the n=1 frames, so the ring climbs without sawtoothing
    # and thumbs_up fires at hold_s + defer_s from a full ring
    seq = [("thumbs_up", 1 / hz, 2 if i % 2 == 0 else 1) for i in range(2 * hz)]
    fired, ring = feed(ReactionState(), seq, hz)
    check(fired, [(0.85, "thumbs_up")], hz)
    assert_full_ring_before_every_fire(fired, ring, hz)
    hold = [p for t, _, p in ring if t < fired[0][0]]
    assert hold == sorted(hold), hold


@pytest.mark.parametrize("hz", RATES)
def test_second_hand_leaving_does_not_skip_deferral(hz):
    # idle second hand in frame for 0.75 s of the hold, then gone: still a 0.85 s hold
    # (the two-hands flag outlives the last n=2 frame by grace_s > defer_s)
    fired, ring = feed(ReactionState(), [("thumbs_up", 0.75, 2), ("thumbs_up", 1.0, 1)], hz)
    check(fired, [(0.85, "thumbs_up")], hz)
    assert_full_ring_before_every_fire(fired, ring, hz)


def test_backwards_timestamp_is_dropped():
    # a stale observation (0.2 after 0.5) must not rewind the clock and let the next
    # interval re-credit time already counted
    st = ReactionState()
    ts = [i / 30 for i in range(16)] + [0.2] + [i / 30 for i in range(16, 40)]
    fires = [t for t in ts if st.update("thumbs_up", t)]
    assert fires == [pytest.approx(0.7)]
