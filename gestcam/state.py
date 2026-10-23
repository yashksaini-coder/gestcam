"""Reaction trigger state machine.

Why this exists: the classifier emits a candidate every ~30 ms and flickers. Feeding
that straight into the effects engine means confetti every time you scratch your nose.
Apple's "hold the gesture and pause for a moment" is a hold timer + cooldown.

Feed it once per NEW observation, with the observation's capture timestamp as `now`
(feeding a stale observation again would count the same evidence twice). A `now`
older than the previous call's is dropped: rewinding the clock would let the next
interval re-credit time already counted (0.5, 0.2, 0.533 fired 167 ms early). dt is
`now` minus the previous call's `now`, clamped to [0, 0.25] so a stall can't fire
anything by itself. Everything is time-based, so 10, 30 and 60 Hz feeds fire at the
same moment to within one update period.

Rules:

* Hold: each reaction has its own accumulator. It grows by dt while that reaction is
  seen, is frozen while it has been unseen for <= `grace_s` (a missed frame or a
  flicker to another label is not a release), and decays at 2*dt after that. A
  reaction fires when its accumulator reaches `hold_s`. Per-reaction accumulators
  replaced an incumbent/challenger rule that let a 50/50 A,B,A,B flicker always
  resolve to whichever label came first.
* Supersets (`SUPERSET`): a two-hand reaction contains its one-hand subset, and a
  subset frame within `grace_s` of a superset frame is the second hand flickering,
  so it counts as the superset too. A 50/50 thumbs_up/fireworks alternation
  therefore fires fireworks.
* Promotion: while a superset is seen its accumulator is max(own, subset's), so
  thumbs-up then a second thumb fires fireworks `hold_s` after the FIRST thumb.
* Deferral: with two or more hands in frame, a one-hand subset needs `hold_s +
  defer_s` and its superset unseen (outside grace) before it fires: the second hand
  may be about to join. With one hand in frame it fires at `hold_s`, so one-handed
  use pays no latency. "Two hands" lasts `grace_s` past the last n_hands >= 2 update:
  n_hands counts every detected hand, an idle one flickers in and out, and keying the
  threshold on the latest update alone made the ring sawtooth (acc/0.85, acc/0.7, ...)
  and fire on the first n=1 frame from ~0.78. grace_s > defer_s, so a second hand
  leaving can't cut the deferral short either.
* Cooldown: after a fire, every accumulator is held at 0 for `cooldown_s`, so each
  fire is preceded by a full visible hold ring. One exception, the upgrade: if a
  subset fired, its superset may still be held to `hold_s` and fire during that
  cooldown (the effects engine retires the subset's effect). Nothing else fires.
* Repeat: the reaction that fired last re-fires only after a release (no candidate)
  lasting `release_s`, measured on the timestamps fed in. release_s = 0.5 is grace
  plus about one inference period, so a tracking dropout doesn't re-arm a held pose.
* `current` is the eligible reaction with the highest accumulator (the ring's glyph);
  `progress` is its accumulator over its threshold. Ineligible reactions sit at 0,
  so the ring never previews something that can't fire.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# two-hand reaction for each one-hand reaction it contains
SUPERSET = {"thumbs_up": "fireworks", "thumbs_down": "rain", "balloons": "confetti"}
SUBSET = {v: k for k, v in SUPERSET.items()}

MAX_DT = 0.25


@dataclass
class ReactionState:
    hold_s: float = 0.7
    cooldown_s: float = 2.5
    grace_s: float = 0.2
    release_s: float = 0.5
    defer_s: float = 0.15

    _acc: dict = field(default_factory=dict, init=False)        # reaction -> held seconds
    _last_seen: dict = field(default_factory=dict, init=False)  # reaction -> last true sighting
    _prev: float | None = field(default=None, init=False)       # previous update's now
    _two_seen: float = field(default=float("-inf"), init=False)  # last n_hands >= 2
    _cooldown_until: float = field(default=float("-inf"), init=False)
    _last_fired: str | None = field(default=None, init=False)
    _armed: bool = field(default=True, init=False)      # may _last_fired fire again?
    _released_since: float | None = field(default=None, init=False)
    _current: str | None = field(default=None, init=False)

    @property
    def current(self) -> str | None:
        """Reaction the hold ring should preview."""
        return self._current

    def progress(self, now: float) -> float:
        """0..1 hold progress of `current`. Reflects the last update; `now` is kept
        for the caller's signature (the ring is drawn every frame, updates come per
        observation)."""
        if self._current is None:
            return 0.0
        return min(1.0, self._acc[self._current] / self._threshold(self._current))

    def _threshold(self, r: str) -> float:
        if r in SUPERSET and self._prev - self._two_seen <= self.grace_s:
            return self.hold_s + self.defer_s
        return self.hold_s

    def _eligible(self, r: str, now: float) -> bool:
        if now < self._cooldown_until:
            return SUPERSET.get(self._last_fired) == r  # the upgrade path
        return r != self._last_fired or self._armed

    def _live(self, r: str | None) -> bool:
        """Was r seen within grace of the previous update (does the interval count)?"""
        seen = self._last_seen.get(r)
        return seen is not None and self._prev is not None and self._prev - seen <= self.grace_s

    def update(self, candidate: str | None, now: float, n_hands: int = 1) -> str | None:
        """Feed one observation; returns the reaction to fire, or None."""
        if self._prev is not None and now < self._prev:
            return None  # out of order: drop it (equal timestamps are fine, dt = 0)
        dt = 0.0 if self._prev is None else min(now - self._prev, MAX_DT)
        if n_hands >= 2:
            self._two_seen = now

        if candidate is None:
            if self._released_since is None:
                self._released_since = now
            if now - self._released_since >= self.release_s:
                self._armed = True
        else:
            self._released_since = None

        seen = set() if candidate is None else {candidate}
        sup = SUPERSET.get(candidate)
        if sup is not None and now - self._last_seen.get(sup, float("-inf")) <= self.grace_s:
            seen.add(sup)  # second-hand flicker: still the two-hand pose

        old = dict(self._acc)  # promotion reads pre-update values: order-independent
        for r in sorted(old.keys() | seen):
            if not self._eligible(r, now):
                self._acc[r] = 0.0
                continue
            acc = old.get(r, 0.0)
            if SUPERSET.get(r) in seen:
                pass  # subset frame inside superset flicker: the superset's evidence, so
                # a held pose can't sneak its subset out after the superset fired
            elif r in seen:
                sub = SUBSET.get(r)
                if sub is not None:
                    acc = max(acc, old.get(sub, 0.0))  # promotion (0 if sub ineligible)
                # credit the interval only if the pose was already up at the previous
                # update, and only the part after the cooldown, so a hold measures from
                # the first sighting (or the cooldown's end) identically at any rate
                if self._live(r) or (sub is not None and self._live(sub) and self._eligible(sub, now)):
                    acc += dt if now < self._cooldown_until else min(dt, now - self._cooldown_until)
            elif now - self._last_seen.get(r, float("-inf")) > self.grace_s:
                acc = max(0.0, acc - 2 * dt)
            self._acc[r] = acc

        if candidate is not None:
            self._last_seen[candidate] = now
        self._prev = now

        fired = None
        for r in sorted(seen, key=lambda r: r not in SUBSET):  # superset first
            # ineligible reactions sit at 0, so this also covers cooldown and repeat
            if self._acc[r] >= self._threshold(r) - 1e-9:  # summed dts lose an ulp
                fired = r
                break
        if fired is not None:
            self._cooldown_until = now + self.cooldown_s
            self._last_fired, self._armed = fired, False
            self._acc = dict.fromkeys(self._acc, 0.0)

        eligible = [r for r, a in self._acc.items() if a > 0 and self._eligible(r, now)]
        self._current = max(eligible, key=lambda r: (self._acc[r], r in SUBSET), default=None)
        return fired
