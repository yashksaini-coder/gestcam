"""Reaction trigger state machine.

Why this exists: the classifier emits a candidate every ~30 ms and flickers. Feeding
that straight into the effects engine means confetti every time you scratch your nose.
Apple's "hold the gesture and pause for a moment" is exactly a hold timer + cooldown.

Rules (all time-based, so they behave the same at 10 fps and 60 fps):

* A candidate must be held continuously for `hold_s` before it fires.
* Brief dropouts (model misses a frame) shorter than `grace_s` don't reset the hold.
* A *different* candidate for less than `grace_s` doesn't reset it either: a two-hand
  reaction dropping to its single-hand subset for a frame is flicker, not a new pose.
* After firing, nothing fires for `cooldown_s`.
* The *same* reaction won't re-fire until the hands have been released (no candidate)
  for `release_s`; a *different* reaction may fire once the cooldown has passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReactionState:
    hold_s: float = 0.7
    cooldown_s: float = 2.5
    grace_s: float = 0.2
    release_s: float = 0.3

    _current: str | None = field(default=None, init=False)
    _since: float = field(default=0.0, init=False)      # when _current started
    _last_seen: float = field(default=0.0, init=False)  # last time _current was observed
    _cooldown_until: float = field(default=0.0, init=False)
    _last_fired: str | None = field(default=None, init=False)
    _released_since: float | None = field(default=None, init=False)
    _armed_for_repeat: bool = field(default=True, init=False)
    _challenger: str | None = field(default=None, init=False)
    _challenger_since: float = field(default=0.0, init=False)

    @property
    def current(self) -> str | None:
        return self._current

    def progress(self, now: float) -> float:
        """0..1 hold progress of the current candidate (for on-screen feedback)."""
        if self._current is None or now < self._cooldown_until:
            return 0.0
        if self._current == self._last_fired and not self._armed_for_repeat:
            return 0.0
        return max(0.0, min(1.0, (now - self._since) / self.hold_s))

    def update(self, candidate: str | None, now: float) -> str | None:
        """Feed one classification; returns the reaction to fire, or None."""
        if candidate is None:
            # Release timer counts from the first missed frame...
            if self._released_since is None:
                self._released_since = now
            if now - self._released_since >= self.release_s:
                self._armed_for_repeat = True
            # ...but the hold survives brief dropouts (flicker guard).
            if self._current is not None and now - self._last_seen <= self.grace_s:
                return None
            self._current = None
            return None

        self._released_since = None
        if candidate != self._current:
            # A different label mid-hold is usually classifier flicker, not a new pose:
            # a two-hand reaction dropping to its single-hand subset for one frame
            # (fireworks<->thumbs_up, confetti<->balloons, rain<->thumbs_down). A
            # newcomer has to out-persist the flicker window before it evicts the
            # incumbent; a genuinely sustained new gesture still wins after grace_s.
            # ponytail: a true 50/50 A,B,A,B alternation never lets B win, because A
            # keeps refreshing _last_seen. Upgrade to a per-candidate leaky
            # accumulator only if that is ever actually observed.
            if self._current is not None and now - self._last_seen <= self.grace_s:
                if candidate != self._challenger:
                    self._challenger, self._challenger_since = candidate, now
                if now - self._challenger_since < self.grace_s:
                    return None          # keep the incumbent's hold alive
            self._current = candidate
            self._since = now
        self._challenger = None
        self._last_seen = now

        if now < self._cooldown_until:
            return None
        if now - self._since < self.hold_s:
            return None
        if candidate == self._last_fired and not self._armed_for_repeat:
            return None

        # Fire.
        self._cooldown_until = now + self.cooldown_s
        self._last_fired = candidate
        self._armed_for_repeat = False
        self._since = now  # require a fresh hold for anything further
        return candidate
