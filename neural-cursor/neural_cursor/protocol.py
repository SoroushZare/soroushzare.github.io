"""Session protocol: which blocks a person runs each day, and how directions are cued.

Why a protocol module at all
----------------------------
The data engine can only be as honest as the labels it is fed.  Labels from a closed loop are
contaminated by construction: the person sees the cursor, so a decoder can "work" by reading the
visual/oculomotor response to the cursor instead of intent (feedback circularity), and the goal is
drawn at a spatial position, so the label is display-identifiable (eyes move toward it).  Two block
kinds exist only to keep the system checkable while it grows:

``cued_blank``      the direction is cued by a spoken word or a CENTRAL symbol; the screen shows only
                    a central fixation mark, no cursor and nothing at any spatial position, and there
                    is no feedback.  The label is defined by the cue alone, so it cannot be recovered
                    from the display or from gaze toward a target.  The data engine's champion test
                    uses the most recent of these windows as its frozen held-out set.
``blank_feedback``  the decoder drives the cursor exactly as in ``closed_loop`` but the cursor is
                    hidden (its end position is revealed only after the trial, i.e. delayed feedback).
                    Comparing accuracy here against the adjacent visible-cursor block is the purity
                    audit's FEEDBACK_CIRCULAR probe.

Every session therefore keeps at least ``MIN_CUED_FRACTION`` of its task time in ``cued_blank`` and at
least one ``blank_feedback`` probe, however good the decoder has become: these are the validity data,
and a learning curve without them cannot distinguish "the decoder improved" from "the decoder learned
the screen".

The remaining block kinds: ``setup`` (impedance check and an eyes-open / eyes-closed rest baseline),
``closed_loop`` (cursor and goal visible, the bulk of training data once a decoder exists),
``free_use`` (closed loop on Zipf-distributed targets, the target statistics of real use) and ``rest``
(breaks; fatigue and dry-electrode drift both grow without them).

Coordinates are normalised screen units in [-1, 1] with +x right and +y up, as in ``task.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------------------------
# Block vocabulary
# ----------------------------------------------------------------------------------------------
BLOCK_KINDS: Tuple[str, ...] = ("setup", "cued_blank", "closed_loop", "blank_feedback", "free_use", "rest")
CUE_MODES: Tuple[str, ...] = ("auditory", "central_symbol", "target", "none")
"""``auditory`` / ``central_symbol`` are the two non-spatial cues allowed in ``cued_blank``;
``target`` means the goal is drawn at its screen position (closed-loop kinds); ``none`` is for blocks
without a direction (setup, rest)."""
FEEDBACK_MODES: Tuple[str, ...] = ("none", "cursor", "delayed")

PROVENANCE_BLOCK: Dict[str, Optional[str]] = {
    "setup": None, "rest": None,                       # no direction, never labelled
    "cued_blank": "cued_blank", "closed_loop": "closed_loop",
    "blank_feedback": "blank_feedback", "free_use": "free",
}
"""Protocol kind -> ``data_engine.BLOCK_KINDS`` name written into ``trial_start`` events."""

_DEFAULT_CUE: Dict[str, str] = {
    "setup": "none", "rest": "none",
    # Auditory is the default for cued_blank: it puts nothing on the screen at all, whereas even a
    # central arrow is a symbolic cue known to shift covert attention (and small saccades) toward
    # the side it points to.
    "cued_blank": "auditory",
    "closed_loop": "target", "blank_feedback": "target", "free_use": "target",
}
_ALLOWED_CUES: Dict[str, Tuple[str, ...]] = {
    "setup": ("none",), "rest": ("none",),
    "cued_blank": ("auditory", "central_symbol"),
    "closed_loop": ("target",), "blank_feedback": ("target",), "free_use": ("target",),
}
_FEEDBACK: Dict[str, str] = {
    "setup": "none", "rest": "none", "cued_blank": "none",
    "closed_loop": "cursor", "free_use": "cursor",
    "blank_feedback": "delayed",                      # cursor hidden; end position shown after the trial
}


@dataclass(frozen=True)
class ScreenItem:
    """One thing drawn on the screen during a block.

    ``spatial`` is True when the item's position depends on the trial's direction or goal (a target,
    the cursor).  A spatial item makes the label recoverable from the display or from gaze, which is
    exactly what ``cued_blank`` must never contain.
    """
    name: str
    position: Optional[Tuple[float, float]]   # None: varies with the trial (target, cursor)
    spatial: bool


FIXATION = ScreenItem("fixation_cross", (0.0, 0.0), spatial=False)


@dataclass(frozen=True)
class Block:
    """One protocol block.

    ``cue`` and ``feedback`` default to the only (or safest) value for the kind and are validated
    against it, so an impossible block -- a cued_blank with a visible cursor, a closed_loop without
    one -- cannot be constructed and silently mislabel a session's data.
    ``seed`` seeds whatever the block randomises (cue order, target draws) so a session can be
    replayed exactly from its plan.
    """
    kind: str
    duration_s: float
    cue: Optional[str] = None
    feedback: Optional[str] = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in BLOCK_KINDS:
            raise ValueError(f"unknown block kind {self.kind!r}; expected one of {BLOCK_KINDS}")
        if not (self.duration_s > 0):
            raise ValueError(f"{self.kind}: duration_s must be > 0, got {self.duration_s}")
        cue = _DEFAULT_CUE[self.kind] if self.cue is None else self.cue
        feedback = _FEEDBACK[self.kind] if self.feedback is None else self.feedback
        if cue not in _ALLOWED_CUES[self.kind]:
            raise ValueError(f"{self.kind}: cue {cue!r} not allowed; use one of {_ALLOWED_CUES[self.kind]}")
        if feedback != _FEEDBACK[self.kind]:
            raise ValueError(f"{self.kind}: feedback must be {_FEEDBACK[self.kind]!r}, got {feedback!r}")
        object.__setattr__(self, "cue", cue)
        object.__setattr__(self, "feedback", feedback)

    @property
    def feedback_visible(self) -> bool:
        """Label provenance flag: was the cursor drawn while the person formed the intent."""
        return self.feedback == "cursor"

    @property
    def provenance_block(self) -> Optional[str]:
        """The ``block`` name the data engine expects, or None for blocks that are never labelled."""
        return PROVENANCE_BLOCK[self.kind]

    @property
    def is_validity(self) -> bool:
        """True for the held-out validity blocks that every session must contain."""
        return self.kind in ("cued_blank", "blank_feedback")

    def screen_items(self, phase: Optional[str] = None) -> Tuple[ScreenItem, ...]:
        """Everything drawn during the block (the renderer's contract, and what tests audit).

        ``phase`` narrows it to one part of a cued_blank trial: ``"cue"`` (cue onset to ``cue_offset``,
        ``CuedTrialTiming.cue_s``) or ``"intent"`` (the labelled span from ``trial_start``).  The
        central symbol is drawn during the cue phase ONLY: its glyph maps one-to-one onto the label,
        so a symbol left on screen into the intent phase would make the champion's test labels
        readable from the display.  ``None`` is the whole block (every phase)."""
        if phase not in (None, "cue", "intent"):
            raise ValueError(f"phase must be None, 'cue' or 'intent', got {phase!r}")
        if self.kind == "cued_blank":
            if self.cue == "central_symbol" and phase != "intent":
                return (FIXATION, ScreenItem("direction_symbol", (0.0, 0.0), spatial=False))
            return (FIXATION,)
        if self.kind in ("closed_loop", "free_use"):
            return (ScreenItem("targets", None, True), ScreenItem("cursor", None, True))
        if self.kind == "blank_feedback":
            # The cursor is absent during the trial; its end position is drawn only after trial_end.
            return (ScreenItem("targets", None, True),)
        return (FIXATION,)                                  # setup baseline and rest: eyes on centre


# ----------------------------------------------------------------------------------------------
# Session plan
# ----------------------------------------------------------------------------------------------
TASK_MIN = 45                     # task minutes per session (excl. setup and rest); ~1 h with breaks
SETUP_MIN = 5                     # impedance check + 60 s eyes-open + 60 s eyes-closed baseline
REST_MIN = 1                      # break between consecutive work units
MIN_CUED_FRACTION = 0.20          # floor on cued_blank share of task time, every session, forever
DAY1_CUED_FRACTION = 0.70         # day 1 is mostly calibration: there is no decoder to close a loop with
CUED_DECAY_DAYS = 3.0             # e-folding time of the cued share toward its floor
PROBE_FRACTION = 0.10             # blank_feedback share of task time (>= 1 block always)
FREE_START_DAY = 3                # free use begins once closed-loop control is established
FREE_STEP = 0.05                  # free-use share added per day from FREE_START_DAY
FREE_MAX_FRACTION = 0.30          # the rest stays structured, so labels stay goal-defined
MAX_BLOCK_MIN: Dict[str, int] = {  # longest single block per kind; attention decays within a block
    "cued_blank": 4, "closed_loop": 5, "blank_feedback": 5, "free_use": 6,
}


def cued_fraction(day: int) -> float:
    """Target cued_blank share of task time on ``day``: 0.70 on day 1 decaying toward 0.20.

    The decay moves time to closed_loop (more data per minute, the user's actual task) as the decoder
    becomes usable; the floor keeps a fresh cue-defined held-out set and a drift reference every day.
    """
    _check_day(day)
    span = DAY1_CUED_FRACTION - MIN_CUED_FRACTION
    return MIN_CUED_FRACTION + span * math.exp(-(day - 1) / CUED_DECAY_DAYS)


def free_fraction(day: int) -> float:
    """Free-use share of task time: zero until FREE_START_DAY, then growing linearly to a cap."""
    _check_day(day)
    if day < FREE_START_DAY:
        return 0.0
    return min(FREE_MAX_FRACTION, FREE_STEP * (day - FREE_START_DAY + 1))


def _check_day(day: int) -> None:
    if int(day) != day or day < 1:
        raise ValueError(f"day must be an integer >= 1, got {day!r}")


def _task_minutes(day: int) -> Dict[str, int]:
    """Whole task minutes per kind.  Validity kinds are rounded UP, so rounding can never take a
    session below its floors; closed_loop absorbs the remainder."""
    cued = math.ceil(cued_fraction(day) * TASK_MIN)
    probe = max(1, math.ceil(PROBE_FRACTION * TASK_MIN))
    free = math.floor(free_fraction(day) * TASK_MIN)
    closed = TASK_MIN - cued - probe - free
    if closed < probe:
        raise ValueError(f"day {day}: {closed} closed_loop min cannot host {probe} min of blank_feedback "
                         f"probes; the plan hyperparameters are inconsistent")
    return {"cued_blank": cued, "closed_loop": closed, "blank_feedback": probe, "free_use": free}


def _split(minutes: int, max_len: int) -> List[int]:
    """Split ``minutes`` into the fewest blocks of at most ``max_len``, lengths differing by <= 1."""
    if minutes <= 0:
        return []
    n = math.ceil(minutes / max_len)
    base, extra = divmod(minutes, n)
    return [base + 1] * extra + [base] * (n - extra)


def session_plan(day: int, seed: int) -> List[Block]:
    """Deterministic block list for one session (``day`` counts from 1 for each person).

    Order: ``setup``; then work units separated by 1 min ``rest``.  A work unit is a single block,
    except that every ``blank_feedback`` probe immediately follows a ``closed_loop`` block with no
    rest in between, so the visible/hidden comparison is made with the same decoder at the same
    fatigue and electrode state.  The session opens with a ``cued_blank`` unit (a fresh drift
    reference before any decoder is trusted).  On day 1 every cued_blank unit comes first -- there
    is no decoder until they are recorded.  From day 2 the cued_blank units are spread evenly over
    the session so the held-out set samples early, middle and late (fatigued) states, and the other
    units are shuffled with ``seed``.  cued_blank blocks alternate auditory / central_symbol cues so
    that a decoder which only learned one cue's evoked response does not pass the held-out test.
    """
    _check_day(day)
    rng = np.random.default_rng([int(seed), int(day)])
    minutes = _task_minutes(day)
    lengths = {kind: _split(m, MAX_BLOCK_MIN[kind]) for kind, m in minutes.items()}

    cued_units = [[("cued_blank", m)] for m in lengths["cued_blank"]]
    closed = list(lengths["closed_loop"])
    probes = list(lengths["blank_feedback"])
    other_units: List[List[Tuple[str, int]]] = []
    for i, m in enumerate(closed):
        unit = [("closed_loop", m)]
        if i < len(probes):
            unit.append(("blank_feedback", probes[i]))
        other_units.append(unit)
    other_units += [[("free_use", m)] for m in lengths["free_use"]]
    order = rng.permutation(len(other_units))
    other_units = [other_units[i] for i in order]

    if day == 1:
        units = cued_units + other_units
    else:
        n_total = len(cued_units) + len(other_units)
        slots = set(np.round(np.linspace(0, n_total - 1, len(cued_units))).astype(int).tolist())
        it_cued, it_other = iter(cued_units), iter(other_units)
        units = [next(it_cued) if k in slots else next(it_other) for k in range(n_total)]

    block_seeds = rng.integers(0, 2**31 - 1, size=sum(len(u) for u in units) + 1)
    plan = [Block("setup", SETUP_MIN * 60.0, seed=int(block_seeds[0]))]
    n_cued, k = 0, 1
    for u_idx, unit in enumerate(units):
        if u_idx > 0:
            plan.append(Block("rest", REST_MIN * 60.0))
        for kind, m in unit:
            cue = None
            if kind == "cued_blank":
                cue = ("auditory", "central_symbol")[n_cued % 2]
                n_cued += 1
            plan.append(Block(kind, m * 60.0, cue=cue, seed=int(block_seeds[k])))
            k += 1
    return plan


def plan_summary(plan: Sequence[Block]) -> Dict[str, float]:
    """Seconds per kind plus ``task_s`` (all labelled kinds) and ``cued_fraction`` of task time."""
    out: Dict[str, float] = {kind: 0.0 for kind in BLOCK_KINDS}
    for b in plan:
        out[b.kind] += b.duration_s
    out["task_s"] = sum(out[k] for k in BLOCK_KINDS if PROVENANCE_BLOCK[k] is not None)
    out["total_s"] = sum(b.duration_s for b in plan)
    out["cued_fraction"] = out["cued_blank"] / out["task_s"] if out["task_s"] else 0.0
    return out


def render_plan_markdown(plan: Sequence[Block], title: Optional[str] = None) -> str:
    """Markdown table of the plan with per-block minutes, start time and a totals footer."""
    lines = [f"### {title}", ""] if title else []
    lines += ["| # | Start (min) | Block | Minutes | Cue | Feedback | Cursor visible | Validity |",
              "|---:|---:|---|---:|---|---|---|---|"]
    t = 0.0
    for i, b in enumerate(plan, 1):
        lines.append(f"| {i} | {t / 60:.0f} | {b.kind} | {b.duration_s / 60:.1f} | {b.cue} | {b.feedback} "
                     f"| {'yes' if b.feedback_visible else 'no'} | {'yes' if b.is_validity else ''} |")
        t += b.duration_s
    s = plan_summary(plan)
    shares = ", ".join(f"{k} {s[k] / s['task_s']:.0%}" for k in BLOCK_KINDS
                       if PROVENANCE_BLOCK[k] is not None and s[k] > 0)
    lines += ["", f"Total {s['total_s'] / 60:.0f} min; task {s['task_s'] / 60:.0f} min ({shares})."]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Cues
# ----------------------------------------------------------------------------------------------
N_DIRECTIONS = 8
_NAMES = ("right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right")
_SYMBOLS = ("→", "↗", "↑", "↖", "←", "↙", "↓", "↘")


@dataclass(frozen=True)
class Cue:
    """One cued direction.  ``vector`` is the unit direction that defines the trial's label."""
    seq: int
    index: int
    name: str
    word: str                      # what the auditory cue speaks
    symbol: str                    # what the central_symbol cue shows at fixation
    vector: Tuple[float, float]

    def presentation(self, mode: str, timing: Optional["CuedTrialTiming"] = None) -> Dict[str, object]:
        """What the stimulus layer must present, and for how long.  Both modes are centred or
        off-screen: a cue that appeared at the direction's screen position would make the label
        display-identifiable.  ``duration_s`` (``timing.cue_s``) bounds the stimulus: the renderer
        removes the symbol (or ends the word) then and logs ``cue_offset``; from there to the end of
        the trial only the fixation mark is drawn (``after_offset``)."""
        duration = (timing or CuedTrialTiming()).cue_s
        if mode == "auditory":
            return {"audio": self.word, "screen": [FIXATION], "duration_s": duration,
                    "after_offset": [FIXATION]}
        if mode == "central_symbol":
            return {"audio": None,
                    "screen": [FIXATION, ScreenItem(f"symbol:{self.symbol}", (0.0, 0.0), spatial=False)],
                    "duration_s": duration, "after_offset": [FIXATION]}
        raise ValueError(f"cue mode {mode!r} is not a cued_blank cue; use 'auditory' or 'central_symbol'")


@dataclass
class CueScheduler:
    """Balanced, block-randomised sequence of the 8 directions.

    Each randomisation block holds every direction ``reps_per_block`` times, shuffled, so counts are
    exactly balanced at every block boundary and never differ by more than ``reps_per_block`` in
    between.  Two repetitions per block (16 cues) rather than one keeps the last cue of a block
    unpredictable: with single permutations a person who counts can name the eighth cue before it
    plays, and anticipation is a preparatory EEG signal the label did not intend.  A shuffle is
    redrawn whenever it would make ``max_run + 1`` identical cues in a row, counting across block
    boundaries, because long runs invite anticipation and adaptation.

    Every issued cue is appended to ``log`` and, when a ``logger`` (``RunLogger``-like
    ``event(kind, payload, t)``) is given, written as a ``"cue"`` event.
    """
    seed: int
    reps_per_block: int = 2
    max_run: int = 2
    max_redraws: int = 10_000          # rejection sampling; with max_run >= reps_per_block only a run
                                       # across the block boundary can be rejected, so almost every draw passes
    logger: Optional[object] = None
    log: List[Dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.reps_per_block < 1 or self.max_run < 1:
            raise ValueError("reps_per_block and max_run must be >= 1")
        self._rng = np.random.default_rng(self.seed)
        self._queue: List[int] = []
        self._history: List[int] = []

    def _valid(self, seq: Sequence[int]) -> bool:
        run, prev = 0, None
        for d in list(self._history[-self.max_run:]) + list(seq):
            run = run + 1 if d == prev else 1
            prev = d
            if run > self.max_run:
                return False
        return True

    def _refill(self) -> None:
        base = np.repeat(np.arange(N_DIRECTIONS), self.reps_per_block)
        for _ in range(self.max_redraws):
            perm = self._rng.permutation(base).tolist()
            if self._valid(perm):
                self._queue = perm
                return
        raise RuntimeError("could not draw a cue block satisfying max_run; relax max_run")

    def next(self, t: Optional[float] = None, mode: Optional[str] = None) -> Cue:
        """Issue and log the next cue; ``t`` is the cue onset on the session clock."""
        if not self._queue:
            self._refill()
        idx = self._queue.pop(0)
        self._history.append(idx)
        ang = 2 * math.pi * idx / N_DIRECTIONS
        cue = Cue(seq=len(self.log), index=idx, name=_NAMES[idx], word=_NAMES[idx].replace("-", " "),
                  symbol=_SYMBOLS[idx], vector=(round(math.cos(ang), 12), round(math.sin(ang), 12)))
        entry = {"seq": cue.seq, "index": idx, "direction": cue.name, "mode": mode, "t": t}
        self.log.append(entry)
        if self.logger is not None:
            self.logger.event("cue", entry, t)
        return cue

    def schedule(self, n: int, mode: Optional[str] = None) -> List[Cue]:
        """Issue ``n`` cues at once (offline planning / simulation; each is still logged)."""
        return [self.next(mode=mode) for _ in range(n)]


@dataclass(frozen=True)
class CuedTrialTiming:
    """Within-trial timing of a cued_blank trial (seconds).

    ``trial_start`` is logged at intent onset, after the cue and a settle period, so no labelled
    window contains the cue-evoked response: different spoken words and symbols evoke different
    sensory potentials that a decoder could otherwise use to "decode" the direction.

    Event order per trial (the contract ``data_engine`` checks): ``cue`` at onset → ``cue_offset``
    (``EVENT_CUE_OFFSET``) after ``cue_s``, when the symbol is removed / the word has ended →
    ``trial_start`` after ``settle_s`` more → ``trial_end`` after ``intent_s``.  A run that logs a
    ``cue`` but no ``cue_offset`` before the trial starts is rejected by the labeler: the stimulus may
    still have been on screen while the labelled intent was recorded.
    """
    cue_s: float = 1.0
    settle_s: float = 0.5
    intent_s: float = 4.0
    iti_s: float = 1.5

    @property
    def trial_s(self) -> float:
        return self.cue_s + self.settle_s + self.intent_s + self.iti_s

    def n_trials(self, block: Block) -> int:
        return int(block.duration_s // self.trial_s)


EVENT_CUE_OFFSET = "cue_offset"
"""Logged when the cue stimulus ends (symbol removed, word finished); see ``CuedTrialTiming``."""

CUED_GOAL_DISTANCE = 0.6
"""Virtual goal distance for cued trials: the data engine labels by the unit vector from the cursor
(at ``start_pos`` = centre, since no cursor is logged) to the goal, so only its direction matters;
0.6 keeps it far outside the engine's ``min_goal_dist`` dwell exclusion."""


def trial_start_payload(block: Block, trial: int, goal: Optional[Sequence[float]],
                        start_pos: Sequence[float] = (0.0, 0.0)) -> Dict[str, object]:
    """The ``trial_start`` event payload ``data_engine.Labeler`` expects, with provenance from the block.

    ``goal`` may be None only for free_use, whose goal is revealed by the selection at trial end.
    """
    if block.provenance_block is None:
        raise ValueError(f"{block.kind} blocks carry no trials")
    if goal is None and block.kind != "free_use":
        raise ValueError(f"{block.kind}: the goal must be known at trial start")
    return {"trial": int(trial), "goal": None if goal is None else [float(g) for g in goal],
            "block": block.provenance_block, "feedback_visible": block.feedback_visible,
            "start_pos": [float(p) for p in start_pos]}


def cued_goal(cue: Cue, distance: float = CUED_GOAL_DISTANCE) -> Tuple[float, float]:
    """Goal position that encodes a cue's direction for the labeler (never drawn on screen)."""
    return (distance * cue.vector[0], distance * cue.vector[1])


# ----------------------------------------------------------------------------------------------
# Free-use targets
# ----------------------------------------------------------------------------------------------
@dataclass
class ZipfTargets:
    """Fixed layout of target locations used with Zipf(``exponent``) frequencies.

    Real interfaces are used this way: a few elements (close button, scroll bar, a favourite app) take
    most clicks.  Free-use blocks reproduce that skew so the decoder meets the class imbalance of
    real use; the same imbalance is why free-use windows never serve as held-out data.  The layout and
    the rank assigned to each location are fixed by ``seed`` so a person's "frequent targets" stay
    where they were across days, as they would on a real desktop.
    """
    seed: int
    n_locations: int = 24
    exponent: float = 1.1
    min_center_dist: float = 0.3       # targets on the start position would need no movement

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        locs: List[np.ndarray] = []
        while len(locs) < self.n_locations:
            p = rng.uniform(-0.9, 0.9, size=2)
            if np.linalg.norm(p) >= self.min_center_dist:
                locs.append(p)
        self.locations = np.asarray(locs)
        ranks = np.arange(1, self.n_locations + 1, dtype=float)
        w = ranks ** -self.exponent
        self.probabilities = w / w.sum()             # location i has rank i + 1

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """(n, 2) goal positions drawn with Zipf frequencies."""
        idx = rng.choice(self.n_locations, size=n, p=self.probabilities)
        return self.locations[idx]
