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

Champion protocol (body-coordinate SMR; ``session_plan_champion``)
-------------------------------------------------------------------
The champion design (``proposals/pure_eeg_tournament/P_smr_modern_FINAL.md``, with the skeptic
corrections) goes one step further: the deployed decoder never trains on a label the display can
identify.  It decodes body-part intent (y = feet + / both hands −, x = right + / left hand −;
``mapping.BODY_VECTORS``) and five more block kinds carry the provenance ``source`` of their labels:

``fixation_probe``     source "L1".  A central, non-directional glyph cues one of four body classes;
                       the glyph→class map is redrawn every session (``glyph_map``), gaze must stay in
                       a 1° fixation window (a break aborts and repeats the trial) and the labelled
                       span is 1.0–4.0 s after the cue.  LABEL = the cued body vector.
``blind_closed_loop``  source "L2b".  The cue is a central glyph or a duration- and loudness-matched
                       body-part word, never a location; the decoder drives a HIDDEN cursor toward a
                       HIDDEN goal; the only feedback is a label-symmetric progress tone
                       (``ProgressTone``: pitch is a function of distance decrease alone, the same for
                       every direction) and a central outcome flash after the trial.  LABEL = the cued
                       body vector, windows ≥ 1.0 s after the cue and before the flash.
``sighted``            source "S".  Cursor and target visible, identity mapping: user practice and the
                       performance endpoint.  No label for the deployed decoder (re-centring only); the
                       never-deployed SHADOW decoder trains here on ReFIT labels as a leak positive
                       control.  15 % of trials silently replay a past trajectory (``ReplayScheduler``).
``heldout_mapping``    source "H", test only.  The cursor moves by ``M · v_body`` with a mapping no
                       decoder was trained under (rot ±90 primary, mirrors); C2c is closed loop with a
                       visible cursor, C2b is open loop (frozen cursor, then a keyed body-part report).
``pixel_replay``       source "R", test only.  Trial-level kind inside S and C2c blocks: the screen
                       replays a past trajectory toward goal A while the person holds goal B.

The "just intend" endpoint (no imagery instruction, identity mapping, frozen decoder, scored only) runs
as ``blind_closed_loop`` / ``sighted`` blocks with ``just_intend=True``; its windows carry source "J"
and are test only.  On day 1 its blind half opens the session before any instruction (the hint-free
block).  Fixation probes and blind closed loop draw nothing at a trial-dependent position
(``Block.screen_items``), and every champion session keeps L1, L2b, at least one held-out-mapping block
and replay trials.

Coordinates are normalised screen units in [-1, 1] with +x right and +y up, as in ``task.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .mapping import BODY_VECTORS, HELDOUT_MAPPINGS, MAPPING_NAMES, TRAINING_MAPPING

# ----------------------------------------------------------------------------------------------
# Block vocabulary
# ----------------------------------------------------------------------------------------------
BLOCK_KINDS: Tuple[str, ...] = ("setup", "cued_blank", "closed_loop", "blank_feedback", "free_use", "rest")
"""The original (screen-coordinate) protocol's kinds; ``PROVENANCE_BLOCK`` maps exactly these."""
CHAMPION_BLOCK_KINDS: Tuple[str, ...] = ("fixation_probe", "blind_closed_loop", "sighted", "heldout_mapping",
                                         "pixel_replay")
"""The champion protocol's kinds; each is also its own ``data_engine`` block name."""
ALL_BLOCK_KINDS: Tuple[str, ...] = BLOCK_KINDS + CHAMPION_BLOCK_KINDS
CUE_MODES: Tuple[str, ...] = ("auditory", "central_symbol", "target", "none",
                              "central_glyph", "body_word", "mixed", "goal_word")
"""``auditory`` / ``central_symbol`` are the two non-spatial cues allowed in ``cued_blank``;
``target`` means the goal is drawn at its screen position (closed-loop kinds); ``none`` is for blocks
without a direction (setup, rest).  Champion cues: ``central_glyph`` (a non-directional glyph at
fixation), ``body_word`` (a spoken body part), ``mixed`` (half of the trials each, balanced per class)
and ``goal_word`` (a spoken screen direction, only in the uninstructed "just intend" blind block)."""
FEEDBACK_MODES: Tuple[str, ...] = ("none", "cursor", "delayed", "tone", "frozen", "replay")
"""``tone``: hidden cursor, label-symmetric progress tone; ``frozen``: cursor drawn but not moving (C2b);
``replay``: cursor drawn but replaying a past trajectory (pixel replay)."""

SOURCES: Tuple[str, ...] = ("L1", "L2b", "S", "H", "R", "J")
"""Label provenance of champion windows (``data_engine`` writes it as provenance ``source``)."""
DEPLOYED_TRAINING_SOURCES: Tuple[str, ...] = ("L1", "L2b")
"""The only sources whose labels nothing on the screen can identify: the deployed decoder's training set."""
TEST_ONLY_SOURCES: Tuple[str, ...] = ("H", "R", "J")
"""Certification and endpoint data: no decoder, deployed or shadow, ever trains on them."""
SOURCES_BY_KIND: Dict[str, Tuple[str, ...]] = {
    "fixation_probe": ("L1",), "blind_closed_loop": ("L2b", "J"), "sighted": ("S", "J"),
    "heldout_mapping": ("H",), "pixel_replay": ("R",),
}
"""Which sources a champion kind can produce (J: the "just intend" variant of the blind/sighted kinds)."""

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
BODY_GLYPH = ScreenItem("body_glyph", (0.0, 0.0), spatial=False)
"""The champion cue glyph, drawn at fixation during the cue phase only (see ``glyph_map``)."""
OUTCOME_FLASH = ScreenItem("outcome_flash", (0.0, 0.0), spatial=False)
"""The blind block's end-of-trial flash.  It is drawn at fixation and says only hit or miss.  Drawing
the hidden cursor's end point instead would put a trial-dependent position on screen, and the person's
eyes would go to it."""
PHASES: Tuple[Optional[str], ...] = (None, "cue", "intent", "outcome")
"""Trial phases ``screen_items`` can be narrowed to: cue onset → cue offset, the labelled span from
``trial_start`` to ``trial_end``, and the outcome display after ``trial_end``.  None means every phase."""

REPLAY_FRACTION = 0.15            # share of S and C2c trials that silently replay a past trajectory (P3)
MAX_REPLAY_FRACTION = 0.25        # more frequent replays get noticed, and a noticed replay is a different test
_REPLAY_HOSTS = ("sighted", "heldout_mapping")
_CHAMPION_DEFAULT_MAPPING = {"fixation_probe": TRAINING_MAPPING, "blind_closed_loop": TRAINING_MAPPING,
                             "sighted": TRAINING_MAPPING, "heldout_mapping": "rot+90",
                             "pixel_replay": TRAINING_MAPPING}
HELDOUT_VARIANTS: Tuple[str, ...] = ("C2b", "C2c")
"""C2b: open loop (cursor frozen, 3 s of imagery, then a keyed body-part report); C2c: closed loop."""


@dataclass(frozen=True)
class Block:
    """One protocol block.

    ``cue`` and ``feedback`` default to the only (or safest) value for the kind and are validated
    against it, so an impossible block -- a cued_blank with a visible cursor, a closed_loop without
    one -- cannot be constructed and silently mislabel a session's data.
    ``seed`` seeds whatever the block randomises (cue order, target draws) so a session can be
    replayed exactly from its plan.

    Champion options are validated per kind the same way.  Original kinds leave them at their
    defaults; only ``setup`` may set ``localiser``.

    ``mapping``          screen mapping name (``mapping.MAPPING_NAMES``).  It must be the identity for
                         every trainable or sighted kind, and a held-out mapping for heldout_mapping
                         (default rot+90, the primary one).
    ``variant``          heldout_mapping only: "C2c" (closed loop, the default) or "C2b" (open loop).
    ``instructed``       whether imagery instruction is given.  It is always True for L1, L2b and H
                         (they stay cued), may be False for S (instruction withdrawn after the
                         withdrawal day), and is always False for "just intend".
    ``just_intend``      blind_closed_loop / sighted only: the uninstructed, scored-only endpoint (source J).
    ``replay_fraction``  share of trials that are pixel replays.  Defaults to ``REPLAY_FRACTION`` in S and
                         C2c blocks and is 0 everywhere else; only those two kinds may host replays.
    ``localiser``        setup only: the session runs the instructed-artefact localiser.
    """
    kind: str
    duration_s: float
    cue: Optional[str] = None
    feedback: Optional[str] = None
    seed: int = 0
    mapping: Optional[str] = None
    variant: Optional[str] = None
    instructed: Optional[bool] = None
    just_intend: bool = False
    replay_fraction: Optional[float] = None
    localiser: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ALL_BLOCK_KINDS:
            raise ValueError(f"unknown block kind {self.kind!r}; expected one of {ALL_BLOCK_KINDS}")
        if not (self.duration_s > 0):
            raise ValueError(f"{self.kind}: duration_s must be > 0, got {self.duration_s}")
        if self.localiser and self.kind != "setup":
            raise ValueError(f"{self.kind}: only a setup block can run the artefact localiser")
        if self.kind in BLOCK_KINDS:
            self._init_original()
        else:
            self._init_champion()

    def _init_original(self) -> None:
        champion_opts = {"mapping": self.mapping, "variant": self.variant, "instructed": self.instructed,
                         "replay_fraction": self.replay_fraction}
        given = sorted(k for k, v in champion_opts.items() if v is not None)
        if given or self.just_intend:
            raise ValueError(f"{self.kind}: champion options {given + (['just_intend'] if self.just_intend else [])} "
                             f"apply only to {CHAMPION_BLOCK_KINDS}")
        cue = _DEFAULT_CUE[self.kind] if self.cue is None else self.cue
        feedback = _FEEDBACK[self.kind] if self.feedback is None else self.feedback
        if cue not in _ALLOWED_CUES[self.kind]:
            raise ValueError(f"{self.kind}: cue {cue!r} not allowed; use one of {_ALLOWED_CUES[self.kind]}")
        if feedback != _FEEDBACK[self.kind]:
            raise ValueError(f"{self.kind}: feedback must be {_FEEDBACK[self.kind]!r}, got {feedback!r}")
        object.__setattr__(self, "cue", cue)
        object.__setattr__(self, "feedback", feedback)

    def _init_champion(self) -> None:
        k = self.kind
        if self.just_intend and k not in ("blind_closed_loop", "sighted"):
            raise ValueError(f"{k}: just_intend applies only to blind_closed_loop and sighted")
        if self.variant is not None and k != "heldout_mapping":
            raise ValueError(f"{k}: variant applies only to heldout_mapping")
        variant = (HELDOUT_VARIANTS[1] if self.variant is None else self.variant) if k == "heldout_mapping" else None
        if k == "heldout_mapping" and variant not in HELDOUT_VARIANTS:
            raise ValueError(f"heldout_mapping: variant must be one of {HELDOUT_VARIANTS}, got {variant!r}")

        mapping = _CHAMPION_DEFAULT_MAPPING[k] if self.mapping is None else self.mapping
        if mapping not in MAPPING_NAMES:
            raise ValueError(f"{k}: unknown mapping {mapping!r}; expected one of {MAPPING_NAMES}")
        if k == "heldout_mapping" and mapping not in HELDOUT_MAPPINGS:
            raise ValueError(f"heldout_mapping: mapping must be held out {HELDOUT_MAPPINGS}, got {mapping!r}")
        if k in ("fixation_probe", "blind_closed_loop", "sighted") and mapping != TRAINING_MAPPING:
            raise ValueError(f"{k}: runs under the {TRAINING_MAPPING!r} mapping only, got {mapping!r}; "
                             f"a held-out mapping belongs in a heldout_mapping block")

        if self.just_intend:
            allowed_cues = ("goal_word",) if k == "blind_closed_loop" else ("target",)
        else:
            allowed_cues = {"fixation_probe": ("central_glyph",),
                            "blind_closed_loop": ("mixed", "central_glyph", "body_word")}.get(k, ("target",))
        cue = allowed_cues[0] if self.cue is None else self.cue
        if cue not in allowed_cues:
            raise ValueError(f"{k}{' (just intend)' if self.just_intend else ''}: cue {cue!r} not allowed; "
                             f"use one of {allowed_cues}")

        expected_fb = {"fixation_probe": "none", "blind_closed_loop": "tone", "sighted": "cursor",
                       "pixel_replay": "replay",
                       "heldout_mapping": "frozen" if variant == "C2b" else "cursor"}[k]
        feedback = expected_fb if self.feedback is None else self.feedback
        if feedback != expected_fb:
            raise ValueError(f"{k}: feedback must be {expected_fb!r}, got {feedback!r}")

        if self.just_intend:
            if self.instructed:
                raise ValueError(f"{k}: a just_intend block is uninstructed by definition")
            instructed = False
        else:
            instructed = True if self.instructed is None else bool(self.instructed)
            if not instructed and k != "sighted":
                raise ValueError(f"{k}: L1, L2b and held-out blocks stay instructed; only S may withdraw it")

        can_host = k in _REPLAY_HOSTS and not self.just_intend and feedback == "cursor"
        replay = (REPLAY_FRACTION if can_host else 0.0) if self.replay_fraction is None else float(self.replay_fraction)
        if not 0.0 <= replay <= MAX_REPLAY_FRACTION:
            raise ValueError(f"{k}: replay_fraction must be in [0, {MAX_REPLAY_FRACTION}], got {replay}")
        if replay > 0 and not can_host:
            raise ValueError(f"{k}: only sighted (not just intend) and C2c held-out blocks host replay trials")

        for name, value in (("cue", cue), ("feedback", feedback), ("mapping", mapping), ("variant", variant),
                            ("instructed", instructed), ("replay_fraction", replay)):
            object.__setattr__(self, name, value)

    @property
    def is_champion(self) -> bool:
        return self.kind in CHAMPION_BLOCK_KINDS

    @property
    def feedback_visible(self) -> bool:
        """Label provenance flag: was the cursor drawn while the person formed the intent (moving,
        frozen or replaying -- in all three the person sees where it is)."""
        return self.feedback in ("cursor", "frozen", "replay")

    @property
    def provenance_block(self) -> Optional[str]:
        """The ``block`` name the data engine expects, or None for blocks that are never labelled."""
        return self.kind if self.is_champion else PROVENANCE_BLOCK[self.kind]

    @property
    def source(self) -> Optional[str]:
        """Label provenance ``source`` of the block's trials (None for the original kinds).  Replay
        trials inside an S or C2c block are "R" (see ``replay_block``)."""
        if not self.is_champion:
            return None
        if self.just_intend:
            return "J"
        return SOURCES_BY_KIND[self.kind][0]

    @property
    def is_test_only(self) -> bool:
        """True for certification / endpoint blocks no decoder (deployed or shadow) may train on."""
        return self.source in TEST_ONLY_SOURCES

    @property
    def is_validity(self) -> bool:
        """True for the validity blocks that every session must contain: cued_blank and blank_feedback
        in the original protocol; L1, L2b, held-out mappings and replays in the champion protocol."""
        if self.is_champion:
            return self.source in ("L1", "L2b", "H", "R")
        return self.kind in ("cued_blank", "blank_feedback")

    def replay_block(self) -> "Block":
        """The trial-level ``pixel_replay`` block for this host's replay trials: same mapping, same
        duration bound, so its screen items and provenance are derived like any block's."""
        if not self.replay_fraction:
            raise ValueError(f"{self.kind} block hosts no replay trials (replay_fraction 0)")
        return Block("pixel_replay", self.duration_s, mapping=self.mapping, seed=self.seed)

    def screen_items(self, phase: Optional[str] = None) -> Tuple[ScreenItem, ...]:
        """Everything drawn during the block (the renderer's contract, and what tests audit).

        ``phase`` narrows it to one part of a trial: ``"cue"`` (cue onset to ``cue_offset``,
        ``CuedTrialTiming.cue_s``), ``"intent"`` (the labelled span from ``trial_start``) or
        ``"outcome"`` (after ``trial_end``).  A cue symbol or glyph is drawn during the cue phase ONLY:
        it maps one-to-one onto the label, so leaving it on screen into the intent phase would make
        the labels readable from the display.  ``None`` is the whole block (every phase).

        fixation_probe and blind_closed_loop draw nothing at a trial-dependent position in any phase:
        a fixation mark, the cue glyph at fixation, and the blind block's central outcome flash."""
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
        if self.kind == "cued_blank":
            if self.cue == "central_symbol" and phase in (None, "cue"):
                return (FIXATION, ScreenItem("direction_symbol", (0.0, 0.0), spatial=False))
            return (FIXATION,)
        if self.kind in ("fixation_probe", "blind_closed_loop"):
            glyph = (BODY_GLYPH,) if self.cue in ("central_glyph", "mixed") and phase in (None, "cue") else ()
            flash = (OUTCOME_FLASH,) if self.kind == "blind_closed_loop" and phase in (None, "outcome") else ()
            return (FIXATION,) + glyph + flash
        if self.kind in ("closed_loop", "free_use", "sighted", "heldout_mapping", "pixel_replay"):
            # pixel_replay: the drawn cursor is a replayed past trajectory; heldout C2b: it is frozen.
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
    """Seconds per kind (original and champion) plus ``task_s`` (all kinds that carry a direction) and
    ``cued_fraction`` (cued_blank share of task time; 0 in a champion plan)."""
    out: Dict[str, float] = {kind: 0.0 for kind in ALL_BLOCK_KINDS}
    for b in plan:
        out[b.kind] += b.duration_s
    out["task_s"] = sum(out[k] for k in ALL_BLOCK_KINDS if _is_task_kind(k))
    out["total_s"] = sum(b.duration_s for b in plan)
    out["cued_fraction"] = out["cued_blank"] / out["task_s"] if out["task_s"] else 0.0
    return out


def _is_task_kind(kind: str) -> bool:
    """Kinds with a direction to intend (everything but setup and rest)."""
    return kind in CHAMPION_BLOCK_KINDS or PROVENANCE_BLOCK[kind] is not None


def render_plan_markdown(plan: Sequence[Block], title: Optional[str] = None) -> str:
    """Markdown table of the plan with per-block minutes, start time and a totals footer.  The last
    two columns (label source, screen mapping) are filled for champion blocks only."""
    lines = [f"### {title}", ""] if title else []
    lines += ["| # | Start (min) | Block | Minutes | Cue | Feedback | Cursor visible | Validity | Source | Mapping |",
              "|---:|---:|---|---:|---|---|---|---|---|---|"]
    t = 0.0
    for i, b in enumerate(plan, 1):
        lines.append(f"| {i} | {t / 60:.0f} | {b.kind} | {b.duration_s / 60:.1f} | {b.cue} | {b.feedback} "
                     f"| {'yes' if b.feedback_visible else 'no'} | {'yes' if b.is_validity else ''} "
                     f"| {b.source or ''} | {b.mapping or ''} |")
        t += b.duration_s
    s = plan_summary(plan)
    shares = ", ".join(f"{k} {s[k] / s['task_s']:.0%}" for k in ALL_BLOCK_KINDS
                       if _is_task_kind(k) and s[k] > 0)
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


def _runs_within(history: Sequence[object], seq: Sequence[object], max_run: int) -> bool:
    """True if appending ``seq`` to ``history`` makes no run of more than ``max_run`` equal items."""
    run, prev = 0, None
    for d in list(history[-max_run:]) + list(seq):
        run = run + 1 if d == prev else 1
        prev = d
        if run > max_run:
            return False
    return True


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

    def _refill(self) -> None:
        base = np.repeat(np.arange(N_DIRECTIONS), self.reps_per_block)
        for _ in range(self.max_redraws):
            perm = self._rng.permutation(base).tolist()
            if _runs_within(self._history, perm, self.max_run):
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
    Champion blocks need provenance this payload lacks: use ``champion_trial_start_payload``.
    """
    if block.is_champion:
        raise ValueError(f"{block.kind} is a champion block; use champion_trial_start_payload")
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


# ----------------------------------------------------------------------------------------------
# Champion protocol: body cues
# ----------------------------------------------------------------------------------------------
BODY_CLASSES: Tuple[str, ...] = ("FEET", "HANDS", "LEFT", "RIGHT")
DIAGONAL_CLASSES: Tuple[str, ...] = ("FEET+RIGHT", "FEET+LEFT", "HANDS+RIGHT", "HANDS+LEFT")
"""Two-part cues for the diagonals: the y part, then the x part (body vector = their sum)."""
L1_CLASSES: Tuple[str, ...] = BODY_CLASSES
"""Fixation probes cue the four single classes (they measure each axis's separability)."""
L2B_CLASSES_Y: Tuple[str, ...] = ("FEET", "HANDS")
"""Blind closed loop for a person who has not earned x: a LEFT cue would leave a hidden cursor that the
decoder (x = 0) cannot move, i.e. a trial with no possible progress."""
L2B_CLASSES_XY: Tuple[str, ...] = BODY_CLASSES + DIAGONAL_CLASSES
BODY_WORDS: Dict[str, str] = {"FEET": "feet", "HANDS": "both hands", "LEFT": "left hand", "RIGHT": "right hand"}
GLYPH_SET: Tuple[str, ...] = ("●", "■", "◆", "✚", "✖", "○", "□", "◇")
"""Candidate cue glyphs.  Each is symmetric under left-right and up-down mirroring, so no glyph points
anywhere.  An arrow or triangle is a symbolic spatial cue, and it moves covert attention and small
saccades toward the side it points to."""
BODY_CUE_MODES: Tuple[str, ...] = ("central_glyph", "body_word")
CUE_LOCKOUT_S = 1.0
"""Labelled windows start at least this long after cue onset (L1, L2b, blind J).  This keeps out the
sensory response evoked by the glyph or word, which differs between classes."""
L1_LABEL_END_S = 4.0
"""Fixation-probe windows end at most this long after cue onset.  Untrained imagery fades after its
1–2 s peak, so later windows add label noise rather than signal."""
WORD_CUE_LEVEL_DB = -20.0
"""Every spoken cue is normalised to this loudness (dBFS, mono, centred) and padded or time-stretched to
the cue duration.  Otherwise a word's length or loudness, which differ between "feet" and "left hand",
would predict the class."""
BLIND_GOAL_DISTANCE = 0.6
"""Distance (task units) from the start to the hidden L2b goal.  It is the same for every class, so the
progress tone has the same range for every direction."""

EVENT_FIXATION_BREAK = "fixation_break"
"""Logged by the presentation layer when gaze leaves the 1° window during a fixation-enforced trial
(L1, L2b, blind J).  The trial is aborted and repeated, and the data engine discards it."""
EVENT_OUTCOME_FLASH = "outcome_flash"
"""Blind-block outcome display.  It must come after ``trial_end``, so no labelled window sees it."""
EVENT_BODY_REPORT = "body_report"
"""C2b keyed report ``{"trial", "class"}``.  It must come after ``trial_end``, because the keypress is a
movement."""


def body_vector(cls: str) -> Tuple[int, int]:
    """Integer body vector (x, y) in {−1, 0, 1}² of a single or two-part class name."""
    parts = cls.split("+")
    if any(p not in BODY_CLASSES for p in parts) or len(set(parts)) != len(parts):
        raise ValueError(f"unknown body class {cls!r}; expected one of {BODY_CLASSES + DIAGONAL_CLASSES}")
    v = np.sum([BODY_VECTORS[p] for p in parts], axis=0)
    return int(round(v[0])), int(round(v[1]))


def glyph_map(day: int, seed: int) -> Dict[str, str]:
    """Glyph for each body class in this session.  The map is redrawn every session from (seed, day).

    Within a session a glyph identifies its class, so the visual response it evokes is class-specific.
    Across sessions the same glyph means different classes.  A decoder that learned glyph-evoked
    activity, or a glyph-specific gaze habit, therefore fails on the next session, while one that learned
    the imagery does not."""
    _check_day(day)
    rng = np.random.default_rng([int(seed), int(day), _GLYPH_STREAM])
    chosen = rng.permutation(len(GLYPH_SET))[:len(BODY_CLASSES)]
    return {cls: GLYPH_SET[i] for cls, i in zip(BODY_CLASSES, chosen)}


@dataclass(frozen=True)
class BodyCue:
    """One cued body class.  ``body`` is the (x, y) label vector in {−1, 0, 1}²."""
    seq: int
    cls: str
    parts: Tuple[str, ...]
    body: Tuple[int, int]
    mode: str                       # "central_glyph" | "body_word"
    glyphs: Tuple[str, ...]         # one per part (shown one after the other at fixation)
    word: str                       # what a body_word cue speaks

    def presentation(self, cue_s: float) -> Dict[str, object]:
        """What the stimulus layer presents for ``cue_s`` seconds (the block's ``trial_timing(b).cue_s``:
        0.75 s for L1, 1.0 s for L2b).  Everything is at fixation or in
        the speaker: a two-part glyph cue shows its glyphs one after the other at the centre, each
        for ``cue_s / len(parts)``.  It never shows them side by side, because there the side would
        carry the x part."""
        if self.mode == "central_glyph":
            per = cue_s / len(self.glyphs)
            return {"audio": None, "screen": [FIXATION, BODY_GLYPH],
                    "glyph_sequence": [(g, per) for g in self.glyphs], "duration_s": cue_s,
                    "after_offset": [FIXATION]}
        if self.mode == "body_word":
            return {"audio": self.word, "audio_level_db": WORD_CUE_LEVEL_DB, "audio_pan": 0.0,
                    "screen": [FIXATION], "duration_s": cue_s, "after_offset": [FIXATION]}
        raise ValueError(f"unknown body cue mode {self.mode!r}")


@dataclass
class BodyCueScheduler:
    """Balanced, block-randomised body cues: every (class, mode) pair ``reps_per_block`` times per
    randomisation block, shuffled, and no more than ``max_run`` equal classes in a row (counted across
    block boundaries).  This is the same design as ``CueScheduler``, for the same reasons.

    ``modes`` is ("central_glyph",) for L1.  For L2b it is both modes, so glyph and word cues are
    balanced within every class.  A decoder that learned one cue modality's evoked response would then
    do no better than chance on the other half.
    ``glyphs`` is this session's ``glyph_map`` and is required whenever glyphs are shown.  Every issued
    cue goes to ``log`` and, with a ``logger``, is written as a ``"cue"`` event whose ``body`` the data
    engine checks against the trial's label.
    """
    seed: int
    classes: Tuple[str, ...] = L1_CLASSES
    modes: Tuple[str, ...] = ("central_glyph",)
    glyphs: Optional[Dict[str, str]] = None
    reps_per_block: int = 2
    max_run: int = 2
    max_redraws: int = 10_000
    logger: Optional[object] = None
    log: List[Dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.reps_per_block < 1 or self.max_run < 1:
            raise ValueError("reps_per_block and max_run must be >= 1")
        if not self.modes or any(m not in BODY_CUE_MODES for m in self.modes):
            raise ValueError(f"modes must be a non-empty subset of {BODY_CUE_MODES}, got {self.modes}")
        self._vectors = [body_vector(c) for c in self.classes]           # validates the class names
        if "central_glyph" in self.modes:
            missing = {p for c in self.classes for p in c.split("+")} - set(self.glyphs or {})
            if missing:
                raise ValueError(f"glyph cues need a glyph for {sorted(missing)}; pass glyphs=glyph_map(day, seed)")
        self._rng = np.random.default_rng(self.seed)
        self._queue: List[Tuple[int, int]] = []
        self._history: List[int] = []

    def _refill(self) -> None:
        base = [(c, m) for c in range(len(self.classes)) for m in range(len(self.modes))] * self.reps_per_block
        for _ in range(self.max_redraws):
            perm = [base[i] for i in self._rng.permutation(len(base))]
            if _runs_within(self._history, [c for c, _ in perm], self.max_run):
                self._queue = perm
                return
        raise RuntimeError("could not draw a cue block satisfying max_run; relax max_run")

    def next(self, t: Optional[float] = None) -> BodyCue:
        """Issue and log the next cue; ``t`` is the cue onset on the session clock."""
        if not self._queue:
            self._refill()
        c, m = self._queue.pop(0)
        self._history.append(c)
        cls = self.classes[c]
        parts = tuple(cls.split("+"))
        cue = BodyCue(seq=len(self.log), cls=cls, parts=parts, body=self._vectors[c], mode=self.modes[m],
                      glyphs=tuple((self.glyphs or {}).get(p, "") for p in parts) if self.modes[m] == "central_glyph" else (),
                      word=" and ".join(BODY_WORDS[p] for p in parts))
        entry = {"seq": cue.seq, "class": cls, "body": list(cue.body), "mode": cue.mode,
                 "glyphs": list(cue.glyphs), "t": t}
        self.log.append(entry)
        if self.logger is not None:
            self.logger.event("cue", entry, t)
        return cue

    def schedule(self, n: int) -> List[BodyCue]:
        return [self.next() for _ in range(n)]


def hidden_goal(cue: BodyCue, start_pos: Sequence[float] = (0.0, 0.0),
                distance: float = BLIND_GOAL_DISTANCE) -> Tuple[float, float]:
    """The L2b hidden goal: ``distance`` from the start along the cued body direction (identity
    mapping).  It is never drawn.  It sets where the hidden cursor must go and how far the progress
    tone can rise."""
    v = np.asarray(cue.body, float)
    g = np.asarray(start_pos, float) + distance * v / np.linalg.norm(v)
    return float(g[0]), float(g[1])


# ----------------------------------------------------------------------------------------------
# Champion protocol: feedback, timing, replay
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ProgressTone:
    """The blind block's only online feedback: a tone whose pitch encodes progress toward the hidden
    goal, ``p = (d0 − d) / d0`` (the fractional decrease of the distance since the trial started).

    It is label-symmetric by construction.  ``pitch_hz`` takes distances only, and the goal distance
    ``d0`` is the same for every class (``BLIND_GOAL_DISTANCE``).  Two trajectories with the same
    distance profile, such as mirror images or rotations of each other toward different classes'
    goals, therefore produce the same sounds.  Loudness, timbre and stereo position are constants.  A
    panned or direction-coloured tone would let the decoder, or the person's attention, recover the
    cued direction from the feedback, and the L2b label would stop being display-blind (audio is a
    display too).

    base_hz        pitch at the start position (440 Hz, comfortably audible on any headphone).
    octave_span    progress 0 → 1 raises the pitch by one octave: large enough to hear a 10 % step.
    min_progress   −0.5: moving away lowers the pitch by at most half an octave, then it saturates,
                   so a runaway trial does not end in a sub-audible drone.
    update_s       one pitch per 0.1 s control step.
    level_db       constant level, dBFS.
    pan            0 (centred, mono).  It is a field only so that the test can state it.
    """
    base_hz: float = 440.0
    octave_span: float = 1.0
    min_progress: float = -0.5
    update_s: float = 0.1
    level_db: float = -20.0
    pan: float = 0.0

    def progress(self, d0: float, d: float) -> float:
        if not d0 > 0:
            raise ValueError(f"the start distance must be > 0, got {d0}")
        return float(np.clip((d0 - d) / d0, self.min_progress, 1.0))

    def pitch_hz(self, d0: float, d: float) -> float:
        """Pitch for current distance ``d`` given start distance ``d0``: a function of the distance
        decrease only."""
        return self.base_hz * 2.0 ** (self.octave_span * self.progress(d0, d))

    def sequence(self, distances: Sequence[float]) -> np.ndarray:
        """Pitches for a trial's per-step distances to the goal (the first is the start distance)."""
        d = np.asarray(distances, float)
        return np.array([self.pitch_hz(d[0], x) for x in d])


def distances_to_goal(positions: Sequence[Sequence[float]], goal: Sequence[float]) -> np.ndarray:
    """Euclidean distance of each (hidden) cursor position to the goal: the tone's only input."""
    return np.linalg.norm(np.asarray(positions, float) - np.asarray(goal, float), axis=1)


@dataclass(frozen=True)
class FixationProbeTiming:
    """L1 trial timing (s, from cue onset).  The glyph is shown for ``cue_s``.  ``trial_start`` is
    logged at ``label_start_s`` and ``trial_end`` at ``label_end_s``.  ``iti_s`` includes time to
    re-acquire fixation.  7.5 s trials give 120 probes in the proposal's 15 minutes."""
    cue_s: float = 0.75
    label_start_s: float = CUE_LOCKOUT_S
    label_end_s: float = L1_LABEL_END_S
    iti_s: float = 3.5

    @property
    def trial_s(self) -> float:
        return self.label_end_s + self.iti_s

    def n_trials(self, block: Block) -> int:
        return int(block.duration_s // self.trial_s)


@dataclass(frozen=True)
class ClosedLoopTiming:
    """L2b / S / C2c / blind-J trial timing (s).  The cue (glyph, word or target onset) starts at 0.
    ``trial_start`` is logged at ``label_start_s``, when control begins.  The trial ends on a hit or at
    ``timeout_s`` of control.  Then the ``outcome_s`` flash is shown (after ``trial_end``) and the
    ``iti_s`` pause follows.  ``n_trials`` assumes every trial times out, so it is a lower bound on
    the number of trials a block holds."""
    cue_s: float = 1.0
    label_start_s: float = CUE_LOCKOUT_S
    timeout_s: float = 10.0
    outcome_s: float = 0.5
    iti_s: float = 1.5

    @property
    def trial_s(self) -> float:
        return self.label_start_s + self.timeout_s + self.outcome_s + self.iti_s

    def n_trials(self, block: Block) -> int:
        return int(block.duration_s // self.trial_s)


@dataclass(frozen=True)
class OpenLoopTiming:
    """C2b trial timing (s).  The target and the frozen cursor are shown at 0.  ``trial_start`` is
    logged at ``preview_s`` and ``trial_end`` after ``imagery_s``.  The keyed ``body_report`` is
    collected within ``report_s`` after that, outside every labelled window."""
    preview_s: float = CUE_LOCKOUT_S
    imagery_s: float = 3.0
    report_s: float = 2.0
    iti_s: float = 1.5

    @property
    def trial_s(self) -> float:
        return self.preview_s + self.imagery_s + self.report_s + self.iti_s

    def n_trials(self, block: Block) -> int:
        return int(block.duration_s // self.trial_s)


def trial_timing(block: Block):
    """The timing object for a champion block's trials."""
    if block.kind == "fixation_probe":
        return FixationProbeTiming()
    if block.kind == "heldout_mapping" and block.variant == "C2b":
        return OpenLoopTiming()
    if block.is_champion:
        return ClosedLoopTiming()
    raise ValueError(f"{block.kind} is not a champion block")


@dataclass
class ReplayScheduler:
    """Which trials of an S or C2c host block are pixel replays (``next() -> bool``).

    Every ``period`` consecutive trials hold exactly ``round(replay_fraction × period)`` replays at
    random positions.  The count is therefore exact however many trials the block ends up running,
    since hits end trials early.  The block's first trial is never a replay, so the person first sees
    the cursor obey them.  Two replays are never adjacent, because back-to-back replays get noticed.
    Deterministic from the block's seed.
    """
    block: Block
    period: int = 20
    max_redraws: int = 10_000

    def __post_init__(self) -> None:
        if not self.block.replay_fraction:
            raise ValueError(f"{self.block.kind} block hosts no replay trials")
        self.n_per_period = int(round(self.block.replay_fraction * self.period))
        if not 1 <= self.n_per_period <= (self.period - 1) // 2:
            raise ValueError(f"period {self.period} cannot hold {self.n_per_period} non-adjacent replays")
        self._rng = np.random.default_rng([int(self.block.seed), _REPLAY_STREAM])
        self._queue: List[bool] = []
        self._prev = True                                    # forbids a replay as the very first trial

    def _refill(self) -> None:
        for _ in range(self.max_redraws):
            pos = np.sort(self._rng.choice(self.period, self.n_per_period, replace=False))
            if (self._prev and pos[0] == 0) or np.any(np.diff(pos) < 2):
                continue
            self._queue = [bool(i in set(pos.tolist())) for i in range(self.period)]
            return
        raise RuntimeError("could not place non-adjacent replays; lower replay_fraction")

    def next(self) -> bool:
        if not self._queue:
            self._refill()
        self._prev = self._queue.pop(0)
        return self._prev

    def schedule(self, n: int) -> List[bool]:
        return [self.next() for _ in range(n)]


# ----------------------------------------------------------------------------------------------
# Champion protocol: trial payloads
# ----------------------------------------------------------------------------------------------
def champion_trial_start_payload(block: Block, trial: int, *, cue: Optional[object] = None,
                                 goal: Optional[Sequence[float]] = None,
                                 start_pos: Sequence[float] = (0.0, 0.0),
                                 replay_goal: Optional[Sequence[float]] = None) -> Dict[str, object]:
    """The ``trial_start`` payload of a champion trial, with the provenance ``data_engine`` requires.

    block        the host block (never ``pixel_replay`` itself: a replay is a trial of an S / C2c block).
    cue          a ``BodyCue`` for L1 and L2b (the label), a ``Cue`` (spoken goal word) for the blind
                 "just intend" block, and None for blocks whose goal is drawn.
    goal         the goal on screen.  For S, H, R and sighted J it is drawn, and for R it is the goal B
                 the person holds.  L1 has none.  For L2b and blind J it defaults to the hidden goal
                 the cue encodes.
    replay_goal  makes this a pixel-replay trial (source R).  It is the goal A that the replayed past
                 trajectory was aimed at.
    """
    if not block.is_champion:
        raise ValueError(f"{block.kind} is not a champion block; use trial_start_payload")
    if block.kind == "pixel_replay":
        raise ValueError("replay trials are issued by their S / C2c host block with replay_goal")
    tb = block.replay_block() if replay_goal is not None else block
    src = tb.source
    start = [float(p) for p in start_pos]
    out: Dict[str, object] = {"trial": int(trial), "block": tb.provenance_block, "source": src,
                              "feedback_visible": tb.feedback_visible, "start_pos": start,
                              "mapping": tb.mapping, "instructed": bool(block.instructed)}
    if src in DEPLOYED_TRAINING_SOURCES:
        if not isinstance(cue, BodyCue):
            raise ValueError(f"{src} trials are labelled by a BodyCue, got {type(cue).__name__}")
        if src == "L1" and len(cue.parts) != 1:
            raise ValueError(f"fixation probes cue single classes {L1_CLASSES}, got {cue.cls!r}")
        if cue.mode not in ((block.cue,) if block.cue != "mixed" else BODY_CUE_MODES):
            raise ValueError(f"{block.kind} block with cue {block.cue!r} cannot present a {cue.mode!r} cue")
        if src == "L1" and goal is not None:
            raise ValueError("fixation probes have no goal")
        if src == "L2b" and goal is None:
            goal = hidden_goal(cue, start)
        out.update(body_cue=list(cue.body), cue_class=cue.cls, cue_mode=cue.mode)
    elif src == "J" and block.kind == "blind_closed_loop":
        if not isinstance(cue, Cue):
            raise ValueError(f"the blind just-intend block is cued by a spoken goal word (Cue), got {type(cue).__name__}")
        if goal is None:
            goal = tuple(np.asarray(start) + np.asarray(cued_goal(cue)))
        out.update(cue_mode="goal_word")
    else:
        if cue is not None:
            raise ValueError(f"{tb.kind} trials are cued by the drawn target, not by {type(cue).__name__}")
        if goal is None:
            raise ValueError(f"{tb.kind}: the drawn goal must be known at trial start")
    out["goal"] = None if goal is None else [float(g) for g in goal]
    if tb.kind == "heldout_mapping":
        out["variant"] = tb.variant
    if replay_goal is not None:
        out["replay_goal"] = [float(g) for g in replay_goal]
    return out


# ----------------------------------------------------------------------------------------------
# Champion protocol: session plan
# ----------------------------------------------------------------------------------------------
CHAMPION_SETUP_MIN = 15           # impedance, a 3-min eyes-open rest (the decoder's re-centring
                                  # reference), and on localiser days the artefact localiser
CHAMPION_SLOT_MIN: Dict[str, int] = {"fixation_probe": 15, "blind_closed_loop": 20, "sighted": 15,
                                     "heldout_mapping": 8, "just_intend": 12}
"""Minutes per slot, as in the proposal's session table (15 / 20 / 15 / 8 / 12 after a 15-min setup;
85 min in total, followed by 5 min of questionnaires outside this plan).  L1 : L2b = 15 : 20 gives the
proposal's ≈ 43 % / 57 % split of labelled time."""
CHAMPION_BLOCK_MIN: Dict[str, int] = {"fixation_probe": 5, "blind_closed_loop": 5, "sighted": 5,
                                      "heldout_mapping": 4, "just_intend": 3}
"""Longest single block per slot.  A block boundary is a natural pause, and attention decays within a
block.  The 8-min held-out slot splits into one 4-min block per mapping.  The just-intend slot splits
into 3-min blocks that alternate blind and sighted."""
LOCALISER_EVERY = 10              # instructed-artefact localiser in sessions 1, 11, 21, ...: session 1
                                  # needs one before the first training (artefact components are projected out)
INSTRUCTION_KEPT_THROUGH: Dict[str, int] = {"deep": 10, "broad": 3}
"""Imagery instruction in S is withdrawn after this session (deep / broad cohort).  L1 and L2b stay cued."""
CLOSED_LOOP_HELDOUT_FROM_DAY = 3
"""C2c (closed-loop held-out mapping) starts in session 3, as in the proposal.  Before that, both
held-out blocks are C2b (open loop, frozen cursor), which needs no closed-loop skill.  This keeps a
held-out-mapping block in every session."""
_GLYPH_STREAM, _REPLAY_STREAM, _CHAMPION_PLAN_STREAM = 1, 2, 3   # independent RNG streams per purpose


def session_plan_champion(day: int, seed: int, cohort: str = "deep", x_earned: bool = False) -> List[Block]:
    """Deterministic champion block list for one session (``day`` counts from 1 for each person).

    It follows the proposal's 85-min session in its order: setup (15) → L1 fixation probes (15) → L2b
    blind closed loop (20) → S sighted identity with 15 % replays (15) → held-out mappings (8) → "just
    intend" (12, alternating blind / sighted, uninstructed, test only).  Every session keeps L1, L2b,
    at least one held-out-mapping block and replay trials.  Seed-dependent choices:

    * held-out mappings.  The first block is a rotation (rot+90 or rot−90, the primary test that
      decorrelates body and screen on both axes).  The second cycles over the other rotation and the
      mirrors, by day.  mirror_x is omitted unless ``x_earned``: a mirror separates body from screen
      only on the axis it flips, and x is 0 for a person who has not earned it.
    * C2b / C2c order (alternating by day from ``CLOSED_LOOP_HELDOUT_FROM_DAY``).
    * whether the just-intend slot starts blind or sighted, and every block's seed.

    Day 1 opens with the blind just-intend blocks right after setup, before any imagery instruction:
    this is the proposal's hint-free block.  The just-intend slot then keeps only its sighted half,
    and the session length does not change.  S is instructed through ``INSTRUCTION_KEPT_THROUGH[cohort]``.
    """
    _check_day(day)
    if cohort not in INSTRUCTION_KEPT_THROUGH:
        raise ValueError(f"cohort must be one of {tuple(INSTRUCTION_KEPT_THROUGH)}, got {cohort!r}")
    rng = np.random.default_rng([int(seed), int(day), _CHAMPION_PLAN_STREAM])
    split = {k: _split(m, CHAMPION_BLOCK_MIN[k]) for k, m in CHAMPION_SLOT_MIN.items()}

    rot = HELDOUT_MAPPINGS[int(rng.integers(2))]                    # "rot+90" or "rot-90"
    other = (("rot-90" if rot == "rot+90" else "rot+90"),) + (("mirror_x",) if x_earned else ()) + ("mirror_y",)
    mappings = (rot, other[(day - 1) % len(other)])
    if day < CLOSED_LOOP_HELDOUT_FROM_DAY:
        variants = ("C2b", "C2b")
    else:
        variants = ("C2b", "C2c") if day % 2 else ("C2c", "C2b")
    blind_first = bool(rng.integers(2))
    ji_kinds = [("blind_closed_loop", "sighted")[(i + (not blind_first)) % 2] for i in range(len(split["just_intend"]))]
    ji = list(zip(ji_kinds, split["just_intend"]))

    specs: List[Tuple[str, int, dict]] = [("setup", CHAMPION_SETUP_MIN, {"localiser": (day - 1) % LOCALISER_EVERY == 0})]
    if day == 1:                                                     # hint-free: blind just-intend first
        specs += [(k, m, {"just_intend": True}) for k, m in ji if k == "blind_closed_loop"]
        ji = [(k, m) for k, m in ji if k != "blind_closed_loop"]
    specs += [("fixation_probe", m, {}) for m in split["fixation_probe"]]
    specs += [("blind_closed_loop", m, {}) for m in split["blind_closed_loop"]]
    instructed = day <= INSTRUCTION_KEPT_THROUGH[cohort]
    specs += [("sighted", m, {"instructed": instructed}) for m in split["sighted"]]
    specs += [("heldout_mapping", m, {"mapping": mp, "variant": v})
              for m, mp, v in zip(split["heldout_mapping"], mappings, variants)]
    specs += [(k, m, {"just_intend": True}) for k, m in ji]

    seeds = rng.integers(0, 2**31 - 1, size=len(specs))
    return [Block(kind, m * 60.0, seed=int(s), **opts) for (kind, m, opts), s in zip(specs, seeds)]


def champion_plan_summary(plan: Sequence[Block]) -> Dict[str, object]:
    """Seconds per label source, the L1 share of labelled (L1 + L2b) time, the held-out mappings and
    variants, and the replay trials the plan guarantees.  A block's trial count is its lower bound,
    with every trial timing out (``trial_timing(b).n_trials``), and the replays are counted with that
    block's own ``ReplayScheduler``."""
    source_s: Dict[str, float] = {s: 0.0 for s in SOURCES}
    replays = 0
    for b in plan:
        if b.source is not None:
            source_s[b.source] += b.duration_s
        if b.replay_fraction:
            replays += sum(ReplayScheduler(b).schedule(trial_timing(b).n_trials(b)))
    labelled = source_s["L1"] + source_s["L2b"]
    return {"total_s": sum(b.duration_s for b in plan), "source_s": source_s,
            "l1_share": source_s["L1"] / labelled if labelled else 0.0,
            "heldout_mappings": [b.mapping for b in plan if b.kind == "heldout_mapping"],
            "heldout_variants": [b.variant for b in plan if b.kind == "heldout_mapping"],
            "min_replay_trials": replays,
            "localiser": any(b.localiser for b in plan)}
