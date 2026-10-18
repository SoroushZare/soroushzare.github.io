"""The data engine: collect → label → retrain → promote, so every hour of ordinary use can make the
decoder better, and an hour can make the deployed decoder worse only if a statistical test on
held-out cue-defined trials is fooled (≈ 2.5 % per comparison by construction).

Pipeline
--------
``SessionStore``  reads RunLogger run directories.  It opens exactly four files per run
                  (``eeg.f32``, ``eeg_ts.f64``, ``events.csv``, ``cursor.csv``) and nothing else, so an
                  auditor stream recorded beside the EEG (eye tracker, EMG, accelerometer) cannot reach
                  a decoder through this module — those belong to ``purity.py`` only.
``Labeler``       turns one run into labelled EEG windows.  Label = unit vector from the cursor *as the
                  person knows it* to the *revealed* goal at the window's end (the logged cursor when
                  it is drawn, the trial's start position when it is hidden); provenance records how
                  the label was obtained.
``Trainer``       fits a ``VelocityDecoder`` on selected provenance, optionally with a population prior
                  (other people's windows) and personal up-weighting ``w_own = max(1, n_pop / 300)``.
                  Visible-cursor labels are re-centred per block and free-use windows are capped
                  (see ``Trainer``): both label sources carry structure set by the deployed decoder.
``ModelRegistry`` append-only model versions (joblib + metadata json) and an append-only champion log.
``ChampionChallenger``  a new model replaces the deployed one only if a trial-level paired bootstrap
                  puts the lower 95 % bound of (challenger − champion) mean cosine above a margin, on
                  held-out cued_blank trials neither model was trained on: the FROZEN set from the
                  last promotion plus the newest cued trials recorded after the champion was trained.
                  A ``BodyCoordinateDecoder`` is judged by the body metric instead: per-axis AUC on
                  held-out L1 fixation probes (the selection probes), same rule.
``DataEngine``    glues them: ``ingest(run_dir)`` and ``maybe_retrain()``; with a ``BodyTrainer`` it
                  runs the champion's deployed decoder end to end (L1/L2b training, body metric).

Event schema (``events.csv``, written by ``RunLogger.event(kind, payload, t)``)
-----------------------------------------------------------------------------
For the original blocks only two kinds are read; every other kind (markers, gate flags, auditor sync
pulses) is ignored.  Champion trials read a few more (see "Champion trials" below).

``trial_start``  payload (JSON object)::

    {"trial": 17,                      # optional int id; pairs the start with its end
     "goal": [0.56, -0.56] | null,     # goal position in normalised screen units; null when the goal
                                       # is only revealed afterwards (free use: the selected target)
     "block": "cued_blank" | "closed_loop" | "blank_feedback" | "free",
     "feedback_visible": true | false, # was the cursor drawn while this trial ran
     "start_pos": [0.0, 0.0]}          # optional cursor position at onset, default [0, 0].  Visible
                                       # cursor: used until the first cursor.csv row inside the trial.
                                       # Hidden cursor (feedback_visible false): the label reference
                                       # for the whole trial, whatever cursor.csv says

Two further kinds are checked, never labelled:

``cue`` / ``cue_offset``  a cued_blank ``trial_start`` that follows a ``cue`` with no ``cue_offset`` in
                 between raises ``ValueError``: the cue stimulus (a central arrow maps one-to-one onto
                 the label) may still have been on screen during the labelled intent.
``realtime_start`` with ``"cursor_source": "model"``  the run logged the real-time loop's internal
                 cursor model, not the cursor that was displayed; its visible-cursor trials cannot be
                 labelled (the label must refer to what the person saw) and ``Labeler`` raises.

``trial_end``    payload (JSON object)::

    {"trial": 17,                      # optional; must match the open trial's id when both are given
     "goal": [x, y] | null,            # optional hindsight reveal, used only if trial_start had null.
                                       # A selection the person undid is not a reveal: write null
     ...}                              # anything else (outcome, dwell, ...) is ignored here

A ``trial_start`` that arrives while another trial is open discards the open one, and a trial with no
``trial_end`` is discarded: without a known end we cannot say which samples carried that intent, and a
crashed run must not inject mislabelled windows.  An unknown ``block`` raises ``ValueError`` — a
protocol that writes a new block kind must be added here deliberately, not silently relabelled.

Champion trials (body-coordinate SMR; ``protocol.session_plan_champion``)
-------------------------------------------------------------------------
Five more ``block`` values (``CHAMPION_BLOCKS``): "fixation_probe", "blind_closed_loop", "sighted",
"heldout_mapping" and "pixel_replay".  Their ``trial_start`` payload (``protocol.
champion_trial_start_payload``) adds::

    {"source": "L1" | "L2b" | "S" | "H" | "R" | "J",   # required, and allowed for the block
                                       # (SOURCES_BY_BLOCK): fixation_probe → L1; blind_closed_loop →
                                       # L2b, or J ("just intend"); sighted → S | J; heldout_mapping
                                       # → H; pixel_replay → R
     "mapping": "identity" | "rot+90" | "rot-90" | "mirror_x" | "mirror_y",
                                       # required: identity for L1/L2b/S/J, held out for H, the host
                                       # block's mapping for R
     "body_cue": [bx, by],             # L1/L2b: the cued body vector in {-1, 0, 1}², THE label
                                       # (L1: one class, so exactly one non-zero entry)
     "cue_class": "FEET+RIGHT", "cue_mode": "central_glyph" | "body_word" | "goal_word",
     "variant": "C2b" | "C2c",         # heldout_mapping: open loop (frozen cursor) or closed loop
     "replay_goal": [x, y],            # pixel_replay: goal A that the replayed trajectory aimed at;
                                       # "goal" is the goal B the person holds
     "instructed": true | false}       # whether imagery instruction was given (logged, not checked)

``feedback_visible`` must be false for fixation_probe / blind_closed_loop and true for the others.
"goal" may be null only for L1 (it has no goal) and L2b (the hidden goal is not needed for the label).
For all the other sources it is required at trial start: no hindsight reveal.  Further events:

``cue``           an L1, L2b or blind-J ``trial_start`` must have a ``cue`` since the previous
                  ``trial_start``, its ``cue_offset`` before the start, and the start at least
                  ``CUE_LOCKOUT_S`` (1.0 s) after cue onset and ``CUE_OFFSET_LOCKOUT_S`` (0.5 s)
                  after cue offset.  Otherwise ``ValueError``: the
                  class-specific cue-evoked response would sit inside the labelled windows.  A ``cue``
                  payload that carries ``"body"`` must equal the trial's ``body_cue``.
``fixation_break``  (gaze left the 1° window; logged by the presentation layer from the eye tracker)
                  discards the open L1 / L2b / blind-J trial.  The proposal aborts and repeats it.  Gaze
                  only selects trials here and never reaches a decoder: this module reads no auditor file.
``outcome_flash`` while a blind_closed_loop trial is open → ``ValueError``.  The flash must follow
                  ``trial_end``, so windows end before it.
``body_report``   ``{"trial", "class"}``, the C2b keyed report or the post-replay report of the held
                  intent, must come after that trial's ``trial_end`` (a keypress inside the window is a
                  movement → ``ValueError``).  It is attached to the held-out or replay trial as
                  ``body_report``.
``trial_end``     with ``"aborted": true`` discards a champion trial.

Labels and provenance of champion windows.  ``labels`` stay unit vectors.  For L1 / L2b they are the
normalised cued body vector (body = screen under the identity, and nothing is drawn).  For blind J they
are start_pos → goal.  For the visible kinds they follow the screen rule above (goal − displayed
cursor).  Cue-locked windows (L1, L2b, blind J) end at most ``CUE_LABEL_END_S`` (4.0 s) after cue onset,
and an L2b window ends no later than the first logged (hidden) cursor row inside the dwell radius
(``min_goal_dist``) of its goal (DESIGN_PURE_EEG §4's L2b window rule).  Provenance adds
``source``, ``mapping`` and ``instructed``; ``body_label`` ([bx, by], ints; the deployed decoder's
label), ``cue_class`` and ``cue_mode`` for L1 / L2b; ``cursor_pos`` / ``cursor_pos_start`` (the displayed cursor at the
window end / start) for visible trials; ``variant`` and ``body_report`` for H; ``replay_goal`` and
``body_report`` for R.  ``display_goal``
is the goal for the drawn-target kinds (``CHAMPION_TARGET_BLOCKS``) and None for L1 / L2b / blind J.
Windows of the original blocks keep exactly their original provenance keys.

Champion trainers: ``BodyTrainer`` fits the deployed ``BodyCoordinateDecoder`` and refuses any window
whose source is not L1 / L2b or whose ``display_goal`` is set.  ``ShadowTrainer`` fits the never-deployed leak positive control on S
windows only, with ReFIT labels mapped to body coordinates (``refit_body_labels``).  The original
``Trainer`` never selects champion blocks: they are not in ``BLOCK_KINDS``.

Windows
-------
EEG is replayed through the online preprocessing (``CausalFilter`` → ``AmplitudeGate`` → CAR over good
channels) so a decoder trained here sees exactly the window the real-time loop will hand it.  Window
length ``cfg.decoder.window_s``; window ends on a ``hop_s`` = 0.25 s grid from ``trial_start +
window_s`` to ``trial_end``, so every labelled window lies wholly inside one trial and windows outside
trials are never labelled.  Windows that start in the first ``settle_s`` of the run (filter transient),
that span a sample gap, or whose reference position is already within ``min_goal_dist`` of the goal
(direction undefined: the user is dwelling) are dropped.

Why hidden-cursor trials are labelled from ``start_pos``: the label must be the direction the person
intends, and with the cursor hidden (cued_blank, blank_feedback) the only cursor position they know
is where it started.  Labelling from the true, unseen cursor would add label noise that exists only
in hidden blocks, and the purity audit would then read the resulting visible-vs-hidden accuracy gap
as feedback circularity.

Provenance per window: ``{"block", "feedback_visible", "session", "t", "trial", "display_goal"}`` where
``t`` is the timestamp of the window's last sample (the time the label refers to), ``session`` the run
directory's name and ``display_goal`` the goal position as drawn on screen ([x, y]; None for
cued_blank, whose goal is a virtual encoding of the cue and is never drawn).  ``display_goal`` exists
for ``purity.py``'s display surrogate and never reaches a decoder.  ``(session, t)`` is the window's
identity everywhere in this module.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from scipy.stats import rankdata

from .body_decoder import AXES as BODY_AXES
from .body_decoder import TRAINABLE_SOURCES, BodyCoordinateDecoder
from .config import Config
from .decoder import VelocityDecoder
from .logger import RunLogger
from .mapping import BODY_VECTORS, HELDOUT_MAPPINGS, MAPPING_NAMES, TRAINING_MAPPING, Mapping
from .signal import AmplitudeGate, CausalFilter, common_average

BLOCK_KINDS: Tuple[str, ...] = ("cued_blank", "closed_loop", "blank_feedback", "free")
"""The original protocol's blocks: the ones ``Trainer`` can select."""
CHAMPION_BLOCKS: Tuple[str, ...] = ("fixation_probe", "blind_closed_loop", "sighted", "heldout_mapping",
                                    "pixel_replay")
ALL_BLOCKS: Tuple[str, ...] = BLOCK_KINDS + CHAMPION_BLOCKS
EVENT_TRIAL_START = "trial_start"
EVENT_TRIAL_END = "trial_end"
EVENT_CUE, EVENT_CUE_OFFSET = "cue", "cue_offset"
EVENT_REALTIME_START = "realtime_start"
EVENT_FIXATION_BREAK = "fixation_break"
EVENT_OUTCOME_FLASH = "outcome_flash"
EVENT_BODY_REPORT = "body_report"
TARGET_BLOCKS: Tuple[str, ...] = ("closed_loop", "blank_feedback", "free")
"""Blocks that draw the targets (and so the goal) at their screen positions (protocol.screen_items)."""
CHAMPION_TARGET_BLOCKS: Tuple[str, ...] = ("sighted", "heldout_mapping", "pixel_replay")
"""Champion blocks that draw the goal; L1 and L2b draw nothing at a trial-dependent position."""
SOURCES_BY_BLOCK: Dict[str, Tuple[str, ...]] = {
    "fixation_probe": ("L1",), "blind_closed_loop": ("L2b", "J"), "sighted": ("S", "J"),
    "heldout_mapping": ("H",), "pixel_replay": ("R",),
}
DEPLOYED_SOURCES: Tuple[str, ...] = tuple(sorted(TRAINABLE_SOURCES))
"""L1 and L2b: labels nothing on the screen can identify (``body_decoder.TRAINABLE_SOURCES``)."""
SHADOW_SOURCES: Tuple[str, ...] = ("S",)
"""The shadow decoder's training data: sighted identity-mapping closed loop."""
FIXATION_BLOCKS: Tuple[str, ...] = ("fixation_probe", "blind_closed_loop")
"""Blocks with gaze-contingent fixation; a ``fixation_break`` discards their open trial."""
CUE_LOCKOUT_S = 1.0
"""Cued champion windows start ≥ this long after cue onset (``protocol.CUE_LOCKOUT_S``)."""
CUE_OFFSET_LOCKOUT_S = 0.5
"""…and ≥ this long after cue offset (``protocol.CUE_OFFSET_LOCKOUT_S``): the offset response of a glyph
or word that identifies its class within a session must not fall inside a labelled window."""
L1_LABEL_END_S = 4.0
"""Fixation-probe windows end ≤ this long after cue onset (``protocol.L1_LABEL_END_S``)."""
CUE_LABEL_END_S = 4.0
"""Every cue-locked window (L1, L2b, blind J) ends ≤ this long after its cue's onset (DESIGN_PURE_EEG §4:
"windows ending 1.0–4.0 s after each cue").  The ERD peaks at 1–2 s and decays (ds004362: feet/fists
d′ 0.667 → 0.610 by 3–4 s), so a blind trial running to its 10-s timeout would otherwise fill the
deployed decoder's training set with decayed windows (79 % of L2b windows ended after 4 s, and L2b
was ~73 % of a person's own training windows).  Re-cues, once the protocol has them, restart it."""
HELDOUT_VARIANTS: Tuple[str, ...] = ("C2b", "C2c")
REPORTED_BLOCKS: Tuple[str, ...] = ("heldout_mapping", "pixel_replay")
"""Trials a keyed ``body_report`` attaches to: the C2b report and the post-replay report of the held
intent; certification keeps only trials whose report confirms the intent (``certify.confirmed_by_report``)."""
RUN_FILES: Tuple[str, ...] = ("eeg.f32", "eeg_ts.f64", "events.csv", "cursor.csv")
"""The only files ``SessionStore`` ever opens in a run directory (pure-EEG rule: auditor streams that
share the directory are invisible to the decoder path by construction, not by convention)."""

Key = Tuple[str, float]


# ----------------------------------------------------------------------------------------------
# Session store
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Trial:
    """One trial as parsed from events.csv; ``goal`` is None when it was never revealed.

    Champion trials also carry their ``source`` and validated champion fields in ``info``
    (mapping, instructed, and per source cue_t / body_cue / cue_class / cue_mode, variant /
    body_report, replay_goal); for the original blocks ``source`` is None and ``info`` empty."""
    session: str
    trial: int
    t_start: float
    t_end: float
    goal: Optional[np.ndarray]
    block: str
    feedback_visible: bool
    start_pos: np.ndarray
    source: Optional[str] = None
    info: Dict[str, object] = field(default_factory=dict)


@dataclass
class Session:
    """Everything the labeler may read from one run: EEG, its clock, events and cursor positions."""
    id: str
    dir: str
    eeg: np.ndarray            # (n_samples, n_channels) float32 raw µV
    ts: np.ndarray             # (n_samples,) float64
    events: List[Tuple[float, str, object]]
    cursor: np.ndarray         # (n_rows, 3): t, x, y — displayed/hidden cursor position per control step
    trials: List[Trial]


class SessionStore:
    """Lists and loads RunLogger run directories under ``root``.

    A run directory is any sub-directory holding ``eeg.f32``; listing is sorted by name, and RunLogger
    names runs ``run_<UTC stamp>[_tag]`` so name order is recording order.
    """

    def __init__(self, root: str, n_channels: int = 32):
        self.root = root
        self.n_channels = n_channels

    def list_runs(self) -> List[str]:
        if not os.path.isdir(self.root):
            return []
        out = []
        for name in sorted(os.listdir(self.root)):
            d = os.path.join(self.root, name)
            if os.path.isfile(os.path.join(d, "eeg.f32")):
                out.append(d)
        return out

    def load(self, run_dir: str) -> Session:
        sid = os.path.basename(os.path.normpath(run_dir))
        eeg, ts = RunLogger.load_eeg(run_dir, self.n_channels)
        events = self.read_events(os.path.join(run_dir, "events.csv"))
        cursor = self.read_cursor(os.path.join(run_dir, "cursor.csv"))
        return Session(sid, run_dir, eeg, ts, events, cursor, parse_trials(sid, events))

    @staticmethod
    def read_events(path: str) -> List[Tuple[float, str, object]]:
        """RunLogger writes ``t,kind,<raw JSON>`` without CSV quoting, so the payload (which may hold
        commas) is everything after the second comma."""
        out = []
        with open(path) as f:
            next(f, None)                                   # header
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                t, kind, payload = line.split(",", 2)
                out.append((float(t), kind, json.loads(payload)))
        out.sort(key=lambda e: e[0])
        return out

    @staticmethod
    def read_cursor(path: str) -> np.ndarray:
        if not os.path.exists(path):
            return np.zeros((0, 3))
        with open(path) as f:
            next(f, None)                                   # header
            lines = [ln for ln in f if ln.strip()]
        if not lines:                                       # cursor hidden and unlogged (cued_blank)
            return np.zeros((0, 3))
        rows = np.loadtxt(lines, delimiter=",", ndmin=2)
        rows = rows[np.argsort(rows[:, 0], kind="stable")]
        return rows[:, :3]


def parse_trials(session: str, events: Sequence[Tuple[float, str, object]]) -> List[Trial]:
    """Pair trial_start/trial_end events per the schema in the module docstring."""
    trials: List[Trial] = []
    open_: Optional[Tuple[float, dict, int, Optional[dict]]] = None
    counter = 0
    cue_on = False                                          # a cue was presented and has not ended
    cue_t: Optional[float] = None                           # onset of the last cue since the last trial_start
    cue_off_t: Optional[float] = None                       # offset of that cue
    cue_body: Optional[list] = None                         # that cue's body vector, when it logged one
    reports: Dict[int, Tuple[int, int]] = {}                # C2b keyed reports by trial id
    for t, kind, payload in events:
        if kind == EVENT_CUE:
            cue_on, cue_t, cue_off_t = True, t, None
            cue_body = payload.get("body") if isinstance(payload, dict) else None
        elif kind == EVENT_CUE_OFFSET:
            cue_on, cue_off_t = False, t
        elif kind == EVENT_FIXATION_BREAK:
            if open_ is not None and open_[1].get("block") in FIXATION_BLOCKS:
                open_ = None                                # aborted: fixation (the label's premise) failed
        elif kind == EVENT_OUTCOME_FLASH:
            if open_ is not None and open_[1].get("block") == "blind_closed_loop":
                raise ValueError(f"{session}: '{EVENT_OUTCOME_FLASH}' at t={t} inside an open blind trial; the "
                                 f"outcome display must follow trial_end so no labelled window sees it")
        elif kind == EVENT_BODY_REPORT:
            q = payload if isinstance(payload, dict) else {}
            if q.get("trial") is None or q.get("class") not in BODY_VECTORS:
                raise ValueError(f"{session}: '{EVENT_BODY_REPORT}' at t={t} needs a trial id and a class "
                                 f"in {tuple(BODY_VECTORS)}, got {payload!r}")
            if open_ is not None and int(q["trial"]) == open_[2]:
                raise ValueError(f"{session}: body report at t={t} inside its own trial; the keypress must "
                                 f"follow trial_end")
            reports[int(q["trial"])] = tuple(int(round(v)) for v in BODY_VECTORS[q["class"]])
        elif kind == EVENT_TRIAL_START:
            p = payload or {}
            block = p.get("block")
            if block not in ALL_BLOCKS:
                raise ValueError(f"{session}: trial_start at t={t} has block {block!r}; "
                                 f"expected one of {ALL_BLOCKS}")
            if block == "cued_blank" and cue_on:
                raise ValueError(f"{session}: cued_blank trial_start at t={t} follows a cue with no "
                                 f"'{EVENT_CUE_OFFSET}' event; the cue may have been on screen during the "
                                 f"labelled intent")
            if "feedback_visible" not in p:
                raise ValueError(f"{session}: trial_start at t={t} lacks 'feedback_visible'")
            info = (_champion_info(session, t, p, cue_on, cue_t, cue_body, cue_off_t)
                    if block in CHAMPION_BLOCKS else None)
            cue_t, cue_off_t, cue_body = None, None, None   # a cue belongs to the one trial it precedes
            tid = int(p["trial"]) if p.get("trial") is not None else counter
            counter += 1
            open_ = (t, p, tid, info)
        elif kind == EVENT_TRIAL_END and open_ is not None:
            t0, p, tid, info = open_
            q = payload or {}
            if q.get("trial") is not None and int(q["trial"]) != tid:
                continue                                    # an end for some other (discarded) trial
            open_ = None
            if info is not None and q.get("aborted"):
                continue                                    # aborted champion trial: repeated, never labelled
            goal = p.get("goal") if p.get("goal") is not None else q.get("goal")
            trials.append(Trial(
                session=session, trial=tid, t_start=t0, t_end=t,
                goal=None if goal is None else np.asarray(goal, float),
                block=p["block"], feedback_visible=bool(p["feedback_visible"]),
                start_pos=np.asarray(p.get("start_pos", (0.0, 0.0)), float),
                source=None if info is None else info["source"], info=info or {}))
    return [replace(tr, info=dict(tr.info, body_report=reports.get(tr.trial)))
            if tr.block in REPORTED_BLOCKS else tr for tr in trials]


def _body_cue(session: str, t: float, value) -> Tuple[int, int]:
    """Validate a ``body_cue``: two integers in {−1, 0, 1}, not both zero."""
    v = np.asarray(value if value is not None else (), float)
    if v.shape != (2,) or not np.isin(v, (-1, 0, 1)).all() or not v.any():
        raise ValueError(f"{session}: trial_start at t={t} needs 'body_cue' [bx, by] in {{-1, 0, 1}}², "
                         f"not both 0; got {value!r}")
    return int(v[0]), int(v[1])


def _champion_info(session: str, t: float, p: dict, cue_on: bool, cue_t: Optional[float],
                   cue_body: Optional[list], cue_off_t: Optional[float] = None) -> dict:
    """Validate a champion ``trial_start`` (module docstring) and return the fields labelling needs."""
    block, src = p["block"], p.get("source")
    where = f"{session}: {block} trial_start at t={t}"
    if src not in SOURCES_BY_BLOCK[block]:
        raise ValueError(f"{where} has source {src!r}; expected one of {SOURCES_BY_BLOCK[block]}")
    if bool(p["feedback_visible"]) != (block in CHAMPION_TARGET_BLOCKS):
        raise ValueError(f"{where}: feedback_visible must be {block in CHAMPION_TARGET_BLOCKS}")
    mapping = p.get("mapping")
    if mapping not in MAPPING_NAMES:
        raise ValueError(f"{where}: mapping {mapping!r} not in {MAPPING_NAMES}")
    if src == "H" and mapping not in HELDOUT_MAPPINGS:
        raise ValueError(f"{where}: a held-out block needs a held-out mapping {HELDOUT_MAPPINGS}, got {mapping!r}")
    if src in ("L1", "L2b", "S", "J") and mapping != TRAINING_MAPPING:
        raise ValueError(f"{where}: source {src} runs under {TRAINING_MAPPING!r} only, got {mapping!r}")
    info: Dict[str, object] = {"source": src, "mapping": mapping, "instructed": p.get("instructed")}
    if src in DEPLOYED_SOURCES or (src == "J" and block == "blind_closed_loop"):
        if cue_t is None:
            raise ValueError(f"{where} has no 'cue' event since the previous trial; its cue onset (and so "
                             f"the {CUE_LOCKOUT_S} s lockout) is unknown")
        if cue_on:
            raise ValueError(f"{where} follows a cue with no '{EVENT_CUE_OFFSET}' event; the cue may still "
                             f"have been presented during the labelled intent")
        if t - cue_t < CUE_LOCKOUT_S - 1e-9:
            raise ValueError(f"{where} is {t - cue_t:.3f} s after cue onset; labelled windows must start "
                             f">= {CUE_LOCKOUT_S} s after it (cue-evoked responses differ by class)")
        if cue_off_t is None or t - cue_off_t < CUE_OFFSET_LOCKOUT_S - 1e-9:
            gap = "no cue_offset" if cue_off_t is None else f"{t - cue_off_t:.3f} s after cue offset"
            raise ValueError(f"{where} is {gap}; labelled windows must start >= {CUE_OFFSET_LOCKOUT_S} s "
                             f"after the cue ends (its offset response identifies the class)")
        info["cue_t"] = cue_t
    if src in DEPLOYED_SOURCES:
        body = _body_cue(session, t, p.get("body_cue"))
        if src == "L1" and sum(v != 0 for v in body) != 1:
            raise ValueError(f"{where}: a fixation probe cues one class, got body_cue {list(body)}")
        if cue_body is not None and tuple(int(v) for v in cue_body) != body:
            raise ValueError(f"{where}: body_cue {list(body)} differs from the presented cue {cue_body}")
        info.update(body_cue=body, cue_class=p.get("cue_class"), cue_mode=p.get("cue_mode"))
    elif p.get("goal") is None:
        raise ValueError(f"{where}: source {src} needs its goal at trial start")
    if src == "H":
        if p.get("variant") not in HELDOUT_VARIANTS:
            raise ValueError(f"{where}: variant must be one of {HELDOUT_VARIANTS}, got {p.get('variant')!r}")
        info["variant"] = p["variant"]
    if src == "R":
        rg = np.asarray(p.get("replay_goal") if p.get("replay_goal") is not None else (), float)
        if rg.shape != (2,):
            raise ValueError(f"{where}: a pixel-replay trial needs 'replay_goal' [x, y]")
        info["replay_goal"] = rg.tolist()
    return info


# ----------------------------------------------------------------------------------------------
# Labelled windows
# ----------------------------------------------------------------------------------------------
@dataclass
class LabelledSet:
    """Windows (list of (n_samples, n_channels) float32 arrays, as ``VelocityDecoder.fit`` takes them),
    unit-vector labels (n, 2) and one provenance dict per window.  Order is ingestion order, and
    time order within a session — "most recent" in this module means "last in this order"."""
    windows: List[np.ndarray] = field(default_factory=list)
    labels: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    provenance: List[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.windows)

    def keys(self) -> List[Key]:
        return [(p["session"], float(p["t"])) for p in self.provenance]

    def take(self, idx: Iterable[int]) -> "LabelledSet":
        idx = list(idx)
        return LabelledSet([self.windows[i] for i in idx], self.labels[idx].reshape(-1, 2),
                           [self.provenance[i] for i in idx])

    def mask(self, m: np.ndarray) -> "LabelledSet":
        return self.take(np.flatnonzero(m))

    @staticmethod
    def concat(sets: Sequence["LabelledSet"]) -> "LabelledSet":
        sets = [s for s in sets if len(s)]
        if not sets:
            return LabelledSet()
        return LabelledSet([w for s in sets for w in s.windows], np.concatenate([s.labels for s in sets]),
                           [p for s in sets for p in s.provenance])

    def mix(self) -> dict:
        """Provenance mix: how the labels were obtained, for registry metadata and audits."""
        return {"n": len(self),
                "block": dict(Counter(p["block"] for p in self.provenance)),
                "source": dict(Counter(p["source"] for p in self.provenance if "source" in p)),
                "feedback_visible": int(sum(bool(p["feedback_visible"]) for p in self.provenance)),
                "sessions": sorted({p["session"] for p in self.provenance})}

    def data_hash(self) -> str:
        """sha256 over keys, labels and window bytes: two models with the same hash saw the same data."""
        h = hashlib.sha256()
        h.update(json.dumps(self.keys()).encode())
        h.update(np.ascontiguousarray(self.labels, dtype=np.float64).tobytes())
        for w in self.windows:
            h.update(np.ascontiguousarray(w, dtype=np.float32).tobytes())
        return h.hexdigest()


class Labeler:
    """Builds labelled windows from one Session by replaying the online preprocessing.

    Hyperparameters (all explicit):
      hop_s          0.25 s between window ends.  Windows overlap by window_s − hop_s; denser hops add
                     near-duplicate windows, not information.
      settle_s       2.0 s: the 0.5 Hz 4th-order high-pass rings for well over a second after the first
                     sample; windows starting earlier carry filter transient, not brain.
      min_goal_dist  cfg.task.target_radius: inside the target the user is dwelling and the cursor→goal
                     direction is noise.
      max_span_excess  0.1 × window_s: a window whose timestamps span more than window_s by this much
                     straddles a dropout and is not the contiguous window the online loop would see.
      filter_chunk_s 60 s: the causal filter is chunk-invariant, so chunking only bounds memory.
    """

    def __init__(self, cfg: Config, hop_s: float = 0.25, settle_s: float = 2.0,
                 min_goal_dist: Optional[float] = None, filter_chunk_s: float = 60.0):
        self.cfg = cfg
        self.hop_s = hop_s
        self.settle_s = settle_s
        self.min_goal_dist = cfg.task.target_radius if min_goal_dist is None else min_goal_dist
        self.filter_chunk_s = filter_chunk_s
        self.fs = cfg.hardware.fs
        self.n_win = int(round(cfg.decoder.window_s * self.fs))
        self.max_span = cfg.decoder.window_s * 1.1

    def signature(self) -> str:
        """Hash of everything that changes the produced windows; used to invalidate label caches."""
        d = dict(label_rule="hidden_cursor_from_start_pos+display_goal+champion_sources_v2_cue_end_4s_dwell",
                 hop_s=self.hop_s,
                 settle_s=self.settle_s,
                 min_goal_dist=self.min_goal_dist,
                 window_s=self.cfg.decoder.window_s, fs=self.fs, pre=repr(self.cfg.pre),
                 hw=repr(self.cfg.hardware))
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def _filtered(self, eeg: np.ndarray) -> np.ndarray:
        filt = CausalFilter(self.cfg.hardware, self.cfg.pre)
        out = np.empty(eeg.shape, dtype=np.float32)
        step = max(1, int(self.filter_chunk_s * self.fs))
        for i in range(0, len(eeg), step):
            out[i:i + step] = filt(eeg[i:i + step])
        return out

    def _reference_pos(self, cursor: np.ndarray, trial: Trial, t: float) -> np.ndarray:
        """Where the person believes the cursor is at time t: with the cursor drawn, the zero-order
        hold of the logged cursor inside the trial (start_pos before its first row); with it hidden,
        start_pos for the whole trial (see the module docstring)."""
        if not trial.feedback_visible:
            return trial.start_pos
        k = int(np.searchsorted(cursor[:, 0], t, side="right")) - 1
        if k >= 0 and cursor[k, 0] >= trial.t_start:
            return cursor[k, 1:3]
        return trial.start_pos

    def _first_inside(self, cursor: np.ndarray, trial: Trial) -> float:
        """Time of the trial's first logged cursor row within ``min_goal_dist`` of its (hidden) goal, or
        +inf.  An L2b window ending after it is not labelled (DESIGN_PURE_EEG §4): the person is dwelling
        on a goal they cannot see and the cue's imagery has done its job."""
        rows = cursor[(cursor[:, 0] >= trial.t_start) & (cursor[:, 0] <= trial.t_end)] if len(cursor) else cursor
        if not len(rows):
            return float("inf")
        inside = np.linalg.norm(rows[:, 1:3] - trial.goal[None], axis=1) < self.min_goal_dist
        return float(rows[np.argmax(inside), 0]) if inside.any() else float("inf")

    @staticmethod
    def check_cursor_source(session: Session) -> None:
        """Refuse visible-cursor trials of a run whose cursor.csv is not the displayed cursor."""
        sources = {p.get("cursor_source") for _, kind, p in session.events
                   if kind == EVENT_REALTIME_START and isinstance(p, dict)}
        if "model" in sources and any(tr.feedback_visible for tr in session.trials):
            raise ValueError(f"{session.id}: cursor.csv holds the real-time loop's internal cursor model "
                             f"(cursor_source 'model'), not the displayed cursor, so its visible-cursor "
                             f"trials cannot be labelled; drive a displaying effector or hide the cursor")

    def label(self, session: Session) -> LabelledSet:
        self.check_cursor_source(session)
        ts = session.ts
        if len(ts) < self.n_win:
            return LabelledSet()
        xf = self._filtered(session.eeg)
        gate = AmplitudeGate(self.cfg.hardware, self.cfg.pre)
        half = 0.5 / self.fs
        t_min = ts[0] + self.settle_s
        windows, labels, prov = [], [], []
        for tr in session.trials:
            cue_labelled = tr.source in DEPLOYED_SOURCES        # L1 / L2b: the label is the cue itself
            if tr.goal is None and not cue_labelled:
                continue
            t_last = tr.t_end
            if tr.info.get("cue_t") is not None:                # L1, L2b, blind J: cue-locked windows
                t_last = min(t_last, float(tr.info["cue_t"]) + CUE_LABEL_END_S)
            if cue_labelled and tr.goal is not None:            # L2b: stop once the hidden cursor dwells
                t_last = min(t_last, self._first_inside(session.cursor, tr))
            n_ends = int(math.floor((t_last - tr.t_start - self.cfg.decoder.window_s) / self.hop_s + 1e-9)) + 1
            for k in range(max(0, n_ends)):
                te = tr.t_start + self.cfg.decoder.window_s + k * self.hop_s
                e = int(np.searchsorted(ts, te + half, side="right"))
                s = e - self.n_win
                if s < 0 or ts[s] < tr.t_start - half or ts[s] < t_min:
                    continue
                if ts[e - 1] - ts[s] > self.max_span:
                    continue
                t_end = float(ts[e - 1])
                ref = None if cue_labelled else self._reference_pos(session.cursor, tr, t_end)
                ref0 = (self._reference_pos(session.cursor, tr, float(ts[s]))
                        if tr.feedback_visible and tr.source is not None else None)
                d = np.asarray(tr.info["body_cue"], float) if cue_labelled else tr.goal - ref
                dist = float(np.linalg.norm(d))
                if not cue_labelled and dist < self.min_goal_dist:
                    continue
                raw = xf[s:e]
                good = gate.update(raw)
                windows.append(common_average(raw, good).astype(np.float32))
                labels.append(d / dist)
                p = {"block": tr.block, "feedback_visible": tr.feedback_visible,
                     "session": session.id, "t": t_end, "trial": tr.trial,
                     "display_goal": (tr.goal.tolist() if tr.block in TARGET_BLOCKS + CHAMPION_TARGET_BLOCKS
                                      else None)}
                if tr.source is not None:
                    p.update(self._champion_provenance(tr, ref, ref0))
                prov.append(p)
        return LabelledSet(windows, np.asarray(labels, float).reshape(-1, 2), prov)

    @staticmethod
    def _champion_provenance(tr: Trial, ref: Optional[np.ndarray], ref_start: Optional[np.ndarray]) -> dict:
        """The champion provenance fields of one window (module docstring); ``ref`` / ``ref_start`` are
        the cursor as the person knows it at the window end / start (None for cue-labelled windows).
        The start position is what certification regresses on in closed loop: the cursor at the window
        end has already been moved by this window's own decoded output (endogenous)."""
        out: Dict[str, object] = {"source": tr.source, "mapping": tr.info["mapping"],
                                  "instructed": tr.info["instructed"]}
        if tr.source in DEPLOYED_SOURCES:
            out.update(body_label=list(tr.info["body_cue"]), cue_class=tr.info["cue_class"],
                       cue_mode=tr.info["cue_mode"])
        if tr.feedback_visible:
            out["cursor_pos"] = [float(v) for v in ref]
            out["cursor_pos_start"] = [float(v) for v in ref_start]
        if tr.source == "H":
            report = tr.info.get("body_report")
            out.update(variant=tr.info["variant"], body_report=None if report is None else list(report))
        if tr.source == "R":
            report = tr.info.get("body_report")
            out.update(replay_goal=list(tr.info["replay_goal"]), body_report=None if report is None else list(report))
        return out


# ----------------------------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------------------------
class WeightedVelocityDecoder(VelocityDecoder):
    """``VelocityDecoder`` with a sample-weighted fit; prediction is inherited unchanged, so the
    real-time loop cannot tell it apart.  The weighted path lives here, not in decoder.py, because
    weighting is a data-engine policy (population prior vs own data), not a decoder property."""

    def fit_weighted(self, windows: Sequence[np.ndarray], targets: np.ndarray,
                     weights: np.ndarray) -> "WeightedVelocityDecoder":
        X = np.stack([self.features(w) for w in windows])
        w = np.asarray(weights, float)
        self._scaler.fit(X, sample_weight=w)
        self._model.fit(self._scaler.transform(X), np.asarray(targets, float), sample_weight=w)
        return self


def check_eeg_windows(windows: Sequence[np.ndarray], n_channels: int, what: str) -> None:
    """Pure-EEG guard at the trainer's door: every window must be (n_samples, n_channels) of the EEG
    montage.  An eye-tracker (2–4 columns) or a mis-wired auditor stream fails loudly here."""
    for w in windows:
        if np.ndim(w) != 2 or np.shape(w)[1] != n_channels:
            raise ValueError(f"{what}: window of shape {np.shape(w)} is not EEG "
                             f"(expected (n_samples, {n_channels})); only EEG may reach a decoder")


def personal_weight(n_pop: int, ref: float = 300.0) -> float:
    """w_own = max(1, n_pop / ref): with ≤ ref population windows the person's own data counts once;
    beyond that, each own window is up-weighted so a large prior cannot drown the individual
    topography (ref = 300 windows ≈ 75 s of labelled use at 4 windows/s).

    Consequence, stated so results are read correctly: once n_pop ≥ ref the population's share of the
    total loss weight is exactly ref / (n_own + ref), whatever n_pop is — the prior acts as a fixed
    ``ref``-window pseudo-count.  Growing the population (``--pop N``) therefore changes which people
    the prior is made of (its composition and diversity), never how strongly it counts; ``ref``
    (``Trainer.pop_ref_windows``) is the knob for the prior's strength and is the one to sweep."""
    return max(1.0, n_pop / ref)


class Trainer:
    """Fits the decoder on windows whose provenance block is in ``blocks``.

    All four blocks are used by default: more labelled data is the point of the engine, and the
    champion test on cued_blank windows (never display-identifiable) is what stops a decoder that only
    learned the displayed cursor from being deployed.  Narrow ``blocks`` when the purity audit flags
    FEEDBACK_CIRCULAR.  Two label sources carry structure set by the *deployed decoder* rather than by
    intent, and are corrected before fitting:

    * Visible-cursor labels (``recenter_visible``, default on).  The label is goal − displayed cursor,
      and the cursor sits wherever the deployed decoder's bias pushed it, so a block's mean label
      points against that bias whatever the EEG holds (an intercept-only predictor can out-score the
      EEG decoder there).  Fitted raw, each champion learns the previous one's bias in reverse and the
      intercept flips between days.  Subtracting each (session, block)'s mean label removes that
      display-driven component; goals are random per trial, so the true block mean is ≈ 0 and little
      intent is lost.  Hidden-cursor labels (cued_blank, blank_feedback) start from ``start_pos`` and
      are left untouched.
    * Free-use windows (``max_free_share``, default 0.2).  Hindsight labels exist only for selections
      the decoder plus assistance completed, so they over-represent windows where the deployed
      decoder already pointed at the goal and reinforce it instead of correcting it.  Their total
      weight is capped at this share of the person's own training weight (they still count, at most
      as much as one fifth of it).

    Weights are normalised to mean 1 so the ridge penalty keeps its calibrated strength
    (cfg.decoder.ridge_alpha) whatever w_own is: with raw weights a 100× up-weighting would act like
    alpha / 100.  (The population prior is a fixed ``pop_ref_windows`` pseudo-count; see
    ``personal_weight``.)
    """

    def __init__(self, cfg: Config, blocks: Tuple[str, ...] = BLOCK_KINDS,
                 feature_set: str = "bandpower", pop_ref_windows: float = 300.0,
                 recenter_visible: bool = True, max_free_share: float = 0.2):
        unknown = set(blocks) - set(BLOCK_KINDS)
        if unknown:
            raise ValueError(f"unknown blocks {unknown}")
        if not 0.0 < max_free_share <= 1.0:
            raise ValueError(f"max_free_share must be in (0, 1], got {max_free_share}")
        self.cfg = cfg
        self.blocks = tuple(blocks)
        self.feature_set = feature_set
        self.pop_ref_windows = pop_ref_windows
        self.recenter_visible = recenter_visible
        self.max_free_share = max_free_share
        self.last_fit_info: dict = {}

    @staticmethod
    def recentred_labels(data: LabelledSet) -> np.ndarray:
        """Labels with each visible-cursor (session, block)'s mean label subtracted (see class doc)."""
        y = np.array(data.labels, float)
        groups: Dict[Tuple[str, str], List[int]] = {}
        for i, p in enumerate(data.provenance):
            if p["feedback_visible"]:
                groups.setdefault((p["session"], p["block"]), []).append(i)
        for idx in groups.values():
            y[idx] -= y[idx].mean(axis=0)
        return y

    def free_weight(self, data: LabelledSet) -> float:
        """Per-window weight of free-use windows that caps their share of own weight at max_free_share."""
        n_free = sum(p["block"] == "free" for p in data.provenance)
        n_other = len(data) - n_free
        if n_free == 0 or n_other == 0 or n_free / len(data) <= self.max_free_share:
            return 1.0
        return self.max_free_share * n_other / ((1.0 - self.max_free_share) * n_free)

    def select(self, data: LabelledSet) -> LabelledSet:
        return data.mask(np.array([p["block"] in self.blocks for p in data.provenance], bool))

    def fit(self, data: LabelledSet,
            population: Optional[Sequence[Tuple[Sequence[np.ndarray], np.ndarray]]] = None
            ) -> WeightedVelocityDecoder:
        own = self.select(data)
        if len(own) == 0:
            raise ValueError("no windows with the selected provenance")
        nch = self.cfg.hardware.n_channels
        check_eeg_windows(own.windows, nch, "own data")
        windows: List[np.ndarray] = list(own.windows)
        targets = [self.recentred_labels(own) if self.recenter_visible else own.labels]
        w_free = self.free_weight(own)
        own_w = np.array([w_free if p["block"] == "free" else 1.0 for p in own.provenance])
        n_pop = 0
        for i, (pw, py) in enumerate(population or ()):
            check_eeg_windows(pw, nch, f"population member {i}")
            py = np.asarray(py, float).reshape(-1, 2)
            if len(pw) != len(py):
                raise ValueError(f"population member {i}: {len(pw)} windows vs {len(py)} labels")
            windows.extend(pw)
            targets.append(py)
            n_pop += len(pw)
        w_own = personal_weight(n_pop, self.pop_ref_windows)
        w = np.concatenate([own_w * w_own, np.ones(n_pop)])
        w *= len(w) / w.sum()
        dec = WeightedVelocityDecoder(self.cfg.decoder, self.cfg.hardware.fs, feature_set=self.feature_set)
        dec.fit_weighted(windows, np.concatenate(targets), w)
        free = np.array([p["block"] == "free" for p in own.provenance], bool)
        self.last_fit_info = {"n_own": len(own), "n_population": n_pop, "w_own": w_own,
                              "blocks": list(self.blocks), "feature_set": self.feature_set,
                              "recenter_visible": self.recenter_visible, "free_weight": w_free,
                              "free_weight_share": float(own_w[free].sum() / own_w.sum()) if free.any() else 0.0,
                              "train_mix": own.mix(), "data_hash": own.data_hash()}
        return dec


# ----------------------------------------------------------------------------------------------
# Champion trainers: the deployed body decoder (L1/L2b only) and the shadow positive control (S only)
# ----------------------------------------------------------------------------------------------
def require_sources(data: LabelledSet, allowed: Sequence[str], who: str) -> None:
    """Raise unless every window's provenance ``source`` is in ``allowed``.

    The trainers refuse mixed data instead of filtering it.  A silent filter would hide a caller that
    passed S, H, R or J windows, and the certification then only tests the decoder that caller thought
    it trained.  ``BodyTrainer.select`` / ``ShadowTrainer.select`` make the selection explicit."""
    bad = Counter(str(p.get("source")) for p in data.provenance if p.get("source") not in allowed)
    if bad:
        raise ValueError(f"{who} trains only on source(s) {list(allowed)}; got {dict(bad)} windows of other "
                         f"sources (None = an original-protocol block). S has no deployed label, H/R/J "
                         f"are test-only")


def body_labels(data: LabelledSet) -> np.ndarray:
    """(n, 2) integer cued body vectors (``body_label``) of L1 / L2b windows."""
    missing = [i for i, p in enumerate(data.provenance) if p.get("body_label") is None]
    if missing:
        raise ValueError(f"{len(missing)} windows carry no cue-defined body label (only L1/L2b do)")
    return np.array([p["body_label"] for p in data.provenance], int).reshape(-1, 2)


def refit_body_labels(data: LabelledSet) -> np.ndarray:
    """ReFIT labels in body coordinates: ``M⁻¹ u`` per window, with ``u`` = the window's screen label
    (goal − displayed cursor, unit) and ``M`` its block's mapping.  This is the body velocity a decoder
    that reads the screen would be taught to output."""
    out = np.empty((len(data), 2))
    for i, p in enumerate(data.provenance):
        if p.get("mapping") is None:
            raise ValueError("ReFIT body labels need the window's screen mapping (champion provenance)")
        out[i] = Mapping(p["mapping"]).screen_to_body(data.labels[i])
    return out


def _source_mask(data: LabelledSet, sources: Sequence[str]) -> np.ndarray:
    return np.array([p.get("source") in sources for p in data.provenance], bool)


class BodyTrainer:
    """Fits the deployed ``BodyCoordinateDecoder`` on fixation probes (L1) and blind closed loop (L2b).

    Those are the only labels nothing on the screen can identify.  ``fit`` raises on any other window
    (``require_sources``) and on any window whose provenance records a drawn goal (``display_goal`` not
    None: a mislabelled source must not slip a screen label in), and so does the decoder itself.  It
    reads ``source``, ``display_goal``, ``body_label`` and ``session`` to train, and ``block``,
    ``feedback_visible`` and ``t`` for its provenance summary and data hash; never a gaze field (a
    fixation break only ever removes a trial in ``parse_trials``).  Sighted data reach the deployed decoder
    only through unsupervised re-centring (``session_references``, rest covariances per session),
    never as labels.  Pass windows of the decoder's length: label with ``Labeler(decoder.loop_config(
    cfg))`` (1.0-s windows by default).

    window_s          1.0 s, the champion's real-time window (the proposal's per-window unit).
    decoder_options   extra ``BodyCoordinateDecoder`` hyperparameters (band, C, x-earning rule).
    """

    def __init__(self, cfg: Config, window_s: float = 1.0, decoder_options: Optional[dict] = None):
        self.cfg = cfg
        self.window_s = window_s
        self.decoder_options = dict(decoder_options or {})
        self.last_fit_info: dict = {}

    @staticmethod
    def select(data: LabelledSet) -> LabelledSet:
        """The L1 / L2b windows of ``data``."""
        return data.mask(_source_mask(data, DEPLOYED_SOURCES))

    def fit(self, data: LabelledSet, session_references: Optional[Dict[str, np.ndarray]] = None, *,
            x_mask: Optional[np.ndarray] = None, neutral_one_hand: bool = False) -> BodyCoordinateDecoder:
        """The deployed decoder on ``data`` (L1 / L2b only).  ``x_mask`` (one bool per window) restricts
        the x model to the windows ``earn_x`` tests (the person's own L1 probes), and ``neutral_one_hand``
        teaches an x earner's y model to output 0 on one-hand imagery (``BodyCoordinateDecoder.fit``)."""
        require_sources(data, DEPLOYED_SOURCES, "the deployed body decoder")
        drawn = sum(p.get("display_goal") is not None for p in data.provenance)
        if drawn:
            raise ValueError(f"the deployed body decoder refuses {drawn} windows with a drawn goal "
                             f"('display_goal' set): its labels must not be identifiable from the screen")
        check_eeg_windows(data.windows, self.cfg.hardware.n_channels, "deployed body decoder")
        dec = BodyCoordinateDecoder(self.cfg, self.cfg.hardware.fs, window_s=self.window_s, **self.decoder_options)
        dec.fit(list(data.windows), body_labels(data), data.provenance, session_references, x_mask=x_mask,
                neutral_one_hand=neutral_one_hand)
        self.last_fit_info = {"role": "deployed", "n": len(data), "sources": list(DEPLOYED_SOURCES),
                              "n_x": None if x_mask is None else int(np.sum(x_mask)),
                              "neutral_one_hand": bool(neutral_one_hand),
                              "train_mix": data.mix(), "data_hash": data.data_hash()}
        return dec


SHADOW_AXIS_MIN = math.sin(math.pi / 8)
"""A ReFIT body label ``u`` gives an axis a class (sign uᵢ) only if |uᵢ| ≥ sin 22.5° ≈ 0.38.  This is
the 8-direction geometry of the L2b cues: a direction within ±22.5° of the other axis counts as 0 on
this one.  Smaller components are too close to 0 to have a sign worth teaching."""


class ShadowBodyDecoder(BodyCoordinateDecoder):
    """The leak positive control (never deployed).  It is the deployed decoder's model (same features,
    re-centring, per-axis logistic classifiers and 2p − 1 output), trained on S windows with ReFIT
    labels instead of cue-defined ones.

    The S label (goal − displayed cursor) is exactly the display-identifiable label the design keeps
    away from the deployed decoder.  If the certification test cannot see this decoder follow the
    screen under held-out mappings and replays (beyond its no-display null), the test is insensitive
    (``certify.certify``).  It is NOT a matched-n twin of the deployed decoder: it trains on S windows
    only — about a third of the person's own L1/L2b window count — and without the population prior.
    ``x_enabled`` is copied from the deployed decoder by ``ShadowTrainer``, so both are scored on the
    same axes.  ``earn_x`` refuses S windows, as it should: x is earned on L1 / L2b only."""

    def fit(self, windows: List[np.ndarray], body_labels: np.ndarray, provenance: Sequence[dict],
            session_references: Optional[Dict[str, np.ndarray]] = None) -> "ShadowBodyDecoder":
        """Per-axis classifiers on S windows; ``body_labels`` in {−1, 0, 1}² (0 = not trained on that axis)."""
        bad = sorted({str(p.get("source")) for p in provenance if p.get("source") not in SHADOW_SOURCES})
        if bad:
            raise ValueError(f"the shadow decoder trains only on {list(SHADOW_SOURCES)} windows, got {bad}")
        y = self._labels(body_labels, len(windows))
        if not (np.any(y[:, 1] > 0) and np.any(y[:, 1] < 0)):
            raise ValueError("the shadow's y axis needs windows with goals both above and below the cursor")
        self._fit_axes(self._feature_matrix(windows, provenance, session_references), y)
        return self


class ShadowTrainer:
    """Fits the ``ShadowBodyDecoder`` on sighted (S) windows only.

    The labels are ReFIT, ``u = (goal − displayed cursor)/‖·‖``, mapped to body coordinates through the
    block's ``M⁻¹`` (``refit_body_labels``; S always runs under the identity) and given a sign per axis
    beyond ``axis_min``.  Same person and same model as the deployed decoder; the label source differs,
    which is the point of the control, and so do the training windows: S only (≈ 1/3 of the deployed
    decoder's own window count at the default allocation) and no population prior.  Its sensitivity is
    therefore that of this smaller, prior-free fit, not of a matched-n twin.

    window_s, decoder_options  as ``BodyTrainer`` (the shadow must match the deployed model).
    axis_min                   ``SHADOW_AXIS_MIN``.
    """

    def __init__(self, cfg: Config, window_s: float = 1.0, axis_min: float = SHADOW_AXIS_MIN,
                 decoder_options: Optional[dict] = None):
        self.cfg = cfg
        self.window_s = window_s
        self.axis_min = axis_min
        self.decoder_options = dict(decoder_options or {})
        self.last_fit_info: dict = {}

    @staticmethod
    def select(data: LabelledSet) -> LabelledSet:
        """The S windows of ``data``."""
        return data.mask(_source_mask(data, SHADOW_SOURCES))

    def axis_labels(self, data: LabelledSet) -> np.ndarray:
        """(n, 2) {−1, 0, 1} labels: the sign of each ReFIT body component with |uᵢ| ≥ ``axis_min``."""
        u = refit_body_labels(data)
        return np.where(np.abs(u) >= self.axis_min, np.sign(u), 0.0).astype(int)

    def fit(self, data: LabelledSet, x_enabled: bool = False,
            session_references: Optional[Dict[str, np.ndarray]] = None) -> ShadowBodyDecoder:
        require_sources(data, SHADOW_SOURCES, "the shadow decoder")
        check_eeg_windows(data.windows, self.cfg.hardware.n_channels, "shadow decoder")
        dec = ShadowBodyDecoder(self.cfg, self.cfg.hardware.fs, window_s=self.window_s, **self.decoder_options)
        dec.fit(list(data.windows), self.axis_labels(data), data.provenance, session_references)
        dec.x_enabled = bool(x_enabled)
        self.last_fit_info = {"role": "shadow", "n": len(data), "sources": list(SHADOW_SOURCES),
                              "axis_min": self.axis_min, "x_enabled": dec.x_enabled,
                              "train_mix": data.mix(), "data_hash": data.data_hash()}
        return dec


# ----------------------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError(type(o))


class ModelRegistry:
    """Append-only model store::

        root/v00001/model.joblib   the fitted decoder
        root/v00001/meta.json      n windows, provenance mix, data hash, metrics, trainer settings
        root/champion.jsonl        one line per promotion: version, frozen held-out keys, score

    A version directory is written under a temporary name and renamed into place, so a crash never
    leaves a half-written version; an existing version is never overwritten (single-writer: one engine
    per registry).
    """

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _vdir(self, v: int) -> str:
        return os.path.join(self.root, f"v{v:05d}")

    def versions(self) -> List[int]:
        return sorted(int(n[1:]) for n in os.listdir(self.root)
                      if n.startswith("v") and n[1:].isdigit() and os.path.isdir(os.path.join(self.root, n)))

    def register(self, model, meta: dict) -> int:
        vs = self.versions()
        v = (vs[-1] + 1) if vs else 1
        final = self._vdir(v)
        if os.path.exists(final):
            raise FileExistsError(final)
        tmp = os.path.join(self.root, f".tmp_v{v:05d}_{os.getpid()}")
        os.makedirs(tmp)
        joblib.dump(model, os.path.join(tmp, "model.joblib"))
        meta = dict(meta, version=v, created_utc=datetime.now(timezone.utc).isoformat())
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=_jsonable)
        os.rename(tmp, final)
        return v

    def load(self, v: int):
        return joblib.load(os.path.join(self._vdir(v), "model.joblib"))

    def meta(self, v: int) -> dict:
        with open(os.path.join(self._vdir(v), "meta.json")) as f:
            return json.load(f)

    def set_champion(self, v: int, record: dict) -> None:
        if v not in self.versions():
            raise KeyError(f"version {v} not registered")
        line = json.dumps(dict(record, version=v, t_utc=datetime.now(timezone.utc).isoformat()),
                          default=_jsonable)
        with open(os.path.join(self.root, "champion.jsonl"), "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def champion_record(self) -> Optional[dict]:
        path = os.path.join(self.root, "champion.jsonl")
        if not os.path.exists(path):
            return None
        last = None
        with open(path) as f:
            for line in f:
                if line.strip():
                    last = line
        return None if last is None else json.loads(last)

    def champion_history(self) -> List[dict]:
        path = os.path.join(self.root, "champion.jsonl")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]


# ----------------------------------------------------------------------------------------------
# Champion / challenger
# ----------------------------------------------------------------------------------------------
def cosines(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-window cosine similarity between decoded and label directions; a zero output scores 0 (no
    direction is no information)."""
    pred = np.asarray(pred, float).reshape(-1, 2)
    labels = np.asarray(labels, float).reshape(-1, 2)
    num = np.sum(pred * labels, axis=1)
    den = np.linalg.norm(pred, axis=1) * np.linalg.norm(labels, axis=1)
    return np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)


def mean_cosine(pred: np.ndarray, labels: np.ndarray) -> float:
    """Mean cosine similarity between decoded and label directions; the null decoder scores exactly 0.

    Not chance-corrected: a constant output u scores u · mean(labels), up to ``constant_ceiling``,
    which is far from 0 on small or direction-imbalanced test sets.  Report it beside every mean
    cosine, and use a trial-level permutation floor (``ChampionChallenger``) to claim skill."""
    cos = cosines(pred, labels)
    return float(np.mean(cos)) if len(cos) else float("nan")


def constant_ceiling(labels: np.ndarray) -> float:
    """Best mean cosine any constant output can reach on ``labels``: |mean of the unit labels|."""
    y = np.asarray(labels, float).reshape(-1, 2)
    if len(y) == 0:
        return float("nan")
    y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    return float(np.linalg.norm(y.mean(axis=0)))


def trial_clusters(provenance: Sequence[dict]) -> np.ndarray:
    """Cluster index per window: windows of one (session, trial) share a cluster."""
    ids: Dict[Tuple[str, int], int] = {}
    return np.array([ids.setdefault((p["session"], p["trial"]), len(ids)) for p in provenance], int)


# ----------------------------------------------------------------------------------------------
# Body metric: per-axis AUC of a body-coordinate decoder on held-out L1 selection probes
# ----------------------------------------------------------------------------------------------
BODY_METRIC = "body_auc"
SELECTION_SOURCE = "L1"
"""The body metric scores fixation probes only: one cued class per trial, nothing on screen, and a
fixed task, so the score does not depend on how much L2b closed loop a day happened to hold."""


def body_outputs(model: Optional[BodyCoordinateDecoder], data: LabelledSet,
                 session_references: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """(n, 2) body-coordinate outputs (x, y), each window re-centred on its own session's rest reference
    when ``session_references`` has one (as the online decoder saw it), else on the model's current
    reference.  A shallow copy carries the reference, so the model itself is untouched.  A non-finite
    row (an invalid window) becomes 0, the zero command the real-time loop sends for it; ``None`` (no
    champion yet) is the null decoder, 0 everywhere."""
    out = np.zeros((len(data), 2))
    if model is None or len(data) == 0:
        return out
    refs = session_references or {}
    groups: Dict[str, List[int]] = {}
    for i, p in enumerate(data.provenance):
        groups.setdefault(p["session"], []).append(i)
    for session, idx in groups.items():
        dec = copy.copy(model).set_reference(reference=refs[session]) if session in refs else model
        out[idx] = dec.predict_many([data.windows[i] for i in idx])
    return np.where(np.all(np.isfinite(out), axis=1, keepdims=True), out, 0.0)


@dataclass(frozen=True)
class _AxisTrials:
    """One axis of a held-out probe set at the trial level, the unit every body statistic resamples.

    ``cl`` maps the axis's windows to trials 0…T−1, ``pos`` says which trials cue the + class
    (FEET on y, RIGHT on x), ``n`` holds windows per trial.  A probe cues one class, so every trial
    is wholly + or − on the axis it tests and absent from the other."""
    rows: np.ndarray
    cl: np.ndarray
    pos: np.ndarray
    n: np.ndarray

    @staticmethod
    def of(provenance: Sequence[dict], labels: np.ndarray, k: int) -> "_AxisTrials":
        rows = np.flatnonzero(labels[:, k] != 0)
        cl = trial_clusters([provenance[i] for i in rows])
        n_tr = int(cl.max()) + 1 if len(cl) else 0
        pos = np.zeros(n_tr, bool)
        pos[cl] = labels[rows, k] > 0
        return _AxisTrials(rows, cl, pos, np.bincount(cl, minlength=n_tr).astype(float))

    def n_trials(self, positive: bool) -> int:
        return int(np.sum(self.pos == positive))

    def pair_counts(self, scores: np.ndarray) -> np.ndarray:
        """(T+, T−) sums over window pairs of 1[s+ > s−] + ½·1[s+ = s−] between each + trial and each
        − trial: the window-level Mann–Whitney numerator, kept per trial pair so a trial bootstrap is
        a weighted sum instead of a re-ranking."""
        s = scores[self.rows]
        pw = self.pos[self.cl]
        g = (s[pw, None] > s[None, ~pw]) + 0.5 * (s[pw, None] == s[None, ~pw])
        tp, tn = np.flatnonzero(self.pos), np.flatnonzero(~self.pos)
        a = (self.cl[pw, None] == tp[None]).astype(float)
        b = (self.cl[~pw, None] == tn[None]).astype(float)
        return a.T @ g @ b

    def auc(self, counts: np.ndarray, m_pos: np.ndarray, m_neg: np.ndarray) -> np.ndarray:
        """AUC per row of trial multiplicities (R, T+) and (R, T−); all ones is the observed AUC."""
        n_pos, n_neg = self.n[self.pos], self.n[~self.pos]
        num = np.einsum("rp,pq,rq->r", m_pos, counts, m_neg)
        return num / ((m_pos @ n_pos) * (m_neg @ n_neg))

    def permutation_aucs(self, scores: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
        """AUC under ``n_draws`` permutations of the trials' class labels (whole trials, class counts
        kept): probe cues are scheduled independently of the EEG, so this is an exact null."""
        ranks = rankdata(scores[self.rows])
        rank_sum = np.bincount(self.cl, weights=ranks, minlength=len(self.n))
        perm = np.argsort(rng.random((n_draws, len(self.pos))), axis=1)
        lab = self.pos[perm].astype(float)
        n1 = lab @ self.n
        n_all = self.n.sum()
        return (lab @ rank_sum - n1 * (n1 + 1) / 2) / (n1 * (n_all - n1))


def _stratified_multiplicities(n_trials: int, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """(n_draws, n_trials) bootstrap multiplicities of one class's trials (resampled with replacement)."""
    m = np.zeros((n_draws, n_trials))
    np.add.at(m, (np.arange(n_draws)[:, None], rng.integers(0, n_trials, (n_draws, n_trials))), 1.0)
    return m


def axis_outputs(model: Optional[BodyCoordinateDecoder], data: LabelledSet,
                 session_references: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """``body_outputs`` of the decoder's axis models without its idle gate (``ungated``): what the body
    metric ranks.  The gate is a separate component with its own pre-declared arming rule
    (``BodyCoordinateDecoder.fit_idle``); the windows it zeroes would tie in an AUC, so a decoder
    whose gate happens to be armed would be ranked on its gate instead of on its body axes."""
    return body_outputs(None if model is None else model.ungated(), data, session_references)


def body_auc(model: Optional[BodyCoordinateDecoder], data: LabelledSet,
             session_references: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
    """Per-axis window AUC of ``model``'s body output (``axis_outputs``) against the cued class of L1
    probes (y: FEET vs HANDS probes; x: RIGHT vs LEFT probes; NaN for an axis without both classes).
    An axis the model does not drive (x not earned) outputs 0 and scores 0.5."""
    labels = body_labels(data)
    out = axis_outputs(model, data, session_references)
    res = {}
    for k, axis in enumerate(BODY_AXES):
        ax = _AxisTrials.of(data.provenance, labels, k)
        both = ax.n_trials(True) > 0 and ax.n_trials(False) > 0
        res[axis] = (float(ax.auc(ax.pair_counts(out[:, k]), np.ones((1, ax.n_trials(True))),
                                  np.ones((1, ax.n_trials(False))))[0]) if both else float("nan"))
    return res


class ChampionChallenger:
    """Promotion rule with held-out cue-defined trials that no compared model was trained on.

    Two metrics, chosen by the decoder (``is_body``):

    * screen decoders (``VelocityDecoder`` and kin): mean cosine on held-out cued_blank windows;
    * the deployed ``BodyCoordinateDecoder``: the BODY metric, per-axis AUC on held-out L1 fixation
      probes (the selection probes; ``candidate_heldout(data, body=True)``).  A body decoder outputs
      2p − 1 per axis with x = 0 until earned, so a 2-D cosine against a screen label would score an
      unearned x as a miss on every LEFT/RIGHT probe and reward a confident y the same as a correct
      one; per-axis AUC is invariant to the output's scale and to an axis that is switched off.

    Protocol for each retrain:
      1. ``candidate_heldout(data)`` = H′, the most recent ``heldout_frac`` (20 %) of cued_blank
         windows (L1 windows for a body decoder) now, extended back to the start of the trial the cut
         falls in (overlapping windows of one trial must not straddle train and test).
      2. The challenger trains on everything EXCEPT the frozen held-out set H and H′: the selection
         probes are never in the training set of the candidate they select.
      3. Before the first champion: the challenger must beat the null decoder on H′ — its score must
         exceed the margin (mean cosine > ``margin``; mean AUC > 0.5 + ``auc_margin``) AND a
         trial-level label-permutation floor (one-sided p ≤ ``alpha``).  Mean cosine is not
         chance-corrected (a constant output scores up to ``constant_ceiling``), and windows of one
         trial are not independent, so "> margin" alone promotes decoders with no signal; permuting
         whole trials' labels keeps both the label distribution and the trial structure, so only
         direction-specific decoding beats it.
      4. Afterwards: both are scored on T = H ∪ F, where F = the windows of H′ from sessions the
         champion was not trained on (fresh trials under current electrode conditions; H alone goes
         stale between promotions).  The challenger is promoted iff the lower bound of a 95 %
         trial-cluster paired bootstrap CI of (challenger − champion) exceeds the margin: the noise of
         a few dozen correlated trials (SE ≈ 0.04 cosine for 34 trials) is larger than a 0.02 margin,
         so a fixed margin alone promotes equal decoders a third of the time.
      5. On promotion H′ is frozen as the new H; by step 2 the new champion has never seen it.
    cued_blank and L1 windows are the test because their labels are cue-defined and nothing on screen
    reveals them: a decoder that learned the displayed cursor (feedback circularity) cannot win here.
    Selection bias: a champion was promoted partly on H′ (F, or all of H′ for the first), which then
    becomes its H, so its score there is slightly optimistic — that only makes the next promotion
    harder (conservative).

    Body metric details.  Axes scored: y always; x when the probes hold both RIGHT and LEFT and either
    compared decoder drives x (an axis both leave at 0 would only halve every difference).  The score
    is the mean AUC over the scored axes; an axis's AUC is over windows (each probe one class), and
    every resample is of whole trials, stratified by class (the trial-pair counts of
    ``_AxisTrials``), so a resample never loses a class.  The first champion's null permutes whole
    trials' classes within each axis.

    margin 0.02 mean cosine and auc_margin 0.02 AUC: the smallest improvement worth the user's
    re-adaptation to a new decoder.  min_heldout 40 windows and min_heldout_trials 8: the cluster
    bootstrap and the permutation need several independent trials; with fewer the CI is itself
    unreliable (too narrow).  A small test set does not let noise through — it makes promotion rarer.
    min_class_trials 3: a body held-out set needs at least 3 FEET and 3 HANDS probes, the fewest for
    which a class-stratified trial bootstrap has any spread.  n_resamples 2000 bootstrap / permutation
    draws; alpha 0.05; ``seed`` makes every verdict deterministic.
    """

    def __init__(self, margin: float = 0.02, heldout_frac: float = 0.2, min_heldout: int = 40,
                 min_heldout_trials: int = 8, n_resamples: int = 2000, alpha: float = 0.05,
                 seed: int = 20261004, auc_margin: float = 0.02, min_class_trials: int = 3):
        self.margin = margin
        self.heldout_frac = heldout_frac
        self.min_heldout = min_heldout
        self.min_heldout_trials = min_heldout_trials
        self.n_resamples = n_resamples
        self.alpha = alpha
        self.seed = seed
        self.auc_margin = auc_margin
        self.min_class_trials = min_class_trials
        self.heldout: Optional[frozenset] = None

    @staticmethod
    def is_body(model) -> bool:
        """True for the deployed body-coordinate decoder (and its subclasses): it is judged by the body
        metric on L1 probes."""
        return isinstance(model, BodyCoordinateDecoder)

    def candidate_heldout(self, data: LabelledSet, body: bool = False) -> Optional[frozenset]:
        """H′ (class protocol step 1): cued_blank windows, or with ``body`` L1 fixation probes, which
        must then hold ≥ ``min_class_trials`` FEET and HANDS probes so the primary axis can be scored."""
        if body:
            idx = [i for i, p in enumerate(data.provenance) if p.get("source") == SELECTION_SOURCE]
        else:
            idx = [i for i, p in enumerate(data.provenance) if p["block"] == "cued_blank"]
        n_hold = int(math.ceil(self.heldout_frac * len(idx)))
        if n_hold < self.min_heldout:
            return None
        cut = len(idx) - n_hold
        trial_of = lambda i: (data.provenance[i]["session"], data.provenance[i]["trial"])
        while cut > 0 and trial_of(idx[cut - 1]) == trial_of(idx[cut]):
            cut -= 1
        if len({trial_of(i) for i in idx[cut:]}) < self.min_heldout_trials:
            return None
        held = data.take(idx[cut:])
        if body:
            y = _AxisTrials.of(held.provenance, body_labels(held), BODY_AXES.index("y"))
            if min(y.n_trials(True), y.n_trials(False)) < self.min_class_trials:
                return None
        return frozenset(held.keys())

    def training_set(self, data: LabelledSet, candidate: Optional[frozenset]) -> LabelledSet:
        excluded = (self.heldout or frozenset()) | (candidate or frozenset())
        return data.mask(np.array([k not in excluded for k in data.keys()], bool))

    @staticmethod
    def subset(data: LabelledSet, keys: frozenset) -> LabelledSet:
        return data.mask(np.array([k in keys for k in data.keys()], bool))

    @staticmethod
    def window_scores(model, data: LabelledSet) -> np.ndarray:
        """Per-window cosine; the null decoder (``None``) scores 0 everywhere."""
        if model is None or len(data) == 0:
            return np.zeros(len(data))
        return cosines(model.predict_many(data.windows), data.labels)

    @classmethod
    def score(cls, model, data: LabelledSet,
              session_references: Optional[Dict[str, np.ndarray]] = None) -> float:
        """Mean cosine, or for a body decoder the mean AUC over the axes it drives (y, and x once
        earned) on L1 probes."""
        if not len(data):
            return 0.0
        if cls.is_body(model):
            auc = body_auc(model, data, session_references)
            axes = [a for a in BODY_AXES if a == "y" or model.x_enabled]
            return float(np.nanmean([auc[a] for a in axes]))
        return float(cls.window_scores(model, data).mean())

    def test_keys(self, candidate: Optional[frozenset],
                  champion_sessions: Optional[Iterable[str]]) -> Tuple[Optional[frozenset], int]:
        """(T, number of fresh windows): H plus the windows of H′ from sessions the champion never
        trained on; before any champion, H′ itself.  ``champion_sessions`` None means unknown, so
        nothing from H′ is added."""
        if self.heldout is None:
            return candidate, 0
        if champion_sessions is None or candidate is None:
            return self.heldout, 0
        seen = set(champion_sessions)
        fresh = frozenset(k for k in candidate if k[0] not in seen) - self.heldout
        return self.heldout | fresh, len(fresh)

    def _permutation_test(self, challenger, test: LabelledSet,
                          rng: np.random.Generator) -> Tuple[float, float, float]:
        """(observed, floor, p) of the challenger's mean cosine against whole-trial label permutations
        (floor = the null's 1 − alpha quantile).  Held-out trials are cued_blank (hidden cursor), so a
        trial's label is constant and trial labels are exchangeable under the null."""
        pred = np.asarray(challenger.predict_many(test.windows), float)
        norm = np.linalg.norm(pred, axis=1, keepdims=True)
        unit = np.where(norm > 1e-12, pred / np.maximum(norm, 1e-12), 0.0)
        cl = trial_clusters(test.provenance)
        n_tr = int(cl.max()) + 1
        P = np.zeros((n_tr, 2))
        np.add.at(P, cl, unit)                                    # Σ unit predictions per trial
        L = np.zeros((n_tr, 2))
        np.add.at(L, cl, test.labels)
        L /= np.maximum(np.linalg.norm(L, axis=1, keepdims=True), 1e-12)   # the trial's label direction
        observed = float(np.einsum("tk,tk->", P, L) / len(test))
        perms = np.stack([rng.permutation(n_tr) for _ in range(self.n_resamples)])
        null = np.einsum("tk,rtk->r", P, L[perms]) / len(test)
        p = float((np.sum(null >= observed) + 1) / (len(null) + 1))
        return observed, float(np.percentile(null, 100 * (1 - self.alpha))), p

    def _paired_bootstrap(self, diff: np.ndarray, provenance: Sequence[dict],
                          rng: np.random.Generator) -> Tuple[float, float]:
        """(1 − alpha) trial-cluster bootstrap CI of the mean of a per-window paired difference."""
        cl = trial_clusters(provenance)
        sums = np.bincount(cl, weights=diff)
        counts = np.bincount(cl).astype(float)
        idx = rng.integers(0, len(sums), (self.n_resamples, len(sums)))
        means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
        lo, hi = np.percentile(means, [100 * self.alpha / 2, 100 * (1 - self.alpha / 2)])
        return float(lo), float(hi)

    def challenge(self, champion, challenger, data: LabelledSet,
                  candidate: Optional[frozenset] = None,
                  champion_sessions: Optional[Iterable[str]] = None,
                  session_references: Optional[Dict[str, np.ndarray]] = None) -> dict:
        """Apply steps 3–4 of the class protocol.  ``champion_sessions`` = the sessions the champion
        was trained on (needed to add fresh trials to the frozen set).  ``session_references``
        ({session: rest covariance}) re-centres a body decoder's test windows per session."""
        keys, n_fresh = self.test_keys(candidate, champion_sessions)
        if keys is None:
            return {"promoted": False, "reason": "no held-out set: too few cued_blank windows or trials "
                                                 "(L1 probes for a body decoder)"}
        test = self.subset(data, keys)
        rng = np.random.default_rng(self.seed)
        if self.is_body(challenger):
            return self._challenge_body(champion, challenger, test, n_fresh, rng, session_references)
        out = {"margin": self.margin, "n_heldout": len(test),
               "n_heldout_trials": int(trial_clusters(test.provenance).max()) + 1 if len(test) else 0,
               "n_fresh": n_fresh, "constant_ceiling": constant_ceiling(test.labels)}
        if self.heldout is None:
            observed, floor, p = self._permutation_test(challenger, test, rng)
            return dict(out, promoted=bool(observed > self.margin and p <= self.alpha),
                        challenger_cosine=observed, champion_cosine=0.0, floor=floor, p_value=p)
        s_ch = self.window_scores(challenger, test)
        s_champ = self.window_scores(champion, test)
        lo, hi = self._paired_bootstrap(s_ch - s_champ, test.provenance, rng)
        return dict(out, promoted=bool(lo > self.margin), challenger_cosine=float(s_ch.mean()),
                    champion_cosine=float(s_champ.mean()), ci95=[lo, hi])

    def _challenge_body(self, champion, challenger: BodyCoordinateDecoder, test: LabelledSet, n_fresh: int,
                        rng: np.random.Generator, session_references: Optional[Dict[str, np.ndarray]]) -> dict:
        """Steps 3–4 with the body metric (class docstring)."""
        bad = Counter(str(p.get("source")) for p in test.provenance if p.get("source") != SELECTION_SOURCE)
        if bad:
            raise ValueError(f"the body metric scores held-out {SELECTION_SOURCE} probes only; the test set "
                             f"holds {dict(bad)} other windows (select H′ with candidate_heldout(data, body=True))")
        if champion is not None and not self.is_body(champion):
            raise ValueError("a body decoder can only be compared with a body-decoder champion")
        labels = body_labels(test)
        outputs = {"challenger": axis_outputs(challenger, test, session_references),
                   "champion": axis_outputs(champion, test, session_references)}
        drives_x = challenger.x_enabled or bool(champion is not None and champion.x_enabled)
        axes = {}
        for k, axis in enumerate(BODY_AXES):
            ax = _AxisTrials.of(test.provenance, labels, k)
            if min(ax.n_trials(True), ax.n_trials(False)) >= 1 and (axis == "y" or drives_x):
                axes[axis] = ax
        out = {"metric": BODY_METRIC, "auc_margin": self.auc_margin, "axes": list(axes),
               "n_heldout": len(test), "n_heldout_trials": int(trial_clusters(test.provenance).max()) + 1,
               "n_fresh": n_fresh,
               "n_class_trials": {a: [ax.n_trials(True), ax.n_trials(False)] for a, ax in axes.items()}}
        if "y" not in axes:
            return dict(out, promoted=False, reason="held-out probes lack FEET or HANDS trials")
        counts = {who: {a: ax.pair_counts(o[:, BODY_AXES.index(a)]) for a, ax in axes.items()}
                  for who, o in outputs.items()}
        ones = {a: (np.ones((1, ax.n_trials(True))), np.ones((1, ax.n_trials(False)))) for a, ax in axes.items()}
        auc = {who: {a: float(ax.auc(counts[who][a], *ones[a])[0]) for a, ax in axes.items()} for who in outputs}
        mean = {who: float(np.mean(list(auc[who].values()))) for who in outputs}
        out = dict(out, auc=auc, challenger_auc=mean["challenger"], champion_auc=mean["champion"])
        if self.heldout is None:
            null = np.mean([ax.permutation_aucs(outputs["challenger"][:, BODY_AXES.index(a)], self.n_resamples, rng)
                            for a, ax in axes.items()], axis=0)
            p = float((np.sum(null >= mean["challenger"]) + 1) / (len(null) + 1))
            return dict(out, promoted=bool(mean["challenger"] > 0.5 + self.auc_margin and p <= self.alpha),
                        floor=float(np.percentile(null, 100 * (1 - self.alpha))), p_value=p)
        diff = np.zeros(self.n_resamples)
        for a, ax in axes.items():
            m_pos = _stratified_multiplicities(ax.n_trials(True), self.n_resamples, rng)
            m_neg = _stratified_multiplicities(ax.n_trials(False), self.n_resamples, rng)
            diff += (ax.auc(counts["challenger"][a], m_pos, m_neg) - ax.auc(counts["champion"][a], m_pos, m_neg))
        diff /= len(axes)
        lo, hi = np.percentile(diff, [100 * self.alpha / 2, 100 * (1 - self.alpha / 2)])
        return dict(out, promoted=bool(lo > self.auc_margin), ci95=[float(lo), float(hi)])

    def freeze(self, keys: frozenset) -> None:
        self.heldout = frozenset(keys)


# ----------------------------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------------------------
class DataEngine:
    """collect → label → retrain → promote.

    ``ingest`` labels a finished run once (idempotent by session id) and caches its windows as .npy
    under ``cache_dir`` (default ``<registry>/labelled/<labeler signature>/``), memory-mapped back, so
    hours of data (32 KB per window, ~460 MB per hour of trials) live on disk rather than in RAM and a
    restarted engine re-ingests at file-read speed.  ``maybe_retrain`` trains when at least
    ``min_new_windows`` arrived since the last attempt and applies the ChampionChallenger rule; every
    challenger is registered (append-only audit trail), promoted or not.  On construction the engine
    restores the champion, its frozen held-out set and the sessions it was trained on (registry
    metadata; needed for the fresh part of the champion test) from the registry.

    With a ``BodyTrainer`` the engine runs the champion's deployed decoder: H′ is the newest L1 probes
    (``candidate_heldout(data, body=True)``), the challenger trains on the L1 / L2b windows of the rest
    (``BodyTrainer.select``: S, H, R, J and original-protocol windows never reach it) plus the
    ``population`` prior, which must then be a ``LabelledSet`` of other people's L1 probes, and the
    judge applies the body metric.  ``session_references`` ({session: rest covariance}, the setup rest
    of each session) re-centres training and test windows per session; the caller adds each session's
    reference as it ingests the session's runs (keys are run directory names).  Label with
    ``Labeler(decoder.loop_config(cfg))`` so windows have the decoder's 1-s length.
    """

    def __init__(self, store: SessionStore, labeler: Labeler, trainer: "Trainer | BodyTrainer",
                 registry: ModelRegistry,
                 judge: Optional[ChampionChallenger] = None, population=None,
                 min_new_windows: int = 200, cache_dir: Optional[str] = None,
                 session_references: Optional[Dict[str, np.ndarray]] = None):
        self.store, self.labeler, self.trainer, self.registry = store, labeler, trainer, registry
        self.judge = judge or ChampionChallenger()
        self.body = isinstance(trainer, BodyTrainer)
        if self.body and population is not None and not isinstance(population, LabelledSet):
            raise ValueError("a body decoder's prior is a LabelledSet of other people's L1 probes")
        self.population = population
        self.session_references = {} if session_references is None else session_references
        self.min_new_windows = min_new_windows
        self.cache_dir = cache_dir or os.path.join(registry.root, "labelled", labeler.signature())
        self._sets: Dict[str, LabelledSet] = {}
        self._n_new = 0
        self.champion = None
        self.champion_version: Optional[int] = None
        self.champion_sessions: Optional[frozenset] = None   # sessions the champion was trained on
        rec = registry.champion_record()
        if rec is not None:
            self.champion_version = int(rec["version"])
            self.champion = registry.load(self.champion_version)
            self.champion_sessions = self._trained_sessions(registry.meta(self.champion_version))
            self.judge.freeze(frozenset((s, float(t)) for s, t in rec["heldout"]))

    @staticmethod
    def _trained_sessions(meta: dict) -> Optional[frozenset]:
        sessions = meta.get("train_mix", {}).get("sessions")
        return None if sessions is None else frozenset(sessions)

    @property
    def data(self) -> LabelledSet:
        return LabelledSet.concat(list(self._sets.values()))

    def _cache_path(self, sid: str) -> str:
        return os.path.join(self.cache_dir, sid)

    def _save(self, sid: str, ls: LabelledSet) -> None:
        final = self._cache_path(sid)
        tmp = final + f".tmp{os.getpid()}"
        os.makedirs(tmp, exist_ok=True)
        win = (np.stack(ls.windows) if len(ls) else
               np.zeros((0, self.labeler.n_win, self.store.n_channels), np.float32))
        np.save(os.path.join(tmp, "windows.npy"), win)
        np.save(os.path.join(tmp, "labels.npy"), ls.labels)
        with open(os.path.join(tmp, "provenance.json"), "w") as f:
            json.dump(ls.provenance, f, default=_jsonable)
        os.rename(tmp, final)

    def _load(self, sid: str) -> LabelledSet:
        d = self._cache_path(sid)
        win = np.load(os.path.join(d, "windows.npy"), mmap_mode="r")
        labels = np.load(os.path.join(d, "labels.npy"))
        with open(os.path.join(d, "provenance.json")) as f:
            prov = json.load(f)
        return LabelledSet([win[i] for i in range(len(win))], labels.reshape(-1, 2), prov)

    def ingest(self, run_dir: str) -> int:
        sid = os.path.basename(os.path.normpath(run_dir))
        if sid in self._sets:
            return 0
        if not os.path.isdir(self._cache_path(sid)):
            self._save(sid, self.labeler.label(self.store.load(run_dir)))
        ls = self._load(sid)
        self._sets[sid] = ls
        self._n_new += len(ls)
        return len(ls)

    def ingest_all(self) -> int:
        return sum(self.ingest(d) for d in self.store.list_runs())

    def maybe_retrain(self, force: bool = False) -> dict:
        if not force and self._n_new < self.min_new_windows:
            return {"promoted": False, "reason": f"{self._n_new} new windows < {self.min_new_windows}"}
        data = self.data
        candidate = self.judge.candidate_heldout(data, body=self.body)
        if self.judge.heldout is None and candidate is None:
            return {"promoted": False, "reason": "too few held-out windows: cued_blank, or L1 probes with "
                                                 "both y classes for a body decoder"}
        train = self.judge.training_set(data, candidate)
        t0 = time.perf_counter()
        if self.body:
            train = BodyTrainer.select(train)
            challenger = self.trainer.fit(LabelledSet.concat([train, self.population or LabelledSet()]),
                                          session_references=self.session_references)
        else:
            challenger = self.trainer.fit(train, self.population)
        fit_s = time.perf_counter() - t0
        verdict = self.judge.challenge(self.champion, challenger, data, candidate, self.champion_sessions,
                                       session_references=self.session_references)
        promoted = verdict["promoted"] and candidate is not None
        bar = (self.judge.score(challenger, self.judge.subset(data, candidate), self.session_references)
               if promoted else None)
        meta = dict(self.trainer.last_fit_info, n_windows=len(train), provenance_mix=train.mix(),
                    metrics=dict(verdict, fit_s=fit_s, new_heldout_cosine=bar),
                    champion_at_training=self.champion_version, promoted=promoted)
        v = self.registry.register(challenger, meta)
        self._n_new = 0
        if promoted:
            self.registry.set_champion(v, {"heldout": sorted(candidate), "heldout_cosine": bar,
                                           "beat": verdict})
            self.judge.freeze(candidate)
            self.champion, self.champion_version = challenger, v
            self.champion_sessions = self._trained_sessions(meta)
        return dict(verdict, promoted=promoted, version=v, champion_version=self.champion_version,
                    n_train=len(train))
