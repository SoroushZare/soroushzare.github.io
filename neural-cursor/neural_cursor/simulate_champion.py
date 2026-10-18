"""Days of the champion protocol in simulation: a body-coordinate SMR decoder that never trains on a
display-identifiable label, certified every day against a deliberately leaky shadow decoder.

Why this simulation exists
--------------------------
The champion design (``proposals/pure_eeg_tournament/P_smr_modern_FINAL.md`` with the skeptic
corrections) makes one load-bearing claim: a decoder trained only on fixation probes (L1) and blind
closed loop (L2b) follows the person's *body* intent, not the screen, and the held-out-mapping and
pixel-replay tests can tell, because a shadow decoder trained on sighted ReFIT labels (the
display-identifiable label) visibly follows the screen in the same blocks.  The skeptic corrections
ask for exactly this rehearsal before data collection ("unless P2/P3 separate them at the planned n,
the tests are redesigned").  This module runs the whole protocol chain on simulated people
(``sim_body.BodyPopulation``) and reports, per simulated day, whether the certification separates the
two decoders, with the screen-leak pathway switched ON by default so the positive control is exercised.

What one simulated day is (per deep participant, offline-paced: no wall clock anywhere)
---------------------------------------------------------------------------------------
``protocol.session_plan_champion(day, seed, x_earned=...)`` gives the blocks; ``hours`` scales every
block of the 85-min session by one factor (never stretching it; more hours → more sessions), so the
protocol's shares are kept.  In plan order:

1. setup: a rest recording through the online preprocessing gives the session's re-centring reference
   (``BodyCoordinateDecoder.reference_from``).  It is the only use of unlabelled data, and every run
   of the session is re-centred on it (training, online decoding, scoring).
2. fixation_probe (L1): central glyph cues (``BodyCueScheduler``, glyph map redrawn per session), the
   person imagines the cued body part from cue onset (``BodySubject.cue``), nothing spatial on screen.
3. blind_closed_loop (L2b): glyph or body-word cues, the deployed decoder drives a hidden cursor toward
   the hidden goal (no assistance: with one hidden goal it would steer for the person), the person
   keeps imagining the cued body part.  y-only classes until x is earned.
4. before the first sighted block of the session the deployed decoder is retrained on ALL of the
   person's L1/L2b windows so far (``data_engine.BodyTrainer``, which refuses any other source) plus
   the population prior, and ``earn_x`` re-decides the x axis on the person's own L1/L2b windows.
   On day 1 the L2b blocks therefore run on the population-prior decoder alone (zero-calibration
   start, as the proposal uses the prior from day 1); with ``pop`` = 0 nothing moves the cursor there.
5. sighted (S): Fitts targets, identity mapping, the deployed decoder blended with the existing
   ``GoalPosterior`` assistance (a = 0.5) through ``task.TrialRunner``.  The person intends
   M⁻¹·unit(target − cursor) (its y component only until x is earned) and looks toward the target
   (``set_screen_direction``).  15 % of trials are pixel replays (``ReplayScheduler``): a past successful S trajectory toward goal A is replayed
   while the person holds goal B.  S windows train only the SHADOW decoder (``ShadowTrainer``, ReFIT
   labels), never the deployed one.
6. heldout_mapping (H): rot±90 / mirrors, C2b (open loop: frozen cursor, 3 s imagery, keyed report)
   in sessions 1–2, C2b/C2c alternating from session 3; C2c hosts replay trials too.  Frozen decoder.
7. "just intend" blocks are NOT simulated (the clock advances): the simulated person has no
   instruction-dependent behaviour, so they would replay L2b/S with another cue and add no
   information here.  Day 1's hint-free block is skipped for the same reason.

At day end the shadow is refit on all S windows so far, and the day's H and R windows give the β
table: ``certify.certify_mapping`` / ``replay_test`` of deployed and shadow outputs on the same
windows, and ``certify.certify``'s verdict.  Every run goes through ``RunLogger`` → ``SessionStore``
→ ``Labeler``, so decoders see exactly the windows a real session would give them.

Endpoints per day
-----------------
* L1 fixed-test-set AUC / r per axis: a pool of ``eval_minutes`` of fixation probes per simulated day,
  recorded once (fresh subject instance, that day's electrode gains, its own rest reference) and never
  trained on; every day's decoder meets the same windows.  The x AUC is of the x model even when x is
  not earned (the deployed output is then 0 on x); "x earned" is reported beside it.
* x earned, with ``earn_x``'s AUC and circular-shift floor.
* sighted success / time-to-target / Fitts throughput (non-replay S trials); blind success (L2b).
* certification: β_body, β_screen with 95 % trial-cluster bootstrap CIs for deployed and shadow, under
  held-out mappings and under replay, and the verdict; also pooled over days per user and over users.

Leak model (the parts of the result that are assumptions, stated)
-----------------------------------------------------------------
* ``screen_leak`` (default 1.0 > 0): ``BodySubject``'s overt pathway — posterior alpha lateralised and
  a frontal gaze potential toward the drawn target's screen direction — active only while targets are
  drawn (S, H, R).  It is what the shadow can learn from its display-identifiable labels.
* ``covert_leak`` (default 0): covert spatial attention toward the screen direction the person imagines
  in BLIND blocks (L1, L2b), where nothing is drawn.  Under the training (identity) mapping that
  direction is the cued body vector, so it is implemented with ``BodySubject.covert_leak`` switched on
  in L1/L2b only; in target blocks spatial attention is the overt pathway toward the drawn target.  This
  is the leak the label rule cannot exclude (skeptic C3): the deployed decoder learns it from its own
  blind labels and then, under a held-out mapping, reads the screen.  (``sim_body``'s own semantics keep
  ``covert_leak`` on in every block, where certification counts it as body signal; here the attention
  follows the intended *screen* direction, which is what makes it a display leak.)
* ``replay_attention`` (default "cursor"): during a replay the overt pathway follows the replayed
  cursor's heading (smooth pursuit of the moving cursor toward A).  With "goal" it follows the held goal
  B, which coincides with the private intent, so replay cannot detect screen reading in this simulator
  by construction.  Held-out mappings are unaffected by this choice.
* Intent is an oracle: the person always imagines exactly the cued body vector (L1/L2b) or
  M⁻¹·unit(target − cursor) (S/H), and remaps instantly under held-out mappings.  β_body is therefore
  an upper bound on what real remapping gives.  Until x is earned the person imagines only the y
  component of M⁻¹·unit(target − cursor): a non-earner is taught 1-D y control (proposal §4,
  "1-D y + dwell + goal posterior"), so one-hand imagery, which would add hand-area ERD that reads
  as HANDS on the y axis, is not used.  L1 probes still cue all four classes, so x can be earned.

Pure-EEG rule: decoders and trainers receive only labelled (n_samples, 32) EEG windows from
``Labeler``; intent, gaze and screen direction are simulator state, used only to drive the simulated
person and to build the certification regressors from logged task geometry.

Run:  python3 -m neural_cursor.simulate_champion --days 3 --hours 1 --users 3 --pop 10
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .body_decoder import AXES, BodyCoordinateDecoder
from .certify import MIN_BOOT, MIN_CLUSTERS, N_BOOT, certify, certify_mapping, replay_test
from .config import DEFAULT, Config
from .data_engine import (EVENT_CUE_OFFSET, EVENT_TRIAL_END, EVENT_TRIAL_START, BodyTrainer, LabelledSet, Labeler,
                          SessionStore, ShadowTrainer, body_labels, trial_clusters)
from .evaluation import pearson_r
from .logger import RunLogger
from .mapping import BODY_VECTORS, TRAINING_MAPPING, Mapping, unit_directions
from .protocol import (BODY_CUE_MODES, CHAMPION_SLOT_MIN, EVENT_BODY_REPORT, EVENT_OUTCOME_FLASH, L1_CLASSES,
                       L2B_CLASSES_XY, L2B_CLASSES_Y, Block, BodyCueScheduler, ClosedLoopTiming, FixationProbeTiming,
                       OpenLoopTiming, ReplayScheduler, champion_trial_start_payload, glyph_map, hidden_goal,
                       session_plan_champion)
from .sim_body import X_DEPTH_RANGE, X_RESPONDER_PROB, Y_DEPTH_RANGE, BodyPopulation, BodySubject
from .simulate_engine import (CLOCK_ORIGIN_S, DAY_OFFSET_S, FITTS_CONDITIONS, PREROLL_S, OnlinePreprocessor,
                              _clean, trial_metrics)
from .task import TrialOutcome, TrialRunner, fitts_targets

CHAMPION_SESSION_S = float(sum(b.duration_s for b in session_plan_champion(1, 0)))
"""Length of one unscaled champion session (85 min), derived from the protocol so it cannot drift."""
BLIND_KINDS: Tuple[str, ...] = ("fixation_probe", "blind_closed_loop")
"""Blocks with nothing spatial on screen: the only ones where ``covert_leak`` acts."""
EVAL_BLOCK_INDEX = 10_000          # block index of the fixed L1 test pool: never collides with a plan block
SESSION_BLOCK_STRIDE = 100         # block index = session × stride + position: unique within a day
REST_HOP_S = 0.5                   # spacing of the rest windows that form the re-centring reference
MIN_REPLAY_STEPS = 20              # a replayable trajectory lasts ≥ 2 s of control, so it yields windows
REPORT_DELAY_S = 0.6               # C2b keyed report: a reaction time after trial_end (a movement, so outside)
_REPLAY_CHOICE_STREAM = 21         # RNG stream for which past trajectory / which held goal B a replay uses
REPLAY_ATTENTION = ("cursor", "goal")
TESTS: Tuple[str, ...] = ("heldout", "replay")
"""Certification tests: held-out mappings (source H, ``certify_mapping``) and pixel replay (source R,
``replay_test``)."""


@dataclass(frozen=True)
class ChampionSimConfig:
    """Hyperparameters of a champion simulation (everything not in ``config.Config``).

    days, hours, users, pop  simulated days; session hours per day (1 = the 85-min plan scaled to 60 min);
                       deep participants; population-prior members (other simulated people).
    seed               population seed: subject i is a pure function of (seed, i).
    dry_penalty        motor d′ loss of the dry cap (``BodySubject``); the CLI sweeps a list of values.
    screen_leak        overt display pathway gain while targets are drawn; > 0 by default so the
                       shadow positive control has something to detect.
    covert_leak        covert attention to the imagined screen direction in blind blocks (module doc).
    replay_attention   what the overt pathway follows during a replay: "cursor" or "goal" (module doc).
    x_responder_prob   probability of lateralised L/R ERD (sim_body: 0.2, the skeptic's 10–25 %).
    y_depth_range, x_depth_range  per-subject ERD depth ranges (sim_body's calibrated defaults); a
                       sensitivity lever, e.g. strong lateralisers to study the x-axis certification.
    user_learning_rate motor depth gain per practice hour (0: the person does not learn).
    pop_minutes        fixation-probe minutes per prior member (≈ 8 probes a minute); with the label
                       hop this sets the prior's size, which is never re-weighted.
    rest_s             setup rest per session for the re-centring reference (the proposal's 3-min rest
                       shortened to 60 s: an 11-channel Riemannian mean is stable well before that).
    eval_minutes       fixation-probe minutes per simulated day in the fixed L1 test pool.
    assist_gain        GoalPosterior blend in sighted and C2c blocks (the existing a = 0.5).
    label_hop_s        0.5 s between window ends: 1-s windows at 0.25 s add half-overlapping
                       near-duplicates (64 KB each) that double memory and training time for little
                       information; the trial-cluster statistics do not depend on it.
    x_permutations     circular-shift null draws of ``earn_x`` (body_decoder default 200).
    n_boot             trial-cluster bootstrap draws of the certification (certify default 2000).
    slot_minutes       ((slot, minutes), ...) per session, overriding the ``hours`` scaling for those slots
                       of ``protocol.CHAMPION_SLOT_MIN`` (empty: the protocol's allocation).  This is the
                       power-analysis lever the skeptic corrections ask for: how many held-out-mapping
                       minutes (and how much S data for the shadow, how much L1 to earn x) a session
                       needs before the certification can separate deployed from shadow.
    """
    days: int = 3
    hours: float = 1.0
    users: int = 3
    pop: int = 10
    seed: int = DEFAULT.eval.seed
    dry_penalty: float = 0.0
    screen_leak: float = 1.0
    covert_leak: float = 0.0
    replay_attention: str = "cursor"
    x_responder_prob: float = X_RESPONDER_PROB
    y_depth_range: Tuple[float, float] = Y_DEPTH_RANGE
    x_depth_range: Tuple[float, float] = X_DEPTH_RANGE
    user_learning_rate: float = 0.0
    pop_minutes: float = 3.0
    rest_s: float = 60.0
    eval_minutes: float = 2.0
    assist_gain: float = 0.5
    label_hop_s: float = 0.5
    x_permutations: int = 200
    n_boot: int = N_BOOT
    slot_minutes: Tuple[Tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.days < 1 or self.hours <= 0 or self.users < 1 or self.pop < 0:
            raise ValueError(f"invalid simulation size: {self}")
        if not 0.0 <= self.dry_penalty < 1.0 or self.screen_leak < 0 or self.covert_leak < 0:
            raise ValueError(f"invalid leak / dry-penalty setting: {self}")
        if self.replay_attention not in REPLAY_ATTENTION:
            raise ValueError(f"replay_attention must be one of {REPLAY_ATTENTION}, got {self.replay_attention!r}")
        if self.eval_minutes * 60.0 < FixationProbeTiming().trial_s or self.pop_minutes < 0 or self.rest_s < 1.0:
            raise ValueError(f"eval_minutes must hold a probe, pop_minutes ≥ 0, rest_s ≥ 1 s: {self}")
        if not 0.0 < self.assist_gain < 1.0 or self.label_hop_s <= 0:
            raise ValueError(f"assist_gain must be in (0, 1) and label_hop_s > 0: {self}")
        if self.n_boot < MIN_BOOT or self.x_permutations < 20:
            raise ValueError(f"n_boot must be ≥ {MIN_BOOT} and x_permutations ≥ 20: {self}")
        for slot, minutes in self.slot_minutes:
            if slot not in CHAMPION_SLOT_MIN or not minutes > 0:
                raise ValueError(f"slot_minutes needs slots in {tuple(CHAMPION_SLOT_MIN)} with minutes > 0, "
                                 f"got {(slot, minutes)}")


def champion_config(cfg: Config = DEFAULT) -> Config:
    """``cfg`` with the body decoder's 1-s window, so the online chain and the labeler hand every
    decoder exactly the window length it is trained on."""
    return BodyCoordinateDecoder(cfg, cfg.hardware.fs).loop_config(cfg)


def _slot(block: Block) -> str:
    """The plan slot a block belongs to (``protocol.CHAMPION_SLOT_MIN``; setup has none)."""
    return "just_intend" if block.just_intend else block.kind


def day_sessions(day: int, hours: float, seed: int, x_earned: bool,
                 slot_minutes: Sequence[Tuple[str, float]] = ()) -> List[List[Block]]:
    """The day's sessions: ``ceil(hours / 85 min)`` champion plans, each scaled by one factor ≤ 1.

    One factor for every block keeps the protocol's shares (L1 : L2b, held-out and replay counts per
    minute); plans are never stretched because the protocol caps block lengths for attention.  Session
    s uses plan seed ``seed + s``, so a second session of the day is another valid arrangement.
    ``slot_minutes`` replaces the factor for the named slots: their blocks are scaled together to that
    many minutes per session (block order, mappings, variants and seeds are the plan's)."""
    n_sessions = max(1, math.ceil(hours * 3600.0 / CHAMPION_SESSION_S - 1e-9))
    scale = hours * 3600.0 / (n_sessions * CHAMPION_SESSION_S)
    target = dict(slot_minutes)
    sessions = []
    for s in range(n_sessions):
        plan = session_plan_champion(day, seed + s, cohort="deep", x_earned=x_earned)
        slot_s = {k: sum(b.duration_s for b in plan if _slot(b) == k) for k in target}
        sessions.append([dataclasses.replace(b, duration_s=b.duration_s * (
            target[_slot(b)] * 60.0 / slot_s[_slot(b)] if _slot(b) in target else scale)) for b in plan])
    return sessions


# ----------------------------------------------------------------------------------------------
# One person's blocks on the simulated clock
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Trajectory:
    """A past successful sighted trial, kept for pixel replay: its targets, goal A and the displayed
    cursor position after every control step."""
    targets: np.ndarray
    goal_idx: int
    positions: np.ndarray


class ChampionRunner:
    """Runs one person's champion blocks for one day as ``RunLogger`` runs on the simulated clock.

    The clock is ``t0 + samples / fs``: the single clock of every EEG timestamp, event and cursor row.
    Each run gets a fresh ``OnlinePreprocessor`` (the per-update chain of ``realtime.RealTimeLoop``),
    because each run is its own recording and the labeler filters it from its first sample.
    ``decoder`` is the deployed decoder (body velocity out; None → zero output); ``x_earned`` picks the
    L2b cue classes; ``trajectories`` is the person's replay pool, shared across days.
    """

    def __init__(self, subject: BodySubject, cfg: Config, sim: ChampionSimConfig, t0: float,
                 trajectories: List[Trajectory]):
        self.subject, self.cfg, self.sim = subject, cfg, sim
        self.fs = cfg.hardware.fs
        self.hop = int(round(self.fs / cfg.task.control_hz))
        self.t0, self.n = t0, 0
        self.trajectories = trajectories
        self.decoder: Optional[BodyCoordinateDecoder] = None
        self.x_earned = False
        timing = ClosedLoopTiming()
        self.sighted_task = dataclasses.replace(cfg.task, assist_gain=sim.assist_gain,
                                                trial_timeout_s=timing.timeout_s)
        self.blind_task = dataclasses.replace(self.sighted_task, assist_gain=0.0)
        self.pre = OnlinePreprocessor(cfg)
        self._logger: Optional[RunLogger] = None

    @property
    def t(self) -> float:
        return self.t0 + self.n / self.fs

    def advance(self, seconds: float) -> None:
        """Clock time with no recording (the rest of setup, skipped just-intend blocks)."""
        self.n += int(round(seconds * self.fs))

    def emit(self, seconds: float = 0.0, samples: Optional[int] = None) -> None:
        n = int(round(seconds * self.fs)) if samples is None else samples
        if n <= 0:
            return
        x = self.subject.generate(n)
        if self._logger is not None:
            self._logger.eeg(x, self.t + np.arange(n) / self.fs)
        self.pre.push(x)
        self.n += n

    def decode(self) -> np.ndarray:
        """Deployed decoder on the latest preprocessed window (body coordinates); 0 if none deployed."""
        return np.zeros(2) if self.decoder is None else np.asarray(self.decoder.predict(self.pre.window()), float)

    # -- person state -----------------------------------------------------------------------------
    def _rest(self) -> None:
        self.subject.cue(0.0, 0.0)
        self.subject.set_screen_direction(None)

    def _imagined(self, body: np.ndarray) -> np.ndarray:
        """The body vector the person imagines for a desired body velocity: all of it once x is earned,
        only its y component before (1-D y control; module docstring)."""
        return body if self.x_earned else np.array([0.0, body[1]])

    def _attend(self, M: Mapping, goal: np.ndarray, pos: np.ndarray, onset: bool = False) -> None:
        """Visible target: look toward it on screen and imagine the body velocity that moves the cursor
        there under ``M``.  ``onset`` restarts the ERD time course (target onset is the cue)."""
        d = np.asarray(goal, float) - pos
        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            return                                             # on the target: keep the current state
        u = d / norm
        self.subject.set_screen_direction(u)
        body = self._imagined(M.screen_to_body(u))
        (self.subject.cue if onset else self.subject.set_intent)(*body)

    def _attend_replay(self, M: Mapping, goal_a: np.ndarray, goal_b: np.ndarray, pos: np.ndarray,
                       onset: bool = False) -> None:
        """Replay: imagine toward the held goal B; the overt pathway follows the replayed cursor's
        heading toward A (``replay_attention`` "cursor") or goal B ("goal")."""
        body = self._imagined(M.screen_to_body(unit_directions(goal_b[None], pos[None])[0]))
        look = goal_a if self.sim.replay_attention == "cursor" else goal_b
        self.subject.set_screen_direction(unit_directions(look[None], pos[None])[0])
        (self.subject.cue if onset else self.subject.set_intent)(*body)

    def _log_cursor(self, pos: np.ndarray, u: np.ndarray = np.zeros(2), p: float = 0.0,
                    inside: Optional[int] = None, dwell: float = 0.0) -> None:
        self._logger.cursor(self.t, pos, u, p, inside, dwell)

    # -- setup ------------------------------------------------------------------------------------
    def rest_reference(self, day: int, index: int, geometry: BodyCoordinateDecoder) -> np.ndarray:
        """The session's re-centring reference: ``rest_s`` of eyes-open rest through the online chain,
        1-s windows every ``REST_HOP_S``, their Riemannian mean (no label, no display involved)."""
        self.subject.begin_block(day, index)                   # at rest, nothing drawn
        self.pre = OnlinePreprocessor(self.cfg)
        self.emit(PREROLL_S)                                   # filter settling, as in every run
        step = int(round(REST_HOP_S * self.fs))
        windows = []
        for _ in range(max(1, int(self.sim.rest_s / REST_HOP_S))):
            self.emit(samples=step)
            windows.append(self.pre.window())
        return geometry.reference_from(windows)

    # -- blocks -----------------------------------------------------------------------------------
    def run_block(self, block: Block, root: str, tag: str, day: int, index: int,
                  glyphs: Dict[str, str]) -> Tuple[str, Dict[str, List[TrialOutcome]]]:
        """One block as one RunLogger run.  Returns (run_dir, outcomes by endpoint: "blind",
        "sighted", "heldout" — replay and open-loop trials have no task outcome)."""
        self.subject.begin_block(day, index)
        self.subject.covert_leak = self.sim.covert_leak if block.kind in BLIND_KINDS else 0.0
        self.pre = OnlinePreprocessor(self.cfg)
        self._logger = RunLogger(root, self.cfg, tag=tag, clock=lambda: self.t)
        start = self.t
        try:
            self._logger.event("block_start", {"kind": block.kind, "mapping": block.mapping, "variant": block.variant,
                                               "duration_s": block.duration_s, "seed": block.seed}, t=self.t)
            self.emit(PREROLL_S)
            if block.kind == "fixation_probe":
                outcomes = self._fixation(block, glyphs)
            elif block.kind == "blind_closed_loop":
                outcomes = {"blind": self._blind(block, glyphs)}
            else:
                outcomes = self._visible(block)
            self._logger.event("block_end", {"kind": block.kind}, t=self.t)
        finally:
            self._logger.close()
        run_dir, self._logger = self._logger.dir, None
        if block.kind != "fixation_probe":
            self.subject.record_practice(self.t - start)      # closed-loop practice (user learning, if on)
        return run_dir, outcomes

    def _fixation(self, block: Block, glyphs: Dict[str, str]) -> Dict[str, List[TrialOutcome]]:
        """L1: glyph cue → cue_offset → trial_start at the 1.0-s lockout → imagery to 4.0 s → ITI.
        The person imagines from cue onset; nothing spatial is drawn; fixation holds."""
        tm = FixationProbeTiming()
        sched = BodyCueScheduler(block.seed, classes=L1_CLASSES, modes=("central_glyph",), glyphs=glyphs,
                                 logger=self._logger)
        for trial in range(tm.n_trials(block)):
            cue = sched.next(t=self.t)
            self.subject.cue(*cue.body)
            self.emit(tm.cue_s)
            self._logger.event(EVENT_CUE_OFFSET, {"seq": cue.seq}, t=self.t)
            self.emit(tm.label_start_s - tm.cue_s)
            self._logger.event(EVENT_TRIAL_START, champion_trial_start_payload(block, trial, cue=cue), t=self.t)
            self.emit(tm.label_end_s - tm.label_start_s)
            self._logger.event(EVENT_TRIAL_END, {"trial": trial, "outcome": "probe"}, t=self.t)
            self._rest()
            self.emit(tm.iti_s)
        return {}

    def _blind(self, block: Block, glyphs: Dict[str, str]) -> List[TrialOutcome]:
        """L2b: cue → control of a hidden cursor toward the hidden goal by the deployed decoder alone →
        trial_end → central outcome flash → ITI.  Trials end on a dwell hit or at the timeout."""
        tm = ClosedLoopTiming()
        classes = L2B_CLASSES_XY if self.x_earned else L2B_CLASSES_Y
        modes = BODY_CUE_MODES if block.cue == "mixed" else (block.cue,)
        sched = BodyCueScheduler(block.seed, classes=classes, modes=modes, glyphs=glyphs, logger=self._logger)
        runner = TrialRunner(self.blind_task, np.random.default_rng(block.seed))
        width = self.cfg.task.target_radius
        outcomes: List[TrialOutcome] = []
        start, trial = self.t, 0
        while self.t - start < block.duration_s:
            cue = sched.next(t=self.t)
            self.subject.cue(*cue.body)
            self.emit(tm.cue_s)
            self._logger.event(EVENT_CUE_OFFSET, {"seq": cue.seq}, t=self.t)
            self.emit(tm.label_start_s - tm.cue_s)
            goal = np.asarray(hidden_goal(cue), float)
            self._logger.event(EVENT_TRIAL_START, champion_trial_start_payload(block, trial, cue=cue), t=self.t)

            def command(_state, _t):
                self.emit(samples=self.hop)
                return self.decode()                           # identity mapping: body = screen

            def on_step(state, u, p, inside, dwell):
                self._log_cursor(state.pos, u, float(np.max(p)), inside, dwell)

            out = runner.run(goal[None], width, 0, command, on_step=on_step)
            self._logger.event(EVENT_TRIAL_END, {"trial": trial, "outcome": "hit" if out.success else "timeout"},
                               t=self.t)
            self._logger.event(EVENT_OUTCOME_FLASH, {"trial": trial, "hit": bool(out.success)}, t=self.t)
            self._rest()
            self.emit(tm.outcome_s + tm.iti_s)
            outcomes.append(out)
            trial += 1
        return outcomes

    def _visible(self, block: Block) -> Dict[str, List[TrialOutcome]]:
        """S and H blocks: Fitts targets drawn; closed-loop (S, C2c), open-loop (C2b) and replay trials."""
        M = Mapping(block.mapping)
        open_loop = block.kind == "heldout_mapping" and block.variant == "C2b"
        rng = np.random.default_rng(block.seed)
        choice_rng = np.random.default_rng([block.seed, _REPLAY_CHOICE_STREAM])
        replays = ReplayScheduler(block) if block.replay_fraction else None
        runner = TrialRunner(self.sighted_task, rng)
        key = "sighted" if block.kind == "sighted" else "heldout"
        outcomes: List[TrialOutcome] = []
        start, trial = self.t, 0
        while self.t - start < block.duration_s:
            distance, width = FITTS_CONDITIONS[trial % len(FITTS_CONDITIONS)]
            targets = fitts_targets(self.sighted_task.n_targets, distance, width, rng)
            goal_idx = int(rng.integers(len(targets)))
            replay = replays is not None and replays.next()
            if open_loop:
                self._open_trial(block, trial, M, targets[goal_idx])
            elif replay and self.trajectories:
                # A scheduled replay with no past trajectory yet runs as a normal trial.
                past = self.trajectories[int(choice_rng.integers(len(self.trajectories)))]
                self._replay_trial(block, trial, M, past, choice_rng)
            else:
                outcomes.append(self._closed_trial(block, trial, M, targets, goal_idx, width, runner))
            trial += 1
        return {key: outcomes}

    def _closed_trial(self, block: Block, trial: int, M: Mapping, targets: np.ndarray, goal_idx: int,
                      width: float, runner: TrialRunner) -> TrialOutcome:
        """Target onset (the cue) → 1.0-s preview → control: screen command = M · decoded body velocity,
        blended with GoalPosterior assistance by ``TrialRunner`` → outcome + ITI."""
        tm = ClosedLoopTiming()
        goal = targets[goal_idx]
        self._attend(M, goal, np.zeros(2), onset=True)
        self.emit(tm.label_start_s)
        self._logger.event(EVENT_TRIAL_START, champion_trial_start_payload(block, trial, goal=goal), t=self.t)
        positions: List[np.ndarray] = []

        def command(state, _t):
            self._attend(M, goal, state.pos)
            self.emit(samples=self.hop)
            return M.body_to_screen(self.decode())

        def on_step(state, u, p, inside, dwell):
            self._log_cursor(state.pos, u, float(np.max(p)), inside, dwell)
            positions.append(state.pos.copy())

        out = runner.run(targets, width, goal_idx, command, on_step=on_step)
        self._logger.event(EVENT_TRIAL_END, {"trial": trial, "outcome": "hit" if out.success else
                                             ("miss" if out.selected_idx is not None else "timeout"),
                                             "selected": out.selected_idx, "movement_time_s": out.movement_time_s},
                           t=self.t)
        self._rest()
        self.emit(tm.outcome_s + tm.iti_s)
        if block.kind == "sighted" and out.success and len(positions) >= MIN_REPLAY_STEPS:
            self.trajectories.append(Trajectory(targets.copy(), goal_idx, np.array(positions)))
        return out

    def _replay_trial(self, block: Block, trial: int, M: Mapping, past: Trajectory,
                      rng: np.random.Generator) -> None:
        """Pixel replay: the past trajectory toward its goal A is shown step by step while the person
        holds another target of the same set, goal B (logged as the trial's goal; A as replay_goal)."""
        tm = ClosedLoopTiming()
        others = [i for i in range(len(past.targets)) if i != past.goal_idx]
        goal_a, goal_b = past.targets[past.goal_idx], past.targets[int(rng.choice(others))]
        pos = np.zeros(2)
        self._attend_replay(M, goal_a, goal_b, pos, onset=True)
        self.emit(tm.label_start_s)
        self._logger.event(EVENT_TRIAL_START, champion_trial_start_payload(block, trial, goal=goal_b,
                                                                           replay_goal=goal_a), t=self.t)
        for nxt in past.positions:
            self._attend_replay(M, goal_a, goal_b, pos)
            self.emit(samples=self.hop)
            pos = nxt
            self._log_cursor(pos)
        self._logger.event(EVENT_TRIAL_END, {"trial": trial, "outcome": "replay"}, t=self.t)
        self._rest()
        self.emit(tm.outcome_s + tm.iti_s)

    def _open_trial(self, block: Block, trial: int, M: Mapping, goal: np.ndarray) -> None:
        """C2b: target and frozen cursor shown → trial_start after the preview → imagery → trial_end →
        keyed body-part report (the class nearest the imagined body vector) → ITI."""
        tm = OpenLoopTiming()
        centre = np.zeros(2)
        self._attend(M, goal, centre, onset=True)
        self.emit(tm.preview_s)
        self._logger.event(EVENT_TRIAL_START, champion_trial_start_payload(block, trial, goal=goal), t=self.t)
        for _ in range(int(round(tm.imagery_s * self.cfg.task.control_hz))):
            self.emit(samples=self.hop)
            self._log_cursor(centre)
        self._logger.event(EVENT_TRIAL_END, {"trial": trial, "outcome": "open_loop"}, t=self.t)
        body = M.screen_to_body(unit_directions(goal[None], centre[None])[0])
        report = max(BODY_VECTORS, key=lambda c: float(np.dot(BODY_VECTORS[c], body)))
        self._rest()
        self.emit(REPORT_DELAY_S)
        self._logger.event(EVENT_BODY_REPORT, {"trial": trial, "class": report}, t=self.t)
        self.emit(tm.report_s - REPORT_DELAY_S + tm.iti_s)


# ----------------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------------
def recentred_outputs(decoder: BodyCoordinateDecoder, data: LabelledSet, refs: Dict[str, np.ndarray],
                      force_x: bool = False) -> np.ndarray:
    """(n, 2) outputs with every window re-centred on its own session's rest reference, exactly as the
    online decoder saw it.  A shallow copy carries the reference (and ``force_x``), so the decoder
    itself is untouched.  ``force_x`` scores the x model even when x is not earned."""
    out = np.zeros((len(data), 2))
    groups: Dict[str, List[int]] = {}
    for i, p in enumerate(data.provenance):
        groups.setdefault(p["session"], []).append(i)
    for session, idx in groups.items():
        d = copy.copy(decoder)
        d.x_enabled = d.x_enabled or force_x
        d.set_reference(reference=refs[session])
        out[idx] = d.predict_many([data.windows[i] for i in idx])
    return out


def l1_scores(decoder: Optional[BodyCoordinateDecoder], pool: LabelledSet, refs: Dict[str, np.ndarray]) -> dict:
    """AUC and r per axis on the fixed L1 pool (y: FEET vs HANDS probes; x: RIGHT vs LEFT probes)."""
    nan = float("nan")
    if decoder is None or len(pool) == 0:
        return {"auc": {a: nan for a in AXES}, "r": {a: nan for a in AXES}, "n": {a: 0 for a in AXES}}
    out = recentred_outputs(decoder, pool, refs, force_x=True)
    y = body_labels(pool)
    res: dict = {"auc": {}, "r": {}, "n": {}}
    for k, axis in enumerate(AXES):
        m = y[:, k] != 0
        both = m.any() and len(np.unique(y[m, k])) == 2
        res["auc"][axis] = float(roc_auc_score(y[m, k] > 0, out[m, k])) if both else nan
        res["r"][axis] = pearson_r(out[m, k], y[m, k].astype(float)) if both else nan
        res["n"][axis] = int(m.sum())
    return res


def regressors(test: str, data: LabelledSet) -> Tuple[np.ndarray, np.ndarray]:
    """(body, screen) regressors in the decoder's output frame, from the logged task geometry only.

    heldout: body = M⁻¹·unit(goal − displayed cursor), screen = M_train⁻¹·unit(same) (= the unit
             direction under the identity): ``mapping.certification_regressors`` per window, since a
             day mixes mappings.
    replay:  body = M⁻¹·unit(held goal B − displayed cursor), screen = M_train⁻¹·unit(replayed goal A −
             displayed cursor): what a decoder reading the replayed cursor would output."""
    goals = np.array([p["display_goal"] for p in data.provenance], float).reshape(-1, 2)
    cursors = np.array([p["cursor_pos"] for p in data.provenance], float).reshape(-1, 2)
    u = unit_directions(goals, cursors)
    body = np.stack([Mapping(p["mapping"]).screen_to_body(v) for p, v in zip(data.provenance, u)]).reshape(-1, 2)
    if test == "heldout":
        screen = Mapping(TRAINING_MAPPING).screen_to_body(u)
    else:
        replayed = np.array([p["replay_goal"] for p in data.provenance], float).reshape(-1, 2)
        screen = Mapping(TRAINING_MAPPING).screen_to_body(unit_directions(replayed, cursors))
    return body, screen


def _beta_summary(entry: dict) -> dict:
    keys = ("beta_body", "ci_body", "beta_screen", "ci_screen", "rel_body", "rel_screen", "rel_ci_screen",
            "scale", "regressor_corr", "identifiable")
    return {k: entry[k] for k in keys}


@dataclass
class CertificationLog:
    """Per-window inputs of one certification test (deployed and shadow outputs, both regressors,
    provenance, and whether x was earned when the window was decoded), accumulated so a test can be
    run per day, pooled over a person's days, or pooled over people (session ids are unique per
    person and run, so trial clusters never merge across them)."""
    deployed: List[np.ndarray] = field(default_factory=list)
    shadow: List[np.ndarray] = field(default_factory=list)
    body: List[np.ndarray] = field(default_factory=list)
    screen: List[np.ndarray] = field(default_factory=list)
    provenance: List[dict] = field(default_factory=list)
    x_enabled: List[np.ndarray] = field(default_factory=list)

    def add(self, deployed: np.ndarray, shadow: np.ndarray, body: np.ndarray, screen: np.ndarray,
            provenance: Sequence[dict], x_enabled: np.ndarray) -> None:
        self.deployed.append(deployed)
        self.shadow.append(shadow)
        self.body.append(body)
        self.screen.append(screen)
        self.provenance.extend(provenance)
        self.x_enabled.append(np.asarray(x_enabled, bool))

    @staticmethod
    def merge(logs: Sequence["CertificationLog"]) -> "CertificationLog":
        out = CertificationLog()
        for log in logs:
            out.deployed += log.deployed
            out.shadow += log.shadow
            out.body += log.body
            out.screen += log.screen
            out.provenance += log.provenance
            out.x_enabled += log.x_enabled
        return out

    def certify(self, test: str, n_boot: int) -> dict:
        """β table and verdict of deployed vs shadow on these windows.  The x axis is tested only if it
        was earned for every window (otherwise the deployed x output is 0 by rule, not by measurement,
        and would only dilute the pooled fit); y is the primary axis."""
        n_trials = int(trial_clusters(self.provenance).max()) + 1 if self.provenance else 0
        if n_trials < MIN_CLUSTERS:
            return {"skipped": f"{n_trials} trials (< {MIN_CLUSTERS} for a trial-cluster bootstrap)",
                    "n_trials": n_trials, "n_windows": len(self.provenance)}
        axes = ("x", "y") if np.concatenate(self.x_enabled).all() else ("y",)
        fn = certify_mapping if test == "heldout" else replay_test
        args = (np.concatenate(self.body), np.concatenate(self.screen), self.provenance)
        dep = fn(np.concatenate(self.deployed), *args, axes=axes, n_boot=n_boot)
        sha = fn(np.concatenate(self.shadow), *args, axes=axes, n_boot=n_boot)
        v = certify(dep, sha)
        return {"verdict": v["verdict"], "reason": v["reason"], "axes": list(axes), "flags": v["flags"],
                "n_trials": n_trials, "n_windows": len(self.provenance),
                "deployed": _beta_summary(v["deployed"]), "shadow": _beta_summary(v["shadow"])}


# ----------------------------------------------------------------------------------------------
# Population prior, fixed L1 pool, one person
# ----------------------------------------------------------------------------------------------
def make_population(sim: ChampionSimConfig, cfg: Config) -> BodyPopulation:
    """Subjects 0 … users−1 are the deep participants; users … users+pop−1 the prior's members.
    ``covert_leak`` is set per block by ``ChampionRunner`` (blind blocks only), so it is 0 here."""
    return BodyPopulation(sim.users + sim.pop, sim.seed, cfg.hardware, x_responder_prob=sim.x_responder_prob,
                          y_depth_range=tuple(sim.y_depth_range), x_depth_range=tuple(sim.x_depth_range),
                          dry_penalty=sim.dry_penalty, screen_leak=sim.screen_leak, covert_leak=0.0,
                          user_learning_rate=sim.user_learning_rate)


def record_fixation(cfg: Config, sim: ChampionSimConfig, subject: BodySubject, day: int, block: Block,
                    index: int, tag: str, root: str, t0: float) -> Tuple[LabelledSet, Dict[str, np.ndarray]]:
    """A rest reference (block ``index``) plus one fixation-probe block (``index + 1``) for ``subject``
    on ``day``, labelled through the logger → store → labeler path (the run directory is deleted
    afterwards).  Returns the windows and {session: reference}.  Used for prior members and for the
    fixed L1 test pool."""
    subject.set_day(day)
    runner = ChampionRunner(subject, cfg, sim, t0, [])
    reference = runner.rest_reference(day, index, BodyCoordinateDecoder(cfg, cfg.hardware.fs))
    run_dir, _ = runner.run_block(block, root, tag, day, index + 1, glyph_map(day, block.seed))
    ls = Labeler(cfg, hop_s=sim.label_hop_s).label(SessionStore(root, cfg.hardware.n_channels).load(run_dir))
    shutil.rmtree(run_dir)
    return ls, {os.path.basename(os.path.normpath(run_dir)): reference}


def build_member(cfg: Config, sim: ChampionSimConfig, member: int, out_dir: str) -> str:
    """One prior member: ``pop_minutes`` of fixation probes on day 1 (cue-defined L1 labels only, so the
    prior carries no display-identifiable label either), saved as .npy for memory-mapped loading."""
    pop = make_population(sim, cfg)
    seed = pop.params[member].seed
    block = Block("fixation_probe", sim.pop_minutes * 60.0, seed=seed)
    ls, refs = record_fixation(cfg, sim, pop.subject(member), 1, block, 0, f"pop{member:03d}",
                               os.path.join(out_dir, "runs"), CLOCK_ORIGIN_S)
    d = os.path.join(out_dir, f"member_{member:03d}")
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "windows.npy"), np.stack(ls.windows))
    np.save(os.path.join(d, "labels.npy"), ls.labels)
    (session, reference), = refs.items()
    np.save(os.path.join(d, "reference.npy"), reference)
    with open(os.path.join(d, "provenance.json"), "w") as f:
        json.dump({"session": session, "provenance": ls.provenance}, f)
    return d


def load_prior(prior_dir: str) -> Tuple[LabelledSet, Dict[str, np.ndarray]]:
    """All members' L1 windows (memory-mapped) and their rest references."""
    sets, refs = [], {}
    for name in sorted(n for n in os.listdir(prior_dir) if n.startswith("member_")):
        d = os.path.join(prior_dir, name)
        win = np.load(os.path.join(d, "windows.npy"), mmap_mode="r")
        with open(os.path.join(d, "provenance.json")) as f:
            meta = json.load(f)
        sets.append(LabelledSet([win[i] for i in range(len(win))], np.load(os.path.join(d, "labels.npy")),
                                meta["provenance"]))
        refs[meta["session"]] = np.load(os.path.join(d, "reference.npy"))
    return LabelledSet.concat(sets), refs


def build_l1_pool(cfg: Config, sim: ChampionSimConfig, pop: BodyPopulation, user: int,
                  root: str) -> Tuple[LabelledSet, Dict[str, np.ndarray]]:
    """The fixed L1 test set of ``user``: ``eval_minutes`` of fixation probes per simulated day, each
    under that day's electrode gains with its own rest reference and noise stream, on a fresh subject
    instance at zero practice, never added to any training set."""
    seed = pop.params[user].seed
    sets, refs = [], {}
    for day in range(1, sim.days + 1):
        block = Block("fixation_probe", sim.eval_minutes * 60.0, seed=seed + EVAL_BLOCK_INDEX + day)
        ls, r = record_fixation(cfg, sim, pop.subject(user), day, block, EVAL_BLOCK_INDEX, f"eval_u{user}_d{day:02d}",
                                os.path.join(root, "eval_runs"), CLOCK_ORIGIN_S + (day - 1) * DAY_OFFSET_S)
        sets.append(ls)
        refs.update(r)
    return LabelledSet.concat(sets), refs


def _by_source(data: LabelledSet, *sources: str) -> LabelledSet:
    return data.mask(np.array([p.get("source") in sources for p in data.provenance], bool))


def simulate_user(cfg: Config, sim: ChampionSimConfig, user: int, prior_dir: Optional[str],
                  root: str) -> Tuple[dict, Dict[str, CertificationLog]]:
    """``sim.days`` simulated days of one deep participant; returns (per-day rows, certification logs)."""
    pop = make_population(sim, cfg)
    params = pop.params[user]
    subject = pop.subject(user)
    geometry = BodyCoordinateDecoder(cfg, cfg.hardware.fs)
    options = {"n_permutations": sim.x_permutations}
    trainer, shadow_trainer = BodyTrainer(cfg, decoder_options=options), ShadowTrainer(cfg, decoder_options=options)
    labeler, runs = Labeler(cfg, hop_s=sim.label_hop_s), os.path.join(root, "runs")
    os.makedirs(runs, exist_ok=True)
    store = SessionStore(runs, cfg.hardware.n_channels)

    pool, refs = build_l1_pool(cfg, sim, pop, user, root)
    prior, prior_refs = load_prior(prior_dir) if prior_dir else (LabelledSet(), {})
    refs.update(prior_refs)
    deployed = trainer.fit(prior, session_references=prior_refs) if len(prior) else None
    prior_l1 = l1_scores(deployed, pool, refs)
    own, sighted = LabelledSet(), LabelledSet()
    trajectories: List[Trajectory] = []
    logs = {t: CertificationLog() for t in TESTS}
    x_report: Optional[dict] = None
    rows = []

    def retrain(new: List[LabelledSet]) -> None:
        nonlocal own, deployed, x_report
        own = LabelledSet.concat([own] + new)
        deployed = trainer.fit(LabelledSet.concat([own, prior]), session_references=refs)
        x_report = deployed.earn_x(list(own.windows), body_labels(own), own.provenance, refs)

    for day in range(1, sim.days + 1):
        subject.set_day(day)
        runner = ChampionRunner(subject, cfg, sim, CLOCK_ORIGIN_S + (day - 1) * DAY_OFFSET_S, trajectories)
        outcomes: Dict[str, List[TrialOutcome]] = {"blind": [], "sighted": [], "heldout": []}
        pending: Dict[str, List[Tuple[LabelledSet, np.ndarray, bool]]] = {t: [] for t in TESTS}
        new_labelled: List[LabelledSet] = []
        counts = {s: 0 for s in ("L1", "L2b", "S", "H", "R")}
        x_at_start = bool(deployed is not None and deployed.x_enabled)
        for s, plan in enumerate(day_sessions(day, sim.hours, params.seed, x_at_start, sim.slot_minutes)):
            glyphs = glyph_map(day, params.seed + s)
            reference = None
            for i, block in enumerate(plan):
                index = s * SESSION_BLOCK_STRIDE + i
                if block.kind == "setup":
                    reference = runner.rest_reference(day, index, geometry)
                    runner.advance(max(0.0, block.duration_s - PREROLL_S - sim.rest_s))
                    if deployed is not None:
                        deployed.set_reference(reference=reference)
                    continue
                if block.just_intend:
                    runner.advance(block.duration_s)
                    continue
                if block.kind == "sighted" and new_labelled:
                    retrain(new_labelled)
                    new_labelled = []
                    deployed.set_reference(reference=reference)
                runner.decoder = deployed
                runner.x_earned = bool(deployed is not None and deployed.x_enabled)
                run_dir, outs = runner.run_block(block, runs, f"u{user}_d{day:02d}_s{s}_b{i:02d}_{block.kind}",
                                                 day, index, glyphs)
                for k, v in outs.items():
                    outcomes[k].extend(v)
                ls = labeler.label(store.load(run_dir))
                shutil.rmtree(run_dir)
                refs[os.path.basename(os.path.normpath(run_dir))] = reference
                for src in counts:
                    counts[src] += sum(p.get("source") == src for p in ls.provenance)
                if block.kind in BLIND_KINDS:
                    new_labelled.append(ls)
                sighted = LabelledSet.concat([sighted, _by_source(ls, "S")])
                for test, src in zip(TESTS, ("H", "R")):
                    part = _by_source(ls, src)
                    if len(part):
                        # The frozen decoder that ran the block scores its windows now, before any retrain.
                        pending[test].append((part, recentred_outputs(deployed, part, refs), runner.x_earned))
        if new_labelled:
            retrain(new_labelled)
        shadow = (shadow_trainer.fit(sighted, x_enabled=deployed.x_enabled, session_references=refs)
                  if len(sighted) else None)
        certification = {}
        for test in TESTS:
            today = CertificationLog()
            for part, dep_out, x_on in pending[test]:
                if shadow is None:
                    continue
                body, screen = regressors(test, part)
                today.add(dep_out, recentred_outputs(shadow, part, refs), body, screen, part.provenance,
                          np.full(len(part), x_on and shadow.x_enabled))
            certification[test] = today.certify(test, sim.n_boot)
            logs[test] = CertificationLog.merge([logs[test], today])
        rows.append({
            "day": day, "cum_hours": day * sim.hours,
            "x_earned": bool(deployed.x_enabled), "x_report": x_report and {
                k: x_report[k] for k in ("auc", "floor", "p_value", "n_windows", "reason")},
            "l1": l1_scores(deployed, pool, refs),
            "sighted": trial_metrics(outcomes["sighted"], fitts=True),
            "blind": trial_metrics(outcomes["blind"], fitts=False),
            "heldout_closed_loop": trial_metrics(outcomes["heldout"], fitts=True),
            "n_windows": counts, "n_train_windows": {"own": len(own), "prior": len(prior), "shadow": len(sighted)},
            "certification": certification,
        })
    pooled = {t: logs[t].certify(t, sim.n_boot) for t in TESTS}
    result = {"user": user, "subject": pop.describe([user])[0], "prior_l1": prior_l1, "days": rows,
              "pooled_days": pooled}
    return result, logs


def _member_job(args: tuple) -> str:
    cfg, sim, member, out_dir = args
    with threadpool_limits(limits=1):
        return build_member(cfg, sim, member, out_dir)


def _user_job(args: tuple) -> Tuple[dict, Dict[str, CertificationLog]]:
    """Process-pool entry point.  BLAS is pinned to one thread: jobs already fill the CPUs, and the
    11 × 11 matrices here are far too small to gain from threading."""
    cfg, sim, user, prior_dir, root = args
    with threadpool_limits(limits=1):
        os.makedirs(root, exist_ok=True)
        return simulate_user(cfg, sim, user, prior_dir, root)


def _map(fn: Callable, jobs: List[tuple], workers: int) -> list:
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(fn, jobs))


def simulate(sim: ChampionSimConfig, cfg: Config = DEFAULT, workers: int = 1) -> dict:
    """One condition: build the population prior (members in parallel), run every deep participant
    (in parallel), and certify pooled over participants.  Run directories live in a temporary
    directory and are deleted as soon as they are labelled."""
    t0 = time.perf_counter()
    cfg = champion_config(cfg)
    pop = make_population(sim, cfg)
    with tempfile.TemporaryDirectory(prefix="nc_champion_") as tmp:
        prior_dir = None
        if sim.pop > 0:
            prior_dir = os.path.join(tmp, "prior")
            _map(_member_job, [(cfg, sim, j, prior_dir) for j in range(sim.users, sim.users + sim.pop)], workers)
        out = _map(_user_job, [(cfg, sim, u, prior_dir, os.path.join(tmp, f"u{u}")) for u in range(sim.users)],
                   workers)
    per_user = [r for r, _ in out]
    pooled = {t: CertificationLog.merge([logs[t] for _, logs in out]).certify(t, sim.n_boot) for t in TESTS}
    return {"config": dataclasses.asdict(sim), "population": pop.describe(range(sim.users, sim.users + sim.pop)),
            "per_user": per_user, "pooled_all_users": pooled, "runtime_s": time.perf_counter() - t0}


def sweep(sim: ChampionSimConfig, dry_penalties: Sequence[float], cfg: Config = DEFAULT, workers: int = 1) -> dict:
    """The dry-penalty sensitivity sweep the skeptic corrections ask for (a d′-loss range, not a point
    prediction): the same people, days and seeds under each ``dry_penalty``."""
    t0 = time.perf_counter()
    conditions = {f"{p:g}": simulate(dataclasses.replace(sim, dry_penalty=float(p)), cfg, workers)
                  for p in dry_penalties}
    return {"dry_penalties": [float(p) for p in dry_penalties], "conditions": conditions,
            "runtime_s": time.perf_counter() - t0}


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def _num(x, nd: int = 3) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _ci(beta: float, ci: Sequence[float]) -> str:
    return f"{_num(beta, 2)} [{_num(ci[0], 2)}, {_num(ci[1], 2)}]"


def _cert_cells(c: dict) -> str:
    if "skipped" in c:
        return f"| – | {c['n_trials']} | – | – | – | – | skipped: {c['skipped']} |"
    d, s = c["deployed"], c["shadow"]
    return (f"| {'+'.join(c['axes'])} | {c['n_trials']} | {_ci(d['beta_body'], d['ci_body'])} "
            f"| {_ci(d['beta_screen'], d['ci_screen'])} | {_ci(s['beta_body'], s['ci_body'])} "
            f"| {_ci(s['beta_screen'], s['ci_screen'])} | **{c['verdict']}** |")


def to_markdown(res: dict) -> str:
    first = next(iter(res["conditions"].values()))
    c = first["config"]
    lines = [f"## Champion protocol (simulated): {c['days']} days × {c['hours']:g} h, {c['users']} deep "
             f"participant(s), population prior N = {c['pop']}",
             "",
             f"seed {c['seed']}; screen leak {c['screen_leak']:g} (targets drawn: S, H, R); covert leak "
             f"{c['covert_leak']:g} (blind L1/L2b blocks); replay attention '{c['replay_attention']}'; x-responder "
             f"probability {c['x_responder_prob']:g}; user learning rate {c['user_learning_rate']:g}; assistance "
             f"a = {c['assist_gain']:g} (sighted, C2c); prior {c['pop_minutes']:g} min of L1 per member; label hop "
             f"{c['label_hop_s']:g} s; runtime {res['runtime_s']:.0f} s.",
             "Deployed decoder: trained on L1 + L2b only (+ prior), re-centred on each session's rest.  Shadow: same "
             "model on sighted ReFIT labels (positive control, never deployed).  L1 AUC: fixed fixation-probe test "
             f"set ({c['eval_minutes']:g} min per simulated day, never trained on); x AUC is the x model's even when x "
             "is not earned.  Sighted: non-replay S trials (success, time-to-target, Fitts throughput); blind: L2b "
             "hidden-cursor success.  β: OLS of decoder output on body intent and on the screen direction "
             "[95 % trial-cluster bootstrap CI], under held-out mappings (H) and pixel replay (R).  Intent is an "
             "oracle (exact cued body vector; instant remapping), so β_body is an upper bound.  'Just intend' "
             "blocks are not simulated.",
             ""]
    for name, cond in res["conditions"].items():
        lines += [f"### dry penalty {name}", "",
                  "| user | day | cum h | x earned (AUC / floor) | L1 AUC y | L1 AUC x | L1 r y | sighted success "
                  "| TTT (s) | TP (bits/s) | blind success | windows L1/L2b/S/H/R |",
                  "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for u in cond["per_user"]:
            pl = u["prior_l1"]["auc"]
            lines.append(f"| {u['user']} | 0 (prior only) | 0 | – | {_num(pl['y'])} | {_num(pl['x'])} "
                         f"| {_num(u['prior_l1']['r']['y'])} | – | – | – | – | – |")
            for r in u["days"]:
                xr = r["x_report"] or {}
                n = r["n_windows"]
                lines.append(
                    f"| {u['user']} | {r['day']} | {r['cum_hours']:g} | {'yes' if r['x_earned'] else 'no'} "
                    f"({_num(xr.get('auc'))} / {_num(xr.get('floor'))}) | {_num(r['l1']['auc']['y'])} "
                    f"| {_num(r['l1']['auc']['x'])} | {_num(r['l1']['r']['y'])} | {_num(r['sighted']['success'], 2)} "
                    f"| {_num(r['sighted']['time_to_target_s'], 1)} | {_num(r['sighted']['throughput_bps'], 2)} "
                    f"| {_num(r['blind']['success'], 2)} | {n['L1']}/{n['L2b']}/{n['S']}/{n['H']}/{n['R']} |")
        lines += ["", "| user | day | test | axes | trials | deployed β_body | deployed β_screen | shadow β_body "
                  "| shadow β_screen | verdict |", "|---:|---|---|---|---:|---|---|---|---|---|"]
        for u in cond["per_user"]:
            for r in u["days"]:
                for test in TESTS:
                    lines.append(f"| {u['user']} | {r['day']} | {test} " + _cert_cells(r["certification"][test]))
            for test in TESTS:
                lines.append(f"| {u['user']} | all | {test} " + _cert_cells(u["pooled_days"][test]))
        for test in TESTS:
            lines.append(f"| all | all | {test} " + _cert_cells(cond["pooled_all_users"][test]))
        lines.append("")
    return "\n".join(lines)


def write_report(res: dict, prefix: str) -> Tuple[str, str]:
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    with open(prefix + ".json", "w") as f:
        json.dump(_clean(res), f, indent=2)
    with open(prefix + ".md", "w") as f:
        f.write(to_markdown(res) + "\n")
    return prefix + ".json", prefix + ".md"


def main(argv: Optional[Sequence[str]] = None) -> dict:
    d = ChampionSimConfig()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=d.days)
    ap.add_argument("--hours", type=float, default=d.hours, help="session hours per day (1 = 85-min plan scaled)")
    ap.add_argument("--users", type=int, default=d.users, help="deep participants")
    ap.add_argument("--pop", type=int, default=d.pop, help="population-prior members; 0 = own data only")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--dry-penalty", type=float, nargs="+", default=[d.dry_penalty],
                    help="motor d′ loss; several values run the dry-penalty sweep on the same people")
    ap.add_argument("--screen-leak", type=float, default=d.screen_leak,
                    help="overt display pathway gain while targets are drawn (> 0 exercises the shadow control)")
    ap.add_argument("--covert-leak", type=float, default=d.covert_leak,
                    help="covert attention to the imagined screen direction in blind (L1/L2b) blocks")
    ap.add_argument("--replay-attention", choices=REPLAY_ATTENTION, default=d.replay_attention)
    ap.add_argument("--x-responder-prob", type=float, default=d.x_responder_prob)
    ap.add_argument("--learning-rate", type=float, default=d.user_learning_rate)
    ap.add_argument("--pop-minutes", type=float, default=d.pop_minutes)
    ap.add_argument("--eval-minutes", type=float, default=d.eval_minutes)
    ap.add_argument("--label-hop", type=float, default=d.label_hop_s)
    ap.add_argument("--x-permutations", type=int, default=d.x_permutations)
    ap.add_argument("--n-boot", type=int, default=d.n_boot)
    ap.add_argument("--slot-minutes", nargs="*", default=[], metavar="SLOT=MIN",
                    help=f"per-session minutes for plan slots {tuple(CHAMPION_SLOT_MIN)}, overriding --hours "
                    "for those slots (power analysis of the session allocation)")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--out", type=str, default="", help="output prefix for .json and .md (default "
                    "results/champion_d<days>_h<hours>_u<users>_pop<N>_dry<...>_leak<...>_covert<...>)")
    a = ap.parse_args(argv)
    sim = ChampionSimConfig(days=a.days, hours=a.hours, users=a.users, pop=a.pop, seed=a.seed,
                            screen_leak=a.screen_leak, covert_leak=a.covert_leak,
                            replay_attention=a.replay_attention, x_responder_prob=a.x_responder_prob,
                            user_learning_rate=a.learning_rate, pop_minutes=a.pop_minutes,
                            eval_minutes=a.eval_minutes, label_hop_s=a.label_hop, x_permutations=a.x_permutations,
                            n_boot=a.n_boot,
                            slot_minutes=tuple((k, float(v)) for k, v in (kv.split("=", 1) for kv in a.slot_minutes)))
    res = sweep(sim, a.dry_penalty, DEFAULT, workers=a.workers)
    prefix = a.out or os.path.join("results", f"champion_d{a.days}_h{a.hours:g}_u{a.users}_pop{a.pop}_dry"
                                   + "-".join(f"{p:g}" for p in a.dry_penalty)
                                   + f"_leak{a.screen_leak:g}_covert{a.covert_leak:g}")
    paths = write_report(res, prefix)
    print(to_markdown(res))
    print(f"\nwrote {paths[0]} and {paths[1]}")
    return res


if __name__ == "__main__":
    main()
