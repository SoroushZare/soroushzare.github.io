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
   (``BodyCoordinateDecoder.reference_from``; one for the central-11 montage, one for the all-32 shadow).
   It is the only use of unlabelled data, and every run of the session is re-centred on it (training,
   online decoding, scoring).  Its windows are also rest windows (source "L0") for the idle gate.
2. fixation_probe (L1): central glyph cues (``BodyCueScheduler``, glyph map redrawn per session), the
   person imagines the cued body part from cue onset (``BodySubject.cue``), nothing spatial on screen.
   The rest between probes (from ``ITI_REST_SETTLE_S`` after the trial end to the next cue) gives more
   L0 windows.
3. blind_closed_loop (L2b): glyph or body-word cues, the deployed decoder drives a hidden cursor toward
   the hidden goal (no assistance: with one hidden goal it would steer for the person), the person
   keeps imagining the cued body part.  y-only classes until x is earned.
4. before the first sighted block of the session the deployed decoder is retrained on ALL of the
   person's L1/L2b windows so far (``data_engine.BodyTrainer``, which refuses any other source) plus
   the population prior, and ``earn_x`` re-decides the x axis on the person's own L1 probes (a fixed
   window population): granted on two consecutive passing grant tests (one retrain a day, so day 2 at
   the earliest), kept unless its keep rule rejects it.  The x model is trained on exactly those own
   L1 probes (``x_mask``: the recipe the grant certified), and an x earner's y model treats one-hand
   windows as neutral (``neutral_one_hand``).  The idle gate
   (``BodyCoordinateDecoder.fit_idle``) is refitted on all L0 windows (own and the prior's) against
   those training windows; it is armed only if it can tell rest from intent, and an armed gate makes
   the output exactly zero at rest.
   On day 1 the L2b blocks therefore run on the population-prior decoder alone (zero-calibration
   start, as the proposal uses the prior from day 1); with ``pop`` = 0 nothing moves the cursor there.
5. sighted (S): Fitts targets, identity mapping, the deployed decoder blended with the existing
   ``GoalPosterior`` assistance (a = 0.5) through ``task.TrialRunner``.  The person intends
   M⁻¹·unit(target − cursor) (its y component only until x is earned) and looks toward the target
   (``set_screen_direction``).  15 % of trials are pixel replays (``ReplayScheduler``): a past successful S trajectory toward goal A is replayed
   while the person holds goal B.  S windows train only the SHADOW decoder (``ShadowTrainer``, ReFIT
   labels), never the deployed one.
6. heldout_mapping (H): rot+90 and rot−90 in every session, C2b (open loop: frozen cursor, 3 s
   imagery, keyed report) in sessions 1–2, then C2b/C2c alternating by session; C2c hosts replay
   trials too, drawn from past rotated C2c trajectories (same diagonal layout) and decided before a
   goal quadrant is drawn, so replays never unbalance the rotated trials' runs of four.  Frozen decoder.  Targets are the four diagonals (``mapping.diagonal_targets``) with the
   goal quadrant balanced in every run of four trials (``mapping.quadrant_schedule``), so under rot±90
   the unreachable vertical offset (the y output's screen regressor) varies independently of the
   reachable horizontal one even for a 1-D y person.  ``identity_fraction`` of the trials (one per
   period, at a random position) are IDENTITY trials (I) on the same targets: the decoder's EXTERNAL
   margin is its slope on body intent in them (``identity_trial_margin``), measured on windows that
   never enter the certification.  The protocol has no "I" source yet, so the simulation logs I trials
   in a parallel run of the same EEG as an identity-mapping sighted block (source "S"), which it keeps
   out of every trainer and counts separately.
7. "just intend" blocks are NOT simulated (the clock advances): the simulated person has no
   instruction-dependent behaviour, so they would replay L2b/S with another cue and add no
   information here.  Day 1's hint-free block is skipped for the same reason.

At day end both shadows are refit on all S windows so far — the central-11 shadow (the deployed
model with ReFIT labels) and the all-32-channel shadow (it sees the posterior and frontal display
signal the central strip barely carries on y) — and the day's H and R windows give the β table:
``certify.certify_mapping`` / ``replay_test`` of deployed and shadow outputs on the same windows per
test unit (``CertificationLog.certify``), combined over units (fixed effect within a person) and over
people (person-then-trial bootstrap) with ``certify.combine_units``, and ``certify.certify``'s verdict
(worst axis, Holm-corrected family, either shadow as the positive control, per-unit veto).  Every run goes through ``RunLogger`` → ``SessionStore``
→ ``Labeler``, so decoders see exactly the windows a real session would give them.

Endpoints per day
-----------------
* L1 fixed-test-set AUC (with its trial-permutation floor) / r per axis: a pool of ``eval_minutes`` of
  fixation probes per simulated day, recorded once (fresh subject instance, that day's electrode gains,
  its own rest reference) and never trained on; every day's decoder meets the same windows.  The x AUC is of the x model even when x is
  not earned (the deployed output is then 0 on x); "x earned" is reported beside it.
* x earned, with ``earn_x``'s AUC and circular-shift floor.
* sighted success / time-to-target / Fitts throughput (non-replay S trials); blind success (L2b).
* certification: β_body, β_screen (relative to the external identity-trial margin) and β_cross with
  95 % CR2 cluster-robust t intervals (trials as clusters) for deployed and both shadows, the largest body–screen regressor
  correlation of any block, under held-out mappings (also C2b / C2c separately) and under replay, and
  the verdict; also combined over days per user and over users.
* idle gate: armed or not (its out-of-fold AUC), and on the fixed pool's rest (L0) and probe windows the
  share of rest windows it zeroes and of intent windows it passes.
* leak transfer: a paired probe of the frozen held-out decoder (``leak_transfer``: identical noise
  streams at screen leak 0 and 1), its output slope on the screen direction ÷ the external margin —
  whether a display leak reaches the DEPLOYED decoder, which a screen-leak sweep at covert leak 0 does
  not test (that sweep measures the shadows' sensitivity).
* report check: held-out and replay trials dropped because the keyed report did not confirm the intent.

Positive-control calibration: with the screen leak on (and ``null_calibration``), every person also
runs at screen leak 0 with the same seeds, noise streams and prior — the same decoder versions and
test units — and each shadow must read the screen BEYOND its result there (``certify``'s
``shadow_null``); at screen leak 0 a run is its own null.  The control loop runs at the champion's
16 Hz (``champion_config``).

Leak model (the parts of the result that are assumptions, stated)
-----------------------------------------------------------------
* ``screen_leak`` (default 1.0 > 0): ``BodySubject``'s overt pathway — posterior alpha lateralised and
  a frontal gaze potential toward the drawn target's screen direction — active only while targets are
  drawn (S, H, R).  It is what the shadow can learn from its display-identifiable labels.
* ``covert_leak`` (default 0): covert spatial attention toward the screen direction the person imagines
  in BLIND blocks (L1, L2b), where nothing is drawn.  At 0 the deployed decoder's training data hold
  no screen-correlated non-motor signal by construction, so every CERTIFIED verdict is conditional on
  it; ``sweep(covert_leaks=…)`` / ``--covert-leak 0 1 2`` measures how much the certification detects.  Under the training (identity) mapping that
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
  M⁻¹·unit(target − cursor) (S/H), and remaps instantly under held-out mappings (so the keyed reports
  always confirm).  β_body is therefore an upper bound on what real remapping gives.  Until x is
  earned the person imagines only the y component of M⁻¹·unit(target − cursor): a non-earner is taught
  1-D y control (proposal §4, "1-D y + dwell + goal posterior"), so one-hand imagery is not used.  An
  earner imagines the full vector, one-hand parts included, against a y model trained to stay neutral
  on them.  L1 probes still cue all four classes, so x can be earned.

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

from .body_decoder import AXES, IDLE_SOURCE, BodyCoordinateDecoder
from .certify import (MIN_BOOT, MIN_CLUSTERS, MIN_REPLAY_CLUSTERS, N_BOOT, SEED, VERDICTS, certify,
                      certify_mapping, cluster_robust_ols, combine_units, confirmed_by_report, replay_test, welch)
from .config import DEFAULT, Config
from .data_engine import (EVENT_CUE_OFFSET, EVENT_TRIAL_END, EVENT_TRIAL_START, BodyTrainer, LabelledSet, Labeler,
                          SessionStore, ShadowTrainer, body_labels, trial_clusters)
from .evaluation import pearson_r, permutation_floor
from .logger import RunLogger
from .mapping import (BODY_VECTORS, DIAGONAL_DIRECTIONS, TRAINING_MAPPING, Mapping, diagonal_targets,
                      quadrant_schedule, unit_directions)
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
L1_FLOOR_PERMUTATIONS = 500     # trial-level label permutations for the L1 AUC floor (p95 stable to ~0.01)
EVAL_BLOCK_INDEX = 10_000          # block index of the fixed L1 test pool: never collides with a plan block
SESSION_BLOCK_STRIDE = 100         # block index = session × stride + position: unique within a day
REST_HOP_S = 0.5                   # spacing of the rest windows that form the re-centring reference
MIN_REPLAY_STEPS = 20              # a replayable trajectory lasts ≥ 2 s of control, so it yields windows
REPORT_DELAY_S = 0.6               # C2b keyed report: a reaction time after trial_end (a movement, so outside)
_REPLAY_CHOICE_STREAM = 21         # RNG stream for which past trajectory / which held goal B a replay uses
_HELDOUT_GEOMETRY_STREAM = 22      # RNG stream for the held-out blocks' goal quadrants and identity trials
ITI_REST_SETTLE_S = 0.5            # L0 windows of the inter-probe rest start this long after trial end
                                   # (post-imagery beta rebound) and end by the next cue
MIN_IDENTITY_TRIALS = 8            # identity trials a decoder version needs for its external margin: its
                                   # CR2 t interval then has ≥ 6 df; the margin's SE (CV 0.3–0.4 at 7–12
                                   # trials) enters every threshold test, so a few trials widen the
                                   # intervals instead of silently moving the 10 % bound
PROBE_LEAK = 1.0                   # screen leak of the paired leak-transfer probe (the default condition's)
PROBE_REPS = 8                     # probe trials per diagonal screen direction (rot+90 / rot−90 alternating)
PROBE_WINDOWS = 5                  # probe windows per trial, every REST_HOP_S from the 1.5-s control onset
PROBE_BLOCK_INDEX = 20_000         # block index of the probe's noise stream: never a plan or pool block
SHADOWS: Tuple[str, ...] = ("central11", "all32")
"""The two sighted-label shadows (positive controls, never deployed): the deployed model on ReFIT labels,
and the same model on all 32 channels (DESIGN_PURE_EEG §5)."""
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
    dry_penalty        dry-cap scaling of the motor ERD depth (``BodySubject``; NOT the d′ loss, which is
                       larger: ``sim_body`` tabulates the measured loss); the CLI sweeps a list of values.
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
    x_time_course      None (x shares the feet/fists ERD time course: optimistic for x) or (peak_s,
                       decay_s, sustain), e.g. ``sim_body.X_ERD_LR`` (the ds004362 L/R decay).
    day_depth_sd       per-day log-normal SD of ERD depth (0: gain-only drift, the best case for
                       re-centring; ``sim_body``).
    slot_minutes       ((slot, minutes), ...) per session, overriding the ``hours`` scaling for those slots
                       of ``protocol.CHAMPION_SLOT_MIN`` (empty: the protocol's allocation).  This is the
                       power-analysis lever the skeptic corrections ask for: how many held-out-mapping
                       minutes (and how much S data for the shadow, how much L1 to earn x) a session
                       needs before the certification can separate deployed from shadow.
    identity_fraction  share of held-out-block trials run under the identity mapping (I trials, one per
                       period of round(1/f) trials): they give the external margin.  0.25 ≈ the design's
                       ~20 I per ~50 rotated trials, at the cost of a quarter of the held-out minutes.
    null_calibration   True: with the screen leak on, the same people also run at screen leak 0 (same
                       seeds, noise streams and prior: the same decoder versions), and each shadow
                       positive control must read the screen beyond its result there
                       (``certify``'s ``shadow_null``).  At screen leak 0 the run is its own null.  False:
                       uncalibrated controls, never CERTIFIED (and half the runtime).
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
    x_time_course: Optional[Tuple[float, float, float]] = None
    day_depth_sd: float = 0.0
    slot_minutes: Tuple[Tuple[str, float], ...] = ()
    identity_fraction: float = 0.25
    null_calibration: bool = True

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
        if self.day_depth_sd < 0 or (self.x_time_course is not None and len(self.x_time_course) != 3):
            raise ValueError(f"day_depth_sd must be >= 0 and x_time_course (peak, decay, sustain): {self}")
        for slot, minutes in self.slot_minutes:
            if slot not in CHAMPION_SLOT_MIN or not minutes > 0:
                raise ValueError(f"slot_minutes needs slots in {tuple(CHAMPION_SLOT_MIN)} with minutes > 0, "
                                 f"got {(slot, minutes)}")
        slots = [slot for slot, _ in self.slot_minutes]
        if len(set(slots)) != len(slots):
            raise ValueError(f"slot_minutes names a slot twice: {slots}")
        if not 0.0 < self.identity_fraction <= 0.5:
            raise ValueError(f"identity_fraction must be in (0, 0.5]: the external margin needs identity "
                             f"trials, and most held-out trials must stay rotated; got {self.identity_fraction}")


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


@dataclass
class BlockRun:
    """One block's recordings: its run directory, the task outcomes by endpoint ("blind", "sighted",
    "heldout", "identity"), the parallel identity run of a held-out block (or None), and the rest (L0)
    windows taken between fixation probes, with provenance whose session is the run's."""
    run_dir: str
    outcomes: Dict[str, List[TrialOutcome]]
    identity_dir: Optional[str]
    idle: LabelledSet


def idle_set(windows: Sequence[np.ndarray], provenance: Sequence[dict]) -> LabelledSet:
    """Rest windows as a ``LabelledSet``: they carry no direction, so the label rows are 0 (never read:
    only ``fit_idle`` takes L0 windows, and it uses the source alone)."""
    return LabelledSet(list(windows), np.zeros((len(windows), 2)), list(provenance))


class ChampionRunner:
    """Runs one person's champion blocks for one day as ``RunLogger`` runs on the simulated clock.

    The clock is ``t0 + samples / fs``: the single clock of every EEG timestamp, event and cursor row.
    Each run gets a fresh ``OnlinePreprocessor`` (the per-update chain of ``realtime.RealTimeLoop``),
    because each run is its own recording and the labeler filters it from its first sample.
    ``decoder`` is the deployed decoder (body velocity out; None → zero output); ``x_earned`` picks the
    L2b cue classes; ``trajectories`` is the person's replay pool by target geometry ("sighted": Fitts
    rings from S blocks; "diagonal": rotated C2c trials), shared across days, so a replay inside a
    held-out block shows a trajectory on the same diagonal layout as its neighbours.

    The control loop runs at the champion's rate (``champion_config``: 16 Hz) on whole samples: one
    update every ``hop`` = round(fs / control_hz) samples, and the task's ``control_hz`` is set to
    fs / hop exactly (16.13 Hz at 500 Hz), so the cursor clock and the EEG clock never drift apart.

    A held-out block writes its EEG to two runs: its own (H trials) and a parallel identity run whose
    events are the interleaved identity trials (I), logged as identity-mapping sighted trials
    (module docstring).  Events and cursor rows of a trial go to ``self._log``, the run it belongs to.
    """

    def __init__(self, subject: BodySubject, cfg: Config, sim: ChampionSimConfig, t0: float,
                 trajectories: Dict[str, List[Trajectory]]):
        self.subject, self.cfg, self.sim = subject, cfg, sim
        self.fs = cfg.hardware.fs
        self.hop = int(round(self.fs / cfg.task.control_hz))
        self.t0, self.n = t0, 0
        self.trajectories = trajectories
        self.decoder: Optional[BodyCoordinateDecoder] = None
        self.x_earned = False
        timing = ClosedLoopTiming()
        self.sighted_task = dataclasses.replace(cfg.task, assist_gain=sim.assist_gain,
                                                trial_timeout_s=timing.timeout_s, control_hz=self.fs / self.hop)
        self.blind_task = dataclasses.replace(self.sighted_task, assist_gain=0.0)
        self.pre = OnlinePreprocessor(cfg)
        self._logger: Optional[RunLogger] = None             # the block's own run
        self._side: Optional[RunLogger] = None               # the held-out block's identity run
        self._log: Optional[RunLogger] = None                # the current trial's run
        self._idle: List[Tuple[np.ndarray, dict]] = []

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
        ts = self.t + np.arange(n) / self.fs
        for logger in (self._logger, self._side):
            if logger is not None:
                logger.eeg(x, ts)
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
        self._log.cursor(self.t, pos, u, p, inside, dwell)

    # -- setup ------------------------------------------------------------------------------------
    def record_rest(self, day: int, index: int) -> List[np.ndarray]:
        """``rest_s`` of eyes-open rest through the online chain, 1-s windows every ``REST_HOP_S`` (no
        label, no display involved): the session's re-centring reference is their Riemannian mean, and
        they are rest (L0) windows for the idle gate."""
        self.subject.begin_block(day, index)                   # at rest, nothing drawn
        self.pre = OnlinePreprocessor(self.cfg)
        self.emit(PREROLL_S)                                   # filter settling, as in every run
        step = int(round(REST_HOP_S * self.fs))
        windows = []
        for _ in range(max(1, int(self.sim.rest_s / REST_HOP_S))):
            self.emit(samples=step)
            windows.append(self.pre.window())
        return windows

    # -- blocks -----------------------------------------------------------------------------------
    def run_block(self, block: Block, root: str, tag: str, day: int, index: int,
                  glyphs: Dict[str, str]) -> BlockRun:
        """One block as one RunLogger run (two for a held-out block: module docstring)."""
        self.subject.begin_block(day, index)
        self.subject.covert_leak = self.sim.covert_leak if block.kind in BLIND_KINDS else 0.0
        self.pre = OnlinePreprocessor(self.cfg)
        self._idle = []
        clock = lambda: self.t
        self._logger = self._log = RunLogger(root, self.cfg, tag=tag, clock=clock)
        if block.kind == "heldout_mapping":
            self._side = RunLogger(root, self.cfg, tag=tag + "_identity", clock=clock)
        start = self.t
        try:
            info = {"kind": block.kind, "mapping": block.mapping, "variant": block.variant,
                    "duration_s": block.duration_s, "seed": block.seed}
            self._logger.event("block_start", info, t=self.t)
            if self._side is not None:
                self._side.event("block_start", dict(info, mapping=TRAINING_MAPPING, identity_trials_of=block.kind),
                                 t=self.t)
            self.emit(PREROLL_S)
            if block.kind == "fixation_probe":
                outcomes = self._fixation(block, glyphs)
            elif block.kind == "blind_closed_loop":
                outcomes = {"blind": self._blind(block, glyphs)}
            else:
                outcomes = self._visible(block)
            for logger in (self._logger, self._side):
                if logger is not None:
                    logger.event("block_end", {"kind": block.kind}, t=self.t)
        finally:
            for logger in (self._logger, self._side):
                if logger is not None:
                    logger.close()
        run_dir = self._logger.dir
        identity_dir = None if self._side is None else self._side.dir
        self._logger = self._side = self._log = None
        if block.kind != "fixation_probe":
            self.subject.record_practice(self.t - start)      # closed-loop practice (user learning, if on)
        session = os.path.basename(os.path.normpath(run_dir))
        idle = idle_set([w for w, _ in self._idle], [dict(p, session=session) for _, p in self._idle])
        return BlockRun(run_dir, outcomes, identity_dir, idle)

    def _fixation(self, block: Block, glyphs: Dict[str, str]) -> Dict[str, List[TrialOutcome]]:
        """L1: glyph cue → cue_offset → trial_start at the 1.0-s lockout → imagery to 4.0 s → ITI.
        The person imagines from cue onset; nothing spatial is drawn; fixation holds.  The ITI rest
        gives L0 windows (``_iti_rest``)."""
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
            self._iti_rest(trial, tm.iti_s)
        return {}

    def _iti_rest(self, trial: int, iti_s: float) -> None:
        """The inter-probe rest: ``iti_s`` of EEG, with a rest window (source L0) every ``REST_HOP_S``
        from ``ITI_REST_SETTLE_S`` + one window after the trial end until the next cue.  Nothing is cued
        or drawn then, so the windows carry no label a display could identify."""
        total = int(round(iti_s * self.fs))
        first = int(round((ITI_REST_SETTLE_S + self.cfg.decoder.window_s) * self.fs))
        hop = int(round(REST_HOP_S * self.fs))
        done = 0
        if first <= total:
            self.emit(samples=first)
            done = first
            while True:
                self._idle.append((self.pre.window(), {"source": IDLE_SOURCE, "block": "fixation_probe",
                                                       "trial": trial, "t": self.t}))
                if done + hop > total:
                    break
                self.emit(samples=hop)
                done += hop
        self.emit(samples=total - done)

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
        """S and H blocks: targets drawn; closed-loop (S, C2c), open-loop (C2b) and replay trials.  In H
        blocks the targets are the four diagonals, the goal quadrant balanced per run of four (separately
        for rotated and identity trials), and one trial per period of round(1/``identity_fraction``) is
        an identity trial logged to the parallel identity run.

        The replay decision comes before the goal is drawn: a replay shows a past trajectory with its
        own targets, so it must not consume a quadrant of the rotated trials' balanced runs of four
        (drawing first left |corr(sign dx, sign dy)| up to 0.5 over a block's non-replay trials, the
        refusal threshold).  A replay draws from the pool of its block's geometry; with an empty pool
        the scheduled replay runs as a normal trial."""
        heldout = block.kind == "heldout_mapping"
        geometry = "diagonal" if heldout else "sighted"
        M = Mapping(block.mapping)
        open_loop = heldout and block.variant == "C2b"
        rng = np.random.default_rng(block.seed)
        choice_rng = np.random.default_rng([block.seed, _REPLAY_CHOICE_STREAM])
        geometry_rng = np.random.default_rng([block.seed, _HELDOUT_GEOMETRY_STREAM])
        replays = ReplayScheduler(block) if block.replay_fraction else None
        runner = TrialRunner(self.sighted_task, rng)
        key = "heldout" if heldout else "sighted"
        identity_block = Block("sighted", block.duration_s, seed=block.seed, replay_fraction=0.0) if heldout else None
        period = max(2, int(round(1.0 / self.sim.identity_fraction)))
        quadrants: Dict[bool, List[int]] = {False: [], True: []}
        identity_slots: List[bool] = []
        outcomes: Dict[str, List[TrialOutcome]] = {key: [], "identity": []}
        trials = {False: 0, True: 0}                          # trial ids per run (rotated / identity)
        pool = self.trajectories.setdefault(geometry, [])
        start = self.t
        while self.t - start < block.duration_s:
            distance, width = FITTS_CONDITIONS[sum(trials.values()) % len(FITTS_CONDITIONS)]
            identity = False
            if heldout:
                if not identity_slots:
                    identity_slots = [False] * period
                    identity_slots[int(geometry_rng.integers(period))] = True
                identity = identity_slots.pop(0)
            replay = replays is not None and not identity and replays.next() and bool(pool)
            trial = trials[identity]
            trials[identity] += 1
            self._log = self._side if identity else self._logger
            if replay:
                self._replay_trial(block, trial, M, pool[int(choice_rng.integers(len(pool)))], choice_rng)
                continue
            if heldout:
                if not quadrants[identity]:
                    quadrants[identity] = list(quadrant_schedule(4, geometry_rng))
                targets, goal_idx = diagonal_targets(distance), int(quadrants[identity].pop(0))
            else:
                targets = fitts_targets(self.sighted_task.n_targets, distance, width, rng)
                goal_idx = int(rng.integers(len(targets)))
            M_trial, host = (Mapping(TRAINING_MAPPING), identity_block) if identity else (M, block)
            if open_loop:
                self._open_trial(host, trial, M_trial, targets[goal_idx])
            else:
                out = self._closed_trial(host, trial, M_trial, targets, goal_idx, width, runner,
                                         keep=None if identity else pool)
                outcomes["identity" if identity else key].append(out)
        self._log = self._logger
        return outcomes

    def _closed_trial(self, block: Block, trial: int, M: Mapping, targets: np.ndarray, goal_idx: int,
                      width: float, runner: TrialRunner, keep: Optional[List[Trajectory]]) -> TrialOutcome:
        """Target onset (the cue) → 1.0-s preview → control: screen command = M · decoded body velocity,
        blended with GoalPosterior assistance by ``TrialRunner`` → outcome + ITI.  ``block`` is the block
        the trial is logged as; a successful trajectory joins the replay pool ``keep`` (None: kept
        nowhere, e.g. identity trials)."""
        tm = ClosedLoopTiming()
        goal = targets[goal_idx]
        self._attend(M, goal, np.zeros(2), onset=True)
        self.emit(tm.label_start_s)
        self._log.event(EVENT_TRIAL_START, champion_trial_start_payload(block, trial, goal=goal), t=self.t)
        positions: List[np.ndarray] = []

        def command(state, _t):
            self._attend(M, goal, state.pos)
            self.emit(samples=self.hop)
            return M.body_to_screen(self.decode())

        def on_step(state, u, p, inside, dwell):
            self._log_cursor(state.pos, u, float(np.max(p)), inside, dwell)
            positions.append(state.pos.copy())

        out = runner.run(targets, width, goal_idx, command, on_step=on_step)
        self._log.event(EVENT_TRIAL_END, {"trial": trial, "outcome": "hit" if out.success else
                                          ("miss" if out.selected_idx is not None else "timeout"),
                                          "selected": out.selected_idx, "movement_time_s": out.movement_time_s},
                        t=self.t)
        self._rest()
        self.emit(tm.outcome_s + tm.iti_s)
        if keep is not None and out.success and len(positions) >= MIN_REPLAY_STEPS:
            keep.append(Trajectory(targets.copy(), goal_idx, np.array(positions)))
        return out

    def _replay_trial(self, block: Block, trial: int, M: Mapping, past: Trajectory,
                      rng: np.random.Generator) -> None:
        """Pixel replay: the past trajectory toward its goal A is shown step by step while the person
        holds another target of the same set, goal B (logged as the trial's goal; A as replay_goal).
        After the trial the person keys the body part they were imagining (the post-trial report of
        the held intent, ``EVENT_BODY_REPORT``): certification keeps only trials it confirms."""
        tm = ClosedLoopTiming()
        others = [i for i in range(len(past.targets)) if i != past.goal_idx]
        goal_a, goal_b = past.targets[past.goal_idx], past.targets[int(rng.choice(others))]
        pos = np.zeros(2)
        self._attend_replay(M, goal_a, goal_b, pos, onset=True)
        self.emit(tm.label_start_s)
        self._log.event(EVENT_TRIAL_START, champion_trial_start_payload(block, trial, goal=goal_b,
                                                                        replay_goal=goal_a), t=self.t)
        for nxt in past.positions:
            self._attend_replay(M, goal_a, goal_b, pos)
            self.emit(samples=self.hop)
            pos = nxt
            self._log_cursor(pos)
        self._log.event(EVENT_TRIAL_END, {"trial": trial, "outcome": "replay"}, t=self.t)
        report = self._report_class(M.screen_to_body(unit_directions(goal_b[None], np.zeros((1, 2)))[0]))
        self._rest()
        self.emit(REPORT_DELAY_S)
        self._log.event(EVENT_BODY_REPORT, {"trial": trial, "class": report}, t=self.t)
        self.emit(tm.outcome_s + tm.iti_s - REPORT_DELAY_S)

    def _report_class(self, body: np.ndarray) -> str:
        """The keyed report: the class nearest the body vector the person IMAGINED (only its y part
        before x is earned).  The simulated person always remaps correctly, so the report always
        confirms the intent; the filter in ``simulate_user`` is the real protocol's."""
        imagined = self._imagined(np.asarray(body, float))
        return max(BODY_VECTORS, key=lambda c: float(np.dot(BODY_VECTORS[c], imagined)))

    def _open_trial(self, block: Block, trial: int, M: Mapping, goal: np.ndarray) -> None:
        """C2b: target and frozen cursor shown → trial_start after the preview → imagery → trial_end →
        keyed body-part report (``_report_class``) → ITI."""
        tm = OpenLoopTiming()
        centre = np.zeros(2)
        self._attend(M, goal, centre, onset=True)
        self.emit(tm.preview_s)
        self._log.event(EVENT_TRIAL_START, champion_trial_start_payload(block, trial, goal=goal), t=self.t)
        for _ in range(int(round(tm.imagery_s * self.fs / self.hop))):
            self.emit(samples=self.hop)
            self._log_cursor(centre)
        self._log.event(EVENT_TRIAL_END, {"trial": trial, "outcome": "open_loop"}, t=self.t)
        report = self._report_class(M.screen_to_body(unit_directions(goal[None], centre[None])[0]))
        self._rest()
        self.emit(REPORT_DELAY_S)
        self._log.event(EVENT_BODY_REPORT, {"trial": trial, "class": report}, t=self.t)
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


def l1_scores(decoder: Optional[BodyCoordinateDecoder], pool: LabelledSet, refs: Dict[str, np.ndarray],
              n_permutations: int = L1_FLOOR_PERMUTATIONS, seed: int = DEFAULT.eval.seed) -> dict:
    """AUC, its chance floor and r per axis on the fixed L1 pool (y: FEET vs HANDS probes; x: RIGHT vs
    LEFT probes), of the axis models without the idle gate (``idle_scores`` reports the gate).

    The floor is the p95 of the AUC under a TRIAL-level label permutation: the windows of one probe
    share its cue and overlap in time, so shuffling windows would give a null far too narrow; the pool's
    cues are scheduled independently of the EEG and never trained on, so permuting whole trials is an
    exact null.  "Above floor" means AUC > floor."""
    nan = float("nan")
    if decoder is None or len(pool) == 0:
        return {"auc": {a: nan for a in AXES}, "floor": {a: nan for a in AXES}, "r": {a: nan for a in AXES},
                "n": {a: 0 for a in AXES}, "n_trials": {a: 0 for a in AXES}}
    # The axis models' own output: an armed idle gate's exact zeros would tie and measure the gate.
    out = recentred_outputs(decoder.ungated(), pool, refs, force_x=True)
    y = body_labels(pool)
    res: dict = {"auc": {}, "floor": {}, "r": {}, "n": {}, "n_trials": {}}
    for k, axis in enumerate(AXES):
        m = y[:, k] != 0
        both = m.any() and len(np.unique(y[m, k])) == 2
        res["auc"][axis] = float(roc_auc_score(y[m, k] > 0, out[m, k])) if both else nan
        res["r"][axis] = pearson_r(out[m, k], y[m, k].astype(float)) if both else nan
        res["n"][axis] = int(m.sum())
        cl = trial_clusters([p for p, keep in zip(pool.provenance, m) if keep])
        res["n_trials"][axis] = int(cl.max()) + 1 if cl.size else 0
        res["floor"][axis] = nan
        if both:
            trial_label = np.zeros(res["n_trials"][axis])
            trial_label[cl] = y[m, k]
            o = out[m, k]
            res["floor"][axis] = permutation_floor(lambda lab: float(roc_auc_score(lab[cl] > 0, o)), trial_label,
                                                   n_permutations, np.random.default_rng([seed, k])).floor_p95
    return res


def identity_margin(decoder: BodyCoordinateDecoder, pool: LabelledSet, refs: Dict[str, np.ndarray]) -> Dict[str, float]:
    """The decoder's L1 margin per axis (secondary, reported beside the external one): the OLS slope of
    its deployed output on its own cued label on the fixed L1 pool (fixation probes no decoder trains
    on).  The certification scales by ``identity_trial_margin`` instead, which is measured in the same
    held-out blocks (moving eyes, a visible cursor) as the windows it scales; x uses the deployed x
    output, so an axis that is off has margin 0 there."""
    out = recentred_outputs(decoder, pool, refs) if len(pool) else np.zeros((0, 2))
    y = body_labels(pool) if len(pool) else np.zeros((0, 2), int)
    margin = {}
    for k, axis in enumerate(AXES):
        m = (y[:, k] != 0) & np.isfinite(out[:, k])
        lab = y[m, k].astype(float)
        margin[axis] = (float(np.cov(out[m, k], lab)[0, 1] / np.var(lab, ddof=1))
                        if m.sum() > 2 and np.var(lab) > 0 else float("nan"))
    return margin


def identity_trial_margin_stats(outputs: np.ndarray, data: LabelledSet, min_trials: int = MIN_IDENTITY_TRIALS
                                ) -> Dict[str, Tuple[float, float, Optional[float]]]:
    """The EXTERNAL margin per axis with its uncertainty: (slope, CR2 SE, df = trials − 2) of a frozen
    decoder's output on the body intent ``unit(goal − cursor at the window start)`` in the identity
    trials interleaved with its held-out trials — identity-mapping windows of the same blocks, never the
    certified ones.  Under the identity body and screen coincide, so the slope also holds any screen
    component: conservative for the 70 % body criterion and < 1 point on the 10 % screen bound
    (DESIGN_PURE_EEG §5).  The SE is clustered on trials like the certification's own (at 7–12 trials
    its CV is 0.3–0.4: ``certify`` propagates it).  NaNs with fewer than ``min_trials`` trials or a
    constant regressor."""
    nan = (float("nan"), float("nan"), None)
    if not len(data):
        return {a: nan for a in AXES}
    cl = trial_clusters(data.provenance)
    goals = np.array([p["display_goal"] for p in data.provenance], float).reshape(-1, 2)
    start = np.array([p["cursor_pos_start"] for p in data.provenance], float).reshape(-1, 2)
    u = unit_directions(goals, start)
    margin = {}
    for k, axis in enumerate(AXES):
        m = np.isfinite(outputs[:, k])
        n_trials = len(np.unique(cl[m]))
        if n_trials < min_trials or m.sum() <= 2 or np.var(u[m, k]) <= 1e-12:
            margin[axis] = nan
            continue
        _, inv = np.unique(cl[m], return_inverse=True)
        beta, cov = cluster_robust_ols(np.column_stack([u[m, k], np.ones(int(m.sum()))]), outputs[m, k], inv)
        margin[axis] = (float(beta[0]), float(np.sqrt(max(cov[0, 0], 0.0))), float(n_trials - 2))
    return margin


def identity_trial_margin(outputs: np.ndarray, data: LabelledSet,
                          min_trials: int = MIN_IDENTITY_TRIALS) -> Dict[str, float]:
    """The EXTERNAL margin per axis (the slope of ``identity_trial_margin_stats``); NaN with fewer than
    ``min_trials`` trials or a constant regressor."""
    return {a: v[0] for a, v in identity_trial_margin_stats(outputs, data, min_trials).items()}


def leak_transfer(cfg: Config, sim: ChampionSimConfig, pop: BodyPopulation, user: int, day: int,
                  decoder: BodyCoordinateDecoder, reference: np.ndarray, margin: Dict[str, float]) -> dict:
    """How much a display leak reaches THIS deployed decoder: a paired probe (review finding).

    The person (a fresh instance of ``user`` on ``day``, at rest re-centred on ``reference``) looks at a
    diagonal screen target and imagines the body vector a rot+90 / rot−90 trial asks for, in two passes
    on IDENTICAL noise streams — screen leak 0 and ``PROBE_LEAK`` — so the paired output difference is the
    leak's own effect on this decoder's axis models.  Its slope on the screen direction, per axis, is the
    leak-transfer gain (relative: ÷ the external margin).  At covert leak 0 the deployed decoder never
    saw a display-correlated label, and a non-zero gain is a display signal it reads anyway (the L1-only
    decoders of the review read −13 to +51 % of their motor slope); the certification's β_screen is the
    held-out estimate of the same quantity, this is its noise-free counterpart."""
    imagined = lambda body: body if decoder.x_enabled else np.array([0.0, body[1]])
    out: Dict[float, np.ndarray] = {}
    screen = []
    for leak in (0.0, PROBE_LEAK):
        subject = pop.subject(user)
        subject.set_day(day)
        subject.screen_leak, subject.covert_leak = leak, 0.0
        runner = ChampionRunner(subject, cfg, sim, CLOCK_ORIGIN_S + (day - 1) * DAY_OFFSET_S, {})
        subject.begin_block(day, PROBE_BLOCK_INDEX)
        runner.pre = OnlinePreprocessor(cfg)
        runner.emit(PREROLL_S)
        dec = decoder.ungated().set_reference(reference=reference)
        rows, screen = [], []
        for rep in range(PROBE_REPS):
            M = Mapping("rot+90" if rep % 2 == 0 else "rot-90")
            for u in DIAGONAL_DIRECTIONS:
                subject.set_screen_direction(u)
                subject.cue(*imagined(M.screen_to_body(u)))
                runner.emit(ClosedLoopTiming().label_start_s)
                for _ in range(PROBE_WINDOWS):
                    runner.emit(REST_HOP_S)
                    rows.append(dec.predict(runner.pre.window()))
                    screen.append(u)
                runner._rest()
                runner.emit(ClosedLoopTiming().outcome_s + ClosedLoopTiming().iti_s)
        out[leak] = np.array(rows)
    d, u = out[PROBE_LEAK] - out[0.0], np.array(screen)
    gain = {a: float(np.nansum(d[:, k] * u[:, k]) / np.sum(u[:, k] ** 2)) for k, a in enumerate(AXES)}
    rel = {a: gain[a] / margin[a] if np.isfinite(margin.get(a, np.nan)) and margin[a] > 0 else float("nan")
           for a in AXES}
    return {"probe_leak": PROBE_LEAK, "gain": gain, "rel": rel, "n_windows": int(len(d))}


def recentred_idle(decoder: BodyCoordinateDecoder, data: LabelledSet, refs: Dict[str, np.ndarray]) -> np.ndarray:
    """(n,) p(idle) per window, each re-centred on its own session's rest reference (0 without an
    armed gate)."""
    out = np.zeros(len(data))
    groups: Dict[str, List[int]] = {}
    for i, p in enumerate(data.provenance):
        groups.setdefault(p["session"], []).append(i)
    for session, idx in groups.items():
        d = copy.copy(decoder)
        d.set_reference(reference=refs[session])
        out[idx] = d.idle_probability([data.windows[i] for i in idx])
    return out


def idle_scores(decoder: Optional[BodyCoordinateDecoder], pool: LabelledSet, pool_idle: LabelledSet,
                refs: Dict[str, np.ndarray]) -> dict:
    """The idle gate on the fixed pool: armed or not, its out-of-fold AUC, and the shares of the pool's
    rest (L0) windows it zeroes and of its fixation-probe (intent) windows it lets through."""
    nan = float("nan")
    rep = (decoder.idle_report if decoder is not None else None) or {}
    res = {"armed": bool(rep.get("armed", False)), "cv_auc": rep.get("cv_auc", nan), "rest_gated": nan,
           "intent_passed": nan}
    if decoder is None or not len(pool) or not len(pool_idle):
        return res
    thr = decoder.idle_threshold
    res["rest_gated"] = float(np.mean(recentred_idle(decoder, pool_idle, refs) > thr))
    res["intent_passed"] = float(np.mean(recentred_idle(decoder, pool, refs) <= thr))
    return res


def _window_direction(goals: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Unit direction to ``goals`` matched to the whole window: the normalised sum of the unit
    directions from the cursor at the window's start and at its end (0 where they cancel)."""
    d = unit_directions(goals, start) + unit_directions(goals, end)
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    return np.where(norm > 1e-9, d / np.maximum(norm, 1e-9), 0.0)


def regressors(test: str, data: LabelledSet) -> Tuple[np.ndarray, np.ndarray]:
    """(body, screen) regressors in the decoder's output frame, from the logged task geometry only.

    heldout: body = M⁻¹·unit(goal − displayed cursor), screen = M_train⁻¹·unit(same) (= the unit
             direction under the identity): ``mapping.certification_regressors`` per window, since a
             day mixes mappings.  The cursor is the one at the WINDOW START: in closed loop (C2c) the
             cursor at the window end has already been moved by this window's own decoded output, so
             a persistent output error would feed back into the body regressor and bias the deployed
             decoder's β_body down (the shadow, scored offline on the same trajectory, has no such
             loop).  In open loop (C2b) the cursor is frozen and both are the same.
    replay:  body = M⁻¹·dir(held goal B), screen = M_train⁻¹·dir(replayed goal A), each the direction
             matched to the window (``_window_direction``): the 1-s window integrates intent along the
             replayed path, so a window-end direction would alias lag into the regressors."""
    goals = np.array([p["display_goal"] for p in data.provenance], float).reshape(-1, 2)
    start = np.array([p["cursor_pos_start"] for p in data.provenance], float).reshape(-1, 2)
    maps = [Mapping(p["mapping"]) for p in data.provenance]
    if test == "heldout":
        u = unit_directions(goals, start)
        screen = Mapping(TRAINING_MAPPING).screen_to_body(u)
    else:
        end = np.array([p["cursor_pos"] for p in data.provenance], float).reshape(-1, 2)
        u = _window_direction(goals, start, end)
        replayed = np.array([p["replay_goal"] for p in data.provenance], float).reshape(-1, 2)
        screen = Mapping(TRAINING_MAPPING).screen_to_body(_window_direction(replayed, start, end))
    body = np.stack([m.screen_to_body(v) for m, v in zip(maps, u)]).reshape(-1, 2)
    return body, screen


MIN_MARGIN = 0.02
"""A decoder whose identity-trial slope (external margin) is below this on an axis has no margin to
scale that axis by (a 10 % screen bound of < 0.002 is not a measurement); that axis is not certified
for that decoder version, and if it is y the version is skipped."""


def _beta_summary(entry: dict) -> dict:
    keys = ("beta_body", "ci_body", "beta_screen", "ci_screen", "beta_cross", "ci_cross", "rel_body",
            "rel_ci_body", "rel_screen", "rel_ci_screen", "scale", "regressor_corr", "max_block_corr",
            "confounded", "collinearity", "identifiable", "n_units", "tau2_screen", "df")
    return {k: entry[k] for k in keys if k in entry}


@dataclass
class CertificationLog:
    """Per-window inputs of one certification test, accumulated so the test can be run per day, over a
    person's days, or over people: deployed and shadow outputs (the central-11 shadow in ``shadow``, any
    other shadow by name in ``extra_shadows``), both regressors, provenance, whether x was earned when
    the window was decoded, the test unit (the deployed decoder version that decoded it), its person, and
    each unit's external margin.  Session ids are unique per person and run, so trial clusters never
    merge across them."""
    deployed: List[np.ndarray] = field(default_factory=list)
    shadow: List[np.ndarray] = field(default_factory=list)
    body: List[np.ndarray] = field(default_factory=list)
    screen: List[np.ndarray] = field(default_factory=list)
    provenance: List[dict] = field(default_factory=list)
    x_enabled: List[np.ndarray] = field(default_factory=list)
    units: List[str] = field(default_factory=list)
    persons: List[str] = field(default_factory=list)
    margins: Dict[str, Dict[str, float]] = field(default_factory=dict)
    extra_shadows: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    margin_se: Dict[str, Dict[str, float]] = field(default_factory=dict)
    margin_df: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)

    def add(self, deployed: np.ndarray, shadow: np.ndarray, body: np.ndarray, screen: np.ndarray,
            provenance: Sequence[dict], x_enabled: np.ndarray, unit: str, person: str,
            margin: Dict[str, float], extra_shadows: Optional[Dict[str, np.ndarray]] = None,
            margin_se: Optional[Dict[str, float]] = None, margin_df: Optional[Dict[str, Optional[float]]] = None) -> None:
        extra = dict(extra_shadows or {})
        if self.deployed and set(extra) != set(self.extra_shadows):
            raise ValueError(f"shadows {sorted(extra)} differ from this log's {sorted(self.extra_shadows)}")
        self.deployed.append(deployed)
        self.shadow.append(shadow)
        self.body.append(body)
        self.screen.append(screen)
        self.provenance.extend(provenance)
        self.x_enabled.append(np.asarray(x_enabled, bool))
        self.units.extend([unit] * len(provenance))
        self.persons.extend([person] * len(provenance))
        self.margins[unit] = dict(margin)
        self.margin_se[unit] = dict(margin_se or {a: 0.0 for a in AXES})
        self.margin_df[unit] = dict(margin_df or {a: None for a in AXES})
        for name, out in extra.items():
            self.extra_shadows.setdefault(name, []).append(out)

    @staticmethod
    def merge(logs: Sequence["CertificationLog"]) -> "CertificationLog":
        out = CertificationLog()
        filled = [log for log in logs if log.deployed]
        names = {frozenset(log.extra_shadows) for log in filled}
        if len(names) > 1:
            raise ValueError(f"logs carry different shadows: {[sorted(n) for n in names]}")
        for log in filled:
            out.deployed += log.deployed
            out.shadow += log.shadow
            out.body += log.body
            out.screen += log.screen
            out.provenance += log.provenance
            out.x_enabled += log.x_enabled
            out.units += log.units
            out.persons += log.persons
            out.margins.update(log.margins)
            out.margin_se.update(log.margin_se)
            out.margin_df.update(log.margin_df)
            for name, outs in log.extra_shadows.items():
                out.extra_shadows.setdefault(name, []).extend(outs)
        return out

    def _subset(self, keep: np.ndarray) -> "CertificationLog":
        idx = np.flatnonzero(keep)
        cat = lambda xs: np.concatenate(xs)[idx]
        out = CertificationLog([cat(self.deployed)], [cat(self.shadow)], [cat(self.body)], [cat(self.screen)],
                               [self.provenance[i] for i in idx], [cat(self.x_enabled)],
                               [self.units[i] for i in idx], [self.persons[i] for i in idx])
        out.margins = {u: self.margins[u] for u in set(out.units)}
        out.margin_se = {u: self.margin_se[u] for u in set(out.units)}
        out.margin_df = {u: self.margin_df[u] for u in set(out.units)}
        out.extra_shadows = {name: [cat(outs)] for name, outs in self.extra_shadows.items()}
        return out

    def shadow_names(self) -> Tuple[str, ...]:
        return (SHADOWS[0],) + tuple(sorted(self.extra_shadows))

    def _unit_margin(self, idx: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Optional[float]]]:
        """External margin of the windows ``idx`` with its SE and df: the window-weighted mean over the
        decoder versions they hold that have one (independent identity-trial sets, so the variance of
        the mean is Σ w²·se², with Welch–Satterthwaite df)."""
        margin, se, df = {}, {}, {}
        versions = [self.units[i] for i in idx]
        for a in AXES:
            have = [v for v in set(versions) if np.isfinite(self.margins[v][a])]
            if not have:
                margin[a], se[a], df[a] = float("nan"), float("nan"), None
                continue
            w = np.array([versions.count(v) for v in have], float)
            w /= w.sum()
            margin[a] = float(np.dot(w, [self.margins[v][a] for v in have]))
            var, df[a] = welch([(wi ** 2 * self.margin_se[v][a] ** 2, self.margin_df[v][a]) for wi, v in zip(w, have)])
            se[a] = float(np.sqrt(var))
        return margin, se, df

    def _unit_results(self, test: str, n_boot: int, unit_of: Sequence[str]) -> Tuple[dict, Dict[str, dict], Dict[str, str]]:
        """Deployed and per-shadow results per test unit, each fitted on its own windows with its own
        external margin and its SE (``_unit_margin``), and the units skipped with the reason."""
        fn, min_cl = (certify_mapping, MIN_CLUSTERS) if test == "heldout" else (replay_test, MIN_REPLAY_CLUSTERS)
        dep_all = np.concatenate(self.deployed)
        shadow_all = {SHADOWS[0]: np.concatenate(self.shadow),
                      **{n: np.concatenate(v) for n, v in self.extra_shadows.items()}}
        body, screen, x_on = np.concatenate(self.body), np.concatenate(self.screen), np.concatenate(self.x_enabled)
        dep: dict = {}
        sha: Dict[str, dict] = {n: {} for n in shadow_all}
        skipped: Dict[str, str] = {}
        for unit in sorted(set(unit_of)):
            idx = np.flatnonzero(np.array(unit_of) == unit)
            prov = [self.provenance[i] for i in idx]
            n_trials = int(trial_clusters(prov).max()) + 1
            if n_trials < min_cl:
                skipped[unit] = f"{n_trials} trials (< {min_cl} for cluster-robust inference)"
                continue
            margin, margin_se, margin_df = self._unit_margin(idx)
            axes = tuple(a for a in AXES if (a == "y" or x_on[idx].all()) and margin[a] >= MIN_MARGIN)
            if "y" not in axes:
                skipped[unit] = (f"no external y margin from identity trials ({margin['y']:.3f}; needs ≥ "
                                 f"{MIN_MARGIN} from ≥ {MIN_IDENTITY_TRIALS} trials)")
                continue
            args = (body[idx], screen[idx], prov)
            kw = dict(axes=axes, margin={a: margin[a] for a in axes}, margin_se={a: margin_se[a] for a in axes},
                      margin_df={a: margin_df[a] for a in axes})
            dep[unit] = fn(dep_all[idx], *args, **kw)
            for name, out in shadow_all.items():
                sha[name][unit] = fn(out[idx], *args, **kw)
        return dep, sha, skipped

    def _combined(self, results: dict, unit_of: Sequence[str], n_boot: int) -> dict:
        """One test's per-unit results combined as the claim needs: one unit → itself; several units of
        one person → fixed effect; several people → person-then-trial bootstrap over each person's
        combination."""
        person_of = {u: self.persons[list(unit_of).index(u)] for u in results}
        people = sorted(set(person_of.values()))
        per_person = {}
        for person in people:
            units = [u for u in results if person_of[u] == person]
            per_person[person] = (results[units[0]] if len(units) == 1 else
                                  combine_units({u: results[u] for u in units}, combination="fixed"))
        if len(people) == 1:
            return per_person[people[0]]
        return combine_units(per_person, combination="bootstrap", n_boot=max(n_boot, MIN_BOOT), seed=SEED)

    def certify(self, test: str, n_boot: int, null: Optional["CertificationLog"] = None) -> dict:
        """β table and verdict of deployed vs both shadows on these windows.

        Test units: for held-out mappings the deployed decoder version (one per session: its own
        slopes, intercepts and external margin, so differing gains or mapping mixes cannot bias another
        unit's β_screen); for replay, with ~10 trials a day, the person.  One unit → its own verdict;
        several units of one person → inverse-variance (fixed-effect) combination; several people →
        a person-then-trial bootstrap over each person's units (``certify.combine_units``).  Any unit's
        own SCREEN_READING vetoes the verdict, every tested axis must pass (x is tested for units whose
        every window had x earned: x pools over earners only, y over all), and the test is sensitive if
        EITHER shadow reads the screen beyond its NULL: ``null`` is the same log from the screen_leak 0
        twin of these people (same seeds and noise streams, so the same units), whose shadow results,
        combined the same way, are ``certify``'s ``shadow_null``.  Without ``null`` the positive control
        is uncalibrated and nothing is CERTIFIED."""
        n_trials = int(trial_clusters(self.provenance).max()) + 1 if self.provenance else 0
        base = {"n_trials": n_trials, "n_windows": len(self.provenance)}
        if not self.provenance:
            return {"skipped": "no windows", **base}
        unit_of = self.units if test == "heldout" else self.persons
        dep, sha, skipped = self._unit_results(test, n_boot, unit_of)
        if not dep:
            return {"skipped": "; ".join(f"{u}: {r}" for u, r in skipped.items()), **base}
        d = self._combined(dep, unit_of, n_boot)
        shadows = {name: self._combined(r, unit_of, n_boot) for name, r in sha.items()}
        shadow_null = None
        if null is not None:
            null_sha = sha if null is self else None
            if null_sha is None and null.provenance:
                null_unit_of = null.units if test == "heldout" else null.persons
                null_sha = null._unit_results(test, n_boot, null_unit_of)[1]
                shadow_null = {n: (null._combined(r, null_unit_of, n_boot) if r else {"test": d["test"]})
                               for n, r in null_sha.items()}
            elif null_sha is not None:
                shadow_null = shadows
            else:
                shadow_null = {n: {"test": d["test"]} for n in shadows}
        v = certify(d, shadows, shadow_null=shadow_null)
        # The axis whose entry is shown: the one that gave the verdict (y on ties); after a unit veto,
        # the worst axis of the combined fit.
        worst = min(v["per_axis"].values(), key=lambda pa: VERDICTS.index(pa["verdict"]))["verdict"]
        axis = next(a for a in ("y", "x") if a in v["per_axis"] and v["per_axis"][a]["verdict"] == worst)
        people = sorted({self.persons[list(unit_of).index(u)] for u in dep})
        return {"verdict": v["verdict"], "reason": v["reason"], "axes": list(d["axes"]), "deciding_axis": axis,
                "per_axis": {a: pa["verdict"] for a, pa in v["per_axis"].items()},
                "flags": {a: pa["flags"] for a, pa in v["per_axis"].items()},
                "unit_verdicts": v.get("unit_verdicts", {}), "vetoed_by": v.get("vetoed_by", []),
                "n_units": len(dep), "n_people": len(people),
                "skipped_units": skipped, **base,
                "deployed": {a: _beta_summary(d[a]) for a in d["axes"]},
                "shadow": {a: _beta_summary(shadows[SHADOWS[0]][a]) for a in d["axes"]},
                "shadows": {n: {a: _beta_summary(s[a]) for a in d["axes"]} for n, s in shadows.items()},
                "shadows_null": None if shadow_null is None else
                {n: {a: _beta_summary(r[a]) for a in d["axes"] if a in r} for n, r in shadow_null.items()},
                "holm": {name: {"p": t["p"], "p_holm": t["p_holm"], "rejected": t["rejected"]}
                         for name, t in v["holm"]["tests"].items()}}

    def certify_by_variant(self, n_boot: int, null: Optional["CertificationLog"] = None) -> Dict[str, dict]:
        """Held-out certification separately for open loop (C2b) and closed loop (C2c): C2c regressors
        come from a cursor the deployed decoder itself moved, so its β_body is not comparable with the
        shadow's or with C2b's even at the window start (``regressors``).  ``null``: as ``certify``."""
        variants = np.array([p.get("variant") for p in self.provenance])
        out = {}
        for v in ("C2b", "C2c"):
            if not (variants == v).any():
                continue
            sub_null = None
            if null is self:
                sub_null = "self"
            elif null is not None:
                nv = np.array([p.get("variant") for p in null.provenance])
                sub_null = null._subset(nv == v) if (nv == v).any() else CertificationLog()
            sub = self._subset(variants == v)
            out[v] = sub.certify("heldout", n_boot, null=sub if sub_null == "self" else sub_null)
        return out


# ----------------------------------------------------------------------------------------------
# Population prior, fixed L1 pool, one person
# ----------------------------------------------------------------------------------------------
def make_population(sim: ChampionSimConfig, cfg: Config) -> BodyPopulation:
    """Subjects 0 … users−1 are the deep participants; users … users+pop−1 the prior's members.
    ``covert_leak`` is set per block by ``ChampionRunner`` (blind blocks only), so it is 0 here."""
    return BodyPopulation(sim.users + sim.pop, sim.seed, cfg.hardware, x_responder_prob=sim.x_responder_prob,
                          y_depth_range=tuple(sim.y_depth_range), x_depth_range=tuple(sim.x_depth_range),
                          dry_penalty=sim.dry_penalty, screen_leak=sim.screen_leak, covert_leak=0.0,
                          user_learning_rate=sim.user_learning_rate, x_time_course=sim.x_time_course,
                          day_depth_sd=sim.day_depth_sd)


def record_fixation(cfg: Config, sim: ChampionSimConfig, subject: BodySubject, day: int, block: Block,
                    index: int, tag: str, root: str, t0: float
                    ) -> Tuple[LabelledSet, Dict[str, np.ndarray], LabelledSet]:
    """A rest recording (block ``index``) plus one fixation-probe block (``index + 1``) for ``subject``
    on ``day``, labelled through the logger → store → labeler path (the run directory is deleted
    afterwards).  Returns the probe windows, {session: reference} and the rest (L0) windows — the rest
    recording and the inter-probe rest, keyed to the probe run's session, whose reference that rest is.
    Used for prior members and for the fixed L1 test pool."""
    subject.set_day(day)
    runner = ChampionRunner(subject, cfg, sim, t0, {})
    rest = runner.record_rest(day, index)
    reference = BodyCoordinateDecoder(cfg, cfg.hardware.fs).reference_from(rest)
    run = runner.run_block(block, root, tag, day, index + 1, glyph_map(day, block.seed))
    ls = Labeler(cfg, hop_s=sim.label_hop_s).label(SessionStore(root, cfg.hardware.n_channels).load(run.run_dir))
    shutil.rmtree(run.run_dir)
    session = os.path.basename(os.path.normpath(run.run_dir))
    rest_set = idle_set(rest, [{"source": IDLE_SOURCE, "block": "setup", "session": session, "trial": i,
                                "t": float(i) * REST_HOP_S} for i in range(len(rest))])
    return ls, {session: reference}, LabelledSet.concat([rest_set, run.idle])


def build_member(cfg: Config, sim: ChampionSimConfig, member: int, out_dir: str) -> str:
    """One prior member: ``pop_minutes`` of fixation probes on day 1 (cue-defined L1 labels only, so the
    prior carries no display-identifiable label either) and its rest windows (L0, for the idle gate),
    saved as .npy for memory-mapped loading."""
    pop = make_population(sim, cfg)
    seed = pop.params[member].seed
    block = Block("fixation_probe", sim.pop_minutes * 60.0, seed=seed)
    ls, refs, rest = record_fixation(cfg, sim, pop.subject(member), 1, block, 0, f"pop{member:03d}",
                                     os.path.join(out_dir, "runs"), CLOCK_ORIGIN_S)
    d = os.path.join(out_dir, f"member_{member:03d}")
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "windows.npy"), np.stack(ls.windows))
    np.save(os.path.join(d, "labels.npy"), ls.labels)
    np.save(os.path.join(d, "rest.npy"), np.stack(rest.windows))
    (session, reference), = refs.items()
    np.save(os.path.join(d, "reference.npy"), reference)
    with open(os.path.join(d, "provenance.json"), "w") as f:
        json.dump({"session": session, "provenance": ls.provenance, "rest_provenance": rest.provenance}, f)
    return d


def load_prior(prior_dir: str) -> Tuple[LabelledSet, Dict[str, np.ndarray], LabelledSet]:
    """All members' L1 windows and rest (L0) windows (memory-mapped) and their rest references."""
    sets, rests, refs = [], [], {}
    for name in sorted(n for n in os.listdir(prior_dir) if n.startswith("member_")):
        d = os.path.join(prior_dir, name)
        win = np.load(os.path.join(d, "windows.npy"), mmap_mode="r")
        rest = np.load(os.path.join(d, "rest.npy"), mmap_mode="r")
        with open(os.path.join(d, "provenance.json")) as f:
            meta = json.load(f)
        sets.append(LabelledSet([win[i] for i in range(len(win))], np.load(os.path.join(d, "labels.npy")),
                                meta["provenance"]))
        rests.append(idle_set([rest[i] for i in range(len(rest))], meta["rest_provenance"]))
        refs[meta["session"]] = np.load(os.path.join(d, "reference.npy"))
    return LabelledSet.concat(sets), refs, LabelledSet.concat(rests)


def build_l1_pool(cfg: Config, sim: ChampionSimConfig, pop: BodyPopulation, user: int,
                  root: str) -> Tuple[LabelledSet, Dict[str, np.ndarray], LabelledSet]:
    """The fixed L1 test set of ``user``: ``eval_minutes`` of fixation probes per simulated day, each
    under that day's electrode gains with its own rest reference and noise stream, on a fresh subject
    instance at zero practice, never added to any training set; with its rest (L0) windows, the idle
    gate's fixed test set."""
    seed = pop.params[user].seed
    sets, rests, refs = [], [], {}
    for day in range(1, sim.days + 1):
        block = Block("fixation_probe", sim.eval_minutes * 60.0, seed=seed + EVAL_BLOCK_INDEX + day)
        ls, r, rest = record_fixation(cfg, sim, pop.subject(user), day, block, EVAL_BLOCK_INDEX,
                                      f"eval_u{user}_d{day:02d}", os.path.join(root, "eval_runs"),
                                      CLOCK_ORIGIN_S + (day - 1) * DAY_OFFSET_S)
        sets.append(ls)
        rests.append(rest)
        refs.update(r)
    return LabelledSet.concat(sets), refs, LabelledSet.concat(rests)


def _by_source(data: LabelledSet, *sources: str) -> LabelledSet:
    return data.mask(np.array([p.get("source") in sources for p in data.provenance], bool))


def all_channels_shadow_options(cfg: Config, options: dict) -> dict:
    """``decoder_options`` of the all-32 shadow: the deployed model on every cap channel."""
    return {**options, "channels": tuple(cfg.hardware.channels)}


def simulate_user(cfg: Config, sim: ChampionSimConfig, user: int, prior_dir: Optional[str],
                  root: str) -> Tuple[dict, Dict[int, Dict[str, CertificationLog]]]:
    """``sim.days`` simulated days of one deep participant; returns (per-day rows, certification logs per
    day and test).  The rows' "certification" entries are filled by ``certify_user`` once the
    screen_leak 0 twin (the shadow's null) has run too."""
    pop = make_population(sim, cfg)
    params = pop.params[user]
    subject = pop.subject(user)
    options = {"n_permutations": sim.x_permutations}
    geometry = {SHADOWS[0]: BodyCoordinateDecoder(cfg, cfg.hardware.fs),
                SHADOWS[1]: BodyCoordinateDecoder(cfg, cfg.hardware.fs, **all_channels_shadow_options(cfg, {}))}
    trainer = BodyTrainer(cfg, decoder_options=options)
    shadow_trainers = {SHADOWS[0]: ShadowTrainer(cfg, decoder_options=options),
                       SHADOWS[1]: ShadowTrainer(cfg, decoder_options=all_channels_shadow_options(cfg, options))}
    labeler, runs = Labeler(cfg, hop_s=sim.label_hop_s), os.path.join(root, "runs")
    os.makedirs(runs, exist_ok=True)
    store = SessionStore(runs, cfg.hardware.n_channels)

    pool, refs, pool_idle = build_l1_pool(cfg, sim, pop, user, root)
    prior, prior_refs, prior_idle = load_prior(prior_dir) if prior_dir else (LabelledSet(), {}, LabelledSet())
    refs.update(prior_refs)
    refs32: Dict[str, np.ndarray] = {}                       # the all-32 shadow's references (own runs only)
    own, sighted, own_idle = LabelledSet(), LabelledSet(), LabelledSet()

    def gate(decoder: BodyCoordinateDecoder) -> None:
        """Refit the idle gate on every rest window so far against the decoder's training windows."""
        rest = LabelledSet.concat([own_idle, prior_idle])
        if len(rest) >= decoder.n_splits:
            decoder.fit_idle(list(rest.windows), rest.provenance, refs)

    deployed = trainer.fit(prior, session_references=prior_refs) if len(prior) else None
    if deployed is not None:
        gate(deployed)
    prior_l1 = l1_scores(deployed, pool, refs)
    trajectories: Dict[str, List[Trajectory]] = {}
    day_logs: Dict[int, Dict[str, CertificationLog]] = {}
    x_report: Optional[dict] = None
    grant_passed = False                                    # the last grant test passed (earn_x's confirm rule)
    rows = []
    version = 0                                             # deployed decoder version: the held-out test unit
    margins: Dict[int, Dict[str, Tuple[float, float, Optional[float]]]] = {}   # external margin per version
    margins_l1: Dict[int, Dict[str, float]] = {}            # L1-pool margin per version (secondary)
    frozen: Dict[int, BodyCoordinateDecoder] = {}           # the decoder each version's held-out blocks ran

    def retrain(new: List[LabelledSet]) -> None:
        """All own L1/L2b + prior → new deployed decoder.  x is re-decided on own L1 probes only (a fixed
        window population; granted on two consecutive passes, kept once earned unless the keep rule
        rejects it), and the x model is trained on exactly those windows (``x_mask``: the recipe the
        grant certified, not one diluted by the prior's mostly non-lateralising members); an x earner's
        y model treats one-hand windows as neutral; idle gate refitted."""
        nonlocal own, deployed, x_report, version, grant_passed
        was_enabled = bool(deployed is not None and deployed.x_enabled)
        own = LabelledSet.concat([own] + new)
        data = LabelledSet.concat([own, prior])
        x_mask = np.r_[np.array([p.get("source") == "L1" for p in own.provenance], bool), np.zeros(len(prior), bool)]
        fit = lambda neutral: trainer.fit(data, session_references=refs, x_mask=x_mask, neutral_one_hand=neutral)
        deployed = fit(was_enabled)
        x_report = deployed.earn_x(list(own.windows), body_labels(own), own.provenance, refs,
                                   previously_enabled=was_enabled, previous_grant_passed=grant_passed)
        grant_passed = bool(x_report["rule"] == "grant" and x_report.get("passed"))
        if deployed.x_enabled != was_enabled:               # the y model's neutral one-hand rule follows x
            refit = fit(deployed.x_enabled)
            refit.x_enabled, refit.x_report = deployed.x_enabled, deployed.x_report
            deployed = refit
        gate(deployed)
        version += 1

    for day in range(1, sim.days + 1):
        subject.set_day(day)
        runner = ChampionRunner(subject, cfg, sim, CLOCK_ORIGIN_S + (day - 1) * DAY_OFFSET_S, trajectories)
        outcomes: Dict[str, List[TrialOutcome]] = {"blind": [], "sighted": [], "heldout": [], "identity": []}
        pending: Dict[str, List[Tuple[LabelledSet, np.ndarray, bool, int]]] = {t: [] for t in TESTS}
        identity: Dict[int, List[Tuple[LabelledSet, np.ndarray]]] = {}
        new_labelled: List[LabelledSet] = []
        counts = {s: 0 for s in ("L1", "L2b", "S", "H", "R", "I", "L0")}
        x_at_start = bool(deployed is not None and deployed.x_enabled)
        day_rest = None
        for s, plan in enumerate(day_sessions(day, sim.hours, params.seed, x_at_start, sim.slot_minutes)):
            glyphs = glyph_map(day, params.seed + s)
            reference: Dict[str, np.ndarray] = {}
            for i, block in enumerate(plan):
                index = s * SESSION_BLOCK_STRIDE + i
                if block.kind == "setup":
                    rest = runner.record_rest(day, index)
                    reference = {name: g.reference_from(rest) for name, g in geometry.items()}
                    key = f"u{user}_d{day:02d}_s{s}_rest"
                    refs[key] = reference[SHADOWS[0]]
                    day_rest = day_rest if day_rest is not None else reference[SHADOWS[0]]
                    own_idle = LabelledSet.concat([own_idle, idle_set(rest, [
                        {"source": IDLE_SOURCE, "block": "setup", "session": key, "trial": j,
                         "t": float(j) * REST_HOP_S} for j in range(len(rest))])])
                    counts["L0"] += len(rest)
                    runner.advance(max(0.0, block.duration_s - PREROLL_S - sim.rest_s))
                    if deployed is not None:
                        deployed.set_reference(reference=reference[SHADOWS[0]])
                    continue
                if block.just_intend:
                    runner.advance(block.duration_s)
                    continue
                if block.kind == "sighted" and new_labelled:
                    retrain(new_labelled)
                    new_labelled = []
                    deployed.set_reference(reference=reference[SHADOWS[0]])
                runner.decoder = deployed
                runner.x_earned = bool(deployed is not None and deployed.x_enabled)
                run = runner.run_block(block, runs, f"u{user}_d{day:02d}_s{s}_b{i:02d}_{block.kind}", day, index,
                                       glyphs)
                for k, v in run.outcomes.items():
                    outcomes[k].extend(v)
                labelled = {}
                for kind, run_dir in (("block", run.run_dir), ("identity", run.identity_dir)):
                    if run_dir is None:
                        continue
                    labelled[kind] = labeler.label(store.load(run_dir))
                    shutil.rmtree(run_dir)
                    session = os.path.basename(os.path.normpath(run_dir))
                    refs[session], refs32[session] = reference[SHADOWS[0]], reference[SHADOWS[1]]
                ls = labelled["block"]
                for src in ("L1", "L2b", "S", "H", "R"):
                    counts[src] += sum(p.get("source") == src for p in ls.provenance)
                own_idle = LabelledSet.concat([own_idle, run.idle])
                counts["L0"] += len(run.idle)
                if block.kind in BLIND_KINDS:
                    new_labelled.append(ls)
                sighted = LabelledSet.concat([sighted, _by_source(ls, "S")])   # never the identity run
                if "identity" in labelled and len(labelled["identity"]):
                    # Identity trials: scored now by the frozen decoder that ran the block, kept apart from
                    # every trainer; they give this version's external margin at day end.
                    ils = labelled["identity"]
                    counts["I"] += len(ils)
                    identity.setdefault(version, []).append((ils, recentred_outputs(deployed, ils, refs)))
                for test, src in zip(TESTS, ("H", "R")):
                    part = _by_source(ls, src)
                    if len(part):
                        # The frozen decoder that ran the block scores its windows now, before any retrain.
                        if version not in margins_l1:
                            margins_l1[version] = identity_margin(deployed, pool, refs)
                        frozen[version] = deployed
                        pending[test].append((part, recentred_outputs(deployed, part, refs), runner.x_earned,
                                              version))
        if new_labelled:
            retrain(new_labelled)
        for v, parts in identity.items():
            margins[v] = identity_trial_margin_stats(np.concatenate([o for _, o in parts]),
                                                     LabelledSet.concat([p for p, _ in parts]))
        shadows, shadow_error = {}, None
        if len(sighted):
            try:
                for name, tr in shadow_trainers.items():
                    shadows[name] = tr.fit(sighted, x_enabled=deployed.x_enabled,
                                           session_references=refs if name == SHADOWS[0] else refs32)
            except ValueError as exc:                     # too little sighted data for a positive control
                shadows, shadow_error = {}, str(exc)
        logs, report_dropped = {}, {}
        for test in TESTS:
            today = CertificationLog()
            dropped = total = 0
            for part, dep_out, x_on, v in pending[test]:
                if not shadows:
                    continue
                body, screen = regressors(test, part)
                keep = confirmed_by_report(part.provenance, body)
                cl = trial_clusters(part.provenance)
                total += len(np.unique(cl))
                dropped += len(np.unique(cl)) - len(np.unique(cl[keep]))
                if not keep.any():
                    continue
                part, dep_out, body, screen = part.mask(keep), dep_out[keep], body[keep], screen[keep]
                outs = {name: recentred_outputs(sh, part, refs if name == SHADOWS[0] else refs32)
                        for name, sh in shadows.items()}
                mg = margins.get(v, {a: (float("nan"), float("nan"), None) for a in AXES})
                today.add(dep_out, outs[SHADOWS[0]], body, screen, part.provenance,
                          np.full(len(part), x_on and deployed.x_enabled), f"u{user}/d{day}/v{v}", f"u{user}",
                          {a: mg[a][0] for a in AXES}, extra_shadows={n: o for n, o in outs.items() if n != SHADOWS[0]},
                          margin_se={a: mg[a][1] for a in AXES}, margin_df={a: mg[a][2] for a in AXES})
            logs[test] = today
            report_dropped[test] = {"trials": total, "dropped_by_report": dropped}
        day_logs[day] = logs
        last = max(margins) if margins else None
        held = max(frozen) if frozen else None
        transfer = None
        if held is not None and day_rest is not None:
            mg = {a: margins.get(held, {}).get(a, (float("nan"),))[0] for a in AXES}
            transfer = leak_transfer(cfg, sim, pop, user, day, frozen[held], day_rest, mg)
        rows.append({
            "day": day, "cum_hours": day * sim.hours,
            "x_earned": bool(deployed.x_enabled), "x_report": x_report and {
                k: x_report[k] for k in ("auc", "floor", "p_value", "auc_purity", "purity_retained", "rule",
                                         "n_windows", "reason", "passed", "grant_pending")},
            "margin": {a: margins[last][a][0] for a in AXES} if last is not None else None,
            "margin_se": {a: margins[last][a][1] for a in AXES} if last is not None else None,
            "margin_l1": margins_l1[last] if last in margins_l1 else None,
            "leak_transfer": transfer,
            "l1": l1_scores(deployed, pool, refs),
            "idle": idle_scores(deployed, pool, pool_idle, refs),
            "sighted": trial_metrics(outcomes["sighted"], fitts=True),
            "blind": trial_metrics(outcomes["blind"], fitts=False),
            "heldout_closed_loop": trial_metrics(outcomes["heldout"], fitts=True),
            "n_windows": counts, "n_train_windows": {"own": len(own), "prior": len(prior), "shadow": len(sighted),
                                                     "idle": len(own_idle) + len(prior_idle)},
            "shadow_error": shadow_error, "report_check": report_dropped,
        })
    result = {"user": user, "subject": pop.describe([user])[0], "prior_l1": prior_l1, "days": rows}
    return result, day_logs


def certify_user(result: dict, day_logs: Dict[int, Dict[str, CertificationLog]],
                 null_logs: Optional[Dict[int, Dict[str, CertificationLog]]], n_boot: int) -> dict:
    """Fill ``result``'s per-day and pooled-days certification from its logs, each against the same
    log of the screen_leak 0 twin (``null_logs``; the logs themselves at screen_leak 0; None: an
    uncalibrated positive control, never CERTIFIED)."""
    def pair(day: int, test: str) -> Tuple[CertificationLog, Optional[CertificationLog]]:
        log = day_logs[day][test]
        if null_logs is None:
            return log, None
        return log, (log if null_logs is day_logs else null_logs.get(day, {}).get(test, CertificationLog()))

    for row in result["days"]:
        day, cert = row["day"], {}
        for test in TESTS:
            log, null = pair(day, test)
            cert[test] = log.certify(test, n_boot, null=null)
            if test == "heldout" and log.provenance:
                cert["heldout_by_variant"] = log.certify_by_variant(n_boot, null=null)
        if row.get("shadow_error"):
            for test in TESTS:
                cert[test] = {"skipped": f"shadow not trained: {row['shadow_error']}", "n_trials": 0, "n_windows": 0}
        row["certification"] = cert
    merged = {t: CertificationLog.merge([day_logs[d][t] for d in sorted(day_logs)]) for t in TESTS}
    merged_null = None if null_logs is None else (
        merged if null_logs is day_logs else
        {t: CertificationLog.merge([null_logs[d][t] for d in sorted(null_logs)]) for t in TESTS})
    pooled = {t: merged[t].certify(t, n_boot, null=None if merged_null is None else merged_null[t]) for t in TESTS}
    pooled["heldout_by_variant"] = (merged["heldout"].certify_by_variant(
        n_boot, null=None if merged_null is None else merged_null["heldout"]) if merged["heldout"].provenance else {})
    result["pooled_days"] = pooled
    return result


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
    (in parallel) and, when ``null_calibration`` and the screen leak is on, the same people at screen
    leak 0 (the twin: same seeds, noise streams and prior, so the same decoder versions and units —
    the shadows' no-display null); then certify per day, per person and pooled over participants.
    Run directories live in a temporary directory and are deleted as soon as they are labelled."""
    t0 = time.perf_counter()
    cfg = champion_config(cfg)
    pop = make_population(sim, cfg)
    twin = dataclasses.replace(sim, screen_leak=0.0) if sim.null_calibration and sim.screen_leak > 0 else None
    with tempfile.TemporaryDirectory(prefix="nc_champion_") as tmp:
        prior_dir = None
        if sim.pop > 0:
            # The prior's members record fixation probes only (nothing drawn), so the twin shares it.
            prior_dir = os.path.join(tmp, "prior")
            _map(_member_job, [(cfg, sim, j, prior_dir) for j in range(sim.users, sim.users + sim.pop)], workers)
        jobs = [(cfg, sim, u, prior_dir, os.path.join(tmp, f"u{u}")) for u in range(sim.users)]
        if twin is not None:
            jobs += [(cfg, twin, u, prior_dir, os.path.join(tmp, f"null_u{u}")) for u in range(sim.users)]
        out = _map(_user_job, jobs, workers)
    main, nulls = out[:sim.users], out[sim.users:]
    per_user = []
    for u, (res, logs) in enumerate(main):
        null_logs = (nulls[u][1] if twin is not None else logs) if sim.null_calibration else None
        per_user.append(certify_user(res, logs, null_logs, sim.n_boot))
    merged = {t: CertificationLog.merge([logs[d][t] for _, logs in main for d in sorted(logs)]) for t in TESTS}
    if not sim.null_calibration:
        merged_null = None
    elif twin is None:
        merged_null = merged
    else:
        merged_null = {t: CertificationLog.merge([logs[d][t] for _, logs in nulls for d in sorted(logs)]) for t in TESTS}
    pooled = {t: merged[t].certify(t, sim.n_boot, null=None if merged_null is None else merged_null[t]) for t in TESTS}
    pooled["heldout_by_variant"] = (merged["heldout"].certify_by_variant(
        sim.n_boot, null=None if merged_null is None else merged_null["heldout"]) if merged["heldout"].provenance else {})
    return {"config": dataclasses.asdict(sim), "population": pop.describe(range(sim.users, sim.users + sim.pop)),
            "per_user": per_user, "pooled_all_users": pooled, "runtime_s": time.perf_counter() - t0}


def sweep(sim: ChampionSimConfig, dry_penalties: Sequence[float], cfg: Config = DEFAULT, workers: int = 1,
          covert_leaks: Optional[Sequence[float]] = None) -> dict:
    """Sensitivity sweeps on the same people, days and seeds: the dry penalty (an ERD-depth scaling
    whose measured d′ loss ``sim_body`` tabulates — a range, not a point prediction, as the skeptic
    corrections ask) and, optionally, the covert leak — the leak the label rule cannot exclude, on
    which every CERTIFIED verdict is conditional.  Condition keys are the dry penalty ("0.35"), with
    "_covert<c>" appended when several covert leaks are swept.  Every setting is validated before the
    first condition runs; a condition that fails for lack of data (a ``ValueError``, e.g. too few
    sighted or held-out trials for a too-short session) is recorded as {"failed": reason} and the sweep
    goes on, so an hour-long sweep never loses its finished conditions."""
    t0 = time.perf_counter()
    covert = [sim.covert_leak] if covert_leaks is None else [float(c) for c in covert_leaks]
    keys = [(float(p), c, f"{p:g}" if len(covert) == 1 else f"{p:g}_covert{c:g}") for p in dry_penalties for c in covert]
    if len({k for _, _, k in keys}) != len(keys):
        raise ValueError(f"dry penalties {list(dry_penalties)} / covert leaks {covert} repeat a condition")
    settings = {k: dataclasses.replace(sim, dry_penalty=p, covert_leak=c) for p, c, k in keys}   # validates
    conditions = {}
    for key, cond in settings.items():
        try:
            conditions[key] = simulate(cond, cfg, workers)
        except ValueError as exc:
            conditions[key] = {"failed": str(exc), "config": dataclasses.asdict(cond)}
    return {"dry_penalties": [float(p) for p in dry_penalties], "covert_leaks": covert, "conditions": conditions,
            "runtime_s": time.perf_counter() - t0}


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def _num(x, nd: int = 3) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _ci(beta: float, ci: Sequence[float]) -> str:
    return f"{_num(beta, 2)} [{_num(ci[0], 2)}, {_num(ci[1], 2)}]"


def _auc(l1: dict, axis: str) -> str:
    return f"{_num(l1['auc'][axis])} ({_num(l1['floor'][axis])})"


def _cert_cells(c: dict, transfer: str) -> str:
    """One certification row: relative βs (÷ the deployed decoder's external margin) of the deciding
    axis for the deployed decoder and both shadows (each with its screen_leak 0 null in brackets), the
    largest per-block body–screen correlation, the deployed decoder's leak-transfer gain, the per-axis
    verdicts and the verdict."""
    if "skipped" in c:
        return f"| – | – | {c['n_trials']} | – | – | – | – | – | – | {transfer} | – | skipped: {c['skipped']} |"
    a = c["deciding_axis"]
    d = c["deployed"][a]
    shadows = c.get("shadows") or {SHADOWS[0]: c["shadow"]}
    nulls = c.get("shadows_null") or {}

    def shadow_cell(n: str) -> str:
        if n not in shadows:
            return "–"
        cell = _ci(shadows[n][a]["rel_screen"], shadows[n][a]["rel_ci_screen"])
        null = nulls.get(n, {}).get(a)
        return cell + (f" (null {_num(null.get('rel_screen'), 2)})" if null else " (no null)")

    cells = " | ".join(shadow_cell(n) for n in SHADOWS)
    per = ", ".join(f"{k}: {v}" for k, v in sorted(c["per_axis"].items()))
    cross = _ci(d["beta_cross"], d["ci_cross"]) if "ci_cross" in d else "–"
    return (f"| {'+'.join(c['axes'])} | {c['n_units']} | {c['n_trials']} | {a}: {_ci(d['rel_body'], d['rel_ci_body'])} "
            f"| {_ci(d['rel_screen'], d['rel_ci_screen'])} | {cross} | {cells} | {_num(d.get('max_block_corr'), 2)} "
            f"| {transfer} | {per} | **{c['verdict']}** |")


def _transfer(rows: Sequence[dict]) -> str:
    """Leak-transfer gain (y / x, ÷ margin) of the decoders behind ``rows``: one day's, or the mean over
    days for a pooled row."""
    vals = {a: [r["leak_transfer"]["rel"][a] for r in rows if r.get("leak_transfer")] for a in ("y", "x")}
    mean = {a: float(np.nanmean(v)) if v and np.isfinite(v).any() else float("nan") for a, v in vals.items()}
    return f"{_num(mean['y'], 2)} / {_num(mean['x'], 2)}"


def _responders(cond: dict) -> str:
    """Responder counts of the deep users and the prior, against the population rate (review: one seed
    drew 7 of 13, P = 0.007 at the stated 20 %)."""
    from scipy.stats import binom
    deep = [u["subject"]["x_responder"] for u in cond["per_user"]]
    prior = [m["x_responder"] for m in cond.get("population", [])]
    p = cond["config"]["x_responder_prob"]
    n, k = len(deep) + len(prior), sum(deep) + sum(prior)
    tail = float(binom.sf(k - 1, n, p)) if n else float("nan")
    return (f"x responders: {sum(deep)} of {len(deep)} deep users, {sum(prior)} of {len(prior)} prior members "
            f"({k} of {n}; the population rate is {p:g}, so P(≥ {k} of {n}) = {tail:.3f}): read x-axis and "
            "prior results as those of this draw, not of the stated population")


def to_markdown(res: dict) -> str:
    ok = {k: v for k, v in res["conditions"].items() if "failed" not in v}
    first = next(iter(ok.values()), next(iter(res["conditions"].values())))
    c = first["config"]
    covert = ", ".join(sorted({f"{v['config']['covert_leak']:g}" for v in res["conditions"].values()}))
    lines = [f"## Champion protocol (simulated): {c['days']} days × {c['hours']:g} h, {c['users']} deep "
             f"participant(s), population prior N = {c['pop']}",
             "",
             f"seed {c['seed']}; screen leak {c['screen_leak']:g} (targets drawn: S, H, R); covert leak "
             f"{covert} (blind L1/L2b "
             f"blocks); replay attention '{c['replay_attention']}'; x-responder probability {c['x_responder_prob']:g}; "
             f"y depth {c['y_depth_range']}, x depth {c['x_depth_range']}; user learning rate "
             f"{c['user_learning_rate']:g}; assistance a = {c['assist_gain']:g} (sighted, C2c); prior "
             f"{c['pop_minutes']:g} min of L1 per member; label hop {c['label_hop_s']:g} s; slot minutes "
             f"{c['slot_minutes'] or 'protocol'}; shadow null calibration "
             f"{'on' if c.get('null_calibration', True) else 'OFF (never CERTIFIED)'}; runtime {res['runtime_s']:.0f} s.",
             ""]
    if "per_user" in first:
        lines += [_responders(first) + ".", ""]
    lines += ["**Every CERTIFIED verdict below is conditional on the covert leak of its condition.**  At covert "
              "leak 0 the deployed decoder's training data (L1/L2b, and a prior of simulated L1) contain no "
              "screen-correlated non-motor signal by construction; covert attention toward the imagined "
              "direction in blind blocks — the leak the label rule cannot exclude (skeptic C3) — is what the "
              "covert-leak sweep (`--covert-leak 0 1 2`) tests.  A screen-leak sweep at covert leak 0 measures "
              "the SHADOW positive controls' sensitivity only (they learn the leak from their display-identifiable "
              "labels; the deployed decoder's training data carry none of it); whether a display leak reaches the "
              "DEPLOYED decoder is the 'leak transfer' column: a paired probe (identical noise streams at screen "
              f"leak 0 and {PROBE_LEAK:g}, rot±90 imagery on diagonal targets) of the frozen held-out decoder, "
              "its output slope on the screen direction ÷ the external margin.",
              "",
              "Deployed decoder: trained on L1 + L2b only (+ prior), re-centred on each session's rest; its x model "
              "only on the person's own L1 probes (the windows x is earned on), and an x earner's y model treats "
              "one-hand windows as neutral.  L2b windows end ≤ 4 s after the cue and before the hidden cursor "
              "reaches the dwell radius.  Shadows: the same model on sighted ReFIT labels (positive controls, "
              "never deployed) — NOT matched-n twins: S windows only (about a third of the deployed decoder's own "
              "window count) and no prior.  L1 AUC: fixed fixation-probe test "
              f"set ({c['eval_minutes']:g} min per simulated day, never trained on), with its floor (p95 of a "
              "trial-level label permutation) in brackets; x AUC is the x model's even when x is not earned.  "
              "x earned: two consecutive passing grant tests on own L1 probes (AUC > 0.60 and > its "
              "circular-shift floor, and ≥ 70 % of the margin kept after regressing out posterior alpha "
              "lateralisation), then kept unless the floor or the purity rule rejects it.  Sighted: non-replay "
              "S trials (success, time-to-target, Fitts throughput); blind: L2b hidden-cursor success.",
              "",
              "Certification: per test unit (held-out: one deployed decoder version, i.e. one session, with its own "
              "slopes and intercepts; replay: one person), output_a = β_body·body_a + β_screen·screen_a + "
              "β_cross·body_other + c_a, CR2 cluster-robust SEs with t(G − k) intervals (trials are the "
              "clusters); units of one person combined by inverse variance, people by a person-then-trial "
              "bootstrap (t with people − 1 df).  βs are relative to the deployed decoder's EXTERNAL margin: its "
              f"slope on body intent in the identity trials ({c.get('identity_fraction', 0.25):g} of each held-out "
              "block, interleaved, never certified), whose SE is propagated into every threshold test; 'margin L1' "
              "is the secondary slope on the fixed L1 pool.  Verdict family (Holm, per axis): β_body > 0, β_body > "
              "50 % of the margin (and the point estimate ≥ 70 %), |β_screen| < 10 % (two one-sided tests), and "
              "each shadow's β_screen above its NULL — the same shadow, same units, at screen leak 0 (in "
              "brackets).  A β_screen significantly different from 0 in EITHER direction and not bounded "
              "inside ±10 % is SCREEN_READING, any unit's own SCREEN_READING vetoes the combined verdict, and a "
              "unit whose body and screen regressors correlate |r| > 0.5 in any block is refused (max |r| block).  "
              "Trials whose keyed report (C2b, post-replay) does not confirm the intent are dropped.  Held-out "
              "blocks are rot+90 and rot−90 in every session with balanced diagonal targets; replays do not "
              "consume a goal quadrant and replay diagonal trajectories inside held-out blocks.  Regressors use "
              "the cursor at the window start; C2b and C2c are also reported separately.  Intent is an oracle "
              "(exact cued body vector; instant remapping), so β_body is an upper bound.  'Just intend' blocks are "
              "not simulated.",
              "",
              "Idle gate: rest windows (L0: setup rest and the rest between probes, which now carries the "
              "post-imagery beta rebound) against the decoder's training windows; armed only if its out-of-fold "
              "AUC ≥ 0.70, and then the output is exactly 0 when p(idle) > 0.7.  'rest gated / intent passed' "
              "are measured on the fixed pool's rest and probe windows.",
              ""]
    for name, cond in res["conditions"].items():
        cc = cond["config"]
        lines += [f"### dry penalty {cc['dry_penalty']:g}, covert leak {cc['covert_leak']:g}", ""]
        if "failed" in cond:
            lines += [f"Condition failed (nothing to report): {cond['failed']}", ""]
            continue
        lines += ["| user | day | cum h | x earned (AUC / floor / purity kept) | L1 AUC y (floor) | L1 AUC x (floor) "
                  "| L1 r y | margin y ± SE / x (I) | margin L1 y | leak transfer y / x | idle gate (AUC): rest gated / "
                  "intent passed | sighted success | TTT (s) | TP (bits/s) | blind success | windows L1/L2b/S/H/R/I/L0 |",
                  "|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|---:|---:|---:|---:|---|"]
        for u in cond["per_user"]:
            pl = u["prior_l1"]
            lines.append(f"| {u['user']} | 0 (prior only) | 0 | – | {_auc(pl, 'y')} | {_auc(pl, 'x')} "
                         f"| {_num(pl['r']['y'])} | – | – | – | – | – | – | – | – | – |")
            for r in u["days"]:
                xr = r["x_report"] or {}
                n = r["n_windows"]
                mg, ml1, idle = r.get("margin") or {}, r.get("margin_l1") or {}, r["idle"]
                mse = r.get("margin_se") or {}
                gate = (f"{'armed' if idle['armed'] else 'off'} ({_num(idle['cv_auc'], 2)}): "
                        f"{_num(idle['rest_gated'], 2)} / {_num(idle['intent_passed'], 2)}")
                earned = "yes" if r["x_earned"] else ("pending" if xr.get("grant_pending") else "no")
                lines.append(
                    f"| {u['user']} | {r['day']} | {r['cum_hours']:g} | {earned} "
                    f"({_num(xr.get('auc'))} / {_num(xr.get('floor'))} / {_num(xr.get('purity_retained'), 2)}) "
                    f"| {_auc(r['l1'], 'y')} | {_auc(r['l1'], 'x')} | {_num(r['l1']['r']['y'])} "
                    f"| {_num(mg.get('y'))} ± {_num(mse.get('y'))} / {_num(mg.get('x'))} | {_num(ml1.get('y'))} "
                    f"| {_transfer([r])} | {gate} | {_num(r['sighted']['success'], 2)} "
                    f"| {_num(r['sighted']['time_to_target_s'], 1)} | {_num(r['sighted']['throughput_bps'], 2)} "
                    f"| {_num(r['blind']['success'], 2)} "
                    f"| {n['L1']}/{n['L2b']}/{n['S']}/{n['H']}/{n['R']}/{n['I']}/{n['L0']} |")
        lines += ["", "| user | day | test | axes | units | trials | deployed β_body/margin | deployed β_screen/margin "
                  "| deployed β_cross | shadow c11 β_screen/margin (null) | shadow all32 β_screen/margin (null) "
                  "| max \\|r\\| block | leak transfer y / x | per axis | verdict |",
                  "|---:|---|---|---|---:|---:|---|---|---|---|---|---:|---|---|---|"]

        def rows_for(who: str, day: str, cert: dict, days: Sequence[dict]) -> None:
            transfer = _transfer(days)
            for test in TESTS:
                lines.append(f"| {who} | {day} | {test} " + _cert_cells(cert[test], transfer))
            for variant, cv in sorted((cert.get("heldout_by_variant") or {}).items()):
                lines.append(f"| {who} | {day} | heldout {variant} " + _cert_cells(cv, transfer))

        for u in cond["per_user"]:
            for r in u["days"]:
                rows_for(str(u["user"]), str(r["day"]), r["certification"], [r])
            rows_for(str(u["user"]), "all", u["pooled_days"], u["days"])
        rows_for("all", "all", cond["pooled_all_users"], [r for u in cond["per_user"] for r in u["days"]])
        dropped = {t: sum(r["report_check"][t]["dropped_by_report"] for u in cond["per_user"] for r in u["days"])
                   for t in TESTS}
        lines += ["", f"Trials dropped because the keyed report did not confirm the intent: held-out "
                      f"{dropped['heldout']}, replay {dropped['replay']} (the simulated person always remaps).", ""]
    return "\n".join(lines)


def write_report(res: dict, prefix: str) -> Tuple[str, str]:
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    with open(prefix + ".json", "w") as f:
        json.dump(_clean(res), f, indent=2)
    with open(prefix + ".md", "w") as f:
        f.write(to_markdown(res) + "\n")
    return prefix + ".json", prefix + ".md"


_PREFIX_KEYS = {"seed": "seed", "replay_attention": "replay", "x_responder_prob": "xresp", "y_depth_range": "y",
                "x_depth_range": "x", "user_learning_rate": "lr", "pop_minutes": "popmin", "rest_s": "rest",
                "eval_minutes": "eval", "assist_gain": "assist", "label_hop_s": "hop", "x_permutations": "xperm",
                "n_boot": "boot", "x_time_course": "xtc", "day_depth_sd": "daydepth", "slot_minutes": "slots",
                "identity_fraction": "idfrac", "null_calibration": "null"}
"""Short names of the ChampionSimConfig fields that enter the default output prefix when they differ
from their defaults (days, hours, users, pop, dry, leak and covert are always in it)."""


def default_prefix(sim: ChampionSimConfig, dry_penalties: Sequence[float], covert_leaks: Sequence[float]) -> str:
    """results/champion_d…_h…_u…_pop…_dry…_leak…_covert… plus every other non-default setting, so runs
    with different settings never overwrite one another's results."""
    fmt = lambda v: "-".join(fmt(x) for x in v) if isinstance(v, (tuple, list)) else f"{v:g}" \
        if isinstance(v, float) else str(v)
    parts = [f"d{sim.days}", f"h{sim.hours:g}", f"u{sim.users}", f"pop{sim.pop}",
             "dry" + "-".join(f"{p:g}" for p in dry_penalties), f"leak{sim.screen_leak:g}",
             "covert" + "-".join(f"{c:g}" for c in covert_leaks)]
    d = ChampionSimConfig()
    for name, short in _PREFIX_KEYS.items():
        value = getattr(sim, name)
        if value != getattr(d, name):
            parts.append(short + ("".join(f"{k.split('_')[0][:2]}{m:g}" for k, m in value) if name == "slot_minutes"
                                  else fmt(value)))
    return os.path.join("results", "champion_" + "_".join(parts))


def _slot_minutes(text: str) -> Tuple[str, float]:
    """argparse type for SLOT=MIN: a known slot and a positive number of minutes."""
    slot, sep, minutes = text.partition("=")
    if not sep or slot not in CHAMPION_SLOT_MIN:
        raise argparse.ArgumentTypeError(f"expected SLOT=MIN with SLOT in {tuple(CHAMPION_SLOT_MIN)}, got {text!r}")
    try:
        value = float(minutes)
    except ValueError:
        raise argparse.ArgumentTypeError(f"minutes must be a number in {text!r}") from None
    if not value > 0:
        raise argparse.ArgumentTypeError(f"minutes must be > 0 in {text!r}")
    return slot, value


def main(argv: Optional[Sequence[str]] = None) -> dict:
    d = ChampionSimConfig()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=d.days)
    ap.add_argument("--hours", type=float, default=d.hours, help="session hours per day (1 = 85-min plan scaled)")
    ap.add_argument("--users", type=int, default=d.users, help="deep participants")
    ap.add_argument("--pop", type=int, default=d.pop, help="population-prior members; 0 = own data only")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--dry-penalty", type=float, nargs="+", default=[d.dry_penalty],
                    help="motor ERD depth scaling, not the d′ loss (sim_body tabulates the measured loss); "
                    "several values sweep")
    ap.add_argument("--screen-leak", type=float, default=d.screen_leak,
                    help="overt display pathway gain while targets are drawn (> 0 exercises the shadow control)")
    ap.add_argument("--covert-leak", type=float, nargs="+", default=[d.covert_leak],
                    help="covert attention to the imagined screen direction in blind (L1/L2b) blocks; "
                    "several values sweep it (CERTIFIED is conditional on it)")
    ap.add_argument("--replay-attention", choices=REPLAY_ATTENTION, default=d.replay_attention)
    ap.add_argument("--x-responder-prob", type=float, default=d.x_responder_prob)
    ap.add_argument("--y-depth-range", type=float, nargs=2, default=list(d.y_depth_range), metavar=("LO", "HI"),
                    help="per-subject feet/hands ERD depth range (sim_body's calibrated default)")
    ap.add_argument("--x-depth-range", type=float, nargs=2, default=list(d.x_depth_range), metavar=("LO", "HI"),
                    help="per-subject one-hand ERD depth range (sim_body's calibrated default)")
    ap.add_argument("--x-time-course", type=float, nargs=3, default=None, metavar=("PEAK", "DECAY", "SUSTAIN"),
                    help="separate x ERD time course, e.g. 0.3 2.8 0 (sim_body.X_ERD_LR, the ds004362 L/R decay)")
    ap.add_argument("--day-depth-sd", type=float, default=d.day_depth_sd,
                    help="per-day log-normal SD of ERD depth (0: gain-only drift)")
    ap.add_argument("--learning-rate", type=float, default=d.user_learning_rate)
    ap.add_argument("--pop-minutes", type=float, default=d.pop_minutes)
    ap.add_argument("--eval-minutes", type=float, default=d.eval_minutes)
    ap.add_argument("--label-hop", type=float, default=d.label_hop_s)
    ap.add_argument("--x-permutations", type=int, default=d.x_permutations)
    ap.add_argument("--n-boot", type=int, default=d.n_boot)
    ap.add_argument("--slot-minutes", nargs="*", default=[], type=_slot_minutes, metavar="SLOT=MIN",
                    help=f"per-session minutes for plan slots {tuple(CHAMPION_SLOT_MIN)}, overriding --hours "
                    "for those slots (power analysis of the session allocation)")
    ap.add_argument("--identity-fraction", type=float, default=d.identity_fraction,
                    help="share of held-out-block trials run under the identity mapping (the external margin)")
    ap.add_argument("--no-null-calibration", action="store_true",
                    help="skip the screen_leak 0 twin: the shadow controls stay uncalibrated, nothing is CERTIFIED")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--out", type=str, default="", help="output prefix for .json and .md (default: "
                    "results/champion_d<days>_h<hours>_u<users>_pop<N>_dry<...>_leak<...>_covert<...> plus every "
                    "other non-default setting)")
    a = ap.parse_args(argv)
    slots = [k for k, _ in a.slot_minutes]
    if len(set(slots)) != len(slots):
        ap.error(f"--slot-minutes names a slot twice: {slots}")
    try:
        sim = ChampionSimConfig(days=a.days, hours=a.hours, users=a.users, pop=a.pop, seed=a.seed,
                                screen_leak=a.screen_leak, covert_leak=a.covert_leak[0],
                                replay_attention=a.replay_attention, x_responder_prob=a.x_responder_prob,
                                y_depth_range=tuple(a.y_depth_range), x_depth_range=tuple(a.x_depth_range),
                                user_learning_rate=a.learning_rate, pop_minutes=a.pop_minutes,
                                eval_minutes=a.eval_minutes, label_hop_s=a.label_hop,
                                x_permutations=a.x_permutations, n_boot=a.n_boot,
                                x_time_course=None if a.x_time_course is None else tuple(a.x_time_course),
                                day_depth_sd=a.day_depth_sd, slot_minutes=tuple(a.slot_minutes),
                                identity_fraction=a.identity_fraction, null_calibration=not a.no_null_calibration)
        # every swept condition is validated now, not after the earlier ones have run for an hour
        conditions = [(p, c) for p in a.dry_penalty for c in a.covert_leak]
        if len(set(conditions)) != len(conditions):
            raise ValueError(f"--dry-penalty {a.dry_penalty} / --covert-leak {a.covert_leak} repeat a condition")
        for p, c in conditions:
            dataclasses.replace(sim, dry_penalty=p, covert_leak=c)
    except ValueError as exc:
        ap.error(str(exc))
    res = sweep(sim, a.dry_penalty, DEFAULT, workers=a.workers, covert_leaks=a.covert_leak)
    prefix = a.out or default_prefix(sim, a.dry_penalty, a.covert_leak)
    paths = write_report(res, prefix)
    print(to_markdown(res))
    print(f"\nwrote {paths[0]} and {paths[1]}")
    return res


if __name__ == "__main__":
    main()
