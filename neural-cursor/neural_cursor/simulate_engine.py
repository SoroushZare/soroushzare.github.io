"""Days × hours of simulated use through the whole data engine: does every hour make the decoder better?

What one simulated day is
-------------------------
1. ``protocol.session_plan(day, seed)`` gives the blocks.  One unscaled plan (58–60 min incl. setup
   and rests) counts as one hour of use.  ``hours`` < 1 scales every block of it down by ``hours``;
   ``hours`` > 1 runs ``ceil(hours)`` sessions scaled to share the time.  Plans are never stretched,
   because the protocol caps block lengths for attention.  Setup and rest blocks advance the clock
   only: they carry no trials, so they cannot change any number below.
2. Every labelled block is one ``RunLogger`` run (raw µV, events, cursor) on a simulated clock — no
   wall clock anywhere, so a day runs as fast as the CPU allows and is exactly reproducible.
   * ``cued_blank``: auditory / central-symbol cues from ``CueScheduler`` with ``CuedTrialTiming``;
     gaze stays at fixation; the latent intent is the cued direction.  These are the calibration
     windows and the champion test set.
   * ``closed_loop`` / ``free_use``: the cursor runs on the deployed decoder blended with
     ``GoalPosterior`` assistance (a = 0.5), through the same per-update chain as
     ``realtime.RealTimeLoop`` (causal filter → amplitude gate → CAR → last window → predict), but
     paced by samples instead of a wall clock.  The simulated person intends toward the goal from
     where the cursor is, and looks at the goal.  By default that intent is exactly the labeler's
     cursor → goal direction, i.e. closed-loop labels are ORACLE labels, and any gain from closed-loop
     data is guaranteed rather than tested.  ``intent_angle_sd_deg`` / ``intent_lapse_prob`` add the
     real-world mismatch (a slowly re-drawn angular error, attention lapses with no intent); report
     curves at several levels before quoting a closed-loop benefit.
   * ``free_use``: the goal is not logged at trial start; a kept selection reveals it at trial end.
     The simulated person undoes wrong selections, and an undone selection reveals nothing.
   * ``blank_feedback``: identical with the cursor hidden; the person can only intend from where the
     cursor started, and ``data_engine.Labeler`` labels hidden-cursor trials from that start position.
3. At day end the ``DataEngine`` ingests the day's runs and ``maybe_retrain`` applies the
   champion/challenger rule; ``PurityAudit`` runs on the day's labelled windows with the simulated
   eye tracker as an auditor stream.

Endpoints per day, as learning curves vs cumulative hours
---------------------------------------------------------
* decoder cosine / r: the decoder *deployed* that day, scored on a FIXED evaluation pool — auditory
  cued_blank blocks of ``eval_minutes`` recorded once per simulated day (under that day's electrode
  gains) and kept out of the engine's store entirely, so no model ever trains on them.  Every day's
  decoder meets the same windows, so the curve moves only when the decoder does (scoring on each
  day's own cued windows mixed "more data" with "a different, often smaller and harder test set").
  Reported with the pool's constant-output ceiling (mean cosine is not chance-corrected: a constant
  output scores up to |mean label|), a trial-cluster 95 % CI and n.  Cosine is also lowered by the
  constant offset day-to-day gain drift adds to every prediction, which r ignores: read the two
  together.  The day's own cued windows are still scored (``today``), for reference only.
* closed-loop success rate, time-to-target and Fitts throughput (``evaluation``).
* Own condition, day 1: nothing is deployed, so cosine and r are 0 by definition, not by measurement;
  "the prior beats own data on day 1" is true by construction.  How much better the prior is depends
  on the population's shared-topography fraction (``heterogeneity_range``, a hard-coded assumption
  until calibrated against real cross-subject transfer): report it as a sweep, not a point.

Two conditions run on the same simulated person with the same block seeds: (i) own data only and
(ii) a population prior from N other simulated subjects' cued calibration.  In (ii), until the first
champion is promoted, the deployed decoder is the population-only decoder (a zero-calibration start);
in (i) there is nothing to deploy on day 1, so the closed loop runs on assistance alone — which is
what a first session without a prior looks like.

Why DISPLAY_LEAK appears every day: closed_loop, blank_feedback and free-use blocks draw the goal
at its screen position, and their labels (cursor or start → goal) are a function of it, so the
display-only surrogate beats its floor whenever such windows are in the audit.  That is the audit
correctly saying "target-display labels cannot certify purity"; the engine's protection is that
champions are chosen, and decoding is claimed, on cued_blank windows only.

Pure-EEG rule: the simulated eye tracker is written next to the EEG as ``eyetracker.f64`` (an auditor
stream).  ``SessionStore`` never opens it, so it cannot reach a decoder; this module reads it only to
hand it to ``PurityAudit``.  Intent and gaze are simulator state, never decoder input.

Run:  python3 -m neural_cursor.simulate_engine --days 5 --hours 1 --pop 20
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from threadpoolctl import threadpool_limits

from .config import DEFAULT, Config
from .data_engine import (ChampionChallenger, DataEngine, LabelledSet, Labeler, ModelRegistry, SessionStore,
                          Trainer, constant_ceiling, cosines, mean_cosine, trial_clusters)
from .decoder import VelocityDecoder
from .evaluation import fitts_throughput, pearson_r, success_rate, time_to_target
from .logger import RunLogger
from .protocol import (EVENT_CUE_OFFSET, Block, CuedTrialTiming, CueScheduler, ZipfTargets, cued_goal,
                       session_plan, trial_start_payload)
from .purity import PurityAudit
from .signal import AmplitudeGate, CausalFilter, common_average
from .sim_population import HETEROGENEITY_RANGE, PopulationSim, SubjectEEG
from .task import TrialOutcome, TrialRunner, fitts_targets

FITTS_CONDITIONS: Tuple[Tuple[float, float], ...] = ((0.5, 0.08), (0.8, 0.08), (0.8, 0.05))
"""(distance, width) cycled over closed-loop trials — the conditions of ``simulate.py``, so numbers
are comparable with the round-1 harness."""
PREROLL_S = 2.5
"""EEG logged at the start of every block run before its first trial: longer than the labeler's 2 s
filter-settling exclusion, so the first trial's windows are not discarded."""
SESSION_HOURS = 1.0               # hours of use one unscaled protocol session counts as
DAY_OFFSET_S = 86_400.0           # simulated clock: day d starts at (d − 1) days
CLOCK_ORIGIN_S = 1_000_000.0      # positive, far from zero, so no timestamp is ever ≤ 0
GAZE_FILE = "eyetracker.f64"      # auditor stream; rows (t_start, t_end, gaze_x, gaze_y)
CONDITIONS = ("own", "population")
EVAL_BLOCK_INDEX = 10_000         # block index of the evaluation pool: never collides with a plan block
INTENT_SEGMENT_S = 1.0            # how often the simulated intent error / lapse is re-drawn
EVAL_BOOTSTRAP = 2000             # trial-cluster bootstrap draws for the evaluation CI
_MISMATCH_STREAM = 17             # RNG stream id of the intent-mismatch draws (per block)


@dataclass(frozen=True)
class SimConfig:
    """Hyperparameters of an engine simulation (everything not in ``config.Config``).

    label_hop_s 0.5 rather than the labeler's 0.25: windows are 0.5 s, so 0.25 s hops add half-
    overlapping near-duplicates that double disk (32 KB per window), training and audit time for
    little information.  pop_minutes 4: ≈ 34 cued trials per population member, enough for that
    member's topography to register in the prior without one member dominating it.
    min_new_windows / min_heldout default to ``data_engine``'s values; tiny test runs lower them.
    audit_max_windows caps the per-day audit (evenly spaced, time order kept): the permutation CV is
    quadratic in n and 1,500 windows already resolve a floor of a few hundredths of cosine.
    eval_minutes 2.0 of auditory cued_blank per simulated day form the fixed evaluation pool (≈ 17
    trials a day; 5 days ≈ 85 trials, enough for a trial-level CI of a few hundredths).
    intent_angle_sd_deg / intent_lapse_prob 0: closed-loop intent equals the label (oracle labels);
    raise them to model intent that deviates from the straight cursor → goal line.
    heterogeneity_range: the population's shared/individual topography mix (sim_population).
    pop_ref_windows: the population prior's pseudo-count (``data_engine.personal_weight``).
    min_heldout_trials: the champion test's trial minimum (``ChampionChallenger``).
    """
    days: int = 5
    hours: float = 1.0
    pop: int = 20
    users: int = 1
    seed: int = DEFAULT.eval.seed
    pop_minutes: float = 4.0
    assist_gain: float = 0.5
    label_hop_s: float = 0.5
    min_new_windows: int = 200
    min_heldout: int = 40
    audit_max_windows: int = 1500
    audit_permutations: int = 200
    ocular_leak_fraction: float = 0.0
    user_learning_rate: float = 0.0
    eval_minutes: float = 2.0
    intent_angle_sd_deg: float = 0.0
    intent_lapse_prob: float = 0.0
    heterogeneity_range: Tuple[float, float] = HETEROGENEITY_RANGE
    pop_ref_windows: float = 300.0
    min_heldout_trials: int = 8

    def __post_init__(self) -> None:
        if self.days < 1 or self.hours <= 0 or self.users < 1 or self.pop < 0:
            raise ValueError(f"invalid simulation size: {self}")
        if self.eval_minutes * 60.0 < CuedTrialTiming().trial_s:
            raise ValueError(f"eval_minutes {self.eval_minutes} holds no complete cued trial")
        if self.intent_angle_sd_deg < 0 or not 0.0 <= self.intent_lapse_prob < 1.0:
            raise ValueError(f"invalid intent mismatch: {self}")


# ----------------------------------------------------------------------------------------------
# Plans
# ----------------------------------------------------------------------------------------------
def day_blocks(day: int, hours: float, seed: int) -> List[Block]:
    """The day's blocks: ``ceil(hours)`` protocol sessions, each scaled by ``hours / n_sessions`` ≤ 1.

    Block durations are scaled by one factor, so the protocol's shares (cued floor, probe share,
    free-use share) are preserved exactly; block seeds are kept, so a scaled plan replays the same
    cue order and targets.  Session s uses plan seed ``seed + s`` so a second session of the day is a
    different (still valid) arrangement rather than a repeat."""
    n_sessions = max(1, math.ceil(hours / SESSION_HOURS - 1e-9))
    scale = hours / (n_sessions * SESSION_HOURS)
    out: List[Block] = []
    for s in range(n_sessions):
        plan = session_plan(day, seed + s)
        out += [Block(b.kind, b.duration_s * scale, cue=b.cue, feedback=b.feedback, seed=b.seed) for b in plan]
    return out


# ----------------------------------------------------------------------------------------------
# Online path (offline-paced) and the auditor log
# ----------------------------------------------------------------------------------------------
class OnlinePreprocessor:
    """The per-update chain of ``realtime.RealTimeLoop`` without its wall-clock scheduler: stateful
    ``CausalFilter`` on every chunk, then at each control step ``AmplitudeGate`` on the latest window
    and CAR over the good channels — the window the labeler will later cut from the same raw log."""

    def __init__(self, cfg: Config):
        self.filt = CausalFilter(cfg.hardware, cfg.pre)
        self.gate = AmplitudeGate(cfg.hardware, cfg.pre)
        self.n_win = int(round(cfg.decoder.window_s * cfg.hardware.fs))
        self.buf = np.zeros((self.n_win, cfg.hardware.n_channels), np.float32)

    def push(self, x: np.ndarray) -> None:
        y = self.filt(x).astype(np.float32)
        if len(y) >= self.n_win:
            self.buf = y[-self.n_win:]
        else:
            self.buf = np.concatenate([self.buf[len(y):], y])

    def window(self) -> np.ndarray:
        good = self.gate.update(self.buf)
        return common_average(self.buf, good).astype(np.float32)


class GazeLog:
    """Simulated eye tracker: one row (t_start, t_end, gx, gy) per emitted EEG chunk (gaze is
    constant within a chunk).  Written beside the EEG; only ``window_gaze`` → ``PurityAudit`` reads it."""

    def __init__(self, run_dir: str):
        self._f = open(os.path.join(run_dir, GAZE_FILE), "ab")

    def write(self, t0: float, t1: float, gaze: np.ndarray) -> None:
        np.array([t0, t1, gaze[0], gaze[1]], np.float64).tofile(self._f)

    def close(self) -> None:
        self._f.close()


def read_gaze(run_dir: str) -> np.ndarray:
    path = os.path.join(run_dir, GAZE_FILE)
    return np.fromfile(path, dtype=np.float64).reshape(-1, 4) if os.path.exists(path) else np.zeros((0, 4))


def window_gaze(rows: np.ndarray, t_end: float, window_s: float) -> np.ndarray:
    """Time-weighted mean gaze over [t_end − window_s, t_end] (the window's span)."""
    lo = t_end - window_s
    overlap = np.clip(np.minimum(rows[:, 1], t_end) - np.maximum(rows[:, 0], lo), 0.0, None)
    tot = overlap.sum()
    return (overlap @ rows[:, 2:4]) / tot if tot > 0 else np.zeros(2)


# ----------------------------------------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------------------------------------
class DayRunner:
    """Runs one subject's labelled blocks for one day on the simulated clock.

    The clock is ``t0 + samples_emitted / fs`` (no float accumulation); it is the clock of every
    EEG timestamp, event and cursor row, exactly as ``RunLogger``'s single-clock contract requires.
    """

    def __init__(self, subject: SubjectEEG, cfg: Config, sim: SimConfig, decoder, t0: float, person_seed: int):
        self.subject, self.cfg, self.sim, self.decoder = subject, cfg, sim, decoder
        self.fs = cfg.hardware.fs
        self.hop = int(round(self.fs / cfg.task.control_hz))
        self.t0, self.n = t0, 0
        self.pre = OnlinePreprocessor(cfg)
        self.task_cfg = dataclasses.replace(cfg.task, assist_gain=sim.assist_gain)
        self.zipf = ZipfTargets(seed=person_seed)      # a person's frequent targets stay put across days
        self.timing = CuedTrialTiming()
        self._logger: Optional[RunLogger] = None
        self._gaze: Optional[GazeLog] = None

    @property
    def t(self) -> float:
        return self.t0 + self.n / self.fs

    def advance(self, seconds: float) -> None:
        """Clock time with no recording (setup, rests)."""
        self.n += int(round(seconds * self.fs))

    def emit(self, seconds: float = 0.0, samples: Optional[int] = None) -> None:
        n = int(round(seconds * self.fs)) if samples is None else samples
        if n <= 0:
            return
        t_start = self.t
        x = self.subject.generate(n)
        self._logger.eeg(x, t_start + np.arange(n) / self.fs)
        self.n += n
        self._gaze.write(t_start, self.t, self.subject.gaze)
        self.pre.push(x)

    def decode(self) -> np.ndarray:
        """The deployed decoder on the latest preprocessed EEG window; zero when nothing is deployed."""
        window = self.pre.window()
        return np.zeros(2) if self.decoder is None else np.asarray(self.decoder.predict(window), float)

    def run_block(self, block: Block, root: str, tag: str, day: int, index: int) -> Tuple[str, List[TrialOutcome]]:
        """One block as one RunLogger run; returns (run_dir, closed-loop trial outcomes)."""
        self.subject.begin_block(day, index)
        self.subject.set_intent(0.0, 0.0)
        self.subject.set_gaze(0.0, 0.0)
        self._logger = RunLogger(root, self.cfg, tag=tag, clock=lambda: self.t)
        self._gaze = GazeLog(self._logger.dir)
        try:
            self._logger.event("block_start", {"kind": block.kind, "cue": block.cue, "feedback": block.feedback,
                                               "duration_s": block.duration_s, "seed": block.seed}, t=self.t)
            self.emit(PREROLL_S)
            outcomes = self._cued(block) if block.kind == "cued_blank" else self._closed(block)
            self._logger.event("block_end", {"kind": block.kind}, t=self.t)
        finally:
            self._logger.close()
            self._gaze.close()
        return self._logger.dir, outcomes

    def _cued(self, block: Block) -> List[TrialOutcome]:
        """Cue (no intent) → cue_offset → settle → trial_start → intent → trial_end → ITI, per
        ``CuedTrialTiming``: the labelled span starts after the cue stimulus has ended and its evoked
        response has settled, and gaze never leaves fixation."""
        sched = CueScheduler(seed=block.seed, logger=self._logger)
        tm = self.timing
        for trial in range(tm.n_trials(block)):
            cue = sched.next(t=self.t, mode=block.cue)
            self.emit(tm.cue_s)
            self._logger.event(EVENT_CUE_OFFSET, {"seq": cue.seq}, t=self.t)   # stimulus removed
            self.emit(tm.settle_s)
            self._logger.event("trial_start", trial_start_payload(block, trial, cued_goal(cue)), t=self.t)
            self.subject.set_intent(*cue.vector)
            self.emit(tm.intent_s)
            self._logger.event("trial_end", {"trial": trial, "outcome": "cued"}, t=self.t)
            self.subject.set_intent(0.0, 0.0)
            self.emit(tm.iti_s)
        return []

    def _closed(self, block: Block) -> List[TrialOutcome]:
        """Closed-loop trials until the block's time is used (the last trial may run past it)."""
        rng = np.random.default_rng(block.seed)
        mismatch = Mismatch(self.sim, np.random.default_rng([block.seed, _MISMATCH_STREAM]), self.fs)
        runner = TrialRunner(self.task_cfg, rng)
        free = block.kind == "free_use"
        visible = block.feedback_visible
        outcomes: List[TrialOutcome] = []
        start, trial = self.t, 0
        while self.t - start < block.duration_s:
            if free:
                targets, width = self.zipf.locations, self.task_cfg.target_radius
                goal_idx = int(rng.choice(len(targets), p=self.zipf.probabilities))
            else:
                distance, width = FITTS_CONDITIONS[trial % len(FITTS_CONDITIONS)]
                targets = fitts_targets(self.task_cfg.n_targets, distance, width, rng)
                goal_idx = int(rng.integers(len(targets)))
            goal = targets[goal_idx]
            self._logger.event("trial_start", trial_start_payload(block, trial, None if free else goal), t=self.t)
            self.subject.set_gaze(*goal)                     # targets are on screen: the eyes go to the goal
            start_pos = np.zeros(2)                          # TrialRunner resets the cursor to the centre

            def command(state, _t, goal=goal):
                ref = state.pos if visible else start_pos    # hidden cursor: only its start is known
                d = goal - ref
                self.subject.set_intent(*mismatch.apply(d / (np.linalg.norm(d) + 1e-9), self.n))
                self.emit(samples=self.hop)
                return self.decode()

            def on_step(state, u, p, inside, dwell):
                self._logger.cursor(self.t, state.pos, u, float(np.max(p)), inside, dwell)

            out = runner.run(targets, width, goal_idx, command, on_step=on_step)
            outcome = "hit" if out.success else ("miss" if out.selected_idx is not None else "timeout")
            end = {"trial": trial, "outcome": outcome, "selected": out.selected_idx,
                   "movement_time_s": out.movement_time_s}
            if free:
                # Hindsight label: a selection the person keeps reveals the goal.  A wrong selection is
                # undone (the person knows what they wanted), and an undone selection reveals nothing,
                # so it must not label the trial toward the target that was wrongly hit.
                if out.selected_idx is not None and not out.success:
                    self._logger.event("undo", {"trial": trial, "selected": out.selected_idx}, t=self.t)
                end["goal"] = targets[out.selected_idx].tolist() if out.success else None
            self._logger.event("trial_end", end, t=self.t)
            outcomes.append(out)
            self.subject.set_intent(0.0, 0.0)
            self.subject.set_gaze(0.0, 0.0)
            self.emit(self.task_cfg.inter_trial_s)
            trial += 1
        self.subject.record_practice(self.t - start)
        return outcomes



class Mismatch:
    """Intent that deviates from the labeler's straight cursor → goal line.

    Every ``INTENT_SEGMENT_S`` of simulated time a new angular error ~ N(0, intent_angle_sd_deg) and a
    lapse (no intent at all, probability ``intent_lapse_prob``) are drawn and held: real people aim
    imprecisely, correct along curved paths and drift off task, so their intent is not the oracle
    vector the label assumes.  With both knobs at 0 the intent is the label direction unchanged."""

    def __init__(self, sim: SimConfig, rng: np.random.Generator, fs: int):
        self.sd = math.radians(sim.intent_angle_sd_deg)
        self.lapse_p = sim.intent_lapse_prob
        self.rng, self.seg = rng, int(round(INTENT_SEGMENT_S * fs))
        self.active = self.sd > 0 or self.lapse_p > 0
        self._segment, self._rot, self._gain = None, np.eye(2), 1.0

    def apply(self, direction: np.ndarray, sample: int) -> np.ndarray:
        if not self.active:
            return direction
        segment = sample // self.seg
        if segment != self._segment:
            a = self.rng.normal(0.0, self.sd) if self.sd > 0 else 0.0
            self._rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            self._gain = 0.0 if self.rng.random() < self.lapse_p else 1.0
            self._segment = segment
        return self._gain * (self._rot @ direction)


# ----------------------------------------------------------------------------------------------
# Population prior
# ----------------------------------------------------------------------------------------------
def build_population_prior(cfg: Config, sim: SimConfig, pop_sim: PopulationSim, members: Sequence[int],
                           root: str) -> str:
    """Cued calibration (``sim.pop_minutes`` of cued_blank, day 1) for each population member, run
    through the same logger → store → labeler path as the target user, so prior windows are
    preprocessed bit-identically.  Cued windows only: they are the members' cleanest labels, and a
    prior built from closed-loop data would import other people's display-driven label structure.
    Windows are cached per member as .npy and memory-mapped by ``load_population_prior``, so a prior
    of thousands of 32 KB windows never has to sit in RAM (and worker processes share the page cache).
    """
    runs = os.path.join(root, "runs")
    os.makedirs(runs, exist_ok=True)
    store = SessionStore(runs, cfg.hardware.n_channels)
    labeler = Labeler(cfg, hop_s=sim.label_hop_s)
    for j in members:
        p = pop_sim.params[j]
        subject = pop_sim.subject(j)
        subject.set_day(1)
        runner = DayRunner(subject, cfg, sim, None, CLOCK_ORIGIN_S, p.seed)
        block = Block("cued_blank", sim.pop_minutes * 60.0, cue="auditory", seed=p.seed)
        run_dir, _ = runner.run_block(block, runs, f"pop{j:03d}", day=1, index=0)
        ls = labeler.label(store.load(run_dir))
        out = os.path.join(root, f"member_{j:03d}")
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, "windows.npy"), np.stack(ls.windows))
        np.save(os.path.join(out, "labels.npy"), ls.labels)
        shutil.rmtree(run_dir)
    return root


def load_population_prior(root: str) -> List[Tuple[List[np.ndarray], np.ndarray]]:
    """[(windows, labels)] per member, windows memory-mapped — the ``Trainer`` population format."""
    out = []
    for name in sorted(n for n in os.listdir(root) if n.startswith("member_")):
        win = np.load(os.path.join(root, name, "windows.npy"), mmap_mode="r")
        out.append(([win[i] for i in range(len(win))], np.load(os.path.join(root, name, "labels.npy"))))
    return out


def fit_prior_decoder(cfg: Config, population: Sequence[Tuple[Sequence[np.ndarray], np.ndarray]]) -> VelocityDecoder:
    """Population-only decoder deployed before the person has a champion (zero-calibration start)."""
    windows = [w for ws, _ in population for w in ws]
    labels = np.concatenate([y for _, y in population])
    return VelocityDecoder(cfg.decoder, cfg.hardware.fs).fit(windows, labels)


# ----------------------------------------------------------------------------------------------
# Day-level measurements
# ----------------------------------------------------------------------------------------------
def build_eval_pool(cfg: Config, sim: SimConfig, pop_sim: PopulationSim, user: int, root: str) -> LabelledSet:
    """The fixed evaluation pool of target ``user``: one auditory cued_blank block of
    ``sim.eval_minutes`` per simulated day, recorded under that day's electrode gains with its own
    noise stream (``EVAL_BLOCK_INDEX``), labelled by the same logger → store → labeler path, and never
    added to the engine's store.  A fresh subject instance, so the main run's state is untouched;
    recorded at zero practice, so with ``user_learning_rate`` > 0 it measures the decoder, not the
    user's learning."""
    runs = os.path.join(root, "eval_runs")
    os.makedirs(runs, exist_ok=True)
    store = SessionStore(runs, cfg.hardware.n_channels)
    labeler = Labeler(cfg, hop_s=sim.label_hop_s)
    params = pop_sim.params[user]
    sets = []
    for day in range(1, sim.days + 1):
        subject = pop_sim.subject(user)
        subject.set_day(day)
        runner = DayRunner(subject, cfg, sim, None, CLOCK_ORIGIN_S + (day - 1) * DAY_OFFSET_S, params.seed)
        block = Block("cued_blank", sim.eval_minutes * 60.0, cue="auditory", seed=params.seed + EVAL_BLOCK_INDEX + day)
        run_dir, _ = runner.run_block(block, runs, f"eval_u{user}_d{day:02d}", day=day, index=EVAL_BLOCK_INDEX)
        sets.append(labeler.label(store.load(run_dir)))
        shutil.rmtree(run_dir)
    return LabelledSet.concat(sets)


def score_decoder(decoder, data: LabelledSet, rng: Optional[np.random.Generator] = None) -> dict:
    """Mean cosine (with its trial-cluster 95 % CI when ``rng`` is given) and mean per-axis Pearson r
    of ``decoder`` on ``data``, the constant-output ceiling of ``data``, and n.  No decoder = zero
    output, whose cosine and r are 0 by definition (``defined_zero``)."""
    if len(data) == 0:
        return {"cosine": float("nan"), "r": float("nan"), "n": 0}
    pred = np.zeros_like(data.labels) if decoder is None else decoder.predict_many(data.windows)
    r = float(np.mean([pearson_r(pred[:, k], data.labels[:, k]) for k in range(2)]))
    out = {"cosine": mean_cosine(pred, data.labels), "r": r, "n": len(data),
           "constant_ceiling": constant_ceiling(data.labels), "defined_zero": decoder is None}
    if rng is not None:
        cl = trial_clusters(data.provenance)
        sums, counts = np.bincount(cl, weights=cosines(pred, data.labels)), np.bincount(cl).astype(float)
        idx = rng.integers(0, len(sums), (EVAL_BOOTSTRAP, len(sums)))
        boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
        out.update(ci95=[float(v) for v in np.percentile(boot, [2.5, 97.5])], n_trials=int(len(sums)))
    return out


def trial_metrics(outcomes: Sequence[TrialOutcome], fitts: bool) -> dict:
    """Success, time-to-target and (Fitts blocks only: free-use targets have no fixed D/W) throughput."""
    if not outcomes:
        return {"n_trials": 0, "success": float("nan"), "time_to_target_s": float("nan"),
                "throughput_bps": float("nan")}
    return {"n_trials": len(outcomes), "success": success_rate(outcomes),
            "time_to_target_s": time_to_target(outcomes),
            "throughput_bps": fitts_throughput(outcomes) if fitts else float("nan")}


def audit_day(audit: PurityAudit, data: LabelledSet, run_dirs: Sequence[str], cfg: Config, sim: SimConfig) -> dict:
    """PurityAudit on the day's windows (evenly subsampled to ``audit_max_windows``, time order kept),
    with the displayed cursor logs and the simulated eye tracker as the auditor stream."""
    if len(data) < 2 * audit.n_splits:
        return {"skipped": f"{len(data)} windows", "flags": [], "whole_set_flags": []}
    idx = np.unique(np.round(np.linspace(0, len(data) - 1, min(len(data), sim.audit_max_windows))).astype(int))
    sub = data.take(idx)
    dirs = {os.path.basename(os.path.normpath(d)): d for d in run_dirs}
    cursor = {sid: SessionStore.read_cursor(os.path.join(d, "cursor.csv")) for sid, d in dirs.items()}
    gaze_rows = {sid: read_gaze(d) for sid, d in dirs.items()}
    gaze = np.stack([window_gaze(gaze_rows[p["session"]], float(p["t"]), cfg.decoder.window_s)
                     for p in sub.provenance])
    rep = audit.run(sub.windows, sub.labels, sub.provenance, auditors={"eyetracker": gaze}, cursor_logs=cursor)
    scores = {k: ({"skipped": v["skipped"]} if "skipped" in v else
                  {"score": v["score"], "floor": v["floor"], "beats_floor": v["beats_floor"]})
              for k, v in rep["reports"].items()}
    cert = rep["certification"]
    cert_scores = ({"skipped": cert["skipped"]} if "skipped" in cert else
                   {k: ({"skipped": v["skipped"]} if "skipped" in v else
                        {"score": v["score"], "floor": v["floor"], "beats_floor": v["beats_floor"]})
                    for k, v in cert["reports"].items()})
    return {"flags": rep["verdict"], "whole_set_flags": rep["flags"], "certification_scores": cert_scores,
            "n": rep["n"], "scores": scores, "feedback": rep["feedback"],
            "by_block": rep["by_block"], "untested": rep["untested"]}


# ----------------------------------------------------------------------------------------------
# One person, one condition
# ----------------------------------------------------------------------------------------------
def make_population(sim: SimConfig, cfg: Config) -> PopulationSim:
    """Subjects 0 … users−1 are the target users; users … users+pop−1 the prior's members."""
    return PopulationSim(sim.users + sim.pop, sim.seed, cfg.hardware, heterogeneity_range=sim.heterogeneity_range,
                         ocular_leak_fraction=sim.ocular_leak_fraction, user_learning_rate=sim.user_learning_rate)


def simulate_user(cfg: Config, sim: SimConfig, user: int, condition: str, prior_dir: Optional[str],
                  root: str, keep_runs: bool = False) -> dict:
    """``sim.days`` days of one target user in one condition; returns the per-day rows."""
    pop_sim = make_population(sim, cfg)
    params = pop_sim.params[user]
    subject = pop_sim.subject(user)
    population = load_population_prior(prior_dir) if condition == "population" else None
    prior = fit_prior_decoder(cfg, population) if population else None
    eval_pool = build_eval_pool(cfg, sim, pop_sim, user, root)
    runs = os.path.join(root, "runs")
    os.makedirs(runs, exist_ok=True)
    engine = DataEngine(SessionStore(runs, cfg.hardware.n_channels), Labeler(cfg, hop_s=sim.label_hop_s),
                        Trainer(cfg, pop_ref_windows=sim.pop_ref_windows), ModelRegistry(os.path.join(root, "registry")),
                        ChampionChallenger(min_heldout=sim.min_heldout, min_heldout_trials=sim.min_heldout_trials,
                                           seed=sim.seed),
                        population=population, min_new_windows=sim.min_new_windows)
    audit = PurityAudit(cfg, n_permutations=sim.audit_permutations)
    rows, task_s = [], 0.0
    for day in range(1, sim.days + 1):
        subject.set_day(day)
        if engine.champion is not None:
            deployed, name = engine.champion, f"v{engine.champion_version}"
        else:
            deployed, name = prior, ("population prior" if prior is not None else "none")
        runner = DayRunner(subject, cfg, sim, deployed, CLOCK_ORIGIN_S + (day - 1) * DAY_OFFSET_S, params.seed)
        run_dirs: List[str] = []
        outcomes: Dict[str, List[TrialOutcome]] = {"closed_loop": [], "blank_feedback": [], "free_use": []}
        for i, block in enumerate(day_blocks(day, sim.hours, params.seed)):
            if block.provenance_block is None:
                runner.advance(block.duration_s)
                continue
            run_dir, outs = runner.run_block(block, runs, f"u{user}_d{day:02d}_b{i:02d}_{block.kind}", day, i)
            run_dirs.append(run_dir)
            outcomes.get(block.kind, []).extend(outs)
            task_s += block.duration_s
        n_new = sum(engine.ingest(d) for d in run_dirs)
        ids = {os.path.basename(os.path.normpath(d)) for d in run_dirs}
        data = engine.data
        today = data.mask(np.array([p["session"] in ids for p in data.provenance], bool))
        cued = today.mask(np.array([p["block"] == "cued_blank" for p in today.provenance], bool))
        score = score_decoder(deployed, eval_pool, np.random.default_rng([sim.seed, user, day]))
        today_score = score_decoder(deployed, cued)
        verdict = engine.maybe_retrain()
        purity = audit_day(audit, today, run_dirs, cfg, sim)
        if not keep_runs:
            for d in run_dirs:
                shutil.rmtree(d)
        rows.append({
            "day": day, "cum_hours": day * sim.hours, "cum_task_hours": task_s / 3600.0,
            "deployed": name, "decoder_cosine": score["cosine"], "decoder_r": score["r"],
            "decoder_cosine_ci95": score["ci95"], "decoder_constant_ceiling": score["constant_ceiling"],
            "decoder_score_defined_zero": score["defined_zero"],
            "n_eval_windows": score["n"], "n_eval_trials": score["n_trials"],
            "today": {k: today_score.get(k) for k in ("cosine", "r", "n", "constant_ceiling")},
            "closed_loop": trial_metrics(outcomes["closed_loop"], fitts=True),
            "blank_feedback": trial_metrics(outcomes["blank_feedback"], fitts=True),
            "free_use": trial_metrics(outcomes["free_use"], fitts=False),
            "n_windows_day": n_new, "n_windows_total": len(data),
            "retrain": {k: verdict.get(k) for k in ("promoted", "reason", "challenger_cosine", "champion_cosine",
                                                    "ci95", "floor", "p_value", "n_heldout", "n_heldout_trials",
                                                    "n_fresh", "n_train", "version") if k in verdict},
            "champion_version": engine.champion_version, "purity": purity,
            "modulation_depth": subject.modulation_depth,
        })
    return {"user": user, "condition": condition, "subject": params.describe(), "days": rows}


def _job(args: tuple) -> dict:
    """Process-pool entry point.  BLAS is pinned to one thread: jobs already fill the CPUs, and the
    matrices here are small enough that threading only adds contention."""
    cfg, sim, user, condition, prior_dir, keep_dir = args
    with threadpool_limits(limits=1):
        if keep_dir:
            root = os.path.join(keep_dir, f"{condition}_u{user}")
            os.makedirs(root, exist_ok=True)
            return simulate_user(cfg, sim, user, condition, prior_dir, root, keep_runs=True)
        with tempfile.TemporaryDirectory(prefix="nc_engine_") as root:
            return simulate_user(cfg, sim, user, condition, prior_dir, root)


# ----------------------------------------------------------------------------------------------
# Whole simulation, aggregation, report
# ----------------------------------------------------------------------------------------------
_MEAN_FIELDS = (("decoder_cosine",), ("decoder_r",), ("decoder_constant_ceiling",), ("n_eval_windows",),
                ("closed_loop", "success"), ("closed_loop", "time_to_target_s"),
                ("closed_loop", "throughput_bps"), ("blank_feedback", "success"), ("free_use", "success"),
                ("purity", "scores", "full", "score"), ("purity", "scores", "full", "floor"),
                ("purity", "certification_scores", "full", "score"), ("purity", "certification_scores", "full", "floor"),
                ("purity", "certification_scores", "slow", "score"), ("purity", "certification_scores", "frontal", "score"))


def _get(d: dict, path: Tuple[str, ...]):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return float("nan")
        d = d[k]
    return float("nan") if d is None else float(d)


def _nanmean(values: Sequence[float]) -> float:
    v = np.asarray(values, float)
    return float(np.mean(v[np.isfinite(v)])) if np.isfinite(v).any() else float("nan")


def aggregate(per_user: Sequence[dict], days: int) -> List[dict]:
    """Per day: the mean of every endpoint across target users (NaN-aware), promotions and flag counts."""
    out = []
    for d in range(days):
        rows = [u["days"][d] for u in per_user]
        agg = {"day": rows[0]["day"], "cum_hours": rows[0]["cum_hours"], "n_users": len(rows),
               "deployed": sorted({r["deployed"] for r in rows}),
               "decoder_score_defined_zero": all(r["decoder_score_defined_zero"] for r in rows),
               "decoder_cosine_ci95": rows[0]["decoder_cosine_ci95"] if len(rows) == 1 else None,
               "promoted": int(sum(bool(r["retrain"].get("promoted")) for r in rows))}
        for path in _MEAN_FIELDS:
            agg[".".join(path)] = _nanmean([_get(r, path) for r in rows])
        flags: Dict[str, int] = {}
        for r in rows:
            for f in r["purity"].get("flags", []):
                flags[f] = flags.get(f, 0) + 1
        agg["purity_flags"] = flags
        out.append(agg)
    return out


def simulate(sim: SimConfig, cfg: Config = DEFAULT, workers: int = 1, keep_dir: Optional[str] = None) -> dict:
    """Run every (user, condition) job and aggregate.  ``workers`` > 1 runs jobs in spawned
    processes (independent jobs; each writes only its own directory)."""
    t0 = time.perf_counter()
    conditions = ["own"] + (["population"] if sim.pop > 0 else [])
    pop_sim = make_population(sim, cfg)
    with tempfile.TemporaryDirectory(prefix="nc_prior_") as tmp:
        prior_dir = None
        if sim.pop > 0:
            prior_dir = build_population_prior(cfg, sim, pop_sim, range(sim.users, sim.users + sim.pop),
                                               os.path.join(keep_dir or tmp, "population"))
        jobs = [(cfg, sim, u, c, prior_dir, keep_dir) for u in range(sim.users) for c in conditions]
        if workers <= 1 or len(jobs) == 1:
            results = [_job(j) for j in jobs]
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
                results = list(ex.map(_job, jobs))
    by_cond = {c: [r for r in results if r["condition"] == c] for c in conditions}
    return {
        "config": dataclasses.asdict(sim),
        "targets": pop_sim.describe(range(sim.users)),
        "population": pop_sim.describe(range(sim.users, sim.users + sim.pop)),
        "conditions": {c: {"mean_by_day": aggregate(rs, sim.days), "per_user": rs} for c, rs in by_cond.items()},
        "runtime_s": time.perf_counter() - t0,
    }


def _fmt(x: float, nd: int = 3) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def to_markdown(res: dict) -> str:
    c = res["config"]
    oracle = c["intent_angle_sd_deg"] == 0 and c["intent_lapse_prob"] == 0
    lines = [f"## Data-engine learning curve (simulated): {c['days']} days × {c['hours']:g} h, "
             f"{c['users']} target user(s), population prior N = {c['pop']}",
             "",
             f"seed {c['seed']}; assistance a = {c['assist_gain']}; ocular-leak fraction {c['ocular_leak_fraction']}; "
             f"user learning rate {c['user_learning_rate']} (0 = users do not learn); topography heterogeneity "
             f"h ~ U{tuple(c['heterogeneity_range'])}; prior pseudo-count {c['pop_ref_windows']:g} windows; "
             f"intent mismatch {c['intent_angle_sd_deg']:g}° sd, lapse p {c['intent_lapse_prob']:g}; "
             f"runtime {res['runtime_s']:.0f} s.",
             "decoder cosine [95 % trial CI] / r: the decoder deployed that day, on a FIXED evaluation pool "
             f"({c['eval_minutes']:g} min of auditory cued_blank per simulated day, never trained on), so every day "
             "meets the same windows.  ceiling: the best mean cosine a constant output reaches on that pool "
             "(cosine is not chance-corrected; day-to-day gain drift also adds an offset that lowers cosine "
             "but not r).  cued_blank audit cosine: out-of-fold cosine of a decoder refit on the day's blank-cue "
             "windows only (nothing spatial on screen, the person fixating), with its permutation floor (p95, "
             "circular shifts within session); its verdict is the purity statement of record.  Cursor-visible "
             "windows are display-identifiable by design and train the decoder but never certify it.  "
             "Success / TTT / TP: closed-loop Fitts trials.",
             "Caveats: '0 (def.)' = nothing deployed, so 0 by definition, not a measurement.  "
             + ("Closed-loop intent equals the label here (oracle closed-loop labels): gains from closed-loop "
                "data are guaranteed by construction, not tested.  " if oracle else "")
             + "The population prior's benefit is set by the assumed heterogeneity range; quote it as a sweep.",
             "",
             "| condition | day | cum. h | deployed | decoder cosine [95 % CI] | ceiling | decoder r | n eval "
             "| CL success | TTT (s) | TP (bits/s) | hidden-cursor success | cued_blank audit cosine (floor) | promoted | purity verdict (cued_blank) |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for cond, block in res["conditions"].items():
        for a in block["mean_by_day"]:
            flags = ", ".join(f"{k}×{v}" if v > 1 else k for k, v in sorted(a["purity_flags"].items())) or "–"
            ci = a["decoder_cosine_ci95"]
            cos = ("0 (def.)" if a["decoder_score_defined_zero"] else
                   _fmt(a["decoder_cosine"]) + (f" [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else ""))
            lines.append(
                f"| {cond} | {a['day']} | {a['cum_hours']:g} | {' / '.join(a['deployed'])} "
                f"| {cos} | {_fmt(a['decoder_constant_ceiling'])} "
                f"| {'0 (def.)' if a['decoder_score_defined_zero'] else _fmt(a['decoder_r'])} "
                f"| {_fmt(a['n_eval_windows'], 0)} | {_fmt(a['closed_loop.success'], 2)} "
                f"| {_fmt(a['closed_loop.time_to_target_s'], 1)} | {_fmt(a['closed_loop.throughput_bps'], 2)} "
                f"| {_fmt(a['blank_feedback.success'], 2)} "
                f"| {_fmt(a['purity.certification_scores.full.score'])} ({_fmt(a['purity.certification_scores.full.floor'])}) "
                f"| {a['promoted']}/{a['n_users']} | {flags} |")
    return "\n".join(lines)


def _clean(o):
    """JSON-safe copy: numpy scalars → Python, NaN/inf → null (strict JSON has no NaN)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def write_report(res: dict, prefix: str) -> Tuple[str, str]:
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    with open(prefix + ".json", "w") as f:
        json.dump(_clean(res), f, indent=2)
    with open(prefix + ".md", "w") as f:
        f.write(to_markdown(res) + "\n")
    return prefix + ".json", prefix + ".md"


def main(argv: Optional[Sequence[str]] = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=SimConfig.days)
    ap.add_argument("--hours", type=float, default=SimConfig.hours, help="hours of use per day")
    ap.add_argument("--pop", type=int, default=SimConfig.pop,
                    help="population-prior size N; 0 runs the own-data condition only")
    ap.add_argument("--users", type=int, default=SimConfig.users, help="target users (curves are averaged)")
    ap.add_argument("--seed", type=int, default=SimConfig.seed)
    ap.add_argument("--pop-minutes", type=float, default=SimConfig.pop_minutes)
    ap.add_argument("--ocular-leak", type=float, default=SimConfig.ocular_leak_fraction,
                    help="fraction of simulated subjects that leak gaze into frontal EEG")
    ap.add_argument("--learning-rate", type=float, default=SimConfig.user_learning_rate,
                    help="user modulation-depth gain per closed-loop hour (default 0: users do not learn)")
    ap.add_argument("--min-new-windows", type=int, default=SimConfig.min_new_windows)
    ap.add_argument("--min-heldout", type=int, default=SimConfig.min_heldout)
    ap.add_argument("--min-heldout-trials", type=int, default=SimConfig.min_heldout_trials)
    ap.add_argument("--eval-minutes", type=float, default=SimConfig.eval_minutes,
                    help="minutes of auditory cued_blank per simulated day in the fixed evaluation pool")
    ap.add_argument("--intent-angle-sd", type=float, default=SimConfig.intent_angle_sd_deg,
                    help="closed-loop intent error (deg sd, re-drawn every 1 s); 0 = oracle labels")
    ap.add_argument("--intent-lapse", type=float, default=SimConfig.intent_lapse_prob,
                    help="probability that a 1 s closed-loop segment carries no intent")
    ap.add_argument("--heterogeneity", type=float, nargs=2, default=SimConfig.heterogeneity_range,
                    metavar=("LO", "HI"), help="topography heterogeneity range h ~ U[LO, HI]")
    ap.add_argument("--pop-ref-windows", type=float, default=SimConfig.pop_ref_windows,
                    help="population-prior pseudo-count (its strength); --pop only changes its composition")
    ap.add_argument("--audit-permutations", type=int, default=SimConfig.audit_permutations)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--out", type=str, default="", help="output prefix for .json and .md "
                    "(default results/sim_engine_d<days>_h<hours>_pop<N>)")
    ap.add_argument("--keep-runs", type=str, default="", help="keep every run/registry under this directory")
    a = ap.parse_args(argv)
    sim = SimConfig(days=a.days, hours=a.hours, pop=a.pop, users=a.users, seed=a.seed, pop_minutes=a.pop_minutes,
                    ocular_leak_fraction=a.ocular_leak, user_learning_rate=a.learning_rate,
                    min_new_windows=a.min_new_windows, min_heldout=a.min_heldout,
                    audit_permutations=a.audit_permutations, min_heldout_trials=a.min_heldout_trials,
                    eval_minutes=a.eval_minutes, intent_angle_sd_deg=a.intent_angle_sd,
                    intent_lapse_prob=a.intent_lapse, heterogeneity_range=tuple(a.heterogeneity),
                    pop_ref_windows=a.pop_ref_windows)
    res = simulate(sim, DEFAULT, workers=a.workers, keep_dir=a.keep_runs or None)
    prefix = a.out or os.path.join("results", f"sim_engine_d{a.days}_h{a.hours:g}_pop{a.pop}")
    paths = write_report(res, prefix)
    print(to_markdown(res))
    print(f"\nwrote {paths[0]} and {paths[1]}")
    return res


if __name__ == "__main__":
    main()
