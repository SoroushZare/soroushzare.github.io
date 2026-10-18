"""A simulated population of EEG users, for asking "does the data engine scale across people?".

Why a population simulator
--------------------------
``acquisition.SimulatedEEG`` is one person with textbook topographies: alpha lateralisation on the
canonical parieto-occipital sites, beta desynchronisation on the canonical central sites.  Under that
model a decoder trained on anyone works on everyone, so a population prior looks free and personal
data looks unnecessary.  Real users differ: the cortical sources that carry intent sit in different
places for different people, dry-electrode gain wanders from day to day, some users leak gaze into
their frontal channels, and few users measurably "learn" the decoder within a week.  Each of those
differences changes what the data engine should do, so each is an explicit, seeded knob here:

``heterogeneity`` h ~ U[0.3, 0.7] per subject
    Intent topography = normalise(shared_pattern·(1 − h) + individual_pattern·h).  The shared part is
    what a population prior can learn from other people; the individual part is what only the user's
    own data can reveal.  The range keeps every subject between "mostly typical" and "mostly
    idiosyncratic" — neither the trivial nor the hopeless case.
per-day gain drift
    A per-channel multiplicative log-normal random walk across days (dry electrodes re-seat
    differently every session).  Day 1 has unit gain, so the first calibration is the reference.
optional ocular leak
    Gaze position adds a signed frontal potential (horizontal EOG F8 − F7, vertical FP1/FP2) for the
    subjects drawn as leaky.  Gaze is set separately from intent, because what the eyes do depends on
    the block: at fixation during cued_blank, on the target during closed loop.  That is exactly the
    distinction the purity audit must detect.
optional ``user_learning_rate`` (default 0)
    Modulation depth grows with closed-loop practice hours.  Off by default: assuming that users learn
    the decoder would credit the engine with improvements the person made, and the evidence for
    within-week non-invasive user learning is weak.

Individual patterns are drawn only from non-frontal channels: in this simulator the frontal channels
carry the ocular confound, and a cortical pattern placed there would make "frontal-only decodable"
mean "ocular leak" by construction, which would make the purity audit uninterpretable.

Every subject is a ``SimulatedEEG`` subclass, so it plugs into anything that takes the base simulator.
Subject ``i`` is a pure function of ``(seed, i)``: it is identical whether the population has 1 or 100
members, so "own data only" and "with population prior" conditions run on the very same person.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import lfilter

from .acquisition import SimulatedEEG
from .config import Hardware

# ----------------------------------------------------------------------------------------------
# Population hyperparameters (all explicit; see the module docstring for the reasoning)
# ----------------------------------------------------------------------------------------------
HETEROGENEITY_RANGE: Tuple[float, float] = (0.3, 0.7)
INTENT_GAIN_RANGE_UV: Tuple[float, float] = (4.0, 8.0)
"""Per-subject intent gain.  8 µV is the base simulator's full-lateralisation reference (r ≈ 0.3–0.4);
4 µV is its 'weak user'.  Drawing across that range gives the spread of decodability real cohorts show."""
FULL_DEPTH_GAIN_UV = 8.0          # intent gain at which the base simulator's alpha modulation is 100 %
DAY_DRIFT_SD = 0.10               # log-gain random-walk step per channel per day (≈ ±10 %/day)
EYE_GAIN_UV = 1.0                 # eye gain of a leaky subject: ≈ 20 µV frontal shift at full gaze
INDIVIDUAL_SUPPORT = 4            # channels in an individual pattern, like the 4–5 of the shared ones
ALPHA_UV = 20.0                   # posterior alpha carrier amplitude per unit topography weight (base sim)
BETA_UV = 10.0                    # central beta carrier amplitude (base sim)
AR_COEF, AR_GAIN = 0.9, 0.44      # background AR(1) of the base simulator, vectorised here

SHARED_CHANNELS: Dict[str, Tuple[str, ...]] = {
    "alpha_left": ("PO7", "PO3", "P7", "P3"),
    "alpha_right": ("PO8", "PO4", "P8", "P4"),
    "central": ("C3", "Cz", "C4", "CP1", "CP2"),
}
"""The canonical intent topographies of ``SimulatedEEG`` — the component every subject shares."""
NON_CORTICAL_SUPPORT: Tuple[str, ...] = ("FP1", "FP2", "AF3", "AF4", "F7", "F8", "F3", "Fz", "F4")
"""Excluded from individual patterns: these carry the simulated ocular confound (see docstring)."""

# Independent random streams per subject, so changing one aspect never reshuffles another.
_TOPO_STREAM, _DRIFT_STREAM, _NOISE_STREAM, _BLINK_STREAM = 1, 2, 3, 4


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def shared_topographies(hw: Hardware) -> Dict[str, np.ndarray]:
    """Unit-norm channel weight vectors of the shared (canonical) intent topographies."""
    idx = {c: k for k, c in enumerate(hw.channels)}
    out = {}
    for name, chans in SHARED_CHANNELS.items():
        v = np.zeros(hw.n_channels)
        v[[idx[c] for c in chans]] = 1.0
        out[name] = _unit(v)
    return out


def individual_topographies(hw: Hardware, rng: np.random.Generator,
                            support: int = INDIVIDUAL_SUPPORT) -> Dict[str, np.ndarray]:
    """Random non-negative unit patterns on ``support`` non-frontal channels, one per component.

    Non-negative because each pattern scales the *amplitude* of an oscillation: a negative weight
    would flip the phase, not the power, and band power cannot see the sign."""
    allowed = np.array([k for k, c in enumerate(hw.channels) if c not in NON_CORTICAL_SUPPORT])
    out = {}
    for name in SHARED_CHANNELS:
        v = np.zeros(hw.n_channels)
        pick = rng.choice(allowed, size=support, replace=False)
        v[pick] = 0.25 + np.abs(rng.standard_normal(support))      # no near-zero weight in the support
        out[name] = _unit(v)
    return out


def mix_topographies(shared: Dict[str, np.ndarray], individual: Dict[str, np.ndarray],
                     h: float) -> Dict[str, np.ndarray]:
    """normalise(shared·(1 − h) + individual·h): unit norm, so h changes *where* the signal is, never
    how strong it is (strength is ``intent_gain_uv``'s job)."""
    return {k: _unit(shared[k] * (1.0 - h) + individual[k] * h) for k in shared}


@dataclass(frozen=True)
class SubjectParams:
    """Everything that defines subject ``index`` of a population; JSON-friendly via ``describe``."""
    index: int
    seed: int
    heterogeneity: float
    intent_gain_uv: float
    ocular_leak: bool
    eye_gain_uv: float
    day_drift_sd: float
    user_learning_rate: float

    def describe(self) -> dict:
        return {"index": self.index, "seed": self.seed, "heterogeneity": round(self.heterogeneity, 4),
                "intent_gain_uv": round(self.intent_gain_uv, 4), "ocular_leak": self.ocular_leak,
                "eye_gain_uv": self.eye_gain_uv, "day_drift_sd": self.day_drift_sd,
                "user_learning_rate": self.user_learning_rate}


@dataclass
class SubjectEEG(SimulatedEEG):
    """One simulated person: ``SimulatedEEG`` with personal topographies, day gains, gaze and practice.

    Lifecycle used by the engine: ``set_day(d)`` at the start of a session, ``begin_block(d, b)`` at
    the start of every block, ``set_intent`` / ``set_gaze`` per control step, ``record_practice(s)``
    after closed-loop time.  ``begin_block`` restarts the noise stream, the background AR state and
    the oscillation phase, so a block's EEG depends only on (subject, day, block, intent and gaze
    sequence, practice) and not on how long earlier blocks ran: the "own data" and "population prior"
    conditions then record identical cued_blank blocks (with the default learning rate 0), and their
    decoders are compared on exactly the same calibration EEG.
    """
    heterogeneity: float = 0.5
    day_drift_sd: float = DAY_DRIFT_SD
    user_learning_rate: float = 0.0          # modulation-depth gain per closed-loop practice hour
    topographies: Dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.heterogeneity <= 1.0:
            raise ValueError(f"heterogeneity must be in [0, 1], got {self.heterogeneity}")
        self.shared = shared_topographies(self.hw)
        self.individual = individual_topographies(self.hw, np.random.default_rng([self.seed, _TOPO_STREAM]))
        self.topographies = mix_topographies(self.shared, self.individual, self.heterogeneity)
        self.gaze = np.zeros(2)
        self.practice_h = 0.0
        self.day = 1
        self.gain = np.ones(self.hw.n_channels)
        self._drift_rng = np.random.default_rng([self.seed, _DRIFT_STREAM])
        self._log_gain_steps: List[np.ndarray] = []
        self._ar_zi = np.zeros((1, self.hw.n_channels))
        self._phase_origin = 0
        self.rng = np.random.default_rng([self.seed, _NOISE_STREAM])
        self._reset_blinks(np.random.default_rng([self.seed, _BLINK_STREAM]))

    # -- state changes ---------------------------------------------------------------------------
    def day_gain(self, day: int) -> np.ndarray:
        """Per-channel gain on ``day``: exp of a random walk whose first step is between days 1 and 2.
        Steps are drawn lazily but always in day order, so the gain of day d never depends on which
        days were queried before."""
        if day < 1:
            raise ValueError(f"day must be >= 1, got {day}")
        while len(self._log_gain_steps) < day - 1:
            self._log_gain_steps.append(self._drift_rng.normal(0.0, self.day_drift_sd, self.hw.n_channels))
        if day == 1:
            return np.ones(self.hw.n_channels)
        return np.exp(np.sum(self._log_gain_steps[:day - 1], axis=0))

    def set_day(self, day: int) -> None:
        self.day = int(day)
        self.gain = self.day_gain(self.day)

    def begin_block(self, day: int, block: int) -> None:
        """Restart noise, blink schedule, AR background and oscillation phase for (day, block); drop
        pending evoked transients.  The zero AR state settles within ~10 samples (pole 0.9), far
        inside any preroll."""
        self.rng = np.random.default_rng([self.seed, _NOISE_STREAM, int(day), int(block)])
        self._reset_blinks(np.random.default_rng([self.seed, _BLINK_STREAM, int(day), int(block)]))
        self._ar_zi = np.zeros((1, self.hw.n_channels))
        self._phase_origin = self.n_sent
        self._pending = []

    def start(self) -> None:
        """As ``SimulatedEEG.start``; the oscillation phase origin is re-based with the sample count."""
        self._phase_origin -= self.n_sent
        super().start()

    def set_gaze(self, gx: float, gy: float) -> None:
        """Gaze position in normalised screen units (0, 0 = fixation).  Only a leaky subject's EEG
        depends on it; the gaze itself is what an eye tracker would record (an auditor stream)."""
        self.gaze = np.clip([gx, gy], -1, 1).astype(float)

    def record_practice(self, seconds: float) -> None:
        self.practice_h += max(0.0, float(seconds)) / 3600.0

    @property
    def modulation_depth(self) -> float:
        """Fractional alpha/beta modulation at full intent, in [0, 1]; grows with practice only when
        ``user_learning_rate`` > 0."""
        depth = self.intent_gain_uv / FULL_DEPTH_GAIN_UV * (1.0 + self.user_learning_rate * self.practice_h)
        return float(np.clip(depth, 0.0, 1.0))

    # -- signal ----------------------------------------------------------------------------------
    def generate(self, n: int) -> np.ndarray:
        """n samples (n, n_channels) µV: the base simulator's signal model on this subject's
        topographies, times the day's per-channel gain.  The AR(1) background is the base model's,
        run through ``lfilter`` with carried state instead of a per-sample Python loop (hours of
        simulated EEG per run make the loop the bottleneck).  Chunk-invariant like the base model."""
        fs, C = self.hw.fs, self.hw.n_channels
        t = (self.n_sent - self._phase_origin + np.arange(n)) / fs
        w = self.rng.standard_normal((n, C)) * self.noise_uv
        x, self._ar_zi = lfilter([AR_GAIN], [1.0, -AR_COEF], w, axis=0, zi=self._ar_zi)
        vx, vy = self.intent
        d = self.modulation_depth
        topo = self.topographies
        alpha = np.sin(2 * np.pi * 10 * t)[:, None]
        x += alpha * (ALPHA_UV * (1 - d * max(vx, 0.0)) * topo["alpha_right"][None]
                      + ALPHA_UV * (1 - d * max(-vx, 0.0)) * topo["alpha_left"][None])
        beta = np.sin(2 * np.pi * 20 * t)[:, None]
        x += beta * (BETA_UV * (1 - 0.5 * d * vy)) * topo["central"][None]
        if self.eye_gain_uv > 0:
            x += self.eye_gain_uv * 20.0 * (self.gaze[0] * self._h_eog[None] + self.gaze[1] * self._fp[None])
        x += self._blinks(n)
        x += self._evoked(n)
        if self.drift:
            x *= self._drift_gain(n)
        x *= self.gain[None]
        self.n_sent += n
        return x.astype(np.float32)


class PopulationSim:
    """``n_subjects`` simulated users drawn from one seeded population.

    Subject ``i``'s parameters come from ``default_rng([seed, i])`` alone, so subject 0 (the target
    user in ``simulate_engine``) is the same person whatever the population size.

    ocular_leak_fraction  probability that a subject leaks gaze into frontal EEG (default 0: the
                          clean case; raise it to test that the purity audit catches leakers).
    user_learning_rate    applied to every subject; default 0 (do not assume users learn).
    """

    def __init__(self, n_subjects: int, seed: int, hw: Hardware = Hardware(),
                 heterogeneity_range: Tuple[float, float] = HETEROGENEITY_RANGE,
                 intent_gain_range_uv: Tuple[float, float] = INTENT_GAIN_RANGE_UV,
                 day_drift_sd: float = DAY_DRIFT_SD, ocular_leak_fraction: float = 0.0,
                 eye_gain_uv: float = EYE_GAIN_UV, user_learning_rate: float = 0.0):
        if n_subjects < 1:
            raise ValueError(f"n_subjects must be >= 1, got {n_subjects}")
        lo, hi = heterogeneity_range
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError(f"heterogeneity_range must lie in [0, 1], got {heterogeneity_range}")
        self.hw = hw
        self.seed = int(seed)
        self.params: List[SubjectParams] = []
        for i in range(n_subjects):
            rng = np.random.default_rng([self.seed, i])
            # Draw every variable for every subject, in a fixed order, so a knob set to 0 does not
            # shift the draws of the others.
            sub_seed = int(rng.integers(0, 2**31 - 1))
            h = float(rng.uniform(lo, hi))
            gain = float(rng.uniform(*intent_gain_range_uv))
            leak = bool(rng.random() < ocular_leak_fraction)
            self.params.append(SubjectParams(i, sub_seed, h, gain, leak, eye_gain_uv if leak else 0.0,
                                             day_drift_sd, user_learning_rate))

    def __len__(self) -> int:
        return len(self.params)

    def __iter__(self) -> Iterator[SubjectEEG]:
        return (self.subject(i) for i in range(len(self)))

    def subject(self, i: int) -> SubjectEEG:
        """A fresh simulator (day 1, no practice) for subject ``i``."""
        p = self.params[i]
        return SubjectEEG(self.hw, seed=p.seed, intent_gain_uv=p.intent_gain_uv, eye_gain_uv=p.eye_gain_uv,
                          heterogeneity=p.heterogeneity, day_drift_sd=p.day_drift_sd,
                          user_learning_rate=p.user_learning_rate)

    def describe(self, indices: Optional[Sequence[int]] = None) -> List[dict]:
        idx = range(len(self)) if indices is None else indices
        return [self.params[i].describe() for i in idx]
