"""Simulated SMR body-part imagery with a switchable screen-leak pathway (INTERFACES_CHAMPION item J).

Why this simulator exists
-------------------------
The champion design decodes *body-part intent* (y = feet vs both hands, x = right vs left hand) and
certifies it against the display with held-out screen mappings and a deliberately leaky shadow
decoder.  That certification can only be rehearsed on a simulator in which (i) the motor signal lives
in body coordinates, (ii) a display-driven signal lives in *screen* coordinates, and (iii) the two can
be switched on and off independently.  Under the identity mapping they coincide (to move up, the person
imagines feet and looks up), so a simulator that only modelled one of them would let every decoder pass
or every decoder fail.  ``BodySubject`` provides both pathways on the existing ``SubjectEEG`` person
(background, blinks, evoked transients, day gain drift, ocular gaze leak and practice are inherited,
not re-implemented).

Motor pathway (body coordinates, always on)
-------------------------------------------
Three sensorimotor generators — left-hemisphere hand area (C3), right-hemisphere hand area (C4) and the
foot area at the vertex (Cz) — each carry a 10 Hz mu and a 20 Hz beta rhythm.  The rhythms are
narrow-band *stochastic* oscillators (AR(2) resonators driven by white noise), not sinusoids: a pure
sinusoid has an almost noiseless power in every window, which would make a 1-s band-power decoder
unrealistically good; a resonator's amplitude waxes and wanes like real mu, and that window-to-window
power variance is what limits real SMR decodability.  Imagery is an event-related desynchronisation
(ERD): a *decrease* of the generator's amplitude.

    FEET   → vertex generator                      (y = +1)
    HANDS  → both hand generators                   (y = −1)
    RIGHT  → C3 generator, ``IPSI_FRACTION`` at C4  (x = +1)   contralateral dominance
    LEFT   → C4 generator, ``IPSI_FRACTION`` at C3  (x = −1)

Diagonal (two-part) cues activate two classes at once.  An x *non-responder* produces the same total
hand ERD for LEFT and RIGHT but with no hemispheric asymmetry, so its x axis carries no information
while its y axis is untouched; ``BodyPopulation`` draws responders with probability 0.2 (the skeptic
correction puts the plausible dry, blind earner rate at 10–25 %, not the proposal's 30–45 %).

The ERD follows a *transient* time course after cue onset (``erd_envelope``): it rises to its peak at
``erd_peak_s`` (1.5 s) and decays toward a sustained fraction ``erd_sustain`` with time constant
``erd_decay_s``.  The defaults reproduce the ds004362 FEET/FISTS time course of 1-s window d′ (AUC
0.554 / 0.667 / 0.629 / 0.610 at 0–1 / 1–2 / 2–3 / 3–4 s): the mean envelope over those windows is
0.32 : 1 : 0.79 : 0.66 of its 1–2 s value, against d′ ratios of 0.31 : 1 : 0.76 : 0.65.  A
sustained-control claim therefore has to survive the decay, exactly as the skeptic corrections require.
By default the x (left/right) pathway shares that time course, which is OPTIMISTIC for x: ds004362's
L/R d′ ratios are 1.20 : 1 : 0.84 : 0.42 (an early peak and a decay to 42 % by 3–4 s), so the default
simulated x keeps far more sustained lateralised ERD in long L2b, S and C2c trials than the data the
proposal concedes is weak.  ``x_time_course = X_ERD_LR`` gives x its own course fitted to that curve
(a sensitivity case; ``X_DEPTH_RANGE`` was calibrated with the shared one).

Dry penalty
-----------
``dry_penalty`` ∈ [0, 1) multiplies the motor ERD depth by ``1 − dry_penalty``.  It is NOT the d′
loss: the loss is larger than proportional, because the background and the resonators' own
window-to-window power variance dilute a shallower ERD more.  Measured with the deployed pipeline on
the windows the champion trains on (6 subjects at y_depth 0.25, heterogeneity 0.5, 40 min of fixation
probes each through ``record_fixation`` → ``Labeler``: 1-s windows 1.0–4.0 s after the cue, local
reference, session rest re-centring, 5-fold contiguous CV, d′ = √2·Φ⁻¹(AUC); paired seeds):

    dry_penalty   0      0.1    0.2    0.3    0.4    0.5    0.7
    y AUC         0.620  0.599  0.577  0.557  0.538  0.521  0.497
    d′ loss       0 %    19 %   37 %   54 %   69 %   83 %   ≈ 100 %

(An earlier table — 48 % at 0.35, 85 % at 0.7 — was measured before the local reference and on 1–2 s
windows; it overstated what remains.)  So the proposal's 20–40 % planning prior is dry_penalty ≈
0.1–0.22, DESIGN_PURE_EEG §8's 20–70 % range is ≈ 0.1–0.4, and 0.7 is a no-signal condition, not
"85 % loss": sweep 0 / 0.2 / 0.4.  It acts on the motor pathway only, keeping the leak at full strength
while the body signal weakens: the conservative case for certification.

Screen-leak pathway (screen coordinates, only while a target is drawn)
----------------------------------------------------------------------
``set_screen_direction(u)`` tells the person where the target is *on screen* (``None``: nothing
spatial drawn, as in L1 fixation probes and L2b blind closed loop).  With ``screen_leak`` > 0 the
person's attention and gaze go toward that direction:

* posterior alpha lateralisation — attending right lowers alpha over the LEFT (contralateral)
  parieto-occipital cortex, and vice versa; attending up lowers ventral (PO/O) alpha, down lowers
  dorsal (P) alpha (the lower visual field maps dorsally), at the same depth as the horizontal
  pathway, because on ds004362 posterior channels decode a top/bottom target at least as well as a
  left/right one (E3: 0.661 vs 0.604);
* a signed frontal gaze potential toward the target (horizontal EOG F8 − F7, vertical FP1/FP2).

The posterior generators volume-conduct onto CP5/CP1 (left) and CP6/CP2 (right) at reduced weight, so
the central-11 montage does see part of the leak in the 8–30 Hz band.  This is deliberate: the skeptic
corrections show central and posterior 8–30 Hz information is largely shared on gel data, and without
that spread a central-montage shadow decoder could never show β_screen > 0 and the certification's
positive control would be untestable.  The gaze potential is slow, so the 8–30 Hz decoders see it only
through their filters' leakage; frontal channels and the purity audit see it in full.

``covert_leak`` (default 0) adds the same posterior alpha lateralisation toward the *body* direction
(the cued body vector read as a spatial direction), with or without a target, and no eye movement.
It models covert spatial attention tied to body-part labels, which the skeptic notes the L1/L2b label
rule does not exclude.  Because it follows the body, not the screen, the certification regressions
count it as body signal: it is the leak those tests *cannot* see, and the simulator makes it explicit.

With ``screen_leak = 0`` nothing in the EEG depends on the screen direction (tested exactly).

Gain drift and day-to-day variability
-------------------------------------
Inherited: per-channel log-normal day-to-day gain walk (``day_drift_sd``) and the base model's global
per-minute ``drift``.  Added: a per-channel within-session log-gain slope (``session_drift_sd`` per
hour, redrawn every ``set_day``), the dry-electrode settling that makes a frozen decoder degrade
within a session and motivates re-centring.  Per-channel gains act on covariances as a congruence,
which is exactly what rest re-centring removes, so gain-only drift is the BEST case for re-centring
and for a frozen decoder across days.  ``day_depth_sd`` (default 0, i.e. that best case) adds a
per-day log-normal factor on the ERD depths (``y_depth``, ``x_depth``), drawn per ``set_day``: the
day-to-day change in imagery strength that re-centring cannot remove.  All of it is applied per sample on the absolute sample
axis, so ``generate`` stays chunk-invariant: ``generate(a)`` then ``generate(b)`` equals
``generate(a + b)``.

Pure-EEG rule: this module produces EEG only; the eye/gaze state it is driven by is what an auditor
would record, and it never returns or exposes a non-EEG stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import lfilter

from .body_decoder import CENTRAL11
from .config import Hardware
from .sim_population import AR_COEF, AR_GAIN, DAY_DRIFT_SD, PopulationSim, SubjectEEG

# ----------------------------------------------------------------------------------------------
# Generators and their scalp projections (weights at the peak channel = 1; volume conduction below)
# ----------------------------------------------------------------------------------------------
MOTOR_TOPOGRAPHIES: Dict[str, Dict[str, float]] = {
    "hand_left_hemi": {"C3": 1.0, "FC5": 0.5, "CP5": 0.5, "FC1": 0.35, "CP1": 0.35, "T7": 0.2,
                       "F3": 0.15, "P3": 0.15},
    "hand_right_hemi": {"C4": 1.0, "FC6": 0.5, "CP6": 0.5, "FC2": 0.35, "CP2": 0.35, "T8": 0.2,
                        "F4": 0.15, "P4": 0.15},
    "feet": {"Cz": 1.0, "FC1": 0.5, "FC2": 0.5, "CP1": 0.5, "CP2": 0.5, "Fz": 0.2, "Pz": 0.2,
             "C3": 0.15, "C4": 0.15},
}
"""Sensorimotor generators: the C3 hand area (right hand), the C4 hand area (left hand), the vertex
foot area.  Weights fall off over neighbouring sites to mimic volume conduction on a 32-channel cap."""

POSTERIOR_TOPOGRAPHIES: Dict[str, Dict[str, float]] = {
    "alpha_left_dorsal": {"P3": 1.0, "P7": 0.7, "Pz": 0.4, "CP5": 0.5, "CP1": 0.4},
    "alpha_left_ventral": {"PO7": 1.0, "PO3": 0.9, "Oz": 0.5, "P7": 0.3},
    "alpha_right_dorsal": {"P4": 1.0, "P8": 0.7, "Pz": 0.4, "CP6": 0.5, "CP2": 0.4},
    "alpha_right_ventral": {"PO8": 1.0, "PO4": 0.9, "Oz": 0.5, "P8": 0.3},
}
"""Posterior alpha generators.  The CP weights are the posterior → central spread that lets a
central-11 decoder see visual attention (see the module docstring); they are set so that, as on
ds004362, the attention signal on the CP sites is of the same order as the motor one, not a trace."""

MU_HZ, BETA_HZ, ALPHA_HZ = 10.0, 20.0, 10.0
MU_BW_HZ, BETA_BW_HZ, ALPHA_BW_HZ = 2.0, 4.0, 2.0
"""Resonator centre frequencies and −3 dB bandwidths.  ~2 Hz for the alpha-band rhythms gives the
waxing/waning bursts of real mu and alpha; beta is broader in real recordings."""
MU_UV, BETA_UV, ALPHA_UV = 6.0, 3.0, 10.0
"""RMS amplitude (µV) of each rhythm at its generator's peak channel.  Against the inherited AR(1)
background (≈ 15 µV broadband, ≈ 7 µV²/Hz at 10 Hz) this gives a visible but not dominant mu peak, a
smaller beta peak and the larger posterior alpha of an awake, eyes-open person."""
WARMUP_S = 1.0
"""Resonator burn-in at every block start (≫ the 0.16 s time constant of a 2 Hz bandwidth), drawn from
the block's own source stream, so the first emitted sample is already stationary."""

IPSI_FRACTION = 0.3
"""Ipsilateral share of a one-hand ERD in an x responder (contralateral = 1): the textbook
contralateral dominance of hand-imagery ERD, with the bilateral component real recordings show."""
ERD_MAX = 0.9
"""Upper bound on any generator's amplitude reduction, so stacked cues never invert a rhythm."""

ERD_PEAK_S, ERD_DECAY_S, ERD_SUSTAIN = 1.5, 1.2, 0.5
"""Default ERD time course after a cue (see ``erd_envelope``); fitted to the ds004362 feet/fists decay."""
REBOUND_PEAK_S = 0.6
"""Post-imagery beta rebound (event-related synchronisation): the beta amplitude of the generators that
were desynchronised rises above baseline after the imagery ends, peaking this long after the offset
(Pfurtscheller's post-movement beta synchronisation peaks ~0.5–1 s after the offset and is over by ~2–3 s)."""
REBOUND_GAIN = 1.0
"""Peak beta amplitude increase per unit of the ERD amplitude reduction at the offset: a y_depth 0.25
person ending 4-s imagery at envelope 0.66 gets +16 % amplitude (≈ +35 % beta power) at the peak.
Without it the ERD vanished at trial end and the rest between probes (the idle gate's L0 windows) was
artificially clean."""
X_ERD_LR: Tuple[float, float, float] = (0.3, 2.8, 0.0)
"""(peak_s, decay_s, sustain) fitted to ds004362's L/R d′ ratios 1.20 : 1 : 0.84 : 0.42 (0–1 … 3–4 s;
the fit gives 1.17 : 1 : 0.70 : 0.49, least squares on a grid): an x pathway that peaks early and fades.
For ``BodySubject.x_time_course``, a sensitivity case (module docstring)."""

ALPHA_LAT_DEPTH = 0.5
"""Contralateral posterior alpha amplitude reduction at ``screen_leak`` = 1 for a fully lateral
target (alpha lateralisation indices of 10–30 % in power ≈ 5–15 % per unit leak of amplitude; 1 is a
strong attender)."""
ALPHA_VERT_DEPTH = ALPHA_LAT_DEPTH
"""Ventral/dorsal alpha reduction for a fully vertical target at ``screen_leak`` = 1: the horizontal
depth.  ds004362's posterior channels decode the displayed feet/fists (top/bottom) target at 0.661 and
the left/right one at 0.604 (skeptic correction C3, E3), so the vertical display signal is not weaker
than the horizontal one.  Only "down" reaches the central-11 montage (dorsal alpha spreads onto CP
sites, ventral does not), so at this depth the up/down leak seen by central 8–30 Hz features is about
as decodable as the left/right one (AUC ≈ 0.73 vs 0.64 at leak 1), and the y-axis positive control
is testable.  At half this depth the central up/down leak stays near chance and a y-only shadow
cannot be shown to read the screen at any practical leak."""
SCREEN_GAZE_UV = 20.0
"""Frontal gaze potential (µV) at ``screen_leak`` = 1 toward a unit screen direction; matches the
inherited ocular-leak scale (≈ 20 µV at full gaze)."""

SESSION_DRIFT_SD = 0.05
"""SD of the per-channel within-session log-gain slope, per hour (≈ ±5 %/h electrode settling)."""

Y_DEPTH_RANGE: Tuple[float, float] = (0.1, 0.4)
"""Per-subject feet/hands ERD depth (fractional amplitude reduction at the time-course peak; 10–40 %
amplitude ≈ 20–65 % power).  Calibrated on gel-equivalent (dry_penalty 0) 1-s windows at 1–2 s with
``BodyCoordinateDecoder`` features, 160 trials, 5-fold CV, mean of 4 people: AUC ≈ 0.54 at 0.1, 0.72
at 0.25, 0.87 at 0.4,
so the population spans ds004362's 1–2 s feet/fists mean of 0.667 with a share of near-floor people.
Calibrated before the decoder's central-11 local reference, which costs ≈ 0.05–0.14 of these AUCs here
(``body_decoder`` docstring: this background is spatially independent, so its common mode is signal):
read the simulated AUCs as a conservative lower range against the real-data anchors.  With the local
reference, on the 1.0–4.0-s windows the champion trains on, 0.25 gives 0.62 (dry-penalty table)."""
X_DEPTH_RANGE: Tuple[float, float] = (0.25, 0.5)
"""Per-subject one-hand ERD depth.  For x responders this gives 1-s AUC ≈ 0.65–0.80 under the same
calibration (0.64 at 0.25, 0.71 at 0.35, 0.79 at 0.5), bracketing the 0.69–0.73 of ds004362's
top-third L/R responders; for non-responders the
depth only sets the (non-lateralised) total hand ERD."""
X_RESPONDER_PROB = 0.2
"""Probability that a subject produces lateralised L/R ERD (skeptic correction: 10–25 % plausible)."""

# Independent random streams per subject (distinct from sim_population's 1–4), so adding a body
# feature never reshuffles the inherited background, blinks or day drift.
_SMR_TOPO_STREAM, _SOURCE_STREAM, _SESSION_DRIFT_STREAM, _BODY_PARAM_STREAM, _DAY_DEPTH_STREAM = 11, 12, 13, 14, 15

_MOTOR = tuple(MOTOR_TOPOGRAPHIES)
_POSTERIOR = tuple(POSTERIOR_TOPOGRAPHIES)


def rebound_envelope(tau_s, peak_s: float = REBOUND_PEAK_S) -> np.ndarray:
    """Post-imagery beta rebound strength at ``tau_s`` seconds after the imagery offset: an alpha
    function (τ/peak)·e^{1 − τ/peak}, 0 at the offset, 1 at ``peak_s``, < 0.02 after ~6 peaks."""
    tau = np.maximum(np.asarray(tau_s, dtype=float), 0.0) / peak_s
    return tau * np.exp(1.0 - tau)


def erd_envelope(tau_s, peak_s: float = ERD_PEAK_S, decay_s: float = ERD_DECAY_S,
                 sustain: float = ERD_SUSTAIN) -> np.ndarray:
    """ERD strength in [0, 1] at ``tau_s`` seconds after cue onset (0 before the cue).

    A raised-cosine rise from 0 at the cue to 1 at ``peak_s`` (smooth, so the band power has no step),
    then an exponential decay with time constant ``decay_s`` toward ``sustain``: untrained imagery
    fades but does not vanish (ds004362 keeps ~65 % of its peak d′ at 3–4 s)."""
    tau = np.asarray(tau_s, dtype=float)
    rise = 0.5 * (1.0 - np.cos(np.pi * np.clip(tau, 0.0, peak_s) / peak_s))
    decay = sustain + (1.0 - sustain) * np.exp(-np.maximum(tau - peak_s, 0.0) / decay_s)
    return np.where(tau < 0, 0.0, np.where(tau < peak_s, rise, decay))


def _resonator(f_hz: float, bw_hz: float, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    """AR(2) resonator at ``f_hz`` with bandwidth ``bw_hz``, scaled to unit output variance for unit
    white-noise input (the scale comes from the impulse response's energy, so amplitudes in µV mean
    RMS whatever the bandwidth)."""
    r = np.exp(-np.pi * bw_hz / fs)
    a = np.array([1.0, -2.0 * r * np.cos(2 * np.pi * f_hz / fs), r * r])
    impulse = np.zeros(int(20 * fs / bw_hz))
    impulse[0] = 1.0
    h = lfilter([1.0], a, impulse)
    return np.array([1.0 / np.sqrt(np.sum(h * h))]), a


def _projection(hw: Hardware, weights: Dict[str, float]) -> np.ndarray:
    idx = {c: k for k, c in enumerate(hw.channels)}
    v = np.zeros(hw.n_channels)
    for ch, w in weights.items():
        v[idx[ch]] = w
    return v


@dataclass
class BodySubject(SubjectEEG):
    """One simulated SMR user: ``SubjectEEG`` with body-part ERD, a transient time course and a
    screen-leak pathway.

    Control (per step, in addition to the inherited ``set_day`` / ``begin_block`` / ``set_gaze`` /
    ``record_practice``):

    ``cue(vx, vy)``          a new imagery episode in body coordinates; restarts the ERD time course.
    ``set_intent(vx, vy)``   update the body intent; the time course runs on from the current episode
                             (it starts one if the person was at rest), as in closed-loop control.
    ``set_screen_direction(u | None)``  the target's direction on screen, or None when nothing
                             spatial is drawn; drives only the screen-leak pathway.

    Parameters (explicit, all with defaults):
    y_depth, x_depth    ERD depth of feet/hands and of one-hand imagery at the time-course peak;
                        the defaults are a mid-range x responder (1-s AUC ≈ 0.72 on y, 0.71 on x).
    x_responder         False → LEFT/RIGHT ERD is bilateral (same total, no asymmetry).
    dry_penalty         dry-cap scaling of the motor ERD depth, in [0, 1): depth × (1 − dry_penalty).  NOT
                        the d′ loss, which is larger (module docstring table: 0.2 → 37 %, 0.4 → 69 %).
    screen_leak         gain of the overt screen pathway (posterior alpha + frontal gaze), ≥ 0.
    covert_leak         gain of body-locked covert attention (posterior alpha only), ≥ 0.
    session_drift_sd    SD of the per-channel within-session log-gain slope per hour.
    erd_peak_s, erd_decay_s, erd_sustain   the ERD time course (``erd_envelope``).
    x_time_course       None (x shares the time course above) or (peak_s, decay_s, sustain) for the
                        one-hand (x) part of the ERD, e.g. ``X_ERD_LR``.
    day_depth_sd        SD of the per-day log-normal factor on both ERD depths (0: identical every day).
    rebound_gain        post-imagery beta rebound per unit of the ERD at the offset (``REBOUND_GAIN``;
                        0 switches it off).
    heterogeneity       (inherited) mixes each generator's canonical projection with an individual one
                        on the same sites, so people differ in where on the strip their ERD peaks.
    user_learning_rate  (inherited) motor depth grows by this fraction per practice hour.

    The inherited ``intent_gain_uv`` drives the base simulator's screen-like intent pathway, which this
    class replaces; body signal strength is ``y_depth`` / ``x_depth`` here.
    """
    y_depth: float = 0.25
    x_depth: float = 0.35
    x_responder: bool = True
    dry_penalty: float = 0.0
    screen_leak: float = 0.0
    covert_leak: float = 0.0
    session_drift_sd: float = SESSION_DRIFT_SD
    erd_peak_s: float = ERD_PEAK_S
    erd_decay_s: float = ERD_DECAY_S
    erd_sustain: float = ERD_SUSTAIN
    x_time_course: Optional[Tuple[float, float, float]] = None
    day_depth_sd: float = 0.0
    rebound_gain: float = REBOUND_GAIN
    projections: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.dry_penalty < 1.0:
            raise ValueError(f"dry_penalty must be in [0, 1), got {self.dry_penalty}")
        for name in ("y_depth", "x_depth", "screen_leak", "covert_leak", "session_drift_sd", "day_depth_sd",
                     "rebound_gain"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        courses = [(self.erd_peak_s, self.erd_decay_s, self.erd_sustain)]
        if self.x_time_course is not None:
            courses.append(tuple(self.x_time_course))
        if not all(len(c) == 3 and c[0] > 0 and c[1] > 0 and 0.0 <= c[2] <= 1.0 for c in courses):
            raise ValueError("ERD time course needs peak_s > 0, decay_s > 0 and sustain in [0, 1]")
        self._day_depth = 1.0
        self._cue_onset: Optional[int] = None
        self._rebound_onset: Optional[int] = None             # sample of the last imagery offset
        self._rebound_depth = np.zeros(len(_MOTOR))           # ERD amplitude reduction at that offset
        self.screen_dir: Optional[np.ndarray] = None
        self._day_origin = 0
        self._session_slope = np.zeros(self.hw.n_channels)
        super().__post_init__()
        self.projections = self._draw_projections()
        self._class_weights = self._class_to_generator()
        self._b10, self._a10 = _resonator(MU_HZ, MU_BW_HZ, self.hw.fs)
        self._b20, self._a20 = _resonator(BETA_HZ, BETA_BW_HZ, self.hw.fs)
        self._b_alpha, self._a_alpha = _resonator(ALPHA_HZ, ALPHA_BW_HZ, self.hw.fs)
        self._reset_sources(np.random.default_rng([self.seed, _SOURCE_STREAM]))
        self.set_day(self.day)

    # -- construction --------------------------------------------------------------------------------
    def _draw_projections(self) -> np.ndarray:
        """(n_generators, n_channels) scalp projections: mu ×3, beta ×3 (motor), alpha ×4 (posterior).

        Motor projections are ``(1 − h)·canonical + h·individual``, peak-normalised, where the
        individual pattern redraws the weights only on the central-11 strip and keeps the canonical
        ones elsewhere: heterogeneity moves the peak along the sensorimotor strip without moving the
        generator onto posterior or frontal sites (which would blur motor and attention signals by
        construction).  Posterior projections are canonical: they model a shared visual system."""
        rng = np.random.default_rng([self.seed, _SMR_TOPO_STREAM])
        strip = np.isin(np.array(self.hw.channels), CENTRAL11)
        motor = []
        for name in _MOTOR:
            canon = _projection(self.hw, MOTOR_TOPOGRAPHIES[name])
            indiv = np.where((canon > 0) & strip, rng.uniform(0.2, 1.0, canon.size), canon)
            mix = (1.0 - self.heterogeneity) * canon + self.heterogeneity * indiv
            motor.append(mix / mix.max())
        posterior = [_projection(self.hw, POSTERIOR_TOPOGRAPHIES[n]) for n in _POSTERIOR]
        return np.stack(motor * 2 + posterior)            # mu and beta share each motor generator's site

    def _class_to_generator(self) -> Dict[str, np.ndarray]:
        """ERD weight of each imagery class on (hand_left_hemi, hand_right_hemi, feet)."""
        if self.x_responder:
            right, left = [1.0, IPSI_FRACTION, 0.0], [IPSI_FRACTION, 1.0, 0.0]
        else:
            both = (1.0 + IPSI_FRACTION) / 2.0                # same total ERD, no hemispheric asymmetry
            right = left = [both, both, 0.0]
        return {"FEET": np.array([0.0, 0.0, 1.0]) * self.y_depth,
                "HANDS": np.array([1.0, 1.0, 0.0]) * self.y_depth,
                "RIGHT": np.array(right) * self.x_depth,
                "LEFT": np.array(left) * self.x_depth}

    def _reset_sources(self, rng: np.random.Generator) -> None:
        """Fresh resonator drive stream and a burnt-in (stationary) resonator state."""
        self._source_rng = rng
        n_motor = len(_MOTOR)
        self._zi_mu = np.zeros((2, n_motor))
        self._zi_beta = np.zeros((2, n_motor))
        self._zi_alpha = np.zeros((2, len(_POSTERIOR)))
        self._resonate(rng.standard_normal((int(WARMUP_S * self.hw.fs), self.projections.shape[0])))

    def _resonate(self, drive: np.ndarray) -> np.ndarray:
        """Unit-variance mu, beta and alpha sources (n, n_generators) from white drive, state carried."""
        m = len(_MOTOR)
        mu, self._zi_mu = lfilter(self._b10, self._a10, drive[:, :m], axis=0, zi=self._zi_mu)
        beta, self._zi_beta = lfilter(self._b20, self._a20, drive[:, m:2 * m], axis=0, zi=self._zi_beta)
        alpha, self._zi_alpha = lfilter(self._b_alpha, self._a_alpha, drive[:, 2 * m:], axis=0,
                                        zi=self._zi_alpha)
        return np.hstack([mu * MU_UV, beta * BETA_UV, alpha * ALPHA_UV])

    # -- state changes -------------------------------------------------------------------------------
    def set_day(self, day: int) -> None:
        """Inherited day gain, plus this day's within-session drift slopes (drift restarts at 0)."""
        super().set_day(day)
        rng = np.random.default_rng([self.seed, _SESSION_DRIFT_STREAM, int(day)])
        self._session_slope = rng.normal(0.0, self.session_drift_sd, self.hw.n_channels)
        self._day_origin = self.n_sent
        depth_rng = np.random.default_rng([self.seed, _DAY_DEPTH_STREAM, int(day)])
        self._day_depth = float(np.exp(depth_rng.normal(0.0, self.day_depth_sd)))

    def begin_block(self, day: int, block: int) -> None:
        """Inherited restart (noise, blinks, AR state, evoked) plus the rhythm generators; the person
        starts the block at rest with nothing drawn, so a block never inherits an earlier cue."""
        super().begin_block(day, block)
        self._reset_sources(np.random.default_rng([self.seed, _SOURCE_STREAM, int(day), int(block)]))
        self.intent = np.zeros(2)
        self._cue_onset = None
        self._rebound_onset = None
        self.screen_dir = None

    def start(self) -> None:
        """As the base ``start``; cue onset and drift origin are re-based with the sample count."""
        shift = self.n_sent
        super().start()
        self._day_origin -= shift
        if self._cue_onset is not None:
            self._cue_onset -= shift
        if self._rebound_onset is not None:
            self._rebound_onset -= shift

    def _end_episode(self) -> None:
        """Imagery stops now: start the post-imagery beta rebound from the ERD it had at this moment."""
        if self._cue_onset is None or not np.any(self.intent):
            return
        env = self.envelope(1)[0]
        env_x = env if self.x_time_course is None else self.envelope(1, tuple(self.x_time_course))[0]
        y_part, x_part = self._motor_depth()
        self._rebound_depth = np.clip(env * y_part + env_x * x_part, 0.0, ERD_MAX)
        self._rebound_onset = self.n_sent

    def cue(self, vx: float, vy: float) -> None:
        """Start an imagery episode for body vector (vx, vy) now (zero vector = rest, which ends the
        current episode and starts its beta rebound; a new episode cancels any rebound)."""
        if np.any((vx, vy)):
            self._rebound_onset = None
        else:
            self._end_episode()
        super().set_intent(vx, vy)
        self._cue_onset = self.n_sent if np.any(self.intent) else None

    def set_intent(self, vx: float, vy: float) -> None:
        """Update the body intent; a move from rest starts an episode, rest ends it (and starts its
        beta rebound)."""
        was_rest = not np.any(self.intent)
        if not np.any((vx, vy)):
            self._end_episode()
        elif was_rest:
            self._rebound_onset = None
        super().set_intent(vx, vy)
        if not np.any(self.intent):
            self._cue_onset = None
        elif was_rest or self._cue_onset is None:
            self._cue_onset = self.n_sent

    def set_screen_direction(self, direction: Optional[Sequence[float]]) -> None:
        """Direction of the drawn target on screen (+x right, +y up), norm clipped to 1; ``None`` when
        nothing spatial is drawn (L1, L2b).  Only ``screen_leak`` lets it reach the EEG."""
        if direction is None:
            self.screen_dir = None
            return
        d = np.asarray(direction, dtype=float).reshape(2)
        norm = float(np.linalg.norm(d))
        self.screen_dir = d / norm if norm > 1.0 else d

    # -- signal --------------------------------------------------------------------------------------
    def envelope(self, n: int, course: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """ERD time course for the next n samples (0 at rest); ``course`` overrides (peak, decay,
        sustain), as the x pathway's ``x_time_course`` does."""
        if self._cue_onset is None:
            return np.zeros(n)
        tau = (self.n_sent + np.arange(n) - self._cue_onset) / self.hw.fs
        peak, decay, sustain = course or (self.erd_peak_s, self.erd_decay_s, self.erd_sustain)
        return erd_envelope(tau, peak, decay, sustain)

    def _rebound(self, n: int) -> np.ndarray:
        """Beta rebound strength (× ``rebound_gain``) for the next n samples; 0 without a recent offset."""
        if self._rebound_onset is None or self.rebound_gain == 0:
            return np.zeros(n)
        tau = (self.n_sent + np.arange(n) - self._rebound_onset) / self.hw.fs
        return self.rebound_gain * rebound_envelope(tau)

    def _motor_depth(self) -> Tuple[np.ndarray, np.ndarray]:
        """Peak ERD depth on each motor generator for the current intent, after dry penalty, practice
        and the day's depth factor, split into its y (feet/hands) and x (one-hand) parts so each can
        follow its own time course: class activations are the positive and negative parts of the body
        vector."""
        vx, vy = self.intent
        w = self._class_weights
        gain = (1.0 - self.dry_penalty) * (1.0 + self.user_learning_rate * self.practice_h) * self._day_depth
        y_part = (max(vy, 0.0) * w["FEET"] + max(-vy, 0.0) * w["HANDS"]) * gain
        x_part = (max(vx, 0.0) * w["RIGHT"] + max(-vx, 0.0) * w["LEFT"]) * gain
        return y_part, x_part

    def attention(self) -> np.ndarray:
        """Spatial attention vector driving posterior alpha: overt (screen target, only when drawn)
        plus covert (body vector read as a direction)."""
        a = self.covert_leak * self.intent
        if self.screen_dir is not None:
            a = a + self.screen_leak * self.screen_dir
        return a

    def _alpha_depth(self) -> np.ndarray:
        """Amplitude reduction of the four posterior generators for the current attention vector:
        contralateral hemisphere for x, ventral (up) / dorsal (down) for y."""
        ax, ay = self.attention()
        right, left, up, down = max(ax, 0.0), max(-ax, 0.0), max(ay, 0.0), max(-ay, 0.0)
        lat, vert = ALPHA_LAT_DEPTH, ALPHA_VERT_DEPTH
        return np.array([lat * right + vert * down,      # left dorsal: attend right, attend down
                         lat * right + vert * up,        # left ventral: attend right, attend up
                         lat * left + vert * down,       # right dorsal
                         lat * left + vert * up])        # right ventral

    def generate(self, n: int) -> np.ndarray:
        """n samples (n, n_channels) float32 µV; chunk-invariant like the base model.

        Inherited background AR(1), blinks, evoked transients, gaze EOG and drift, plus the rhythm
        generators with their ERD / attention modulation, the screen gaze potential and the
        within-session per-channel gain slope."""
        C = self.hw.n_channels
        w = self.rng.standard_normal((n, C)) * self.noise_uv
        x, self._ar_zi = lfilter([AR_GAIN], [1.0, -AR_COEF], w, axis=0, zi=self._ar_zi)
        sources = self._resonate(self._source_rng.standard_normal((n, self.projections.shape[0])))
        env = self.envelope(n)[:, None]
        env_x = env if self.x_time_course is None else self.envelope(n, tuple(self.x_time_course))[:, None]
        y_part, x_part = self._motor_depth()
        motor = env * y_part[None] + env_x * x_part[None]
        beta = motor - self._rebound(n)[:, None] * self._rebound_depth[None]   # rebound: negative reduction
        reduction = np.hstack([np.clip(motor, 0.0, ERD_MAX), np.clip(beta, -ERD_MAX, ERD_MAX),
                               np.clip(np.broadcast_to(self._alpha_depth(), (n, len(_POSTERIOR))), 0.0, ERD_MAX)])
        x += (sources * (1.0 - reduction)) @ self.projections
        if self.screen_dir is not None and self.screen_leak > 0:
            g = self.screen_leak * SCREEN_GAZE_UV * self.screen_dir
            x += g[0] * self._h_eog[None] + g[1] * self._fp[None]
        if self.eye_gain_uv > 0:
            x += self.eye_gain_uv * 20.0 * (self.gaze[0] * self._h_eog[None] + self.gaze[1] * self._fp[None])
        x += self._blinks(n)
        x += self._evoked(n)
        if self.drift:
            x *= self._drift_gain(n)
        hours = (self.n_sent + np.arange(n) - self._day_origin) / self.hw.fs / 3600.0
        x *= self.gain[None] * np.exp(np.outer(hours, self._session_slope))
        self.n_sent += n
        return x.astype(np.float32)


@dataclass(frozen=True)
class BodyParams:
    """Per-subject body-model draws (beside ``sim_population.SubjectParams``)."""
    index: int
    y_depth: float
    x_depth: float
    x_responder: bool

    def describe(self) -> dict:
        return {"index": self.index, "y_depth": round(self.y_depth, 4), "x_depth": round(self.x_depth, 4),
                "x_responder": self.x_responder}


class BodyPopulation(PopulationSim):
    """``PopulationSim`` whose subjects are ``BodySubject`` people.

    The inherited draws (seed, heterogeneity, ocular leak, drift, learning) are kept unchanged; the
    body draws (``y_depth``, ``x_depth``, ``x_responder``) come from their own stream
    ``default_rng([seed, i, 14])``, so subject ``i`` is the same person whatever the population size
    and whatever the body knobs.  ``dry_penalty``, ``screen_leak``, ``covert_leak``, ``session_drift_sd``,
    ``x_time_course`` and ``day_depth_sd`` are condition-level settings, identical for every subject, so a sweep over
    them runs on the very same people.
    """

    def __init__(self, n_subjects: int, seed: int, hw: Hardware = Hardware(), *,
                 x_responder_prob: float = X_RESPONDER_PROB,
                 y_depth_range: Tuple[float, float] = Y_DEPTH_RANGE,
                 x_depth_range: Tuple[float, float] = X_DEPTH_RANGE,
                 dry_penalty: float = 0.0, screen_leak: float = 0.0, covert_leak: float = 0.0,
                 session_drift_sd: float = SESSION_DRIFT_SD, day_drift_sd: float = DAY_DRIFT_SD,
                 x_time_course: Optional[Tuple[float, float, float]] = None, day_depth_sd: float = 0.0,
                 **population_kw):
        if not 0.0 <= x_responder_prob <= 1.0:
            raise ValueError(f"x_responder_prob must be in [0, 1], got {x_responder_prob}")
        for name, (lo, hi) in (("y_depth_range", y_depth_range), ("x_depth_range", x_depth_range)):
            if not 0.0 <= lo <= hi <= ERD_MAX:
                raise ValueError(f"{name} must lie in [0, {ERD_MAX}], got {(lo, hi)}")
        super().__init__(n_subjects, seed, hw, day_drift_sd=day_drift_sd, **population_kw)
        self.dry_penalty = float(dry_penalty)
        self.screen_leak = float(screen_leak)
        self.covert_leak = float(covert_leak)
        self.session_drift_sd = float(session_drift_sd)
        self.x_time_course = None if x_time_course is None else tuple(float(v) for v in x_time_course)
        self.day_depth_sd = float(day_depth_sd)
        self.body: List[BodyParams] = []
        for i in range(n_subjects):
            rng = np.random.default_rng([self.seed, i, _BODY_PARAM_STREAM])
            y = float(rng.uniform(*y_depth_range))
            x = float(rng.uniform(*x_depth_range))
            responder = bool(rng.random() < x_responder_prob)
            self.body.append(BodyParams(i, y, x, responder))

    def subject(self, i: int) -> BodySubject:
        """A fresh ``BodySubject`` (day 1, no practice) for subject ``i``."""
        p, b = self.params[i], self.body[i]
        return BodySubject(self.hw, seed=p.seed, eye_gain_uv=p.eye_gain_uv, heterogeneity=p.heterogeneity,
                           day_drift_sd=p.day_drift_sd, user_learning_rate=p.user_learning_rate,
                           y_depth=b.y_depth, x_depth=b.x_depth, x_responder=b.x_responder,
                           dry_penalty=self.dry_penalty, screen_leak=self.screen_leak,
                           covert_leak=self.covert_leak, session_drift_sd=self.session_drift_sd,
                           x_time_course=self.x_time_course, day_depth_sd=self.day_depth_sd)

    def describe(self, indices: Optional[Sequence[int]] = None) -> List[dict]:
        idx = range(len(self)) if indices is None else indices
        return [{**self.params[i].describe(), **self.body[i].describe()} for i in idx]
