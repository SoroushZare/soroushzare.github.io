"""Body-coordinate SMR decoder: the champion's deployed decoder (INTERFACES_CHAMPION item G).

What it decodes, and why in body coordinates
--------------------------------------------
Imagined movement of a body part suppresses 8–30 Hz (mu/beta) power over the sensorimotor cortex that
represents it: both feet → the vertex (Cz); both hands → bilateral C3/C4; one hand → the opposite
hemisphere.  The decoder therefore outputs velocity in *body* coordinates,

    y = feet (+1) vs both hands (−1)     primary axis, always on
    x = right (+1) vs left hand (−1)     earned per person (``earn_x``), else 0

and a separate 2×2 mapping M turns that into screen velocity.  Decoding the body, not the screen, is
what lets held-out mappings (rot ±90°, mirrors) test whether the decoder follows the person's motor
intent or the display: a decoder that learned the screen would follow the screen under a mapping it
never saw.

Label rule (the reason ``fit`` refuses data)
--------------------------------------------
Only windows whose provenance ``source`` is "L1" (fixation probe) or "L2b" (blind closed loop) carry a
label that nothing on the screen can identify.  "S" (sighted) has no label for this decoder, and "H"
(held-out mapping) and "R" (pixel replay) are test blocks; training on any of them would let the
decoder learn the display and void the certification, so ``fit`` and ``earn_x`` raise instead of
silently filtering them out (a silent filter would hide a mis-wired trainer).

Pipeline (per axis, identical online and offline)
-------------------------------------------------
window (n_samples, 32) EEG µV, already CAR-referenced by the real-time chain
  → last ``window_s`` seconds, central-11 channels
  → LOCAL re-reference: the central-11 mean is subtracted (``local_reference``, below)
  → zero-phase 8–30 Hz Butterworth band-pass *within the window* (window-local, so the online path
    and offline re-scoring see bit-identical features; no state to carry)
  → OAS-shrunk covariance (10 dimensions × 500 samples: shrinkage is needed and OAS picks it in closed form)
  → re-centred on a rest reference R: C ↦ R^{-1/2} C R^{-1/2}  (Zanini et al. 2018)
  → tangent space at the identity (matrix log, upper triangle, √2 off-diagonal weights)
  → standardise → L2 logistic regression feet-vs-hands (y) and right-vs-left (x)
  → output 2p − 1 per axis, or exactly 0 on both axes when the idle gate says the person is at rest.

Why a local reference.  The real-time chain re-references every channel to the common average of
ALL good channels (``signal.common_average``), and FP1/FP2 are gate-exempt, so blink, saccade and
frontal-EMG potentials enter every central channel at ~1/n of their size: "central-11 excludes the
frontal channels by construction" is false after CAR.  Subtracting the central-11 mean removes any
signal common to the 11 channels, and the global CAR term is exactly such a signal, so it cancels
exactly: the decoder's input no longer depends on any channel outside its montage.  The re-referenced
data have rank k − 1 (they sum to zero), so they are expressed in an orthonormal basis of the
zero-sum subspace (``scipy.linalg.helmert``): the same information as "subtract the mean", but a
full-rank covariance whose matrix log stays finite.  A reference covariance given in channel
coordinates (k × k) is mapped into that basis.  The price is the strip's common mode, which also holds
part of every broad motor generator (the central mean is ~25 % C3 hand area, ~30 % vertex): in
``sim_body`` the 1-s cross-session AUC falls by ≈ 0.05–0.14 on both axes (strong lateraliser: x
0.83 → 0.72).  That is an upper bound: the simulator's background is spatially independent, so its
common mode is nearly all signal; real scalp background is volume-conducted, its common mode mostly
noise (with 80 % of the simulated background made spatially smooth, the cost was within seed noise).

Idle state (``fit_idle``).  A logistic 2p − 1 output is never 0 at rest, so without an idle state the
cursor and a robot arm drift whenever the person is not imagining anything.  Rest windows (source
"L0": the setup rest and the rest between fixation probes, both with nothing cued) against the
decoder's own intent windows (L1 / L2b) train an idle-vs-intent logistic gate on the same
re-centred tangent features; whenever p(idle) exceeds the pre-declared ``idle_threshold`` the output
is exactly (0, 0).  Rest has no direction, so an L0 window carries no display-identifiable label.

Signal validity.  A window whose central channels are not finite, are flagged bad by the real-time
amplitude gate (``predict(window, good=...)``), whose band power after re-centring sits more than
``POWER_LIMIT`` (30 dB) away from the training windows' median, or an eighth of which is dead
(``MIN_SEGMENT_SHARE``) gives NaN instead of a command: a
flat-lined amplifier (power → 0) would otherwise push the tangent features to extremes and saturate the
logistic output at full speed, and a dropped electrode held at its last value reads as ERD.
``realtime.RealTimeLoop`` turns a NaN output into a counted zero command (``n_invalid``).

Re-centring is the only use of unlabelled data: it moves each session's rest covariance to the
identity, so a cap re-fit or an electrode gain change between sessions — which acts on covariances as
a congruence — is largely removed before the classifier sees the data.  It needs no label and no
display, so it may be computed from any EEG block, sighted ones included (the champion design's only
use of sighted data).  With no reference set R = I, and re-centring is the identity map.

Pure-EEG rule: the only arrays that reach this class are EEG windows; provenance is read only for the
label-source check and for the session key that selects a rest reference, never as a feature.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Dict, Hashable, List, Mapping, Optional, Sequence

import numpy as np
from pyriemann.geometry.mean import mean_riemann
from scipy import signal as sps
from scipy.linalg import helmert
from sklearn.covariance import oas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .features import _sym_fn
from .purity import circular_shift_permutations

#: Central-11 montage on the g.tec 32 cap: the sensorimotor strip and its neighbours.  It leaves out the
#: frontal (ocular) and parieto-occipital (visual attention) channels; with the local reference (module
#: docstring) no other channel reaches the decoder even after the real-time chain's global CAR.  Purity
#: still rests on the label rule and the certification tests, because CP channels share posterior sources.
CENTRAL11 = ("FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6")

#: Provenance sources whose labels the display cannot identify; the only ones a deployed decoder may
#: train on.  S (sighted), H (held-out mapping) and R (pixel replay) are refused.
TRAINABLE_SOURCES = frozenset({"L1", "L2b"})

#: Provenance source of rest windows (setup rest, rest between fixation probes): nothing is cued and
#: nothing spatial is drawn.  Only the idle gate trains on them (``fit_idle``); ``fit`` refuses them.
IDLE_SOURCE = "L0"

#: p(idle) above which the output is exactly zero.  Pre-declared, not tuned on test data: 0.7 asks for
#: odds of at least 7 : 3 for rest under a class-balanced gate, so a window the gate cannot place
#: (p ≈ 0.5, e.g. a weak responder's decayed late-trial ERD) still drives the cursor, and the cost of
#: the gate falls on rest, where it belongs.
IDLE_THRESHOLD = 0.7

#: Output axes, in the order ``predict`` returns them.
AXES = ("x", "y")

#: Update rate of the deployed loop (DESIGN_PURE_EEG §7: 1-s windows every 62.5 ms).  ``predict`` takes
#: ~3 ms (p95 ~4 ms), so 16 Hz leaves the loop most of its period; ``loop_config`` sets it.
LOOP_HZ = 16.0

#: A window whose re-centred band power differs from the training median by more than this factor
#: (either way) is not EEG the decoder knows: a flat line is ~1e-8, a disconnected electrode or a
#: saturated amplifier is far above.  30 dB is well outside the window-to-window spread of real EEG
#: (a few dB) and of re-centred session differences, so a valid window is never refused.
POWER_LIMIT = 1e3

#: The window is also cut into ``N_SEGMENTS`` equal parts (0.125 s of a 1-s window); if any part
#: carries less than ``MIN_SEGMENT_SHARE`` of the window's mean 8–30 Hz power (or none at all), the
#: window is invalid.  Scale-free, so it needs no reference: it catches an amplifier that has JUST
#: flat-lined (the newest ~0.2 s are dead while the window's total power still looks normal), which the
#: 30-dB check above only catches once the whole window is flat — about one second of full-speed
#: commands later.  Broadband 8–30 Hz EEG over 11 channels does not lose 99 % of its power for 0.125 s.
N_SEGMENTS = 8
MIN_SEGMENT_SHARE = 0.01

#: Posterior channels of the x-purity check: left (P3, PO7) and right (P4, PO8) parieto-occipital alpha,
#: the proposal's S7 lateralisation auditor.  Covert spatial attention lateralises alpha there, and
#: through volume conduction onto CP5/CP6 it can masquerade as a lateralised SMR x signal.
POSTERIOR_LEFT = ("P3", "PO7")
POSTERIOR_RIGHT = ("P4", "PO8")


def _tangent_at_identity(c: np.ndarray) -> np.ndarray:
    """Upper triangle of log(c) with √2 off-diagonal weights, so the Euclidean norm of the vector
    equals the Frobenius norm of the matrix log (the affine-invariant distance to the identity)."""
    s = _sym_fn(c, np.log)
    iu = np.triu_indices(s.shape[0])
    return s[iu] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))


def _residualise(scores: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """``scores`` minus their least-squares fit on ``covariate`` (with an intercept)."""
    c = covariate - covariate.mean()
    denom = float(c @ c)
    return scores - (float(c @ (scores - scores.mean())) / denom) * c if denom > 0 else scores


def _order_key(p: Mapping, j: int) -> tuple:
    """(session, time, index) of a window: the session/time order contiguous folds follow."""
    return str(p.get("session")), float(p.get("t", j)), j


def _check_sources(provenance: Sequence[Mapping], n: int) -> None:
    """Raise unless every window is labelled by a source the display cannot identify."""
    if len(provenance) != n:
        raise ValueError(f"provenance has {len(provenance)} entries, windows {n}")
    bad = sorted({str(p.get("source")) for p in provenance if p.get("source") not in TRAINABLE_SOURCES})
    if bad:
        raise ValueError(f"the deployed decoder trains only on {sorted(TRAINABLE_SOURCES)} windows; got "
                         f"source(s) {bad}. S has no decoder label (display-identifiable), H and R are "
                         f"test-only blocks.")


@dataclass
class BodyCoordinateDecoder:
    """Per-axis Riemannian SMR decoder from 1-s EEG windows to body velocity in [−1, 1]².

    Satisfies the ``predict(window) -> (2,)`` interface of ``realtime.RealTimeLoop``; run the loop
    with ``loop_config(cfg)`` so it hands over windows of this decoder's length.

    Hyperparameters (explicit, pre-declared):
    band_hz            8–30 Hz: mu and beta, where hand/foot imagery ERD lives; excludes the < 8 Hz
                       band that carries ocular and slow-potential leaks, and > 30 Hz EMG.
    filter_order       4th-order Butterworth band-pass, applied forward-backward inside the window.
    C                  inverse L2 strength of the logistic regression.  0.1 because the tangent vector
                       has 66 dimensions against a few hundred correlated windows per axis.
    x_auc_min          0.60: the pre-declared minimum cross-validated AUC for x to be earned.
    floor_percentile   95: x must also beat the 95th percentile of its circular-shift null.
    x_purity_retain    0.70: x must keep ≥ 70 % of its AUC margin above 0.5 after its out-of-fold scores
                       are regressed on posterior (S7) alpha lateralisation (the proposal's P1 increment
                       rule): an x axis carried by covert attention is not a motor axis.
    alpha_band_hz      8–12 Hz: the band of that posterior lateralisation index.
    n_permutations     200 circular shifts for that null (p95 resolution 0.005).
    n_splits           5 contiguous (time-block) folds for the x-earning cross-validation: only a fold
                       boundary can split a trial, so a test window is almost never scored by a model
                       trained on the other windows of its own trial.
    seed               seed of the circular-shift draws, so ``earn_x`` is deterministic.
    local_reference    True: re-reference the montage to its own mean (module docstring), so the
                       real-time CAR cannot carry frontal or ocular potentials into the decoder.  The
                       all-32 shadow keeps it too (there it is a second CAR, which only removes the
                       rank deficiency).
    idle_threshold     ``IDLE_THRESHOLD``: p(idle) above which the output is exactly zero (``fit_idle``).
    idle_auc_min       0.70: the gate is armed only if its out-of-fold rest-vs-intent AUC reaches this;
                       below it about one rest/intent pair in three is misordered, and gating would
                       remove about as much control as drift.
    idle_C             inverse L2 strength of the idle gate's logistic regression (as ``C``: same
                       features, same dimension problem).
    """
    cfg: Config
    fs: int
    channels: Sequence[str] = CENTRAL11
    window_s: float = 1.0
    band_hz: tuple = (8.0, 30.0)
    filter_order: int = 4
    C: float = 0.1
    x_auc_min: float = 0.60
    floor_percentile: float = 95.0
    n_permutations: int = 200
    n_splits: int = 5
    seed: int = 20261004
    x_purity_retain: float = 0.70
    alpha_band_hz: tuple = (8.0, 12.0)
    local_reference: bool = True
    idle_threshold: float = IDLE_THRESHOLD
    idle_auc_min: float = 0.70
    idle_C: float = 0.1
    x_enabled: bool = False
    x_report: Optional[dict] = None
    idle_report: Optional[dict] = None
    neutral_one_hand: bool = field(default=False, init=False)
    _ref_isqrt: np.ndarray = field(init=False, repr=False)
    _models: Dict[str, Optional[Pipeline]] = field(init=False, repr=False)

    def __post_init__(self):
        if int(self.fs) != int(self.cfg.hardware.fs):
            # A mismatch builds the window length and the band-pass for the wrong rate: the decoder
            # would silently decode part of its window in the wrong band.
            raise ValueError(f"fs {self.fs} differs from cfg.hardware.fs {self.cfg.hardware.fs}")
        if not 0.5 <= self.idle_threshold <= 1.0:
            raise ValueError(f"idle_threshold must be in [0.5, 1] (a p(idle) above it zeroes the output), "
                             f"got {self.idle_threshold}")
        names = list(self.cfg.hardware.channels)
        missing = [c for c in self.channels if c not in names]
        if missing:
            raise ValueError(f"channels {missing} are not on the cap {self.cfg.hardware.channels}")
        if len(set(self.channels)) != len(self.channels) or len(self.channels) < 2:
            raise ValueError(f"channels must be at least two distinct cap channels, got {tuple(self.channels)}")
        self._idx = np.array([names.index(c) for c in self.channels])
        k = len(self.channels)
        # (k, k − 1) orthonormal basis of the zero-sum subspace: x @ basis is "x minus its montage mean"
        # in coordinates where the covariance has full rank.  Without a local reference: the identity.
        self._basis = helmert(k, full=False).T if self.local_reference else np.eye(k)
        self.n_dims = self._basis.shape[1]
        self.n_win = int(round(self.window_s * self.fs))
        self._sos = sps.butter(self.filter_order, self.band_hz, "bandpass", fs=self.fs, output="sos")
        self._ref_isqrt = np.eye(self.n_dims)
        self._models = {axis: None for axis in AXES}
        self._idle_model: Optional[Pipeline] = None
        self._intent_features: Optional[np.ndarray] = None   # last fit's training features (idle gate)
        self._intent_keys: List[tuple] = []                  # their (session, time) order keys
        iu = np.triu_indices(self.n_dims)
        self._diag = iu[0] == iu[1]                   # tangent entries that are log-eigen "powers"
        self._log_power_ref = 0.0                     # training median of the re-centred log power

    # -- features -----------------------------------------------------------------------------------
    def _window(self, window: np.ndarray) -> np.ndarray:
        """Validate a raw (n_samples, 32) EEG window and return its last ``n_win`` central samples.

        A shorter window is an error, not a silently noisier covariance: it means the real-time loop
        runs with another ``decoder.window_s`` than the one this decoder was trained on."""
        w = np.asarray(window, dtype=np.float64)
        n_ch = self.cfg.hardware.n_channels
        if w.ndim != 2 or w.shape[1] != n_ch:
            raise ValueError(f"expected an EEG window of shape (n_samples, {n_ch}), got {w.shape}")
        if w.shape[0] < self.n_win:
            raise ValueError(f"window has {w.shape[0]} samples, the decoder needs {self.n_win} "
                             f"({self.window_s} s at {self.fs} Hz); run the loop with loop_config(cfg)")
        return w[-self.n_win:, self._idx]

    def _referenced(self, x: np.ndarray) -> np.ndarray:
        """Montage samples (n, k) → locally re-referenced coordinates (n, n_dims)."""
        return x @ self._basis

    def _central_cov(self, x: np.ndarray) -> np.ndarray:
        return oas(sps.sosfiltfilt(self._sos, self._referenced(x), axis=0))[0]

    def covariance(self, window: np.ndarray) -> np.ndarray:
        """8–30 Hz OAS covariance of the montage after the local reference (``n_dims`` × ``n_dims``),
        before re-centring."""
        return self._central_cov(self._window(window))

    def reference_from(self, rest_windows: Sequence[np.ndarray]) -> np.ndarray:
        """Riemannian (affine-invariant) mean of the rest windows' covariances: the point re-centring
        moves to the identity.  It is the centre of the geometry the tangent space is taken in (Zanini
        et al. 2018), and, being a geometric rather than arithmetic mean, a few high-power artefact
        windows in the rest block pull it less."""
        if len(rest_windows) == 0:
            raise ValueError("need at least one rest window for a reference")
        return mean_riemann(np.stack([self.covariance(w) for w in rest_windows]))

    def set_reference(self, rest_windows: Optional[Sequence[np.ndarray]] = None, *,
                      reference: Optional[np.ndarray] = None) -> "BodyCoordinateDecoder":
        """Re-centre every later ``predict`` (and windows of sessions without their own reference in
        ``fit``) on this rest reference.  Pass rest EEG windows (e.g. the session's 3-min rest) or a
        ready covariance ``reference``; with neither the reference returns to the identity."""
        if rest_windows is not None and reference is not None:
            raise ValueError("pass rest_windows or reference, not both")
        if rest_windows is not None:
            reference = self.reference_from(rest_windows)
        # The identity of the decoder's own coordinates: n_dims (k − 1 with the local reference), not k.
        self._ref_isqrt = self._isqrt(reference) if reference is not None else np.eye(self.n_dims)
        return self

    def _isqrt(self, reference: np.ndarray) -> np.ndarray:
        """R^{-1/2} of a reference given in the decoder's coordinates (``n_dims`` square, as
        ``reference_from`` returns it) or, with the local reference, in channel coordinates (k × k,
        mapped by the basis: the covariance of x @ basis is basisᵀ R basis)."""
        r = np.asarray(reference, dtype=np.float64)
        k, d = len(self.channels), self.n_dims
        if r.shape == (k, k) and d != k:
            r = self._basis.T @ r @ self._basis
        if r.shape != (d, d) or not np.allclose(r, r.T) or np.linalg.eigvalsh(r).min() <= 0:
            raise ValueError(f"reference must be a symmetric positive-definite {d}×{d} matrix "
                             f"(or {k}×{k} in channel coordinates)")
        return _sym_fn(r, lambda w: w ** -0.5)

    def _tangent(self, cov: np.ndarray, ref_isqrt: np.ndarray) -> np.ndarray:
        return _tangent_at_identity(ref_isqrt @ cov @ ref_isqrt)

    def features(self, window: np.ndarray) -> np.ndarray:
        """Tangent vector of one window under the current reference (the classifiers' input)."""
        return self._tangent(self.covariance(window), self._ref_isqrt)

    def _feature_matrix(self, windows: Sequence[np.ndarray], provenance: Sequence[Mapping],
                        session_references: Optional[Mapping[Hashable, np.ndarray]]) -> np.ndarray:
        """Training features: each window re-centred on its own session's rest reference when one is
        given, else on the decoder's current reference."""
        refs = {s: self._isqrt(r) for s, r in (session_references or {}).items()}
        return np.stack([self._tangent(self.covariance(w), refs.get(p.get("session"), self._ref_isqrt))
                         for w, p in zip(windows, provenance)])

    # -- training -----------------------------------------------------------------------------------
    def _logistic(self) -> LogisticRegression:
        return LogisticRegression(C=self.C, max_iter=2000)

    def _classifier(self) -> Pipeline:
        return make_pipeline(StandardScaler(), self._logistic())

    @staticmethod
    def _labels(body_labels: np.ndarray, n: int) -> np.ndarray:
        y = np.asarray(body_labels)
        if y.shape != (n, 2) or not np.isin(y, (-1, 0, 1)).all():
            raise ValueError(f"body_labels must be ({n}, 2) with entries in {{-1, 0, 1}}, got shape {y.shape}")
        return y.astype(int)

    def fit(self, windows: List[np.ndarray], body_labels: np.ndarray, provenance: Sequence[Mapping],
            session_references: Optional[Mapping[Hashable, np.ndarray]] = None, *,
            x_mask: Optional[np.ndarray] = None, neutral_one_hand: bool = False) -> "BodyCoordinateDecoder":
        """Train the y (feet vs hands) and, when both classes are present, x (right vs left) models.

        body_labels         (n, 2) cued body vector (x, y) in {−1, 0, 1}²; a window enters an axis's
                            training set where that axis's label is non-zero (a diagonal cue enters
                            both, a LEFT cue only x), plus the neutral windows below.
        provenance          one dict per window; ``source`` must be L1 or L2b for every window.
        session_references  optional {session: rest covariance} so each training session is re-centred
                            on its own rest; the decoder's current reference is used otherwise.
        x_mask              optional (n,) bool: the x model trains only on these windows.  ``earn_x``
                            certifies a recipe — the person's own L1 probes — so the x model that drives
                            the cursor must be trained on exactly that population; trained on own
                            L1 + L2b + a population prior of mostly non-responders, an "earned" x was
                            below its own fixed-pool chance floor on 4 of 4 earned person-days (review).
        neutral_one_hand    True for an x earner: one-hand windows (label (±1, 0)) also enter the y model
                            as NEUTRAL — once per class at half weight, so the loss is minimised at
                            p = 0.5, an output of 0.  Trained only on feet/hands, the y model reads
                            one-hand ERD on the hand generators exactly like HANDS (mean y −0.16 on both
                            in the review), so an earner imagining LEFT drove the cursor down.  Off for
                            non-earners: they never imagine one hand in use, and their one-hand ERD is
                            bilateral (a weaker HANDS), which a neutral target would fight.
        ``x_enabled`` is not changed here: only ``earn_x`` grants or revokes the x axis.  The training
        features are kept as the idle gate's intent class (``fit_idle``); a gate fitted earlier stays
        until ``fit_idle`` is called again.
        """
        _check_sources(provenance, len(windows))
        y = self._labels(body_labels, len(windows))
        if not (np.any(y[:, 1] > 0) and np.any(y[:, 1] < 0)):
            raise ValueError("the y axis needs both feet (+1) and hands (−1) windows")
        if x_mask is not None and np.asarray(x_mask).shape != (len(windows),):
            raise ValueError(f"x_mask must be ({len(windows)},) bool, got {np.asarray(x_mask).shape}")
        X = self._feature_matrix(windows, provenance, session_references)
        self.neutral_one_hand = bool(neutral_one_hand)
        self._fit_axes(X, y, None if x_mask is None else np.asarray(x_mask, bool))
        intent = np.any(y != 0, axis=1)
        self._intent_features = X[intent]
        self._intent_keys = [_order_key(p, j) for j, (p, i) in enumerate(zip(provenance, intent)) if i]
        return self

    def fit_idle(self, windows: Sequence[np.ndarray], provenance: Sequence[Mapping],
                 session_references: Optional[Mapping[Hashable, np.ndarray]] = None) -> dict:
        """Train the idle-vs-intent gate (module docstring), arm it only if it can tell rest from
        intent, and return its report.

        windows / provenance  rest windows (source ``IDLE_SOURCE`` "L0": the idle class) and, optionally,
                              intent windows (L1 / L2b).  Without intent windows the intent class is the
                              last ``fit``'s training windows, whose features are kept for this purpose.
                              Any other source is refused: S / H / R / J windows show a moving cursor or
                              are test-only, so they may not shape what the decoder treats as rest.
        session_references    {session: rest covariance}, as in ``fit``.

        The gate is a class-balanced L2 logistic regression on the re-centred tangent features (rest
        and intent windows differ a lot in count, and the threshold must mean the same odds for every
        person).  Arming rule (pre-declared, like ``earn_x``): the gate's out-of-fold AUC over
        ``n_splits`` contiguous folds in (session, time) order must reach ``idle_auc_min``.  Rest vs
        intent rests on ABSOLUTE band power, which waxes and wanes from window to window far more than
        the feet-vs-hands contrast between sites does, so for many people a 1-s window cannot place it;
        an unarmed gate leaves the output untouched instead of zeroing intent windows at random.
        Report: n_idle, n_intent, cv_auc, auc_min, armed, threshold and the out-of-fold shares of rest
        windows gated and of intent windows passed at that threshold.
        """
        if len(provenance) != len(windows):
            raise ValueError(f"provenance has {len(provenance)} entries, windows {len(windows)}")
        allowed = TRAINABLE_SOURCES | {IDLE_SOURCE}
        bad = sorted({str(p.get("source")) for p in provenance if p.get("source") not in allowed})
        if bad:
            raise ValueError(f"the idle gate trains only on {IDLE_SOURCE} (rest) and {sorted(TRAINABLE_SOURCES)} "
                             f"(intent) windows; got source(s) {bad}")
        idle = np.array([p.get("source") == IDLE_SOURCE for p in provenance], bool)
        X = (self._feature_matrix(list(windows), list(provenance), session_references) if len(windows)
             else np.zeros((0, self._diag.size)))
        keys = [_order_key(p, j) for j, p in enumerate(provenance)]
        X_idle, k_idle = X[idle], [k for k, i in zip(keys, idle) if i]
        X_intent, k_intent = X[~idle], [k for k, i in zip(keys, idle) if not i]
        if not len(X_intent):
            if self._intent_features is None:
                raise ValueError("no intent windows: pass L1/L2b windows or call fit first")
            X_intent, k_intent = self._intent_features, self._intent_keys
        if len(X_idle) < self.n_splits or len(X_intent) < self.n_splits:
            raise ValueError(f"the idle gate needs ≥ {self.n_splits} windows per class, got {len(X_idle)} rest "
                             f"and {len(X_intent)} intent")
        Xg = np.vstack([X_idle, X_intent])
        yg = np.r_[np.ones(len(X_idle), bool), np.zeros(len(X_intent), bool)]
        all_keys = k_idle + list(k_intent)
        order = np.array(sorted(range(len(yg)), key=all_keys.__getitem__))
        Xg, yg = Xg[order], yg[order]
        oof = np.full(len(yg), np.nan)
        for tr, te in KFold(self.n_splits, shuffle=False).split(Xg):
            if yg[tr].all() or not yg[tr].any():
                continue                                    # a fold that saw one class scores nothing
            oof[te] = self._idle_classifier().fit(Xg[tr], yg[tr]).predict_proba(Xg[te])[:, 1]
        ok = np.isfinite(oof)
        both = ok.any() and yg[ok].any() and not yg[ok].all()
        auc = float(roc_auc_score(yg[ok], oof[ok])) if both else float("nan")
        armed = bool(np.isfinite(auc) and auc >= self.idle_auc_min)
        self._idle_model = self._idle_classifier().fit(Xg, yg) if armed else None
        thr = self.idle_threshold
        self.idle_report = {"n_idle": int(yg.sum()), "n_intent": int((~yg).sum()), "cv_auc": auc,
                            "auc_min": self.idle_auc_min, "armed": armed, "threshold": thr,
                            "cv_idle_gated": float(np.mean(oof[ok & yg] > thr)) if both else float("nan"),
                            "cv_intent_passed": float(np.mean(oof[ok & ~yg] <= thr)) if both else float("nan")}
        return self.idle_report

    def _idle_classifier(self) -> Pipeline:
        return make_pipeline(StandardScaler(), LogisticRegression(C=self.idle_C, class_weight="balanced",
                                                                  max_iter=2000))

    def idle_probability(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        """(n,) p(idle) of each window under the current reference; 0 without a gate, NaN where the
        window is not valid EEG."""
        F = self._valid_features(windows, None)
        out = np.full(len(F), np.nan)
        ok = np.isfinite(F).all(axis=1)
        out[ok] = self._idle_model.predict_proba(F[ok])[:, 1] if self._idle_model is not None and ok.any() else 0.0
        return out

    def _fit_axes(self, X: np.ndarray, y: np.ndarray, x_mask: Optional[np.ndarray] = None) -> None:
        """Per-axis classifiers on the windows labelled on that axis (an axis with one class gets none;
        x only inside ``x_mask``; one-hand windows as neutral y targets when ``neutral_one_hand``), and
        the training median of the re-centred log power that ``predict``'s validity check uses."""
        for k, axis in enumerate(AXES):
            m = y[:, k] != 0
            if axis == "x" and x_mask is not None:
                m &= x_mask
            if len(np.unique(y[m, k])) != 2:
                self._models[axis] = None
                continue
            Xa, ya, wa = X[m], y[m, k] > 0, np.ones(int(m.sum()))
            neutral = (y[:, k] == 0) & (y[:, 1 - k] != 0) if axis == "y" and self.neutral_one_hand else None
            if neutral is not None and neutral.any():
                n = int(neutral.sum())
                Xa = np.vstack([Xa, X[neutral], X[neutral]])
                ya = np.r_[ya, np.ones(n, bool), np.zeros(n, bool)]
                wa = np.r_[wa, np.full(2 * n, 0.5)]
            self._models[axis] = self._classifier().fit(Xa, ya, standardscaler__sample_weight=wa,
                                                        logisticregression__sample_weight=wa)
        self._log_power_ref = float(np.median(self._log_power(X)))

    def _log_power(self, F: np.ndarray) -> np.ndarray:
        """Mean log-eigenvalue of each re-centred covariance (log band power relative to the rest
        reference), read off the tangent vectors' diagonal entries."""
        return F[:, self._diag].mean(axis=1)

    def earn_x(self, windows: List[np.ndarray], body_labels: np.ndarray, provenance: Sequence[Mapping],
               session_references: Optional[Mapping[Hashable, np.ndarray]] = None, *,
               sources: Sequence[str] = ("L1",), previously_enabled: bool = False,
               previous_grant_passed: bool = False) -> dict:
        """Decide, by a pre-declared rule, whether this person has a usable lateralised MOTOR (x) signal.

        The statistic is the cross-validated x AUC (right vs left, ``n_splits`` contiguous folds in
        session/time order) on the windows of ``sources`` only — by default the fixation probes (L1,
        1–4 s after the cue).  A fixed window population matters: x was once judged on L1 alone on day 1
        and on L1 plus far more numerous long closed-loop L2b windows (where the ERD has decayed) later,
        so a genuine responder's x flipped on and off with the mix.  Other trainable sources are left
        out, not refused; S / H / R / J windows are refused (``_check_sources``).

        Rules (all pre-declared):
          grant   AUC > ``x_auc_min`` AND AUC > the ``floor_percentile`` of a null in which the x labels
                  are circularly shifted within each session (``purity.circular_shift_permutations``),
                  AND the purity rule below.  Why both thresholds: the floor alone passes a
                  statistically real but useless 0.55 with enough data, and 0.60 alone passes chance
                  with few trials or autocorrelated windows; a rotation, unlike a shuffle, keeps the
                  trial structure, so overlapping windows of one trial cannot fake skill.
          confirm the grant test is repeated after every retrain, so one pass at its 5 % size is not
                  enough: x is granted only when this grant test passes AND the previous one did
                  (``previous_grant_passed``, i.e. the last report's ``passed``).  Without it a
                  non-responder had a fresh 5 % chance every day (one false grant in the shipped
                  three-day results); two consecutive passes on cumulative data cost a true responder
                  one retrain.  Simulated (100 non-responders, 3 × 3.5 min of L1 a day, cumulative,
                  one test a day): the cumulative false-grant rate by days 1/2/3 is 7/10/11 % with a
                  single pass and 0/1/1 % with the confirm rule.  A pass that is not yet confirmed is reported as ``grant_pending``.
          keep    (``previously_enabled``) AUC > the floor AND the purity rule: once earned, x stays
                  unless a separate test — no longer above chance, or no longer motor — rejects it, so
                  sampling noise around 0.60 does not toggle the person's control scheme daily.
          purity  the out-of-fold x scores, regressed on the posterior (S7) alpha lateralisation index
                  (8–12 Hz log power right − left, ``POSTERIOR_RIGHT`` / ``POSTERIOR_LEFT``) of the same
                  windows, keep ≥ ``x_purity_retain`` of their AUC margin above 0.5 (the proposal's P1
                  increment rule).  Covert attention lateralises posterior alpha and spreads onto
                  CP5/CP6, so a non-responder can otherwise earn an attention axis through the
                  label-pure path.  The regression uses no labels, so it cannot fit the label.
        The same folds and classifier are used for the observed and every null AUC, and they are the
        deployed x model's recipe: fit the deployed decoder with ``x_mask`` on these same windows, so the
        model that drives the cursor is the one this test certified.  Sets ``x_enabled`` and ``x_report``
        and returns the numbers (auc, floor, p_value, auc_purity, purity_retained, rule, passed,
        grant_pending, …).
        """
        _check_sources(provenance, len(windows))
        y = self._labels(body_labels, len(windows))
        allowed = set(sources)
        m = np.flatnonzero((y[:, 0] != 0) & np.array([p.get("source") in allowed for p in provenance], bool))
        rule = "keep" if previously_enabled else "grant"
        report = {"enabled": False, "auc": float("nan"), "floor": float("nan"), "p_value": float("nan"),
                  "auc_purity": float("nan"), "purity_retained": float("nan"), "rule": rule,
                  "sources": sorted(allowed), "auc_min": self.x_auc_min, "floor_percentile": self.floor_percentile,
                  "purity_retain_min": self.x_purity_retain, "n_permutations": self.n_permutations,
                  "n_windows": int(len(m)), "n_sessions": len({provenance[i].get("session") for i in m}),
                  "passed": False, "grant_pending": False, "previous_grant_passed": bool(previous_grant_passed)}
        labels = y[m, 0] > 0
        if len(m) < 2 * self.n_splits or labels.all() or not labels.any():
            report["reason"] = "too few x-labelled windows or only one x class"
            self.x_enabled, self.x_report = False, report
            return report
        prov = [provenance[i] for i in m]
        order = np.array(sorted(range(len(m)), key=lambda j: (str(prov[j].get("session")),
                                                               float(prov[j].get("t", j)), j)))
        win = [windows[m[j]] for j in order]
        X = self._feature_matrix(win, [prov[j] for j in order], session_references)
        labels, prov = labels[order], [prov[j] for j in order]
        # The scaler never sees labels, so each fold is standardised once and reused by every null
        # draw: identical to refitting the deployed pipeline per draw, at a fraction of the cost.
        folds = []
        for tr, te in KFold(self.n_splits, shuffle=False).split(X):
            scaler = StandardScaler().fit(X[tr])
            folds.append((tr, te, scaler.transform(X[tr]), scaler.transform(X[te])))
        scores = self._cv_scores(labels, folds)
        auc = float(roc_auc_score(labels, scores))
        perms = circular_shift_permutations(prov, self.n_permutations, np.random.default_rng(self.seed))
        null = np.array([roc_auc_score(labels[p], self._cv_scores(labels[p], folds)) for p in perms])
        floor = float(np.percentile(null, self.floor_percentile))
        auc_pure = float(roc_auc_score(labels, _residualise(scores, self._posterior_lateralisation(win))))
        retained = (auc_pure - 0.5) / (auc - 0.5) if auc > 0.5 else float("nan")
        pure = bool(np.isfinite(retained) and retained >= self.x_purity_retain)
        above = auc > floor
        passed = bool(above and pure and (previously_enabled or auc > self.x_auc_min))
        enabled = bool(passed and (previously_enabled or previous_grant_passed))
        report.update(auc=auc, floor=floor, p_value=float((1 + np.sum(null >= auc)) / (1 + len(null))),
                      auc_purity=auc_pure, purity_retained=float(retained), enabled=enabled, passed=passed,
                      grant_pending=bool(passed and not enabled))
        if enabled:
            report["reason"] = "x earned (second consecutive grant pass)" if rule == "grant" else "x kept"
        elif passed:
            report["reason"] = (f"grant test passed (AUC {auc:.3f}); x is granted when the next grant test "
                                "passes too")
        elif not pure:
            report["reason"] = (f"x skill does not survive regressing out posterior alpha lateralisation "
                                f"(AUC {auc:.3f} → {auc_pure:.3f}; keeps {retained:.0%} < "
                                f"{self.x_purity_retain:.0%} of its margin): not a motor axis")
        elif rule == "keep":
            report["reason"] = f"AUC {auc:.3f} no longer exceeds the p{self.floor_percentile:g} floor {floor:.3f}"
        else:
            report["reason"] = (f"AUC {auc:.3f} must exceed both {self.x_auc_min:.2f} and the p"
                                f"{self.floor_percentile:g} circular-shift floor {floor:.3f}")
        self.x_enabled, self.x_report = enabled, report
        return report

    def _posterior_lateralisation(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        """S7 index per window: 8–12 Hz log power over the right minus the left parieto-occipital
        channels, last ``window_s``.  Rises with covert attention to the right (left-hemisphere alpha
        falls).  An auditor-type EEG feature for the purity rule only, never a decoder input."""
        names = list(self.cfg.hardware.channels)
        missing = [c for c in POSTERIOR_LEFT + POSTERIOR_RIGHT if c not in names]
        if missing:
            raise ValueError(f"the x-purity check needs posterior channels {missing} on the cap")
        left = [names.index(c) for c in POSTERIOR_LEFT]
        right = [names.index(c) for c in POSTERIOR_RIGHT]
        sos = sps.butter(self.filter_order, self.alpha_band_hz, "bandpass", fs=self.fs, output="sos")
        out = np.empty(len(windows))
        for i, w in enumerate(windows):
            a = sps.sosfiltfilt(sos, np.asarray(w, np.float64)[-self.n_win:], axis=0)
            p = np.mean(a ** 2, axis=0) + 1e-12
            out[i] = np.log(p[right].mean()) - np.log(p[left].mean())
        return out

    def _cv_scores(self, labels: np.ndarray, folds: list) -> np.ndarray:
        """Out-of-fold decision scores over pre-standardised folds ``(train, test, X_train, X_test)``.
        A training fold with one class scores its test fold 0 (no information) rather than being
        dropped, so every label set is scored on the same windows."""
        score = np.zeros(len(labels))
        for tr, te, x_tr, x_te in folds:
            if labels[tr].all() or not labels[tr].any():
                continue
            score[te] = self._logistic().fit(x_tr, labels[tr]).decision_function(x_te)
        return score

    # -- decoding -----------------------------------------------------------------------------------
    def _valid_features(self, windows: Sequence[np.ndarray], good: Optional[np.ndarray]) -> np.ndarray:
        """Tangent features per window, a NaN row where the window is not valid EEG (module docstring):
        non-finite or gated central samples, a dead segment, or re-centred band power ``POWER_LIMIT``
        away from the training median."""
        central = [self._window(w) for w in windows]          # shape errors first: they are wiring bugs
        if self._models["y"] is None:
            raise RuntimeError("decoder is not fitted")
        central_bad = good is not None and not np.all(np.asarray(good, bool)[self._idx])
        F = np.full((len(central), int(self._diag.size)), np.nan)
        for i, x in enumerate(central):
            if central_bad or not np.isfinite(x).all():
                continue
            xf = sps.sosfiltfilt(self._sos, self._referenced(x), axis=0)
            seg = np.array([np.mean(part ** 2) for part in np.array_split(xf, N_SEGMENTS)])
            if not seg.mean() > 0 or seg.min() < MIN_SEGMENT_SHARE * seg.mean():
                continue
            F[i] = self._tangent(oas(xf)[0], self._ref_isqrt)
        off = np.abs(self._log_power(F) - self._log_power_ref) > np.log(POWER_LIMIT)
        F[off] = np.nan
        return F

    def _velocity(self, F: np.ndarray) -> np.ndarray:
        out = np.full((len(F), 2), np.nan)
        ok = np.isfinite(F).all(axis=1)
        out[ok] = 0.0
        if not ok.any():
            return out
        for k, axis in enumerate(AXES):
            model = self._models[axis]
            if model is None or (axis == "x" and not self.x_enabled):
                continue
            out[ok, k] = 2.0 * model.predict_proba(F[ok])[:, 1] - 1.0
        if self._idle_model is not None:
            rows = np.flatnonzero(ok)
            idle = self._idle_model.predict_proba(F[rows])[:, 1] > self.idle_threshold
            out[rows[idle]] = 0.0                      # exactly zero: neither cursor nor arm drifts at rest
        return out

    def predict(self, window: np.ndarray, good: Optional[np.ndarray] = None) -> np.ndarray:
        """(2,) body velocity (x, y) in [−1, 1]: 2p − 1 per axis; x is 0 unless earned; exactly (0, 0)
        when the idle gate puts p(idle) above ``idle_threshold``.  NaN (both axes) when the window is
        not valid EEG; ``good`` is the real-time loop's per-channel mask (amplitude gate and repaired
        samples), and a bad central channel makes the window invalid."""
        return self._velocity(self._valid_features([window], good))[0]

    def predict_many(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        """(n, 2) — the same as ``predict`` window by window (no state is carried between windows); an
        invalid window gives a NaN row instead of aborting the whole call."""
        return self._velocity(self._valid_features(windows, None))

    def ungated(self) -> "BodyCoordinateDecoder":
        """A copy without the idle gate: the axis models' own output, for offline discriminability (e.g.
        an AUC that zeroed windows would tie).  Never a command source.  The copy has its own model
        table, so fitting it can never replace the axis models of the decoder it came from (the fitted
        pipelines themselves are shared read-only: ``fit`` builds new ones, it never refits them)."""
        d = copy.copy(self)
        d._models = dict(self._models)
        d._idle_model = None
        return d

    def loop_config(self, cfg: Optional[Config] = None) -> Config:
        """``cfg`` (default: the decoder's) with ``decoder.window_s`` set to this decoder's window, so
        ``realtime.RealTimeLoop`` hands ``predict`` exactly the window length it was trained on, and
        ``task.control_hz`` set to ``LOOP_HZ`` (the design's 16-Hz update; the config default is 10 Hz)."""
        cfg = cfg or self.cfg
        return replace(cfg, decoder=replace(cfg.decoder, window_s=self.window_s),
                       task=replace(cfg.task, control_hz=LOOP_HZ))
