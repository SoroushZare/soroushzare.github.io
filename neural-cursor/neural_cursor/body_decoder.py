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
  → zero-phase 8–30 Hz Butterworth band-pass *within the window* (window-local, so the online path
    and offline re-scoring see bit-identical features; no state to carry)
  → OAS-shrunk covariance (11 channels × 500 samples: shrinkage is needed and OAS picks it in closed form)
  → re-centred on a rest reference R: C ↦ R^{-1/2} C R^{-1/2}  (Zanini et al. 2018)
  → tangent space at the identity (matrix log, upper triangle, √2 off-diagonal weights)
  → standardise → L2 logistic regression feet-vs-hands (y) and right-vs-left (x)
  → output 2p − 1 per axis.

Re-centring is the only use of unlabelled data: it moves each session's rest covariance to the
identity, so a cap re-fit or an electrode gain change between sessions — which acts on covariances as
a congruence — is largely removed before the classifier sees the data.  It needs no label and no
display, so it may be computed from any EEG block, sighted ones included (the champion design's only
use of sighted data).  With no reference set R = I, and re-centring is the identity map.

Pure-EEG rule: the only arrays that reach this class are EEG windows; provenance is read only for the
label-source check and for the session key that selects a rest reference, never as a feature.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Hashable, List, Mapping, Optional, Sequence

import numpy as np
from pyriemann.geometry.mean import mean_riemann
from scipy import signal as sps
from sklearn.covariance import oas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .features import _sym_fn
from .purity import circular_shift_permutations

#: Central-11 montage on the g.tec 32 cap: the sensorimotor strip and its neighbours.  It excludes the
#: frontal (ocular) and parieto-occipital (visual attention) channels by construction; purity still
#: rests on the label rule and the certification tests, because CP channels share posterior sources.
CENTRAL11 = ("FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6")

#: Provenance sources whose labels the display cannot identify; the only ones a deployed decoder may
#: train on.  S (sighted), H (held-out mapping) and R (pixel replay) are refused.
TRAINABLE_SOURCES = frozenset({"L1", "L2b"})

#: Output axes, in the order ``predict`` returns them.
AXES = ("x", "y")


def _tangent_at_identity(c: np.ndarray) -> np.ndarray:
    """Upper triangle of log(c) with √2 off-diagonal weights, so the Euclidean norm of the vector
    equals the Frobenius norm of the matrix log (the affine-invariant distance to the identity)."""
    s = _sym_fn(c, np.log)
    iu = np.triu_indices(s.shape[0])
    return s[iu] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))


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
    n_permutations     200 circular shifts for that null (p95 resolution 0.005).
    n_splits           5 contiguous (time-block) folds for the x-earning cross-validation: only a fold
                       boundary can split a trial, so a test window is almost never scored by a model
                       trained on the other windows of its own trial.
    seed               seed of the circular-shift draws, so ``earn_x`` is deterministic.
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
    x_enabled: bool = False
    x_report: Optional[dict] = None
    _ref_isqrt: np.ndarray = field(init=False, repr=False)
    _models: Dict[str, Optional[Pipeline]] = field(init=False, repr=False)

    def __post_init__(self):
        names = list(self.cfg.hardware.channels)
        missing = [c for c in self.channels if c not in names]
        if missing:
            raise ValueError(f"channels {missing} are not on the cap {self.cfg.hardware.channels}")
        self._idx = np.array([names.index(c) for c in self.channels])
        self.n_win = int(round(self.window_s * self.fs))
        self._sos = sps.butter(self.filter_order, self.band_hz, "bandpass", fs=self.fs, output="sos")
        self._ref_isqrt = np.eye(len(self.channels))
        self._models = {axis: None for axis in AXES}

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

    def covariance(self, window: np.ndarray) -> np.ndarray:
        """8–30 Hz OAS covariance of the central channels (before re-centring)."""
        x = sps.sosfiltfilt(self._sos, self._window(window), axis=0)
        return oas(x)[0]

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
        self._ref_isqrt = self._isqrt(reference) if reference is not None else np.eye(len(self.channels))
        return self

    def _isqrt(self, reference: np.ndarray) -> np.ndarray:
        r = np.asarray(reference, dtype=np.float64)
        k = len(self.channels)
        if r.shape != (k, k) or not np.allclose(r, r.T) or np.linalg.eigvalsh(r).min() <= 0:
            raise ValueError(f"reference must be a symmetric positive-definite {k}×{k} matrix")
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
            session_references: Optional[Mapping[Hashable, np.ndarray]] = None) -> "BodyCoordinateDecoder":
        """Train the y (feet vs hands) and, when both classes are present, x (right vs left) models.

        body_labels         (n, 2) cued body vector (x, y) in {−1, 0, 1}²; a window enters an axis's
                            training set only where that axis's label is non-zero (a diagonal cue
                            enters both, a LEFT cue only x).
        provenance          one dict per window; ``source`` must be L1 or L2b for every window.
        session_references  optional {session: rest covariance} so each training session is re-centred
                            on its own rest; the decoder's current reference is used otherwise.
        ``x_enabled`` is not changed here: only ``earn_x`` grants or revokes the x axis.
        """
        _check_sources(provenance, len(windows))
        y = self._labels(body_labels, len(windows))
        if not (np.any(y[:, 1] > 0) and np.any(y[:, 1] < 0)):
            raise ValueError("the y axis needs both feet (+1) and hands (−1) windows")
        X = self._feature_matrix(windows, provenance, session_references)
        for k, axis in enumerate(AXES):
            m = y[:, k] != 0
            if len(np.unique(y[m, k])) < 2:
                self._models[axis] = None
                continue
            self._models[axis] = self._classifier().fit(X[m], y[m, k] > 0)
        return self

    def earn_x(self, windows: List[np.ndarray], body_labels: np.ndarray, provenance: Sequence[Mapping],
               session_references: Optional[Mapping[Hashable, np.ndarray]] = None) -> dict:
        """Decide, by a pre-declared rule, whether this person has a usable lateralised (x) signal.

        x is enabled iff its cross-validated AUC (right vs left, contiguous folds in session/time
        order) both exceeds ``x_auc_min`` and beats the ``floor_percentile`` of a null in which the x
        labels are circularly shifted within each session (``purity.circular_shift_permutations``).
        Why both: the floor alone passes a statistically real but useless 0.55 with enough data, and
        the 0.60 alone passes chance with few trials or autocorrelated windows; a rotation, unlike a
        shuffle, keeps the trial structure, so overlapping windows of one trial cannot fake skill.
        The same folds and classifier are used for the observed and every null AUC.  Sets
        ``x_enabled`` and ``x_report`` and returns the numbers.
        """
        _check_sources(provenance, len(windows))
        y = self._labels(body_labels, len(windows))
        m = np.flatnonzero(y[:, 0] != 0)
        report = {"enabled": False, "auc": float("nan"), "floor": float("nan"), "p_value": float("nan"),
                  "auc_min": self.x_auc_min, "floor_percentile": self.floor_percentile,
                  "n_permutations": self.n_permutations, "n_windows": int(len(m)),
                  "n_sessions": len({provenance[i].get("session") for i in m})}
        labels = y[m, 0] > 0
        if len(m) < 2 * self.n_splits or labels.all() or not labels.any():
            report["reason"] = "too few x-labelled windows or only one x class"
            self.x_enabled, self.x_report = False, report
            return report
        prov = [provenance[i] for i in m]
        order = np.array(sorted(range(len(m)), key=lambda j: (str(prov[j].get("session")),
                                                               float(prov[j].get("t", j)), j)))
        X = self._feature_matrix([windows[i] for i in m], prov, session_references)[order]
        labels, prov = labels[order], [prov[j] for j in order]
        # The scaler never sees labels, so each fold is standardised once and reused by every null
        # draw: identical to refitting the deployed pipeline per draw, at a fraction of the cost.
        folds = []
        for tr, te in KFold(self.n_splits, shuffle=False).split(X):
            scaler = StandardScaler().fit(X[tr])
            folds.append((tr, te, scaler.transform(X[tr]), scaler.transform(X[te])))
        auc = self._cv_auc(labels, folds)
        perms = circular_shift_permutations(prov, self.n_permutations, np.random.default_rng(self.seed))
        null = np.array([self._cv_auc(labels[p], folds) for p in perms])
        floor = float(np.percentile(null, self.floor_percentile))
        report.update(auc=auc, floor=floor, p_value=float((1 + np.sum(null >= auc)) / (1 + len(null))),
                      enabled=bool(auc > floor and auc > self.x_auc_min))
        report["reason"] = ("x earned" if report["enabled"] else
                            f"AUC {auc:.3f} must exceed both {self.x_auc_min:.2f} and the p"
                            f"{self.floor_percentile:g} circular-shift floor {floor:.3f}")
        self.x_enabled, self.x_report = report["enabled"], report
        return report

    def _cv_auc(self, labels: np.ndarray, folds: list) -> float:
        """Pooled out-of-fold AUC over pre-standardised folds ``(train, test, X_train, X_test)``.  A
        training fold with one class scores its test fold 0 (no information) rather than being dropped,
        so every label set is scored on the same windows."""
        score = np.zeros(len(labels))
        for tr, te, x_tr, x_te in folds:
            if labels[tr].all() or not labels[tr].any():
                continue
            score[te] = self._logistic().fit(x_tr, labels[tr]).decision_function(x_te)
        return float(roc_auc_score(labels, score))

    # -- decoding -----------------------------------------------------------------------------------
    def _velocity(self, F: np.ndarray) -> np.ndarray:
        if self._models["y"] is None:
            raise RuntimeError("decoder is not fitted")
        out = np.zeros((len(F), 2))
        for k, axis in enumerate(AXES):
            model = self._models[axis]
            if model is None or (axis == "x" and not self.x_enabled):
                continue
            out[:, k] = 2.0 * model.predict_proba(F)[:, 1] - 1.0
        return out

    def predict(self, window: np.ndarray) -> np.ndarray:
        """(2,) body velocity (x, y) in [−1, 1]: 2p − 1 per axis; x is 0 unless earned."""
        return self._velocity(self.features(window)[None])[0]

    def predict_many(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        """(n, 2) — the same as ``predict`` window by window (no state is carried between windows)."""
        return self._velocity(np.stack([self.features(w) for w in windows]))

    def loop_config(self, cfg: Optional[Config] = None) -> Config:
        """``cfg`` (default: the decoder's) with ``decoder.window_s`` set to this decoder's window, so
        ``realtime.RealTimeLoop`` hands ``predict`` exactly the window length it was trained on."""
        cfg = cfg or self.cfg
        return replace(cfg, decoder=replace(cfg.decoder, window_s=self.window_s))
