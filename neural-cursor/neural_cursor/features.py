"""Windowed features.  Two families, both cheap enough for a 10 Hz control loop on a laptop CPU.

* log band power per channel and band (the Wolpaw/SMR lineage, and alpha lateralisation)
* covariance → tangent-space vector (the Riemannian lineage; shrinkage, Euclidean reference —
  the deployed RLNF pipeline's reference was the arithmetic mean too, and that is stated here, not hidden)
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .config import Decoder


def log_band_power(window: np.ndarray, fs: int, bands=Decoder().bands_hz) -> np.ndarray:
    """window: (n_samples, n_channels) → (n_channels × n_bands,) log10 power via Welch."""
    nper = min(len(window), fs)
    f, p = sps.welch(window, fs=fs, nperseg=nper, axis=0)
    out = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        out.append(np.log10(p[m].mean(axis=0) + 1e-12))
    return np.concatenate(out)


def slow_shift(window: np.ndarray, frac: float = 0.2) -> np.ndarray:
    """Signed slow potential per channel: mean of the last `frac` of the window minus the first.

    Band power discards the sign of a slow potential, so a gaze-position EOG leak is largely invisible
    to it; a time-domain decoder (any 'kinematics from low-frequency EEG' design) sees it directly.
    This feature exists so the EOG-only surrogate can *demonstrate* that leak, and so a decoder that
    uses it can be shown to fail its ocular control.
    """
    k = max(1, int(len(window) * frac))
    return window[-k:].mean(axis=0) - window[:k].mean(axis=0)


def shrink_cov(window: np.ndarray, shrinkage: float) -> np.ndarray:
    x = window - window.mean(axis=0, keepdims=True)
    c = x.T @ x / max(1, len(x) - 1)
    mu = np.trace(c) / c.shape[0]
    return (1 - shrinkage) * c + shrinkage * mu * np.eye(c.shape[0])


def _sym_fn(c: np.ndarray, fn) -> np.ndarray:
    w, v = np.linalg.eigh(c)
    w = np.clip(w, 1e-10, None)
    return (v * fn(w)) @ v.T


def tangent_space(c: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """log-map of c at the reference ref, upper triangle with √2 off-diagonal weighting."""
    isq = _sym_fn(ref, lambda w: w ** -0.5)
    s = _sym_fn(isq @ c @ isq, np.log)
    iu = np.triu_indices(s.shape[0])
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return s[iu] * coef


class OnlineRecenter:
    """Reference covariance that drifts toward the session: ref = blend(train_ref, session_mean).

    Blend weight n/(n + prior) so the reference moves only once the buffer holds more than a few
    windows.  Arithmetic (Euclidean) mean of the buffer, stated plainly.
    """

    def __init__(self, train_ref: np.ndarray, buffer: int, prior: float):
        self.train_ref = train_ref
        self.buffer, self.prior = buffer, prior
        self._covs: list = []

    def push(self, c: np.ndarray) -> None:
        self._covs.append(c)
        if len(self._covs) > self.buffer:
            self._covs.pop(0)

    def reference(self) -> np.ndarray:
        n = len(self._covs)
        if n == 0:
            return self.train_ref
        w = n / (n + self.prior)
        return (1 - w) * self.train_ref + w * np.mean(self._covs, axis=0)
