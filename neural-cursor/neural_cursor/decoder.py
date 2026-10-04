"""Decoders.  Two layers, both deliberately simple and linear so every number can be audited:

* ``VelocityDecoder``  continuous: 0.5 s window → log band power (+ optional tangent-space) → ridge
                       regression to a 2-D command.  This is the layer the bitter lesson scales:
                       swap ``features`` for a pretrained encoder's embeddings without touching the rest.
* ``EvidenceDecoder``  discrete: 0.2–0.8 s post-action window → tangent space → logistic regression
                       giving p(accept).  Online covariance re-centring with a stated Euclidean reference.

Both expose ``fit(X, y)`` / ``predict(window)`` and nothing else.  Any decoder that respects the
interface can be dropped into ``TrialRunner`` as ``command_fn`` / ``evidence_fn``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .config import Decoder as DecoderCfg
from .features import OnlineRecenter, log_band_power, shrink_cov, slow_shift, tangent_space


@dataclass
class VelocityDecoder:
    cfg: DecoderCfg
    fs: int
    channels: Optional[np.ndarray] = None      # boolean mask of channels used (surrogates set this)
    feature_set: str = "bandpower"             # "bandpower" or "bandpower+slow" (time-domain, leak-prone)
    _scaler: StandardScaler = field(default_factory=StandardScaler)
    _model: Ridge = field(init=False)

    def __post_init__(self):
        self._model = Ridge(alpha=self.cfg.ridge_alpha)

    def features(self, window: np.ndarray) -> np.ndarray:
        if self.channels is not None:
            window = window[:, self.channels]
        f = log_band_power(window, self.fs, self.cfg.bands_hz)
        if self.feature_set == "bandpower+slow":
            f = np.concatenate([f, slow_shift(window)])
        return f

    def fit(self, windows: List[np.ndarray], targets: np.ndarray) -> "VelocityDecoder":
        X = np.stack([self.features(w) for w in windows])
        X = self._scaler.fit_transform(X)
        self._model.fit(X, np.asarray(targets, float))
        return self

    def predict(self, window: np.ndarray) -> np.ndarray:
        x = self._scaler.transform(self.features(window)[None])
        return np.clip(self._model.predict(x)[0], -1, 1)

    def predict_many(self, windows: List[np.ndarray]) -> np.ndarray:
        X = self._scaler.transform(np.stack([self.features(w) for w in windows]))
        return np.clip(self._model.predict(X), -1, 1)


@dataclass
class EvidenceDecoder:
    """p(accept) from a post-action epoch.

    feature_set "erp": the epoch decimated to 25 Hz, channels × time flattened (a phase-locked
    transient lives in the time course, not in the covariance) — the xDAWN/ERP lineage, kept linear.
    feature_set "cov": covariance → tangent space (induced power changes; blind to phase-locked ERPs).
    Both go through a scaler and an L2 logistic regression; the online covariance re-centring applies
    to "cov" only.
    """
    cfg: DecoderCfg
    feature_set: str = "erp"
    erp_hz: int = 25
    fs: int = 500
    channels: Optional[np.ndarray] = None
    _ref: Optional[np.ndarray] = None
    _recenter: Optional[OnlineRecenter] = None
    _scaler: StandardScaler = field(default_factory=StandardScaler)
    _model: LogisticRegression = field(init=False)
    threshold: float = 0.5

    def __post_init__(self):
        self._model = LogisticRegression(C=self.cfg.evidence_C, max_iter=2000)

    def _cov(self, window: np.ndarray) -> np.ndarray:
        if self.channels is not None:
            window = window[:, self.channels]
        return shrink_cov(window, self.cfg.tangent_shrinkage)

    def _erp(self, window: np.ndarray) -> np.ndarray:
        if self.channels is not None:
            window = window[:, self.channels]
        q = max(1, self.fs // self.erp_hz)
        n = (len(window) // q) * q
        return window[:n].reshape(n // q, q, -1).mean(axis=1).ravel()   # block-average then flatten

    def fit(self, windows: List[np.ndarray], labels: np.ndarray) -> "EvidenceDecoder":
        if self.feature_set == "erp":
            X = self._scaler.fit_transform(np.stack([self._erp(w) for w in windows]))
            self._model.fit(X, np.asarray(labels, int))
            return self
        covs = [self._cov(w) for w in windows]
        self._ref = np.mean(covs, axis=0)               # Euclidean reference, stated
        X = np.stack([tangent_space(c, self._ref) for c in covs])
        X = self._scaler.fit_transform(X)
        self._model.fit(X, np.asarray(labels, int))
        self._recenter = OnlineRecenter(self._ref, self.cfg.recenter_buffer, self.cfg.recenter_prior)
        return self

    def predict_proba(self, window: np.ndarray, recenter: bool = True) -> float:
        if self.feature_set == "erp":
            return float(self._model.predict_proba(self._scaler.transform(self._erp(window)[None]))[0, 1])
        c = self._cov(window)
        ref = self._ref
        if recenter and self._recenter is not None:
            self._recenter.push(c)
            ref = self._recenter.reference()
        x = self._scaler.transform(tangent_space(c, ref)[None])
        return float(self._model.predict_proba(x)[0, 1])

    def predict_many(self, windows: List[np.ndarray]) -> np.ndarray:
        """Batch scoring against the frozen training reference — a *different* decoder from the
        sequential online path (RLNF C-P008); use it only for offline cross-validation."""
        if self.feature_set == "erp":
            return self._model.predict_proba(self._scaler.transform(np.stack([self._erp(w) for w in windows])))[:, 1]
        X = self._scaler.transform(np.stack([tangent_space(self._cov(w), self._ref) for w in windows]))
        return self._model.predict_proba(X)[:, 1]
