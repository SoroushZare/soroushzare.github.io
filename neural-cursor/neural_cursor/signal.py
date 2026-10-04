"""Causal preprocessing with persistent filter state, re-referencing and amplitude gating.

Every filter is stateful (``sosfilt`` with ``zi``) so online and offline paths are bit-identical:
the offline trainer streams the recording through the same object in the same chunk order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal as sps

from .config import Hardware, Preprocessing


@dataclass
class CausalFilter:
    hw: Hardware
    pre: Preprocessing
    _sos: list = field(default_factory=list)
    _zi: list = field(default_factory=list)

    def __post_init__(self):
        fs = self.hw.fs
        self._sos = [
            sps.butter(self.pre.filter_order, self.pre.hp_hz, "highpass", fs=fs, output="sos"),
            sps.butter(self.pre.filter_order, self.pre.lp_hz, "lowpass", fs=fs, output="sos"),
        ]
        if self.pre.notch_hz:
            b, a = sps.iirnotch(self.pre.notch_hz, 30.0, fs=fs)
            self._sos.append(sps.tf2sos(b, a))
        self._zi = [np.repeat(sps.sosfilt_zi(s)[:, None, :], self.hw.n_channels, axis=1) for s in self._sos]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (n_samples, n_channels) → filtered, same shape, state carried across calls."""
        y = x.T.astype(np.float64)
        for k, sos in enumerate(self._sos):
            y, self._zi[k] = sps.sosfilt(sos, y, axis=-1, zi=self._zi[k])
        return y.T


def common_average(x: np.ndarray, good: Optional[np.ndarray] = None) -> np.ndarray:
    """CAR over the good channels only; bad channels still get the reference subtracted."""
    ref = x[:, good].mean(axis=1, keepdims=True) if good is not None else x.mean(axis=1, keepdims=True)
    return x - ref


@dataclass
class AmplitudeGate:
    """Rule 53: flag a channel whose window RMS exceeds factor × the session's median channel RMS.

    The median is tracked online from the windows seen so far (robust to a single railed channel).
    FP1/FP2 are exempt because blinks are physiology there, not faults.
    """
    hw: Hardware
    pre: Preprocessing
    _history: list = field(default_factory=list)

    def __post_init__(self):
        self.exempt = np.array([c in self.pre.gate_exempt for c in self.hw.channels])

    def update(self, window: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(window.astype(np.float64) ** 2, axis=0))
        self._history.append(rms)
        if len(self._history) > 600:
            self._history.pop(0)
        session_median = np.median(np.stack(self._history))     # one scalar across channels and time
        bad = rms > self.pre.gate_rms_factor * max(session_median, 1e-6)
        bad &= ~self.exempt
        return ~bad                                             # True = good


def eog_proxy(x: np.ndarray, hw: Hardware) -> np.ndarray:
    """(n, 2): vertical ≈ mean(FP1, FP2), horizontal ≈ F8 − F7.  A proxy, not an EOG pair.

    Use it for the EOG-only surrogate decoder and for regression; never claim it isolates the eyes.
    """
    i = {c: k for k, c in enumerate(hw.channels)}
    v = 0.5 * (x[:, i["FP1"]] + x[:, i["FP2"]])
    h = x[:, i["F8"]] - x[:, i["F7"]]
    return np.stack([v, h], axis=1)


def decimate(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Plain subsampling after the causal 40 Hz low-pass; no new filter, so no new group delay."""
    q = fs_in // fs_out
    assert fs_in % fs_out == 0, "decimation factor must be an integer"
    return x[::q]
