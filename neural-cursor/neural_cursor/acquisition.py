"""EEG sources: a Lab Streaming Layer inlet and a deterministic simulator.

Both expose the same interface so every downstream component is tested headless:

    src.start(); chunk, ts = src.pull()  # (n_samples, n_channels) float32 µV, LSL timestamps
    src.ring.latest(n)                   # last n samples, oldest first

The simulator embeds a known, small, *cortical-like* signal for a given latent intent so the whole
pipeline can be regression-tested for exact numbers; it is never a model of real EEG.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import Hardware


class RingBuffer:
    """Fixed-length sample buffer; samples are rows, channels columns."""

    def __init__(self, n_samples: int, n_channels: int):
        self.buf = np.zeros((n_samples, n_channels), dtype=np.float32)
        self.ts = np.zeros(n_samples, dtype=np.float64)
        self.n = n_samples
        self.head = 0
        self.filled = 0

    def push(self, x: np.ndarray, ts: np.ndarray) -> None:
        k = len(x)
        if k == 0:
            return
        if k >= self.n:                 # keep only the newest n
            x, ts, k = x[-self.n:], ts[-self.n:], self.n
        idx = (self.head + np.arange(k)) % self.n
        self.buf[idx] = x
        self.ts[idx] = ts
        self.head = (self.head + k) % self.n
        self.filled = min(self.n, self.filled + k)

    def latest(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        n = min(n, self.filled)
        idx = (self.head - n + np.arange(n)) % self.n
        return self.buf[idx], self.ts[idx]


@dataclass
class SimulatedEEG:
    """Deterministic surrogate EEG at the cap's sampling rate.

    Background: pink-ish noise per channel (1/f via cumulative AR), 10 Hz alpha with a posterior
    topography, blinks on FP1/FP2.  Intent: a 2-D latent (vx, vy) in [−1, 1]² that the caller sets;
    it modulates (i) a small lateralised alpha asymmetry and (ii) a small low-frequency potential on
    central channels, both with effect sizes chosen so a linear decoder reaches r ≈ 0.3–0.4, the range
    the non-invasive literature reports — never better.
    """
    hw: Hardware = Hardware()
    seed: int = 0
    noise_uv: float = 15.0
    intent_gain_uv: float = 8.0   # 8 → full lateralisation at |vx| = 1; halve it for a 'weak user'
    eye_gain_uv: float = 0.0      # set > 0 to simulate gaze-following EOG (the confound to detect)
    errp_uv: float = 6.0          # peak of the frontocentral transient evoked by a wrong agent action
    drift: float = 0.0            # per-minute multiplicative gain drift (dry-electrode non-stationarity)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.t0 = time.time()
        self.n_sent = 0
        self.intent = np.zeros(2)
        self.ring = RingBuffer(self.hw.fs * 60, self.hw.n_channels)
        ch = list(self.hw.channels)
        self._i = {c: ch.index(c) for c in ch}
        self._state = np.zeros(self.hw.n_channels)
        # Topographies (unit vectors over channels)
        self._alpha_L = self._topo(["PO7", "PO3", "P7", "P3"])
        self._alpha_R = self._topo(["PO8", "PO4", "P8", "P4"])
        self._central = self._topo(["C3", "Cz", "C4", "CP1", "CP2"])
        self._fp = self._topo(["FP1", "FP2"])
        self._h_eog = self._topo(["F8"]) - self._topo(["F7"])
        self._fcz = self._topo(["Fz", "FC1", "FC2", "Cz"]) * 2.0   # peak amplitude at each of 4 sites
        self._pending: list = []      # (onset_sample, amplitude) of queued evoked transients
        self._wall = 0.0

    def _topo(self, names):
        v = np.zeros(self.hw.n_channels)
        for n in names:
            v[self._i[n]] = 1.0
        return v / max(1.0, np.linalg.norm(v))

    def start(self):
        self.t0 = time.time()
        self.n_sent = 0

    def set_intent(self, vx: float, vy: float) -> None:
        self.intent = np.clip([vx, vy], -1, 1)

    def evoke(self, wrong: bool) -> None:
        """Queue the response to an agent action.  A wrong action evokes a biphasic frontocentral
        transient (negative peak ≈0.30 s, positive ≈0.45 s), amplitude jittered ±40 %; a correct
        action evokes a quarter-size version so the classes are not trivially 'signal vs nothing'."""
        amp = self.errp_uv * (1.0 if wrong else 0.25) * self.rng.uniform(0.6, 1.4)
        self._pending.append((self.n_sent, amp))

    def _evoked(self, n: int) -> np.ndarray:
        if not self._pending:
            return 0.0
        k = (self.n_sent + np.arange(n))[:, None]
        out = np.zeros((n, self.hw.n_channels))
        keep = []
        for onset, amp in self._pending:
            dt = (k - onset) / self.hw.fs
            wave = -amp * np.exp(-((dt - 0.30) / 0.05) ** 2) + 0.6 * amp * np.exp(-((dt - 0.45) / 0.07) ** 2)
            out += wave * self._fcz[None]
            if (self.n_sent + n - onset) / self.hw.fs < 0.9:
                keep.append((onset, amp))
        self._pending = keep
        return out

    def generate(self, n: int) -> np.ndarray:
        """n samples of (n, n_channels) µV for the current intent (pure function of RNG state)."""
        fs = self.hw.fs
        t = (self.n_sent + np.arange(n)) / fs
        # background: AR(1)-smoothed white noise ≈ 1/f-ish
        w = self.rng.standard_normal((n, self.hw.n_channels)) * self.noise_uv
        x = np.empty_like(w)
        s = self._state
        for k in range(n):
            s = 0.9 * s + 0.44 * w[k]
            x[k] = s
        self._state = s
        # alpha with posterior topography (≈10 µV per occipital site, realistic); lateralised by
        # horizontal intent: alpha falls contralateral to the attended side by up to `lat` × 100 %
        alpha = np.sin(2 * np.pi * 10 * t)[:, None]
        vx, vy = self.intent
        base, lat = 20.0, min(1.0, self.intent_gain_uv / 8.0)
        x += alpha * (base * (1 - lat * max(vx, 0)) * self._alpha_R[None]
                      + base * (1 - lat * max(-vx, 0)) * self._alpha_L[None])
        # central beta (20 Hz) event-related desynchronisation scaled by vertical intent — a
        # power change, because a signed slow potential is invisible to a band-power feature
        beta = np.sin(2 * np.pi * 20 * t)[:, None]
        x += beta * (10.0 * (1 - 0.5 * lat * vy)) * self._central[None]
        # the ocular confound, off by default: gaze follows intent, a SIGNED frontal potential
        # (≈ eye_gain × 20 µV at |intent| = 1; real EOG is ≈ 10–20 µV per degree of gaze)
        if self.eye_gain_uv > 0:
            x += self.eye_gain_uv * 20.0 * (vx * self._h_eog[None] + vy * self._fp[None])
        # occasional blinks on FP
        if self.rng.random() < n / fs * 0.25:     # ≈ one blink every 4 s
            k0 = self.rng.integers(0, n)
            blink = 150.0 * np.exp(-((np.arange(n) - k0) / (0.08 * fs)) ** 2)
            x += blink[:, None] * self._fp[None] * np.sqrt(2)
        x += self._evoked(n)
        if self.drift:
            x *= 1.0 + self.drift * (self.n_sent / fs / 60.0)
        self.n_sent += n
        return x.astype(np.float32)

    def pull(self, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Real-time emulation: returns the samples that 'arrived' since the last call."""
        due = int((time.time() - self.t0) * self.hw.fs) - self.n_sent
        if max_samples is not None:
            due = min(due, max_samples)
        if due <= 0:
            return np.zeros((0, self.hw.n_channels), np.float32), np.zeros(0)
        ts = self.t0 + (self.n_sent + np.arange(due)) / self.hw.fs
        x = self.generate(due)
        self.ring.push(x, ts)
        return x, ts


class LSLEEG:
    """Thin wrapper over pylsl; imported lazily so the package runs without LSL installed."""

    def __init__(self, hw: Hardware = Hardware(), timeout_s: float = 10.0):
        self.hw = hw
        self.timeout_s = timeout_s
        self.inlet = None
        self.ring = RingBuffer(hw.fs * 60, hw.n_channels)

    def start(self):
        from pylsl import StreamInlet, resolve_byprop  # type: ignore
        streams = resolve_byprop("type", self.hw.lsl_stream_type, timeout=self.timeout_s)
        if not streams:
            raise RuntimeError(f"no LSL stream of type {self.hw.lsl_stream_type!r}")
        self.inlet = StreamInlet(streams[0], max_buflen=60, recover=True)
        info = self.inlet.info()
        if info.channel_count() != self.hw.n_channels or int(info.nominal_srate()) != self.hw.fs:
            raise RuntimeError(f"stream is {info.channel_count()} ch @ {info.nominal_srate()} Hz, "
                               f"config expects {self.hw.n_channels} @ {self.hw.fs}")

    def pull(self, max_samples: Optional[int] = None):
        chunk, ts = self.inlet.pull_chunk(timeout=0.0, max_samples=max_samples or 4096)
        if not chunk:
            return np.zeros((0, self.hw.n_channels), np.float32), np.zeros(0)
        x = np.asarray(chunk, dtype=np.float32)
        ts = np.asarray(ts, dtype=np.float64) + self.inlet.time_correction()
        self.ring.push(x, ts)
        return x, ts
