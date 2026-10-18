"""EEG sources: a Lab Streaming Layer inlet and a deterministic simulator.

Both expose the same interface so every downstream component is tested headless:

    src.start(); chunk, ts = src.pull()  # (n_samples, n_channels) float32 µV, LSL timestamps
    src.ring.latest(n)                   # last n samples, oldest first

The simulator embeds a known, small, *cortical-like* signal for a given latent intent so the whole
pipeline can be regression-tested for exact numbers; it is never a model of real EEG.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import Hardware

BLINK_RATE_HZ = 0.25      # ≈ one blink every 4 s at rest
BLINK_UV = 150.0          # peak frontopolar blink amplitude
BLINK_WIDTH_S = 0.08      # Gaussian width parameter w of exp(-(Δt / w)²)
BLINK_SPAN_WIDTHS = 3.0   # a blink is rendered over ±3 w (its tail there is e⁻⁹ × 150 µV ≈ 0.02 µV)
_BLINK_STREAM = 7_919     # RNG stream id of blink timing, separate from the noise stream


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
        self._reset_blinks(np.random.default_rng([self.seed, _BLINK_STREAM]))

    def _topo(self, names):
        v = np.zeros(self.hw.n_channels)
        for n in names:
            v[self._i[n]] = 1.0
        return v / max(1.0, np.linalg.norm(v))

    def start(self):
        """Restart the real-time clock.  Sample indices restart at 0, so queued blinks and evoked
        transients (held at absolute sample indices) are re-based with them rather than lost."""
        shift = self.n_sent
        self.t0 = time.time()
        self.n_sent = 0
        self._pending = [(onset - shift, amp) for onset, amp in self._pending]
        self._blink_centres = [c - shift for c in self._blink_centres]
        self._next_blink -= shift

    def _reset_blinks(self, rng: np.random.Generator) -> None:
        """Start a fresh blink schedule: a Poisson process (rate ``BLINK_RATE_HZ``) of blink centres at
        absolute sample indices, drawn from its own RNG.  The first centre lies at least one half-span
        ahead, so no blink ever starts in samples that were already emitted."""
        fs = self.hw.fs
        self._blink_rng = rng
        self._blink_half = BLINK_SPAN_WIDTHS * BLINK_WIDTH_S * fs
        self._blink_centres: list = []
        self._next_blink = self.n_sent + self._blink_half + rng.exponential(fs / BLINK_RATE_HZ)

    def _blinks(self, n: int) -> np.ndarray:
        """Blink potential on FP1/FP2 for the next n samples, (n, n_channels) µV.

        Blinks are scheduled on the absolute sample axis and kept queued until their tail has been
        emitted, so a blink that straddles two ``generate`` calls is rendered whole across them.  The
        output therefore does not depend on how the stream is chunked: 5-sample real-time pushes and
        one-hour offline blocks see the same blinks (placing each blink inside the current block, as
        an earlier version did, turned blinks in small blocks into truncated broadband spikes)."""
        end = self.n_sent + n
        while self._next_blink - self._blink_half < end:
            self._blink_centres.append(self._next_blink)
            self._next_blink += self._blink_rng.exponential(self.hw.fs / BLINK_RATE_HZ)
        out = np.zeros((n, self.hw.n_channels))
        if not self._blink_centres:
            return out
        k = self.n_sent + np.arange(n)
        width = BLINK_WIDTH_S * self.hw.fs
        trace = np.zeros(n)
        for c in self._blink_centres:
            d = k - c
            trace += np.where(np.abs(d) < self._blink_half, BLINK_UV * np.exp(-(d / width) ** 2), 0.0)
        self._blink_centres = [c for c in self._blink_centres if c + self._blink_half > end]
        return trace[:, None] * self._fp[None] * np.sqrt(2)

    def _drift_gain(self, n: int) -> np.ndarray:
        """Per-sample multiplicative drift factor (n, 1); per sample so chunking cannot change it."""
        minutes = (self.n_sent + np.arange(n)) / self.hw.fs / 60.0
        return (1.0 + self.drift * minutes)[:, None]

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
        """n samples of (n, n_channels) µV for the current intent (pure function of RNG state).

        Chunk-invariant: for a fixed intent, ``generate(a)`` then ``generate(b)`` returns exactly the
        samples of ``generate(a + b)`` — the noise stream, AR state, oscillation phase, blinks and
        drift all live on the absolute sample axis."""
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
        x += self._blinks(n)                      # occasional blinks on FP
        x += self._evoked(n)
        if self.drift:
            x *= self._drift_gain(n)
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
    """Thin wrapper over pylsl; imported lazily so the package runs without LSL installed.

    The stream is resolved by type ``hw.lsl_stream_type`` and, when given, also by ``name`` and/or
    ``source_id``.  Type alone picks whichever "EEG" stream answers first, which on a lab network with
    a second cap or a simulator running can be the wrong person's brain; pinning ``source_id`` (the
    amplifier's serial, or ``SimLSLStreamer.source_id``) makes the choice unambiguous.

    ``start`` also checks the stream's montage: when the stream describes its channels
    (``desc/channels/channel/label`` and ``unit``), the labels must equal ``hw.channels`` in order
    and the unit must be microvolts.  Channel count and rate alone accept a cap that orders its
    montage differently, and then the gate exemption (FP1/FP2), CAR and the decoder's topographies
    silently apply to the wrong electrodes; a stream in volts would be gated and decoded at a scale
    10⁶ off.  A stream without channel metadata is accepted with a warning (many drivers omit it).

    ``close()`` (idempotent; also the context-manager exit) unsubscribes the inlet.  A failed or
    repeated ``start`` closes the previous inlet first, because an inlet opened with
    ``recover=True`` keeps buffering, and reconnecting, until it is closed.
    """

    MICROVOLT_UNITS = ("microvolts", "microvolt", "uv", "µv", "μv")

    def __init__(self, hw: Hardware = Hardware(), timeout_s: float = 10.0,
                 name: Optional[str] = None, source_id: Optional[str] = None):
        self.hw = hw
        self.timeout_s = timeout_s
        self.name = name
        self.source_id = source_id
        self.inlet = None
        self.ring = RingBuffer(hw.fs * 60, hw.n_channels)

    def predicate(self) -> str:
        """The LSL XPath predicate this inlet resolves (type, plus name / source_id when pinned)."""
        def quote(v: str) -> str:
            if "'" in v:
                raise ValueError(f"LSL predicate values cannot contain a single quote: {v!r}")
            return f"'{v}'"
        terms = [f"type={quote(self.hw.lsl_stream_type)}"]
        if self.name is not None:
            terms.append(f"name={quote(self.name)}")
        if self.source_id is not None:
            terms.append(f"source_id={quote(self.source_id)}")
        return " and ".join(terms)

    def start(self):
        from pylsl import StreamInlet, resolve_bypred  # type: ignore
        self.close()
        streams = resolve_bypred(self.predicate(), 1, self.timeout_s)
        if not streams:
            raise RuntimeError(f"no LSL stream matching {self.predicate()}")
        self.inlet = StreamInlet(streams[0], max_buflen=60, recover=True)
        try:
            # Subscribe now: an inlet only starts buffering at its first pull/open, so without this
            # the samples between start() and the first pull() would be lost.
            self.inlet.open_stream(timeout=self.timeout_s)
            self._check_stream(self.inlet.info())
        except BaseException:
            self.close()
            raise

    def _check_stream(self, info) -> None:
        if info.channel_count() != self.hw.n_channels or int(info.nominal_srate()) != self.hw.fs:
            raise RuntimeError(f"stream is {info.channel_count()} ch @ {info.nominal_srate()} Hz, "
                               f"config expects {self.hw.n_channels} @ {self.hw.fs}")
        labels, units = [], []
        ch = info.desc().child("channels").child("channel")
        while not ch.empty():
            labels.append(ch.child_value("label"))
            units.append(ch.child_value("unit"))
            ch = ch.next_sibling()
        if not any(labels):
            warnings.warn("LSL stream has no channel labels; the montage order cannot be checked and is "
                          f"assumed to be {self.hw.channels[:3]}…", RuntimeWarning, stacklevel=3)
            return
        if [lab.upper() for lab in labels] != [c.upper() for c in self.hw.channels]:
            raise RuntimeError(f"stream channel labels {labels} do not match the configured montage "
                               f"{list(self.hw.channels)} (order matters)")
        bad_units = sorted({u for u in units if u and u.strip().lower() not in self.MICROVOLT_UNITS})
        if bad_units:
            raise RuntimeError(f"stream units {bad_units} are not microvolts; the pipeline expects µV")

    def close(self) -> None:
        """Unsubscribe and drop the inlet; idempotent."""
        if self.inlet is not None:
            inlet, self.inlet = self.inlet, None
            inlet.close_stream()

    def __enter__(self) -> "LSLEEG":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def pull(self, max_samples: Optional[int] = None):
        chunk, ts = self.inlet.pull_chunk(timeout=0.0, max_samples=max_samples or 4096)
        if not chunk:
            return np.zeros((0, self.hw.n_channels), np.float32), np.zeros(0)
        x = np.asarray(chunk, dtype=np.float32)
        ts = np.asarray(ts, dtype=np.float64) + self.inlet.time_correction()
        self.ring.push(x, ts)
        return x, ts
