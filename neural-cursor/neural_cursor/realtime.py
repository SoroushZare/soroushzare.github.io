"""Real-time closed loop: EEG source → causal preprocessing → decoder → effector, on a wall clock.

Two pieces:

* ``SimLSLStreamer``  a thread that publishes a ``SimulatedEEG`` over a real LSL outlet at the cap's
                      sampling rate, in small chunks, so the online path (``LSLEEG`` inlet, time
                      correction, chunking, thread scheduling) is exercised end to end without an
                      amplifier.  Tests steer the simulated intent through ``set_intent``.
* ``RealTimeLoop``    the control loop.  Every period (``cfg.task.control_hz``) it pulls everything
                      the source has, filters it with the same stateful ``CausalFilter`` the offline
                      trainer uses, gates bad channels, re-references to the average of the good
                      channels, decodes the last ``cfg.decoder.window_s`` and sends the command.

Timing model.  The scheduler sleeps on ``time.monotonic`` until an absolute deadline
``t_start + k / rate`` (absolute, so jitter does not accumulate into drift); jitter is the wake-up
time minus the deadline.  If a tick overruns past one or more deadlines those deadlines are skipped
and counted, never executed in a burst — a burst would send several stale commands back to back.

Latency is measured on the *sample* clock (the clock the source's timestamps live on: LSL
``local_clock`` for an LSL inlet, whose timestamps ``LSLEEG`` already time-corrects): from the
timestamp of the newest sample that entered the window to the moment ``effector.send`` returns.  It
therefore includes amplifier/stream transport, waiting for the tick, preprocessing, decoding and
the effector call — the end-to-end number a user feels, not just the compute time.

Failure handling (an amplifier unplugs, a driver emits NaN, the process stalls):

* **Non-finite samples** are replaced, per channel, by that channel's last finite value before they
  reach the causal filter (one NaN would otherwise poison the IIR state of that channel for the rest
  of the run), the channel is forced "bad" for the amplitude gate while a repaired sample is inside
  the decoder window, and the count is logged.  CAR always excludes bad channels.  A held channel
  still has samples, so a decoder that reads it would see a fake signal (high-passed, it becomes
  ≈ −CAR, which an SMR decoder reads as ERD): decoders whose ``predict`` accepts ``good`` get the mask
  and refuse a window with a bad channel they use (``BodyCoordinateDecoder`` returns NaN → a counted
  zero command); a decoder without that parameter decodes the CAR'd window as before.  The repaired
  samples are what ``RunLogger`` records, so an offline replay sees exactly what the loop saw.
* **Stale input**: if no new sample has arrived for ``stale_s`` (default two control periods), the
  tick sends a zero command instead of re-decoding the frozen window, and is counted; after
  ``stale_abort_s`` without samples the run ends early with ``stats["aborted"]`` set.
* **Overruns**: the ``dt`` handed to the effector is capped at ``MAX_DT_PERIODS`` control periods, so a
  stall is not integrated as one burst-sized jump.
* **Invalid output**: a non-finite decoder output (e.g. the body decoder's refusal of a flat or gated
  window) sends a zero command and is counted (``n_invalid``).
* **Exit**: however ``run`` ends (normally, early, or by exception) it sends a final zero command, so
  a velocity-controlled device is left stopped.  The effector is not closed (the caller owns it).

Logged cursor: ``cursor.csv`` gets the position the person actually saw — ``effector.task_position()``
when the effector draws a cursor (``ScreenCursor``), else the loop's ``task.Cursor`` model; the source
is written as ``cursor_source`` ("effector" / "model") in the ``realtime_start`` event.  The data
engine refuses visible-cursor labels from a "model" run, because the person never saw that cursor.
The assist uses the same position.

Screen mapping.  A body-coordinate decoder's output is body velocity; ``mapping`` (``mapping.Mapping``,
default identity) turns it into screen velocity ``M · v`` before the assist and the effector, which is
how the champion's held-out-mapping blocks (rot ±90) run live.  ``decoded`` stays in body coordinates
(what certification regresses), ``sent`` is the screen command, and ``realtime_start`` logs the mapping.

Pure-EEG rule: the only arrays that reach ``decoder.predict`` are the preprocessed EEG window and the
loop's own per-channel validity mask of that window.
``intent_fn`` is a hook for a simulator or cue generator; its return value is logged as a label and
never reaches the decoder, the assist or the effector.  Auditor streams (eye tracker, EMG) are not
accepted here at all; they belong to ``purity.py``.
"""
from __future__ import annotations

import inspect
import logging
import threading
import time
import uuid
from typing import Callable, Optional

import numpy as np

from .acquisition import LSLEEG, SimulatedEEG
from .config import Config
from .mapping import TRAINING_MAPPING, Mapping
from .signal import AmplitudeGate, CausalFilter, common_average
from .task import Cursor

log = logging.getLogger(__name__)


class SimLSLStreamer(threading.Thread):
    """Publishes a ``SimulatedEEG`` over an LSL outlet in real time.

    Samples are released only when they are due on LSL's ``local_clock`` (``t0 + i / fs``), and each
    chunk is pushed with the nominal capture time of its newest sample.  Pushing *late* therefore
    shows up as latency downstream instead of being hidden by a ``now`` timestamp — the simulator
    behaves like an amplifier whose clock is the truth.

    Small chunks (``chunk_s``, default 10 ms) mirror what real amplifier drivers deliver; a coarse
    chunk would add up to ``chunk_s`` of latency that a real cap would not have.  Samples are
    generated exactly when they fall due, so ``set_intent`` takes effect from the next chunk; this is
    safe because ``SimulatedEEG.generate`` is chunk-invariant (10 ms pushes produce the same EEG as
    the offline calibration's 100 ms blocks).
    """

    def __init__(self, sim: SimulatedEEG, chunk_s: float = 0.010,
                 stream_name: str = "NeuralCursorSim", source_id: Optional[str] = None):
        super().__init__(name="SimLSLStreamer", daemon=True)
        if chunk_s <= 0:
            raise ValueError(f"chunk_s must be > 0, got {chunk_s}")
        self.sim = sim
        self.hw = sim.hw
        self.chunk_s = float(chunk_s)
        self.stream_name = stream_name
        # A unique source_id per instance: two simulators on one host must never be confused by
        # an inlet that recovers a dropped connection by source_id.
        self.source_id = source_id or f"neural_cursor_sim_{uuid.uuid4().hex[:12]}"
        self.n_pushed = 0
        self._lock = threading.Lock()          # guards sim state shared with set_intent
        self._stop_evt = threading.Event()
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self.info = self._make_info()

    def _make_info(self):
        from pylsl import StreamInfo, cf_float32  # type: ignore
        info = StreamInfo(self.stream_name, self.hw.lsl_stream_type, self.hw.n_channels, self.hw.fs,
                          cf_float32, self.source_id)
        chans = info.desc().append_child("channels")
        for label in self.hw.channels:
            ch = chans.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")
        info.desc().append_child_value("manufacturer", "neural_cursor.SimulatedEEG")
        return info

    def set_intent(self, vx: float, vy: float) -> None:
        """Set the simulated user's latent intent; takes effect from the next pushed chunk."""
        with self._lock:
            self.sim.set_intent(vx, vy)

    def start(self) -> None:
        """Start the thread and block until the outlet exists, so a subsequent resolve finds it."""
        super().start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("SimLSLStreamer outlet did not come up within 5 s")
        if self._error is not None:
            raise RuntimeError("SimLSLStreamer failed to start") from self._error

    def run(self) -> None:
        from pylsl import StreamOutlet, local_clock  # type: ignore
        outlet = None
        try:
            outlet = StreamOutlet(self.info, chunk_size=0, max_buffered=60)
            fs = self.hw.fs
            t0 = local_clock()
            k = 0
            self._ready.set()
            while not self._stop_evt.is_set():
                k += 1
                delay = t0 + k * self.chunk_s - local_clock()
                if delay > 0 and self._stop_evt.wait(delay):
                    break
                due = int((local_clock() - t0) * fs) + 1 - self.n_pushed   # samples 0..i with t0+i/fs ≤ now
                if due <= 0:
                    continue
                with self._lock:
                    x = self.sim.generate(due)
                newest_ts = t0 + (self.n_pushed + due - 1) / fs
                outlet.push_chunk(x, newest_ts)
                self.n_pushed += due
        except BaseException as exc:                 # noqa: BLE001 - surfaced to start()/stop()
            self._error = exc
            self._ready.set()
            log.exception("SimLSLStreamer crashed")
        finally:
            del outlet                               # destroys the liblsl outlet in this thread

    def stop(self, timeout: float = 2.0) -> None:
        """Stop publishing and wait for the thread; idempotent."""
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout)
        if self.is_alive():
            raise RuntimeError("SimLSLStreamer did not stop")
        if self._error is not None:
            raise RuntimeError("SimLSLStreamer crashed while streaming") from self._error

    def __enter__(self) -> "SimLSLStreamer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _sample_clock(source) -> Callable[[], float]:
    """The clock the source's timestamps live on; latency is only meaningful on that clock."""
    if isinstance(source, LSLEEG):
        from pylsl import local_clock  # type: ignore
        return local_clock
    if isinstance(source, SimulatedEEG):
        return time.time                             # SimulatedEEG.pull stamps with time.time
    raise TypeError(f"unknown source type {type(source).__name__}; pass sample_clock explicitly")


class RealTimeLoop:
    """Fixed-rate closed loop from an EEG source to an effector.

    Parameters
    ----------
    source      object with ``pull() -> (x (n, n_channels) µV, ts (n,))``; ``LSLEEG`` or ``SimulatedEEG``,
                already started.
    decoder     object with ``predict(window (n_win, n_channels)) -> (2,)`` in [−1, 1].
    effector    ``effectors.Effector``; receives ``send(u, dt)`` once per update.
    cfg         ``config.Config``; uses hardware, pre, decoder.window_s and task (rate, cursor, assist).
    logger      optional ``RunLogger``: EEG (non-finite samples repaired, before filtering), events and
                cursor rows.
    assist      optional goal posterior (``task.GoalPosterior`` interface: ``update_velocity``,
                ``assist_command``, ``p``).  The command becomes ``(1 − g) u + g · assist`` with
                ``g = cfg.task.assist_gain``; the posterior sees only the decoded (EEG) command.
    rate_hz     overrides ``cfg.task.control_hz``.
    warmup_s    EEG pulled and filtered before the timed run starts.  One second fills the decoder
                window twice over, gives the amplitude gate a session median built from several
                windows, and lets the causal filters' onset transient decay, so the first decoded
                command is not computed on a half-empty or ringing window.
    sample_clock clock of the source's timestamps (default: inferred from the source type).
    stale_s     no new sample for this long (wall time) → the tick sends zero instead of decoding.
                Default two control periods: LSL chunks arrive every few ms, so a gap of two whole
                periods is a fault, not jitter.
    stale_abort_s no new sample for this long → the run ends early (``stats["aborted"]``).
    mapping     screen mapping applied to the decoder's output before assist and effector (module
                docstring); None → the identity.
    """

    #: a gap between consecutive sample timestamps above this many sample periods counts as a drop
    DROP_GAP_PERIODS = 1.5
    #: longest dt (in control periods) handed to the effector; a stall is never integrated in one step
    MAX_DT_PERIODS = 2.0

    def __init__(self, source, decoder, effector, cfg: Config, logger=None, assist=None,
                 rate_hz: Optional[float] = None, warmup_s: float = 1.0,
                 sample_clock: Optional[Callable[[], float]] = None,
                 stale_s: Optional[float] = None, stale_abort_s: float = 2.0,
                 mapping: Optional[Mapping] = None):
        self.source = source
        self.decoder = decoder
        self.mapping = mapping if mapping is not None else Mapping(TRAINING_MAPPING)
        try:
            self._predict_takes_good = "good" in inspect.signature(decoder.predict).parameters
        except (TypeError, ValueError):                      # builtins / C extensions: no signature
            self._predict_takes_good = False
        self.effector = effector
        self.cfg = cfg
        self.logger = logger
        self.assist = assist
        self.rate_hz = float(rate_hz if rate_hz is not None else cfg.task.control_hz)
        if self.rate_hz <= 0:
            raise ValueError(f"rate_hz must be > 0, got {self.rate_hz}")
        if assist is not None and cfg.task.assist_gain <= 0:
            raise ValueError("an assist was given but cfg.task.assist_gain is 0, so it would be ignored; "
                             "set assist_gain > 0 or pass assist=None")
        self.warmup_s = float(warmup_s)
        self.stale_s = 2.0 / self.rate_hz if stale_s is None else float(stale_s)
        self.stale_abort_s = float(stale_abort_s)
        if not 0 < self.stale_s <= self.stale_abort_s:
            raise ValueError(f"need 0 < stale_s <= stale_abort_s, got {self.stale_s}, {self.stale_abort_s}")
        self.clock = sample_clock or _sample_clock(source)
        hw = cfg.hardware
        self.fs = hw.fs
        self.n_win = int(round(cfg.decoder.window_s * hw.fs))
        self.filt = CausalFilter(hw, cfg.pre)
        self.gate = AmplitudeGate(hw, cfg.pre)
        self.cursor = Cursor(cfg.task)       # task-space model of the commanded motion (no display)
        self._buf = np.zeros((0, hw.n_channels))
        self._last_ts: Optional[float] = None
        self._last_arrival = time.monotonic()
        self._last_good = np.zeros(hw.n_channels, np.float32)   # last finite sample per channel
        self._repaired_at = np.full(hw.n_channels, -self.n_win - 1, np.int64)  # last repaired sample index
        self._stop_evt = threading.Event()
        self.n_samples = 0
        self.dropped_samples = 0
        self.nonfinite_samples = 0
        self.decoded: np.ndarray = np.zeros((0, 2))   # per-update decoder output (body frame, before mapping)
        self.sent: np.ndarray = np.zeros((0, 2))      # per-update command handed to the effector (screen frame)

    # -- acquisition --------------------------------------------------------------------------------
    def _ingest(self) -> Optional[float]:
        """Pull until the source is empty; repair, log, filter and buffer.  Returns the newest
        timestamp seen, or None when nothing arrived."""
        newest = None
        while True:
            x, ts = self.source.pull()
            if len(x) == 0:
                return newest
            self._last_arrival = time.monotonic()
            x = self._repair(np.asarray(x, dtype=np.float32))
            if self.logger is not None:
                self.logger.eeg(x, ts)                       # µV before any filtering
            self._count_drops(ts)
            y = self.filt(x)
            self._buf = np.concatenate([self._buf, y])[-self.n_win:]
            self.n_samples += len(x)
            newest = float(ts[-1])

    def _repair(self, x: np.ndarray) -> np.ndarray:
        """Hold each channel's last finite value over non-finite samples (see the module docstring)."""
        bad = ~np.isfinite(x)
        if bad.any():
            rows = np.arange(len(x))[:, None]
            src = np.maximum.accumulate(np.where(bad, -1, rows), axis=0)    # last finite row so far
            cols = np.arange(x.shape[1])[None]
            x = np.where(src >= 0, x[np.maximum(src, 0), cols], self._last_good[None]).astype(np.float32)
            chans = np.flatnonzero(bad.any(axis=0))
            last_bad = len(bad) - 1 - np.argmax(bad[::-1], axis=0)
            self._repaired_at[chans] = self.n_samples + last_bad[chans]
            self.nonfinite_samples += int(bad.sum())
            log.warning("realtime: %d non-finite samples on %s held at the last finite value",
                        int(bad.sum()), [self.cfg.hardware.channels[c] for c in chans])
            if self.logger is not None:
                self.logger.event("nonfinite", {"n": int(bad.sum()),
                                                "channels": [self.cfg.hardware.channels[c] for c in chans]},
                                  t=self.clock())
        self._last_good = x[-1].copy()
        return x

    def _count_drops(self, ts: np.ndarray) -> None:
        prev = np.concatenate([[self._last_ts], ts]) if self._last_ts is not None else ts
        gaps = np.diff(prev) * self.fs
        self.dropped_samples += int(np.sum(np.rint(gaps[gaps > self.DROP_GAP_PERIODS]) - 1))
        self._last_ts = float(ts[-1])

    def _warmup(self) -> None:
        """Pull for ``warmup_s`` and until the window is full (bounded at 10× warmup_s + 5 s); returns
        early if ``stop`` is called."""
        t_end = time.monotonic() + self.warmup_s
        t_fail = t_end + 9 * self.warmup_s + 5.0
        while time.monotonic() < t_end or len(self._buf) < self.n_win:
            if self._stop_evt.is_set():
                return
            if time.monotonic() > t_fail:
                raise RuntimeError(f"no full {self.n_win}-sample window after warm-up: "
                                   f"{len(self._buf)} samples received")
            self._ingest()
            time.sleep(0.005)
        # gate history from the warm-up so the first live window is judged against a session median
        self.gate.update(self._buf)

    # -- control ------------------------------------------------------------------------------------
    def stop(self) -> None:
        """Ask ``run`` to finish after the current tick (thread-safe).  A stop issued before ``run``
        starts is kept: that run returns at once.  ``run`` clears it when it returns."""
        self._stop_evt.set()

    def _display_pos(self) -> np.ndarray:
        """The cursor position the person sees: the effector's, else the internal model's."""
        pos = self.effector.task_position()
        return self.cursor.state.pos.copy() if pos is None else np.asarray(pos, float)

    def _step(self, dt: float) -> Optional[tuple]:
        """One update on the current buffer; returns (decoded, sent, good, p_goal_max), or None when
        the window or the decoder output is non-finite (a zero command was sent instead)."""
        window = self._buf
        good = self.gate.update(window)
        good &= self._repaired_at < self.n_samples - self.n_win     # repaired samples in the window
        if self.cfg.pre.reference == "car":
            window = common_average(window, good)
        if not np.isfinite(window).all():
            self.effector.send(np.zeros(2), dt)
            return None
        if self._predict_takes_good:
            decoded = np.asarray(self.decoder.predict(window, good=good), float).reshape(-1)
        else:
            decoded = np.asarray(self.decoder.predict(window), float).reshape(-1)
        if not np.isfinite(decoded).all():
            self.effector.send(np.zeros(2), dt)
            return None
        u = self.mapping.body_to_screen(decoded)              # screen velocity
        pos = self._display_pos()
        p_goal_max = float("nan")
        if self.assist is not None:
            self.assist.update_velocity(pos, u)
            g = self.cfg.task.assist_gain
            u = (1 - g) * u + g * np.asarray(self.assist.assist_command(pos), float)
            p_goal_max = float(np.max(self.assist.p))
        self.effector.send(u, dt)
        return decoded, u, good, p_goal_max

    def run(self, duration_s: float, intent_fn: Optional[Callable[[float], object]] = None) -> dict:
        """Run for ``duration_s`` seconds of timed updates (after warm-up) and return statistics.

        ``intent_fn(t)`` (t = seconds since the timed run started) is called at the top of every tick
        to drive a simulator or cue; a non-None return is logged as an ``intent`` event and goes
        nowhere else.  ``n_updates`` counts decoded updates; ticks that sent a zero command because
        input was stale (``n_stale``) or non-finite (``n_invalid``) are counted separately.  The
        effector is left stopped (a final zero command) however the run ends.
        """
        lat, jit, dec_rows, sent_rows = [], [], [], []
        n_gated = n_overruns = n_stale = n_invalid = 0
        aborted: Optional[str] = None
        period = 1.0 / self.rate_hz
        n_expected = int(round(duration_s * self.rate_hz))
        lg = self.logger
        t_start = time.monotonic()
        try:
            self._warmup()
            if lg is not None:
                lg.event("realtime_start", {"rate_hz": self.rate_hz, "window_s": self.cfg.decoder.window_s,
                                            "warmup_s": self.warmup_s,
                                            "effector": type(self.effector).__name__,
                                            "mapping": self.mapping.name,
                                            "cursor_source": ("model" if self.effector.task_position() is None
                                                              else "effector")}, t=self.clock())
            t_start = time.monotonic()
            last_send = None
            k = 0
            newest_ts = self._last_ts
            stale_logged = False
            while k < n_expected and not self._stop_evt.is_set():
                deadline = t_start + k * period
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                woke = time.monotonic()
                jit.append(woke - deadline)
                if intent_fn is not None:
                    label = intent_fn(woke - t_start)
                    if label is not None and lg is not None:
                        lg.event("intent", np.asarray(label, float), t=self.clock())
                got = self._ingest()
                newest_ts = got if got is not None else newest_ts
                dt = period if last_send is None else min(woke - last_send, self.MAX_DT_PERIODS * period)
                last_send = woke
                age = time.monotonic() - self._last_arrival
                if age > self.stale_s:
                    self.effector.send(np.zeros(2), dt)
                    n_stale += 1
                    u = np.zeros(2)
                    if lg is not None and not stale_logged:
                        lg.event("stale", {"age_s": age}, t=self.clock())
                    stale_logged = True
                    if age > self.stale_abort_s:
                        aborted = f"stale: no EEG samples for {age:.2f} s"
                        log.warning("realtime: %s; run aborted", aborted)
                        if lg is not None:
                            lg.event("realtime_abort", {"reason": aborted}, t=self.clock())
                        break
                    p_goal_max = float("nan")
                else:
                    stale_logged = False
                    out = self._step(dt)
                    if out is None:
                        n_invalid += 1
                        u, p_goal_max = np.zeros(2), float("nan")
                    else:
                        decoded, u, good, p_goal_max = out
                        lat.append(self.clock() - newest_ts)
                        dec_rows.append(decoded)
                        sent_rows.append(u)
                        if not good.all():
                            n_gated += 1
                            if lg is not None:
                                bad = [c for c, ok in zip(self.cfg.hardware.channels, good) if not ok]
                                lg.event("gate", {"bad": bad}, t=self.clock())
                self.cursor.step(u, dt)
                if lg is not None:
                    lg.cursor(self.clock(), self._display_pos(), u, p_goal_max, None, 0.0)
                # next deadline; skip (and count) any we already missed instead of bursting
                k += 1
                behind = int((time.monotonic() - t_start) / period) - k
                if behind > 0:
                    n_overruns += behind
                    k += behind
        finally:
            self._rest_effector()
            self._stop_evt.clear()
            self.decoded = np.array(dec_rows).reshape(-1, 2)
            self.sent = np.array(sent_rows).reshape(-1, 2)
            stats = self._stats(lat, jit, n_expected, n_gated, n_overruns, time.monotonic() - t_start)
            stats.update(n_stale=n_stale, n_invalid=n_invalid, aborted=aborted)
            if lg is not None:
                lg.event("realtime_stop", stats, t=self.clock())
        return stats

    def _rest_effector(self) -> None:
        """Final zero command (never raises: it runs while another exception may be propagating)."""
        try:
            self.effector.send(np.zeros(self.effector.dim), 0.0)
        except Exception as exc:                                  # noqa: BLE001 - see docstring
            log.warning("realtime: could not send the final stop to %s: %s", type(self.effector).__name__, exc)

    def _stats(self, lat, jit, n_expected, n_gated, n_overruns, elapsed) -> dict:
        def ms(v, q):
            return float(np.percentile(v, q) * 1e3) if len(v) else float("nan")
        lat, jit = np.asarray(lat), np.asarray(jit)
        return {
            "n_updates": int(len(self.decoded)),
            "n_expected": int(n_expected),
            "n_overruns": int(n_overruns),
            "n_gated": int(n_gated),
            "n_samples": int(self.n_samples),
            "dropped_samples": int(self.dropped_samples),
            "nonfinite_samples": int(self.nonfinite_samples),
            "elapsed_s": float(elapsed),
            "latency_ms": {"p50": ms(lat, 50), "p95": ms(lat, 95), "max": ms(lat, 100),
                           "mean": float(lat.mean() * 1e3) if len(lat) else float("nan")},
            "jitter_ms": {"p50": ms(jit, 50), "p95": ms(jit, 95), "max": ms(jit, 100),
                          "mean": float(jit.mean() * 1e3) if len(jit) else float("nan")},
            "decoded_mean": self.decoded.mean(axis=0).tolist() if len(self.decoded) else [float("nan")] * 2,
        }
