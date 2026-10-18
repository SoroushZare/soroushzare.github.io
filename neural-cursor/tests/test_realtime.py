"""Real-time loop tests: a simulated cap streamed over a real LSL outlet on localhost, decoded live.

The decoder is trained offline on simulated calibration windows (``simulate.calibration``, same
filter + CAR as the online path) from a *different* simulator seed than the streamed one, so the
online test is a genuine transfer, not a replay.
"""
import dataclasses
import threading
import time
import uuid

import numpy as np
import pytest

from neural_cursor.acquisition import LSLEEG, SimulatedEEG
from neural_cursor.config import DEFAULT
from neural_cursor.decoder import VelocityDecoder
from neural_cursor.effectors import Effector, NullEffector, ScreenCursor
from neural_cursor.logger import RunLogger
from neural_cursor.realtime import RealTimeLoop, SimLSLStreamer
from neural_cursor.simulate import SimUser, calibration
from neural_cursor.task import GoalPosterior

CAL_SEED, STREAM_SEED = 11, 23
N_CAL_WINDOWS = 240        # enough for a stable ridge fit on 160 band-power features
RUN_S = 5.0                # timed run after the loop's 1 s warm-up: ≈ 6 s of streaming in total
INTENT_GAIN_UV = 8.0       # the simulator's 'full-strength user'


class SpyDecoder:
    """Records every array the decoder receives, to assert the pure-EEG rule."""

    def __init__(self, inner):
        self.inner = inner
        self.shapes = set()
        self.all_finite = True

    def predict(self, window):
        self.shapes.add(window.shape)
        self.all_finite &= bool(np.isfinite(window).all())
        return self.inner.predict(window)


@pytest.fixture(scope="module")
def decoder():
    user = SimUser(DEFAULT, seed=CAL_SEED, gain=INTENT_GAIN_UV, eye=0.0)
    ws, ys = calibration(user, N_CAL_WINDOWS, np.random.default_rng(CAL_SEED))
    return VelocityDecoder(DEFAULT.decoder, DEFAULT.hardware.fs).fit(ws, ys)


def test_streamer_default_metadata():
    streamer = SimLSLStreamer(SimulatedEEG(DEFAULT.hardware, seed=0))
    info = streamer.info
    assert info.type() == "EEG" and info.channel_count() == 32 and info.nominal_srate() == 500
    ch = info.desc().child("channels").child("channel")
    labels = []
    while not ch.empty():
        labels.append(ch.child_value("label"))
        ch = ch.next_sibling()
    assert tuple(labels) == DEFAULT.hardware.channels


def test_lsl_inlet_predicate_pins_the_stream():
    """Type alone resolves whichever 'EEG' stream answers first; name / source_id pin one device."""
    assert LSLEEG(DEFAULT.hardware).predicate() == "type='EEG'"
    src = LSLEEG(DEFAULT.hardware, name="Cap", source_id="sn-42")
    assert src.predicate() == "type='EEG' and name='Cap' and source_id='sn-42'"
    with pytest.raises(ValueError):
        LSLEEG(DEFAULT.hardware, source_id="x' or type='EEG").predicate()


def test_lsl_closed_loop_latency_rate_and_direction(decoder, tmp_path):
    cfg = DEFAULT                                    # production stream type 'EEG'
    sim = SimulatedEEG(cfg.hardware, seed=STREAM_SEED, intent_gain_uv=INTENT_GAIN_UV)
    threads_before = set(threading.enumerate())
    # a unique source_id, pinned by the inlet, so a real cap (or another test run) on this host
    # is never resolved instead of our simulator
    streamer = SimLSLStreamer(sim, chunk_s=0.010, source_id=f"nc-test-{uuid.uuid4().hex[:12]}")
    streamer.set_intent(1.0, 0.0)                    # the simulated user pushes right throughout
    streamer.start()
    src = LSLEEG(cfg.hardware, timeout_s=5.0, source_id=streamer.source_id)
    logger = RunLogger(str(tmp_path), cfg, tag="rt")
    spy = SpyDecoder(decoder)
    try:
        src.start()
        loop = RealTimeLoop(src, spy, NullEffector(), cfg, logger=logger)
        stats = loop.run(RUN_S, intent_fn=lambda t: (1.0, 0.0))
    finally:
        logger.close()
        src.close()                                  # close the consumer first: an inlet with
        src.close()                                  # recover=True would chase a vanished outlet
        streamer.stop()

    expected = int(RUN_S * cfg.task.control_hz)
    assert stats["n_expected"] == expected
    assert stats["n_updates"] >= 0.9 * expected, stats
    assert stats["latency_ms"]["p95"] < 50.0, stats
    assert stats["latency_ms"]["p50"] > 0.0, stats          # timestamps are causal, not in the future
    assert stats["dropped_samples"] == 0, stats
    assert stats["jitter_ms"]["p95"] < 20.0, stats
    assert loop.decoded[:, 0].mean() > 0.0, stats["decoded_mean"]

    # pure-EEG rule: the decoder only ever saw full-length, finite 32-channel EEG windows
    assert spy.shapes == {(int(cfg.decoder.window_s * cfg.hardware.fs), cfg.hardware.n_channels)}
    assert spy.all_finite

    # raw EEG logged before filtering, one timestamp per sample, cursor row per update
    x, ts = RunLogger.load_eeg(logger.dir, cfg.hardware.n_channels)
    assert len(x) == stats["n_samples"] and np.all(np.diff(ts) > 0)
    rows = np.loadtxt(f"{logger.dir}/cursor.csv", delimiter=",", skiprows=1, ndmin=2)
    assert len(rows) == stats["n_updates"]
    events = open(f"{logger.dir}/events.csv").read()
    assert "realtime_start" in events and "realtime_stop" in events and ",intent," in events

    # clean shutdown: the streamer thread is gone and nothing new is left running
    assert not streamer.is_alive()
    assert not [t for t in set(threading.enumerate()) - threads_before if t.is_alive()]


def test_assist_blend_and_validation(decoder):
    """In-process source (no LSL): the assist blends toward the posterior and needs a non-zero gain."""
    sim = SimulatedEEG(DEFAULT.hardware, seed=STREAM_SEED)
    targets = np.array([[0.6, 0.0], [-0.6, 0.0]])
    with pytest.raises(ValueError):
        RealTimeLoop(sim, decoder, NullEffector(), DEFAULT, assist=GoalPosterior(targets, DEFAULT.task))

    cfg = dataclasses.replace(DEFAULT, task=dataclasses.replace(DEFAULT.task, assist_gain=0.5))
    sim.set_intent(1.0, 0.0)
    sim.start()
    post = GoalPosterior(targets, cfg.task)
    loop = RealTimeLoop(sim, decoder, NullEffector(), cfg, assist=post, warmup_s=0.6)
    stats = loop.run(1.0)
    assert stats["n_updates"] >= 9
    g = cfg.task.assist_gain
    # sent = (1 − g)·decoded + g·assist, so the residual is a vector of norm ≤ 1 scaled by g
    resid = loop.sent - (1 - g) * loop.decoded
    assert np.all(np.linalg.norm(resid, axis=1) <= g + 1e-9)
    assert not np.allclose(loop.sent, loop.decoded)


# ------------------------------------------------------------------------------- failure handling
class RecordingEffector(Effector):
    """Records every (command, dt) the loop sends."""

    def __init__(self):
        super().__init__(dim=2, max_speed=1.0)
        self.sends = []

    def _apply(self, u, dt):
        self.sends.append((u.copy(), dt))


class FaultySource:
    """A SimulatedEEG that stops delivering after ``stop_after`` s, or emits one NaN after
    ``nan_after`` s (an unplugged amplifier; a driver's lost-sample marker)."""

    def __init__(self, sim, stop_after=None, nan_after=None, nan_channel=5):
        self.sim, self.stop_after, self.nan_after, self.ch = sim, stop_after, nan_after, nan_channel
        self.t0 = None
        sim.start()

    def pull(self):
        x, ts = self.sim.pull()
        self.t0 = self.t0 or time.monotonic()
        elapsed = time.monotonic() - self.t0
        if self.stop_after is not None and elapsed > self.stop_after:
            return np.zeros((0, 32), np.float32), np.zeros(0)
        if len(x) and self.nan_after is not None and elapsed > self.nan_after:
            x = x.copy()
            x[0, self.ch] = np.nan
            self.nan_after = None
        return x, ts


def _loop(decoder, source, effector, **kw):
    return RealTimeLoop(source, decoder, effector, DEFAULT, warmup_s=0.6, sample_clock=time.time, **kw)


def test_stale_input_sends_zero_then_aborts_and_leaves_the_effector_stopped(decoder):
    eff = RecordingEffector()
    loop = _loop(decoder, FaultySource(SimulatedEEG(DEFAULT.hardware, seed=1), stop_after=1.0), eff,
                 stale_abort_s=0.6)
    stats = loop.run(3.0)
    assert stats["aborted"] and stats["aborted"].startswith("stale"), stats
    assert stats["n_stale"] >= 3 and stats["n_updates"] < stats["n_expected"]
    assert stats["elapsed_s"] < 2.0                                   # ended early, not after 3 s
    final_u, final_dt = eff.sends[-1]
    assert np.all(final_u == 0) and final_dt == 0.0                  # left at rest
    stale_sends = eff.sends[-1 - stats["n_stale"]:-1]
    assert all(np.all(u == 0) for u, _ in stale_sends)               # never re-sent a frozen command
    assert any(np.any(u != 0) for u, _ in eff.sends[:-1 - stats["n_stale"]])


def test_nan_sample_is_repaired_and_never_reaches_filter_or_decoder(decoder, tmp_path):
    spy = SpyDecoder(decoder)
    logger = RunLogger(str(tmp_path), DEFAULT, tag="nan")
    loop = _loop(spy, FaultySource(SimulatedEEG(DEFAULT.hardware, seed=2), nan_after=0.9), NullEffector(),
                 logger=logger)
    stats = loop.run(1.5)
    logger.close()
    assert stats["nonfinite_samples"] == 1 and stats["n_invalid"] == 0
    assert stats["n_updates"] >= 0.9 * stats["n_expected"] and spy.all_finite
    assert np.all(np.isfinite(loop.filt._zi[0]))                     # the IIR state was not poisoned
    assert np.all(np.isfinite(loop._buf[:, 5]))
    x, _ = RunLogger.load_eeg(logger.dir, 32)
    assert np.all(np.isfinite(x))                                    # the log is what the loop used
    assert ",nonfinite," in open(f"{logger.dir}/events.csv").read()


def test_a_stall_is_not_integrated_as_one_jump_and_stop_before_run_is_kept(decoder):
    class Stalling:
        def __init__(self):
            self.calls = 0

        def predict(self, w):
            self.calls += 1
            if self.calls == 3:
                time.sleep(0.5)                                      # a GC pause / swap / slow model
            return decoder.predict(w)

    eff = RecordingEffector()
    loop = _loop(Stalling(), SimulatedEEG(DEFAULT.hardware, seed=3), eff)
    loop.source.start()
    stats = loop.run(1.2)
    period = 1.0 / DEFAULT.task.control_hz
    assert stats["n_overruns"] >= 3
    assert max(dt for _, dt in eff.sends) <= RealTimeLoop.MAX_DT_PERIODS * period + 1e-9
    early = _loop(decoder, SimulatedEEG(DEFAULT.hardware, seed=4), RecordingEffector())
    early.source.start()
    early.stop()                                                     # e-stop pressed during setup
    t0 = time.monotonic()
    st = early.run(2.0)
    assert st["n_updates"] == 0 and time.monotonic() - t0 < 0.5


def test_cursor_log_is_the_displayed_cursor(decoder, tmp_path):
    """cursor.csv must hold what the person saw: the ScreenCursor's position in task units, not the
    loop's smoothed internal model; a run without a displaying effector says so."""
    for eff, source in ((ScreenCursor(1920, 1080, gain_px_s=600, backend="log"), "effector"),
                        (NullEffector(), "model")):
        logger = RunLogger(str(tmp_path), DEFAULT, tag=source)
        sim = SimulatedEEG(DEFAULT.hardware, seed=5)
        sim.set_intent(1.0, 0.0)
        sim.start()
        loop = RealTimeLoop(sim, decoder, eff, DEFAULT, logger=logger, warmup_s=0.6)
        loop.run(1.0)
        logger.close()
        events = open(f"{logger.dir}/events.csv").read()
        assert f'"cursor_source": "{source}"' in events
        rows = np.loadtxt(f"{logger.dir}/cursor.csv", delimiter=",", skiprows=1, ndmin=2)
        if source == "effector":
            assert np.allclose(rows[-1, 1:3], eff.task_position(), atol=1e-4)
            assert not np.allclose(rows[-1, 1:3], loop.cursor.state.pos, atol=1e-3)


def _outlet(labels, units, fs=500):
    from pylsl import StreamInfo, StreamOutlet, cf_float32
    sid = f"nc-test-{uuid.uuid4().hex[:12]}"
    info = StreamInfo("NCTest", "EEG", len(labels), fs, cf_float32, sid)
    chans = info.desc().append_child("channels")
    for lab, unit in zip(labels, units):
        ch = chans.append_child("channel")
        ch.append_child_value("label", lab)
        ch.append_child_value("unit", unit)
    return StreamOutlet(info), sid


@pytest.mark.parametrize("case", ["reversed_montage", "volts", "wrong_rate"])
def test_lsl_inlet_rejects_a_mismatched_stream_and_does_not_leak(case):
    labels, units = list(DEFAULT.hardware.channels), ["microvolts"] * 32
    hw = DEFAULT.hardware
    if case == "reversed_montage":
        labels = labels[::-1]
    elif case == "volts":
        units = ["V"] * 32
    else:
        hw = dataclasses.replace(hw, fs=250)
    outlet, sid = _outlet(labels, units)
    src = LSLEEG(hw, timeout_s=5.0, source_id=sid)
    with pytest.raises(RuntimeError, match="labels|microvolts|Hz"):
        src.start()
    assert src.inlet is None                                         # failed start closed its inlet
    src.close()                                                      # idempotent
    del outlet


def test_lsl_inlet_accepts_a_matching_stream_and_closes():
    outlet, sid = _outlet(list(DEFAULT.hardware.channels), ["microvolts"] * 32)
    with LSLEEG(DEFAULT.hardware, timeout_s=5.0, source_id=sid) as src:
        first = src.inlet
        src.start()                                                  # a second start replaces, not leaks
        assert src.inlet is not None and src.inlet is not first
    assert src.inlet is None
    del outlet
