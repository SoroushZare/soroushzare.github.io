"""Simulated EEG sources: the generated signal must not depend on how the stream is chunked.

Real-time code pulls a few samples at a time while offline calibration generates whole blocks; if the
simulator's output depended on block size (as blink placement once did), a decoder calibrated
offline would meet a different signal online and the realtime tests would measure the simulator,
not the pipeline."""
import numpy as np

from neural_cursor.acquisition import BLINK_RATE_HZ, BLINK_UV, SimulatedEEG
from neural_cursor.config import DEFAULT
from neural_cursor.sim_population import PopulationSim

HW = DEFAULT.hardware
FP1 = list(HW.channels).index("FP1")
DURATION_S = 20                     # ≈ 5 blinks at 0.25 Hz: several straddle chunk boundaries


def _chunked(sim, sizes):
    return np.concatenate([sim.generate(k) for k in sizes])


def _sizes(total, rng):
    out = []
    while sum(out) < total:
        out.append(int(rng.integers(1, 40)))
    out[-1] -= sum(out) - total
    return out


def test_simulated_eeg_is_chunk_invariant_including_blinks_and_drift():
    n = DURATION_S * HW.fs
    whole = SimulatedEEG(HW, seed=5, drift=0.05)
    whole.set_intent(0.7, -0.4)
    a = whole.generate(n)
    parts = SimulatedEEG(HW, seed=5, drift=0.05)
    parts.set_intent(0.7, -0.4)
    b = _chunked(parts, _sizes(n, np.random.default_rng(0)))
    assert b.shape == a.shape
    assert np.allclose(a, b, atol=1e-3)
    # blinks are present, whole (peak ≈ BLINK_UV above background), and at a plausible rate
    peaks = np.flatnonzero((a[1:-1, FP1] > 100) & (a[1:-1, FP1] >= a[:-2, FP1]) & (a[1:-1, FP1] >= a[2:, FP1]))
    assert len(peaks) >= 1 and a[:, FP1].max() > 0.6 * BLINK_UV
    assert len(whole._blink_centres) <= 2                           # finished blinks leave the queue
    expected = DURATION_S * BLINK_RATE_HZ
    assert abs(len(set(np.round(peaks / HW.fs))) - expected) < 4 * np.sqrt(expected) + 1


def test_start_rebases_queued_blinks_without_losing_them():
    sim = SimulatedEEG(HW, seed=9)
    sim.generate(HW.fs)
    queued = sim._next_blink - sim.n_sent
    sim.start()
    assert sim.n_sent == 0 and np.isclose(sim._next_blink, queued)


def test_population_subject_is_chunk_invariant_within_a_block():
    pop = PopulationSim(1, seed=2)
    n = 8 * HW.fs
    s1, s2 = pop.subject(0), pop.subject(0)
    for s in (s1, s2):
        s.set_day(2)
        s.begin_block(2, 3)
        s.set_intent(-0.5, 0.5)
    a = s1.generate(n)
    b = _chunked(s2, _sizes(n, np.random.default_rng(1)))
    assert np.allclose(a, b, atol=1e-3)
