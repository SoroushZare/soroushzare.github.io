"""Headless regression tests: the simulator → preprocessing → decoders → task → evaluation chain."""
import numpy as np

from neural_cursor.acquisition import SimulatedEEG, RingBuffer
from neural_cursor.config import DEFAULT
from neural_cursor.decoder import EvidenceDecoder, VelocityDecoder
from neural_cursor.evaluation import fitts_throughput, permutation_floor, pearson_r, wolpaw_itr_bits_per_trial
from neural_cursor.features import tangent_space, shrink_cov
from neural_cursor.signal import AmplitudeGate, CausalFilter, common_average, eog_proxy
from neural_cursor.task import TrialRunner, fitts_targets


def _windows(sim, filt, n, win_s, rng):
    """n windows of filtered simulated EEG with random intents; returns (windows, intents)."""
    fs = sim.hw.fs
    ws, ys = [], []
    for _ in range(n):
        y = rng.uniform(-1, 1, 2)
        sim.set_intent(*y)
        x = filt(sim.generate(int(win_s * fs)))
        ws.append(common_average(x))
        ys.append(y)
    return ws, np.array(ys)


def test_ring_buffer_order():
    rb = RingBuffer(5, 2)
    rb.push(np.arange(8, dtype=np.float32).reshape(4, 2), np.arange(4.0))
    x, ts = rb.latest(3)
    assert ts.tolist() == [1.0, 2.0, 3.0] and x[-1, 0] == 6.0


def test_causal_filter_is_stateful_and_chunk_invariant():
    cfg = DEFAULT
    sim = SimulatedEEG(cfg.hardware, seed=1)
    x = sim.generate(cfg.hardware.fs * 2)
    f1 = CausalFilter(cfg.hardware, cfg.pre)
    whole = f1(x)
    f2 = CausalFilter(cfg.hardware, cfg.pre)
    chunks = np.concatenate([f2(x[:333]), f2(x[333:900]), f2(x[900:])])
    assert np.allclose(whole, chunks, atol=1e-6)


def test_causal_filter_has_no_onset_transient_from_dc_offset():
    """Dry-electrode DC offsets of ±20 mV: started from the unit-input state they rang through the
    high-pass for ~6 s (1.9 mV residual at 1 s).  Starting in the first sample's steady state, the
    offset leaves no trace."""
    cfg = DEFAULT
    x = SimulatedEEG(cfg.hardware, seed=4).generate(cfg.hardware.fs * 3).astype(np.float64)
    dc = np.random.default_rng(0).uniform(-2e4, 2e4, cfg.hardware.n_channels)
    with_dc = CausalFilter(cfg.hardware, cfg.pre)(x + dc)
    without = CausalFilter(cfg.hardware, cfg.pre)(x)
    assert np.abs(with_dc - without).max() < 1e-3
    f = CausalFilter(cfg.hardware, cfg.pre)                     # still chunk-invariant
    assert np.allclose(np.concatenate([f(x[:7] + dc), f(x[7:] + dc)]), with_dc, atol=1e-6)


def test_amplitude_gate_flags_railed_channel_but_not_fp():
    cfg = DEFAULT
    gate = AmplitudeGate(cfg.hardware, cfg.pre)
    rng = np.random.default_rng(0)
    for _ in range(20):
        gate.update(rng.standard_normal((250, 32)) * 15)
    bad = rng.standard_normal((250, 32)) * 15
    oz = list(cfg.hardware.channels).index("Oz"); fp1 = list(cfg.hardware.channels).index("FP1")
    bad[:, oz] *= 50; bad[:, fp1] *= 50
    good = gate.update(bad)
    assert not good[oz] and good[fp1]


def test_tangent_space_at_reference_is_zero():
    rng = np.random.default_rng(0)
    c = shrink_cov(rng.standard_normal((500, 6)), 0.1)
    assert np.allclose(tangent_space(c, c), 0, atol=1e-8)


def test_velocity_decoder_learns_embedded_signal_and_eog_surrogate_does_not():
    cfg = DEFAULT
    rng = np.random.default_rng(0)
    sim = SimulatedEEG(cfg.hardware, seed=2)
    filt = CausalFilter(cfg.hardware, cfg.pre)
    filt(sim.generate(cfg.hardware.fs * 2))                     # settle the filter
    ws, ys = _windows(sim, filt, 400, cfg.decoder.window_s, rng)
    tr, te = slice(0, 300), slice(300, 400)
    dec = VelocityDecoder(cfg.decoder, cfg.hardware.fs).fit(ws[tr], ys[tr])
    r = pearson_r(dec.predict_many(ws[te])[:, 0], ys[te][:, 0])
    # EOG-only surrogate: frontal channels only.  With eye_gain_uv = 0 there is nothing to find.
    mask = np.array([c in cfg.hardware.eog_proxy for c in cfg.hardware.channels])
    sur = VelocityDecoder(cfg.decoder, cfg.hardware.fs, channels=mask).fit(ws[tr], ys[tr])
    r_sur = pearson_r(sur.predict_many(ws[te])[:, 0], ys[te][:, 0])
    assert r > 0.3, r
    assert abs(r_sur) < 0.25, r_sur
    # permutation floor at the same n
    pred = dec.predict_many(ws[te])[:, 0]
    fl = permutation_floor(lambda lab: pearson_r(pred, lab), ys[te][:, 0], 300, rng)
    assert fl.observed > fl.floor_p95


def test_evidence_decoder_cv_above_chance():
    cfg = DEFAULT
    rng = np.random.default_rng(1)
    sim = SimulatedEEG(cfg.hardware, seed=3)
    filt = CausalFilter(cfg.hardware, cfg.pre)
    filt(sim.generate(cfg.hardware.fs * 2))
    ws, ys = _windows(sim, filt, 300, 0.6, rng)
    lab = (ys[:, 0] > 0).astype(int)                             # stands in for accept/veto
    dec = EvidenceDecoder(cfg.decoder, feature_set="cov").fit(ws[:200], lab[:200])
    p = dec.predict_many(ws[200:])
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(lab[200:], p) > 0.6


def test_trial_runner_reaches_goal_with_oracle_and_times_out_with_noise():
    cfg = DEFAULT
    rng = np.random.default_rng(0)
    tr = TrialRunner(cfg.task, rng)
    tg = fitts_targets(8, 0.6, cfg.task.target_radius, rng)
    goal = 3
    oracle = lambda st, t: (tg[goal] - st.pos) / (np.linalg.norm(tg[goal] - st.pos) + 1e-9)
    o = tr.run(tg, cfg.task.target_radius, goal, oracle)
    assert o.success and o.movement_time_s < 5
    noise = lambda st, t: rng.uniform(-1, 1, 2)
    o2 = tr.run(tg, cfg.task.target_radius, goal, noise)
    assert o2.movement_time_s <= cfg.task.trial_timeout_s + 1e-9
    assert np.isnan(fitts_throughput([o]))                       # one success: no spread, no TP
    o3 = tr.run(tg, cfg.task.target_radius, goal, oracle)
    assert fitts_throughput([o, o3]) > 0
    assert 0.9 < wolpaw_itr_bits_per_trial(8, 0.95) < 3.0


def test_reported_decoder_r_and_its_floor_are_the_same_statistic():
    """simulate.py reports decoder_r_mean (mean over x and y of Pearson r) next to its floor; the floor
    must be the null of that statistic, not of the x-axis r alone.  Here x is decoded perfectly and y
    is anti-decoded: the reported r is 0, the x-only r would be 1."""
    from neural_cursor.simulate import decoder_r_with_floor, mean_axis_r
    rng = np.random.default_rng(0)
    ys = rng.standard_normal((60, 2))
    pred = np.column_stack([ys[:, 0], -ys[:, 1]])
    fl = decoder_r_with_floor(pred, ys, 300, rng)
    assert fl.observed == mean_axis_r(pred, ys) and abs(fl.observed) < 1e-12
    assert fl.floor_p95 < 0.3                                      # the null of the mean-axis r is near 0
    good = decoder_r_with_floor(ys + 0.5 * rng.standard_normal(ys.shape), ys, 300, rng)
    assert good.observed > good.floor_p95
