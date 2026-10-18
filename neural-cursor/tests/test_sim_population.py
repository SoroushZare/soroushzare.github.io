"""Population simulator and the days × hours engine simulation.

Subject model: topographies are the stated shared/individual mix, subject i does not depend on the
population size, day gain drift is per-channel multiplicative, gaze reaches EEG only for leaky
subjects, users do not learn unless told to, and a block's EEG does not depend on earlier blocks.
Engine: a tiny 2-day run through protocol → logger → data engine → purity, via the CLI, produces the
learning-curve report, learns a decodable signal, and lets only 32-channel EEG reach any decoder."""
import json

import numpy as np
import pytest

from neural_cursor import simulate_engine as se
from neural_cursor.config import DEFAULT
from neural_cursor.data_engine import RUN_FILES
from neural_cursor.decoder import VelocityDecoder
from neural_cursor.protocol import MAX_BLOCK_MIN, plan_summary, session_plan
from neural_cursor.purity import PurityAudit
from neural_cursor.sim_population import (HETEROGENEITY_RANGE, NON_CORTICAL_SUPPORT, PopulationSim,
                                          mix_topographies)

HW = DEFAULT.hardware
CH = list(HW.channels)


def test_topographies_are_the_shared_individual_mix():
    pop = PopulationSim(8, seed=3)
    subs = list(pop)
    lo, hi = HETEROGENEITY_RANGE
    frontal = [CH.index(c) for c in NON_CORTICAL_SUPPORT]
    for p, s in zip(pop.params, subs):
        assert lo <= p.heterogeneity <= hi and s.heterogeneity == p.heterogeneity
        expect = mix_topographies(s.shared, s.individual, p.heterogeneity)
        for name, topo in s.topographies.items():
            assert np.isclose(np.linalg.norm(topo), 1.0)
            assert np.allclose(topo, expect[name])
            assert np.all(s.individual[name][frontal] == 0)          # cortical patterns stay off the EOG sites
            assert np.all(topo >= 0)
    # individual patterns differ between people; the shared part does not
    a, b = subs[0], subs[1]
    assert np.allclose(a.shared["central"], b.shared["central"])
    assert not np.allclose(a.topographies["central"], b.topographies["central"])
    # subject i is the same person whatever the population size (paired own vs prior conditions)
    small = PopulationSim(2, seed=3)
    assert small.params[1] == pop.params[1]
    assert np.allclose(small.subject(1).topographies["alpha_left"], subs[1].topographies["alpha_left"])
    assert PopulationSim(8, seed=4).params[0] != pop.params[0]


def test_day_gain_drift_is_per_channel_multiplicative_and_order_free():
    pop = PopulationSim(1, seed=11)
    s1, s3 = pop.subject(0), pop.subject(0)
    assert np.all(s1.day_gain(1) == 1.0)
    g3 = s3.day_gain(3)                                             # queried first, before day 2
    assert np.allclose(s1.day_gain(2), s1.day_gain(2)) and np.allclose(s1.day_gain(3), g3)
    assert np.std(np.log(g3)) > 0.05 and not np.allclose(g3, g3[0])  # per channel, not one global factor
    s1.set_day(1)
    s3.set_day(3)
    for s in (s1, s3):
        s.begin_block(1, 0)
        s.set_intent(0.5, -0.5)
    x1, x3 = s1.generate(1000), s3.generate(1000)
    assert np.allclose(x3, x1 * g3[None].astype(np.float32), rtol=1e-4, atol=1e-3)


def test_gaze_reaches_eeg_only_for_leaky_subjects():
    leaky = PopulationSim(2, seed=5, ocular_leak_fraction=1.0)
    clean = PopulationSim(2, seed=5)
    assert all(p.ocular_leak and p.eye_gain_uv > 0 for p in leaky.params)
    assert not any(p.ocular_leak for p in clean.params) and all(p.eye_gain_uv == 0 for p in clean.params)

    def pair(pop):
        out = []
        for gaze in ((0.0, 0.0), (0.8, 0.0)):
            s = pop.subject(0)
            s.begin_block(1, 0)
            s.set_gaze(*gaze)
            out.append(s.generate(500))
        return out[1] - out[0]

    d = pair(leaky).mean(axis=0)
    assert d[CH.index("F8")] > 10 and d[CH.index("F7")] < -10        # signed horizontal EOG, ≈ 16 µV
    others = [i for i, c in enumerate(CH) if c not in ("F7", "F8")]
    assert np.allclose(d[others], 0, atol=1e-3)
    assert np.allclose(pair(clean), 0)


def test_users_do_not_learn_by_default():
    s = PopulationSim(1, seed=2).subject(0)
    assert s.user_learning_rate == 0.0
    before = s.modulation_depth
    s.record_practice(10 * 3600)
    assert s.modulation_depth == before
    learner = PopulationSim(1, seed=2, user_learning_rate=0.5).subject(0)
    assert learner.modulation_depth == before < 1.0
    learner.record_practice(3600)
    assert before < learner.modulation_depth <= 1.0


def test_block_eeg_does_not_depend_on_earlier_blocks():
    """Cued blocks are identical across conditions only if a block's EEG ignores how long the
    previous (closed-loop) blocks ran."""
    pop = PopulationSim(1, seed=9)
    a, b = pop.subject(0), pop.subject(0)
    a.begin_block(2, 0)
    a.set_intent(1.0, 0.0)
    a.generate(1234)                                                 # an earlier block of another length
    out = []
    for s in (a, b):
        s.set_day(2)
        s.begin_block(2, 3)
        s.set_intent(0.0, 1.0)
        out.append(s.generate(800))
    assert np.array_equal(out[0], out[1])


def test_day_blocks_scale_the_protocol_without_breaking_it():
    one = plan_summary(session_plan(2, 7))
    for hours in (0.1, 1.0):
        s = plan_summary(se.day_blocks(2, hours, seed=7))
        assert np.isclose(s["total_s"], hours * one["total_s"])       # one plan = one hour of use
        assert np.isclose(s["cued_fraction"], one["cued_fraction"]) and s["blank_feedback"] > 0
    two = se.day_blocks(2, 2.0, seed=7)                              # two unscaled sessions, caps respected
    assert sum(b.kind == "setup" for b in two) == 2 and two != se.day_blocks(2, 1.0, seed=7) * 2
    assert all(b.duration_s <= MAX_BLOCK_MIN[b.kind] * 60 + 1e-6 for b in two if b.kind in MAX_BLOCK_MIN)


def test_engine_simulation_cli_end_to_end(tmp_path, monkeypatch):
    """2 days × 0.1 h, population of 3: the full protocol → logger → engine → audit path via the CLI."""
    shapes, audits = set(), []
    features = VelocityDecoder.features
    run = PurityAudit.run

    def checked_features(self, window):
        shapes.add(np.shape(window))
        return features(self, window)

    def recorded_run(self, windows, labels, provenance, auditors=None, cursor_logs=None):
        audits.append((len(windows), {k: np.shape(v) for k, v in (auditors or {}).items()}))
        return run(self, windows, labels, provenance, auditors=auditors, cursor_logs=cursor_logs)

    monkeypatch.setattr(VelocityDecoder, "features", checked_features)
    monkeypatch.setattr(PurityAudit, "run", recorded_run)
    prefix = str(tmp_path / "out" / "engine")
    res = se.main(["--days", "2", "--hours", "0.1", "--pop", "3", "--pop-minutes", "1", "--workers", "1",
                   "--min-new-windows", "50", "--min-heldout", "20", "--min-heldout-trials", "4",
                   "--eval-minutes", "0.5", "--out", prefix])

    # Pure-EEG rule: every window that reached any decoder (fit, closed-loop predict, champion test,
    # purity surrogates) was a 32-channel EEG window; the eye tracker went to the audit only.
    win = int(round(DEFAULT.decoder.window_s * HW.fs))
    assert shapes == {(win, HW.n_channels)}
    assert se.GAZE_FILE not in RUN_FILES
    assert len(audits) == 4 and all(a[1] == {"eyetracker": (a[0], 2)} for a in audits)

    with open(prefix + ".json") as f:
        saved = json.load(f)
    md = open(prefix + ".md").read()
    assert set(saved["conditions"]) == {"own", "population"} and len(saved["population"]) == 3
    assert md.count("\n| own |") == 2 and md.count("\n| population |") == 2
    own = res["conditions"]["own"]["per_user"][0]["days"]
    prior = res["conditions"]["population"]["per_user"][0]["days"]
    for rows in (own, prior):
        for r in rows:
            assert 0.0 <= r["closed_loop"]["success"] <= 1.0 and r["closed_loop"]["n_trials"] > 0
            assert r["n_windows_day"] > 0 and r["purity"]["flags"]
            assert r["n_eval_windows"] > 0
        assert rows[1]["cum_hours"] == pytest.approx(0.2)
    # Day 1 without a prior: nothing deployed, so the decoder scores exactly zero and cannot select.
    assert own[0]["deployed"] == "none" and own[0]["decoder_cosine"] == 0.0
    assert own[0]["closed_loop"]["success"] == 0.0
    # The engine learned from day 1: day 2's decoder works on day 2's unseen cued windows.
    assert own[1]["deployed"].startswith("v") and own[1]["decoder_cosine"] > 0.1
    # A population prior gives a usable decoder before any own data (zero-calibration start).
    assert prior[0]["deployed"] == "population prior" and prior[0]["decoder_cosine"] > 0.05
    # A fixed evaluation pool: every day and both conditions are scored on the very same windows,
    # never trained on, with the pool's constant-output ceiling and a trial-level CI.
    n_eval = {r["n_eval_windows"] for rows in (own, prior) for r in rows}
    assert len(n_eval) == 1 and n_eval.pop() > 0
    for rows in (own, prior):
        for r in rows:
            lo, hi = r["decoder_cosine_ci95"]
            assert lo <= r["decoder_cosine"] <= hi and 0.0 <= r["decoder_constant_ceiling"] < 1.0
            assert r["n_eval_trials"] > 1 and "cosine" in r["today"]
    assert own[0]["decoder_score_defined_zero"] and not own[1]["decoder_score_defined_zero"]
    assert "0 (def.)" in md and "oracle closed-loop labels" in md and "ceiling" in md
    assert res["targets"] == saved["targets"]


def test_intent_mismatch_model():
    """Default: closed-loop intent is exactly the label direction (oracle labels).  With an angular
    error and lapses, intent deviates from the label, re-drawn once per segment."""
    fs = HW.fs
    d = np.array([0.6, 0.8])
    oracle = se.Mismatch(se.SimConfig(), np.random.default_rng(0), fs)
    assert all(np.array_equal(oracle.apply(d, k * 50), d) for k in range(100))
    noisy = se.Mismatch(se.SimConfig(intent_angle_sd_deg=30.0, intent_lapse_prob=0.2), np.random.default_rng(0), fs)
    out = np.array([noisy.apply(d, k * 50) for k in range(2000)])          # 200 one-second segments
    seg = out.reshape(200, 10, 2)
    assert np.allclose(seg, seg[:, :1])                                    # held within a segment
    lapsed = np.linalg.norm(seg[:, 0], axis=1) == 0
    assert 0.1 < lapsed.mean() < 0.3
    cos = seg[~lapsed, 0] @ d
    assert np.allclose(np.linalg.norm(seg[~lapsed, 0], axis=1), 1.0) and 0.8 < cos.mean() < 0.95
    with pytest.raises(ValueError):
        se.SimConfig(intent_lapse_prob=1.0)
