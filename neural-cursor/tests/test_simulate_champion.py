"""Tests for the champion simulation (INTERFACES_CHAMPION item K).

Two simulated sessions of one person are the fixtures.  The person is a strong lateraliser (x ERD depth
0.8), so x is earned within one session, and the certification can use the x axis.  The horizontal
screen-leak pathway reaches the central-11 montage much more strongly than the vertical one, so a y-only
participant gives a near-insensitive positive control at test scale (see the module docstring and the
simulation report).  The session allocation (``slot_minutes``) moves minutes from the protocol's
85-min plan toward L1, S and held-out blocks, so that one day holds enough trials for the trial-cluster
bootstrap.  The two sessions differ only in the covert leak (and in S minutes, which train only the
shadow).

What is asserted, and why it is the claim of the design:
* the whole chain (prior → L1 → L2b → retrain → S → shadow → held-out mappings → β table and verdict)
  runs in under a minute for a tiny configuration;
* only L1/L2b windows reach the deployed trainer, only S windows the shadow trainer, and every decoder
  sees only (500, 32) EEG windows (pure-EEG rule, checked with spies);
* with screen_leak > 0 the shadow reads the screen (its β_screen CI lies above 0), so the verdict is not
  TEST_INSENSITIVE: the positive control is exercised;
* with a covert leak in the blind blocks, the deployed decoder learns the attention pattern from its own
  cue-defined labels and then reads the screen under held-out mappings: no longer CERTIFIED.
"""
import dataclasses
import json
import time

import numpy as np
import pytest

from neural_cursor.body_decoder import BodyCoordinateDecoder
from neural_cursor.data_engine import BodyTrainer, ShadowTrainer
from neural_cursor.protocol import CHAMPION_SLOT_MIN, session_plan_champion
from neural_cursor.simulate_champion import (CHAMPION_SESSION_S, ChampionSimConfig, day_sessions, simulate,
                                             to_markdown, write_report)

SLOTS = (("fixation_probe", 6.0), ("blind_closed_loop", 1.0), ("sighted", 6.0), ("heldout_mapping", 8.0))
BASE = ChampionSimConfig(days=1, hours=0.1, users=1, pop=2, seed=46, pop_minutes=1.0, rest_s=20.0,
                         eval_minutes=1.0, screen_leak=2.0, covert_leak=0.0, x_responder_prob=1.0,
                         x_depth_range=(0.8, 0.8), x_permutations=50, n_boot=1000, slot_minutes=SLOTS)
COVERT = dataclasses.replace(BASE, covert_leak=2.0,
                             slot_minutes=tuple((k, 2.0 if k == "sighted" else m) for k, m in SLOTS))
"""S minutes only train the shadow (the deployed decoder is retrained before S, and the held-out blocks
are seeded per block), so the covert run keeps 2 min of S for speed."""


def _run_with_spies(sim):
    """simulate() in-process, recording the provenance sources each trainer receives and the shape of
    every window any body decoder (deployed or shadow) is asked to decode."""
    seen = {"deployed": set(), "shadow": set(), "shapes": set()}
    fit_deployed, fit_shadow = BodyTrainer.fit, ShadowTrainer.fit
    predict, predict_many = BodyCoordinateDecoder.predict, BodyCoordinateDecoder.predict_many

    def spy_deployed(self, data, session_references=None):
        seen["deployed"].update(p.get("source") for p in data.provenance)
        return fit_deployed(self, data, session_references)

    def spy_shadow(self, data, x_enabled=False, session_references=None):
        seen["shadow"].update(p.get("source") for p in data.provenance)
        return fit_shadow(self, data, x_enabled, session_references)

    def spy_predict(self, window):
        seen["shapes"].add(np.shape(window))
        return predict(self, window)

    def spy_predict_many(self, windows):
        seen["shapes"].update(np.shape(w) for w in windows)
        return predict_many(self, windows)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(BodyTrainer, "fit", spy_deployed)
        mp.setattr(ShadowTrainer, "fit", spy_shadow)
        mp.setattr(BodyCoordinateDecoder, "predict", spy_predict)
        mp.setattr(BodyCoordinateDecoder, "predict_many", spy_predict_many)
        t0 = time.perf_counter()
        res = simulate(sim, workers=1)
        return res, seen, time.perf_counter() - t0


@pytest.fixture(scope="module")
def baseline():
    return _run_with_spies(BASE)


@pytest.fixture(scope="module")
def covert():
    return simulate(COVERT, workers=1)


def _heldout(res):
    return res["per_user"][0]["days"][0]["certification"]["heldout"]


def test_tiny_configuration_runs_the_whole_chain_under_a_minute(baseline):
    res, _, elapsed = baseline
    assert elapsed < 60.0, elapsed
    day = res["per_user"][0]["days"][0]
    assert all(day["n_windows"][s] > 0 for s in ("L1", "L2b", "S", "H")), day["n_windows"]
    assert day["n_train_windows"]["prior"] > 0 and day["n_train_windows"]["shadow"] == day["n_windows"]["S"]
    assert 0.0 <= day["sighted"]["success"] <= 1.0 and 0.0 <= day["blind"]["success"] <= 1.0
    assert day["sighted"]["n_trials"] > 0 and day["blind"]["n_trials"] > 0
    assert set(day["l1"]["auc"]) == {"x", "y"} and np.isfinite(day["l1"]["auc"]["y"])
    assert day["x_earned"] and day["x_report"]["auc"] > 0.6     # strong lateraliser: x earned by the rule
    cert = _heldout(res)
    assert cert["axes"] == ["x", "y"] and cert["n_trials"] >= 8
    for role in ("deployed", "shadow"):
        lo, hi = cert[role]["ci_screen"]
        assert lo <= cert[role]["beta_screen"] <= hi


def test_only_permitted_sources_and_eeg_windows_reach_the_decoders(baseline):
    _, seen, _ = baseline
    assert seen["deployed"] == {"L1", "L2b"}, seen["deployed"]   # prior members' windows are L1 too
    assert seen["shadow"] == {"S"}, seen["shadow"]
    assert seen["shapes"] == {(500, 32)}, seen["shapes"]


def test_screen_leak_exercises_the_shadow_positive_control(baseline):
    cert = _heldout(baseline[0])
    assert cert["flags"]["shadow_reads_screen"], cert["shadow"]
    assert cert["verdict"] != "TEST_INSENSITIVE"
    assert cert["flags"]["deployed_body_significant"] and not cert["flags"]["deployed_screen_significant"]


def test_covert_leak_in_blind_blocks_breaks_the_certification(baseline, covert):
    clean, leaky = _heldout(baseline[0]), _heldout(covert)
    assert leaky["verdict"] != "CERTIFIED"
    assert leaky["flags"]["deployed_screen_significant"], leaky["deployed"]
    assert leaky["deployed"]["beta_screen"] > clean["deployed"]["ci_screen"][1]


def test_day_sessions_scale_the_plan_and_honour_slot_minutes():
    plan = session_plan_champion(2, 7)
    one_hour = day_sessions(2, 1.0, 7, False)
    assert len(one_hour) == 1 and sum(b.duration_s for b in one_hour[0]) == pytest.approx(3600.0)
    assert [(b.kind, b.mapping, b.variant, b.seed) for b in one_hour[0]] == \
           [(b.kind, b.mapping, b.variant, b.seed) for b in plan]
    three = day_sessions(2, 3.0, 7, False)                         # never stretched: more sessions instead
    assert len(three) == 3 and all(sum(b.duration_s for b in s) <= CHAMPION_SESSION_S + 1e-6 for s in three)
    custom = day_sessions(2, 1.0, 7, False, (("heldout_mapping", 20.0),))[0]
    assert sum(b.duration_s for b in custom if b.kind == "heldout_mapping") == pytest.approx(1200.0)
    sighted = [b.duration_s for b in custom if b.kind == "sighted" and not b.just_intend]
    assert sum(sighted) == pytest.approx(CHAMPION_SLOT_MIN["sighted"] * 60.0 * 3600.0 / CHAMPION_SESSION_S)


@pytest.mark.parametrize("bad", [dict(slot_minutes=(("rest", 5.0),)), dict(slot_minutes=(("sighted", 0.0),)),
                                 dict(replay_attention="target"), dict(covert_leak=-0.1), dict(n_boot=500),
                                 dict(dry_penalty=1.0)])
def test_invalid_configuration_is_refused(bad):
    with pytest.raises(ValueError):
        dataclasses.replace(BASE, **bad)


def test_report_round_trips_to_json_and_markdown(baseline, tmp_path):
    res = {"dry_penalties": [0.0], "conditions": {"0": baseline[0]}, "runtime_s": baseline[2]}
    js, md = write_report(res, str(tmp_path / "champion_tiny"))
    with open(js) as f:
        back = json.load(f)                                        # strict JSON: NaN written as null
    assert back["conditions"]["0"]["per_user"][0]["days"][0]["certification"]["heldout"]["verdict"] == \
           _heldout(baseline[0])["verdict"]
    text = open(md).read()
    assert text == to_markdown(res) + "\n" and _heldout(baseline[0])["verdict"] in text
