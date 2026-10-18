"""Tests for the champion simulation (INTERFACES_CHAMPION item K).

Two simulated sessions of one person are the fixtures.  The person is a strong lateraliser (x ERD depth
0.8); with the decoder's central-11 local reference (which removes the central common mode, and with it
about half of this simulator's lateral contrast) 6 min of probes are not enough to earn x, so the tested
axes follow whatever earn_x decides.  The session allocation (``slot_minutes``) moves minutes from
the protocol's 85-min plan toward L1, S and held-out blocks, so that one day holds enough trials for the
trial-cluster bootstrap.  The two sessions differ only in the covert leak (and in S minutes, which train
only the shadow).  At this scale (one short day, a few dozen held-out trials per rotation) every axis is
tested on its own against the trained margin, so verdicts are mostly TEST_INSENSITIVE / INCONCLUSIVE:
the assertions are about the β estimates and the flags, not about a CERTIFIED headline.

What is asserted, and why it is the claim of the design:
* the whole chain (prior → L1 → L2b → retrain → idle gate → S → both shadows → held-out mappings with
  interleaved identity trials → β table and verdict) runs for a tiny configuration, with the external
  identity-trial margin as the certification's ruler and diagonal held-out targets (per-block body–screen
  correlation ≤ 0.5);
* only L1/L2b windows reach the deployed trainer, only S windows the shadow trainers (central-11 and
  all-32; identity trials never), only rest (L0) windows the idle gate, and every decoder sees only
  (500, 32) EEG windows (pure-EEG rule, checked with spies);
* with screen_leak > 0 a shadow reads the screen on y (its β_screen > 0 in the Holm family): the
  positive control is exercised on the primary axis;
* with a covert leak in the blind blocks, the deployed decoder learns the attention pattern from its own
  cue-defined labels and its β_screen under held-out mappings rises above the clean run's CI; and the x
  axis, now carried partly by posterior attention, is refused by earn_x's purity rule.
The certification plumbing (units, people, veto, window-start regressors) and the CLI are tested on
synthetic logs without a simulation.  The screen_leak = 0 null run is in ``test_champion_null.py``.
"""
import dataclasses
import json
import time

import numpy as np
import pytest

from neural_cursor.body_decoder import CENTRAL11, BodyCoordinateDecoder
from neural_cursor.data_engine import BodyTrainer, LabelledSet, ShadowTrainer
from neural_cursor.mapping import DIAGONAL_DIRECTIONS as DIAGONAL
from neural_cursor.mapping import Mapping, unit_directions
from neural_cursor.protocol import CHAMPION_SLOT_MIN, session_plan_champion
import neural_cursor.simulate_champion as sc
from neural_cursor.mapping import diagonal_targets
from neural_cursor.sim_body import BodySubject
from neural_cursor.simulate_champion import (CHAMPION_SESSION_S, MIN_IDENTITY_TRIALS, CertificationLog, ChampionRunner,
                                             ChampionSimConfig, Trajectory, champion_config, certify_user, day_sessions,
                                             default_prefix, identity_trial_margin, identity_trial_margin_stats, main,
                                             regressors, simulate, sweep, to_markdown, write_report)

SLOTS = (("fixation_probe", 6.0), ("blind_closed_loop", 1.0), ("sighted", 6.0), ("heldout_mapping", 8.0))
BASE = ChampionSimConfig(days=1, hours=0.1, users=1, pop=2, seed=46, pop_minutes=1.0, rest_s=20.0,
                         eval_minutes=1.0, screen_leak=2.0, covert_leak=0.0, x_responder_prob=1.0,
                         x_depth_range=(0.8, 0.8), x_permutations=50, n_boot=1000, slot_minutes=SLOTS)
COVERT = dataclasses.replace(BASE, covert_leak=2.0, null_calibration=False,
                             slot_minutes=tuple((k, 2.0 if k == "sighted" else m) for k, m in SLOTS))
"""S minutes only train the shadow (the deployed decoder is retrained before S, and the held-out blocks
are seeded per block), so the covert run keeps 2 min of S for speed."""


def _run_with_spies(sim):
    """simulate() in-process, recording the provenance sources each trainer (and the idle gate)
    receives, the montage size of each shadow, and the shape of every window any body decoder (deployed
    or shadow) is asked to decode or the gate is fitted on."""
    seen = {"deployed": set(), "shadow": set(), "shapes": set(), "idle": set(), "shadow_channels": set(),
            "x_sources": set(), "x_sessions": set()}
    fit_deployed, fit_shadow = BodyTrainer.fit, ShadowTrainer.fit
    predict, predict_many = BodyCoordinateDecoder.predict, BodyCoordinateDecoder.predict_many
    fit_idle = BodyCoordinateDecoder.fit_idle

    def spy_deployed(self, data, session_references=None, **kw):
        seen["deployed"].update(p.get("source") for p in data.provenance)
        if kw.get("x_mask") is not None:                      # the x model's windows: own L1 probes only
            seen["x_sources"].update(p.get("source") for p, m in zip(data.provenance, kw["x_mask"]) if m)
            seen["x_sessions"].update(p.get("session") for p, m in zip(data.provenance, kw["x_mask"]) if m)
        return fit_deployed(self, data, session_references, **kw)

    def spy_shadow(self, data, x_enabled=False, session_references=None):
        seen["shadow"].update(p.get("source") for p in data.provenance)
        seen["shadow_channels"].add(len(self.decoder_options.get("channels", CENTRAL11)))
        return fit_shadow(self, data, x_enabled, session_references)

    def spy_idle(self, windows, provenance, session_references=None):
        seen["idle"].update(p.get("source") for p in provenance)
        seen["shapes"].update(np.shape(w) for w in windows)
        return fit_idle(self, windows, provenance, session_references)

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
        mp.setattr(BodyCoordinateDecoder, "fit_idle", spy_idle)
        t0 = time.perf_counter()
        res = simulate(sim, workers=1)
        return res, seen, time.perf_counter() - t0


@pytest.fixture(scope="module")
def baseline():
    """BASE without its screen_leak 0 twin (the spies watch this process only, and the twin would double
    the time); the twin-calibrated run is ``tests/test_champion_twin.py``."""
    return _run_with_spies(dataclasses.replace(BASE, null_calibration=False))


@pytest.fixture(scope="module")
def covert():
    return simulate(COVERT, workers=1)


def _heldout(res):
    return res["per_user"][0]["days"][0]["certification"]["heldout"]


def test_tiny_configuration_runs_the_whole_chain(baseline):
    res, _, _ = baseline
    day = res["per_user"][0]["days"][0]
    assert all(day["n_windows"][s] > 0 for s in ("L1", "L2b", "S", "H")), day["n_windows"]
    assert day["n_train_windows"]["prior"] > 0 and day["n_train_windows"]["shadow"] == day["n_windows"]["S"]
    assert 0.0 <= day["sighted"]["success"] <= 1.0 and 0.0 <= day["blind"]["success"] <= 1.0
    assert day["sighted"]["n_trials"] > 0 and day["blind"]["n_trials"] > 0
    assert set(day["l1"]["auc"]) == {"x", "y"} and np.isfinite(day["l1"]["auc"]["y"])
    assert all(0.5 < day["l1"]["floor"][a] < 1.0 for a in ("x", "y")), day["l1"]["floor"]   # trial-level null
    # x is judged by the rule on own L1 probes; the tested axes follow its decision
    assert day["x_report"]["rule"] == "grant" and day["x_report"]["n_windows"] > 0
    assert day["margin"]["y"] > 0 and day["margin_l1"]["y"] > 0
    # identity trials: interleaved in the held-out blocks, counted apart, never shadow training data
    assert day["n_windows"]["I"] > 0 and day["n_windows"]["L0"] > 0
    assert day["n_train_windows"]["idle"] >= day["n_windows"]["L0"]
    idle = day["idle"]
    assert isinstance(idle["armed"], bool) and 0.0 <= idle["rest_gated"] <= 1.0 and 0.0 <= idle["intent_passed"] <= 1.0
    cert = _heldout(res)
    axes = ["x", "y"] if day["x_earned"] else ["y"]
    assert cert["axes"] == axes and cert["n_trials"] >= 8 and cert["n_units"] == 1
    assert set(cert["per_axis"]) == set(axes) and set(cert["shadows"]) == {"central11", "all32"}
    for entries in (cert["deployed"], cert["shadows"]["central11"], cert["shadows"]["all32"]):
        for axis in axes:
            e = entries[axis]
            lo, hi = e["ci_screen"]
            assert lo <= e["beta_screen"] <= hi and "beta_cross" in e
            # the ruler is the external identity-trial margin, not the self-estimate
            assert e["scale"] == pytest.approx(day["margin"][axis])
            assert abs(e["scale"] - (e["beta_body"] + e["beta_screen"])) > 1e-6
            # balanced diagonal targets: body and screen regressors uncorrelated in every block
            assert e["max_block_corr"] <= 0.5 and not e["confounded"]
    assert cert["flags"]["y"]["margin_external"]
    assert set(res["per_user"][0]["days"][0]["certification"]["heldout_by_variant"]) == {"C2b"}   # day 1


def test_only_permitted_sources_and_eeg_windows_reach_the_decoders(baseline):
    _, seen, _ = baseline
    assert seen["deployed"] == {"L1", "L2b"}, seen["deployed"]   # prior members' windows are L1 too
    # the x model trains on the windows earn_x tests: the person's own L1 probes, never the prior's
    assert seen["x_sources"] == {"L1"} and seen["x_sessions"] and all("pop" not in str(x) for x in seen["x_sessions"])
    assert seen["shadow"] == {"S"}, seen["shadow"]
    assert seen["shadow_channels"] == {len(CENTRAL11), 32}       # the central-11 and the all-32 shadow
    assert seen["idle"] == {"L0"}, seen["idle"]                  # intent class: the deployed training set
    assert seen["shapes"] == {(500, 32)}, seen["shapes"]


def test_screen_leak_exercises_the_shadow_positive_control(baseline):
    """Uncalibrated here (tested against 0, so never CERTIFIED); against the twin's null in
    ``test_champion_twin.py``."""
    cert = _heldout(baseline[0])
    y = cert["flags"]["y"]
    assert not y["shadow_null_calibrated"] and cert["shadows_null"] is None
    assert y["shadow_reads_screen"], cert["shadows"]
    assert set(y["shadows_read_screen"]) == {"central11", "all32"} and y["shadows_read_screen"]["all32"]
    assert y["deployed_body_significant"] and not y["deployed_screen_significant"]
    assert cert["per_axis"]["y"] != "TEST_INSENSITIVE"
    assert all(v != "SCREEN_READING" for v in cert["per_axis"].values())


def test_covert_leak_in_blind_blocks_breaks_the_certification(baseline, covert):
    clean, leaky = _heldout(baseline[0]), _heldout(covert)
    assert leaky["verdict"] != "CERTIFIED"
    assert leaky["deployed"]["y"]["beta_screen"] > clean["deployed"]["y"]["ci_screen"][1]
    # the same strong lateraliser no longer earns x: its x skill is partly posterior attention
    day = covert["per_user"][0]["days"][0]
    assert not day["x_earned"] and "posterior" in day["x_report"]["reason"]


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
                                 dict(slot_minutes=(("sighted", 3.0), ("sighted", 4.0))),
                                 dict(replay_attention="target"), dict(covert_leak=-0.1), dict(n_boot=500),
                                 dict(dry_penalty=1.0), dict(day_depth_sd=-0.1), dict(x_time_course=(1.0, 1.0))])
def test_invalid_configuration_is_refused(bad):
    with pytest.raises(ValueError):
        dataclasses.replace(BASE, **bad)


@pytest.mark.parametrize("arg", ["heldout_mapping", "heldout_mapping=abc", "rest=5", "sighted=0",
                                 "heldout_mapping=8 heldout_mapping=9"])
def test_cli_refuses_a_malformed_slot_with_a_usage_error(arg, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--days", "1", "--slot-minutes", *arg.split()])
    assert exc.value.code == 2 and "slot" in capsys.readouterr().err


def test_default_output_prefix_names_every_non_default_setting():
    """Runs that differ in any setting must not overwrite each other's results."""
    plain = default_prefix(ChampionSimConfig(days=2, users=2, pop=6), [0.0], [0.0])
    assert plain == "results/champion_d2_h1_u2_pop6_dry0_leak1_covert0"
    tuned = ChampionSimConfig(days=2, users=2, pop=6, y_depth_range=(0.3, 0.4), eval_minutes=8.0,
                              slot_minutes=(("heldout_mapping", 24.0),), seed=7)
    name = default_prefix(tuned, [0.0, 0.35], [0.0, 1.0])
    for part in ("dry0-0.35", "covert0-1", "seed7", "y0.3-0.4", "eval8", "slotshe24"):
        assert part in name, (part, name)


def _synthetic_log(person, versions, n_trials, w_screen=0.0, gain=0.4, seed=0, extra=None):
    """Held-out windows of one person over several decoder versions (each one session with rot+90 and
    rot−90): deployed output = gain·body + w_screen·screen + noise; shadow = 0.3·body + 0.3·screen;
    ``extra`` {name: (w_body, w_screen)} adds more shadows."""
    rng = np.random.default_rng(seed)
    log = CertificationLog()
    for v in range(versions):
        dep, sha, body, screen, prov = [], [], [], [], []
        for t in range(n_trials):
            m = Mapping(("rot+90", "rot-90")[t % 2])
            a = rng.uniform(0, 2 * np.pi)
            u = np.tile([np.cos(a), np.sin(a)], (6, 1))
            b, s = m.screen_to_body(u), u
            noise = rng.normal(0, 0.3, (6, 2)) + rng.normal(0, 0.1, 2)
            dep.append(gain * b + w_screen * s + noise)
            sha.append(0.3 * b + 0.3 * s + noise)
            body.append(b)
            screen.append(s)
            prov += [{"session": f"{person}_v{v}", "trial": t, "t": float(k), "source": "H", "variant": "C2b"}
                     for k in range(6)]
        b_all, s_all = np.concatenate(body), np.concatenate(screen)
        more = {name: wb * b_all + ws * s_all + rng.normal(0, 0.3, b_all.shape)
                for name, (wb, ws) in (extra or {}).items()}
        log.add(np.concatenate(dep), np.concatenate(sha), b_all, s_all, prov,
                np.ones(len(prov), bool), f"{person}/v{v}", person, {"x": gain, "y": gain}, extra_shadows=more)
    return log


def test_certification_fits_each_decoder_version_and_vetoes_on_any_unit():
    """Units are decoder versions (own slopes, intercepts and external margin), combined within a person
    by inverse variance and across people by a person-then-trial bootstrap; one version that reads the
    screen makes the whole verdict SCREEN_READING even when the other people are clean."""
    clean = CertificationLog.merge([_synthetic_log("u0", 2, 60, seed=1), _synthetic_log("u1", 2, 60, seed=2)])
    res = clean.certify("heldout", 1000)
    assert res["n_units"] == 4 and res["n_people"] == 2 and res["verdict"] != "SCREEN_READING"
    assert set(res["unit_verdicts"]) == {"u0", "u1"} and res["vetoed_by"] == []   # people: top-level units
    # two people give a population (person bootstrap, t with 1 df) interval far too wide to certify
    assert res["deployed"]["y"]["n_units"] == 2 and res["verdict"] != "CERTIFIED"
    leaky = CertificationLog.merge([clean, _synthetic_log("u2", 1, 120, w_screen=0.2, seed=3)])
    res = leaky.certify("heldout", 1000)
    assert res["verdict"] == "SCREEN_READING" and res["vetoed_by"] == ["u2"], res["reason"]
    few = _synthetic_log("u3", 1, 5, seed=4).certify("heldout", 1000)
    assert "skipped" in few


def test_heldout_regressors_use_the_cursor_at_the_window_start():
    """In closed loop the window-end cursor has been moved by the window's own output (endogenous)."""
    goal = [0.6, 0.0]
    prov = [{"display_goal": goal, "cursor_pos": [0.5, 0.3], "cursor_pos_start": [0.0, 0.0], "mapping": "rot+90",
             "replay_goal": [-0.6, 0.0]}]
    data = LabelledSet([np.zeros((500, 32), np.float32)], np.zeros((1, 2)), prov)
    body, screen = regressors("heldout", data)
    np.testing.assert_allclose(screen[0], [1.0, 0.0])
    np.testing.assert_allclose(body[0], Mapping("rot+90").screen_to_body(np.array([1.0, 0.0])))
    _, replayed = regressors("replay", data)                     # replay: matched to the whole window
    expect = unit_directions(np.array([[-0.6, 0.0]] * 2), np.array([[0.0, 0.0], [0.5, 0.3]])).sum(axis=0)
    np.testing.assert_allclose(replayed[0], expect / np.linalg.norm(expect))


def test_report_round_trips_to_json_and_markdown(baseline, tmp_path):
    res = {"dry_penalties": [0.0], "conditions": {"0": baseline[0]}, "runtime_s": baseline[2]}
    js, md = write_report(res, str(tmp_path / "champion_tiny"))
    with open(js) as f:
        back = json.load(f)                                        # strict JSON: NaN written as null
    assert back["conditions"]["0"]["per_user"][0]["days"][0]["certification"]["heldout"]["verdict"] == \
           _heldout(baseline[0])["verdict"]
    text = open(md).read()
    assert text == to_markdown(res) + "\n" and _heldout(baseline[0])["verdict"] in text


def test_identity_trials_give_the_external_margin():
    """The margin is the frozen decoder's slope on body intent in the identity trials of its own
    held-out blocks: measured on windows the certification never sees, from the window-start cursor."""
    rng = np.random.default_rng(3)
    prov, outs = [], []
    for t in range(12):
        goal = DIAGONAL[t % 4] * 0.6
        for k in range(5):
            start = goal * k / 10
            u = (goal - start) / np.linalg.norm(goal - start)
            prov.append({"session": "id", "trial": t, "display_goal": goal.tolist(), "cursor_pos_start": start.tolist()})
            outs.append(0.35 * u + rng.normal(0, 0.1, 2))
    data = LabelledSet([np.zeros((500, 32), np.float32)] * len(prov), np.zeros((len(prov), 2)), prov)
    m = identity_trial_margin(np.array(outs), data)
    assert m["x"] == pytest.approx(0.35, abs=0.05) and m["y"] == pytest.approx(0.35, abs=0.05)
    stats = identity_trial_margin_stats(np.array(outs), data)
    for a in ("x", "y"):                                          # the margin carries its trial-clustered SE
        slope, se, df = stats[a]
        assert slope == m[a] and 0 < se < 0.1 and df == 12 - 2
    few = data.mask(np.array([p["trial"] < MIN_IDENTITY_TRIALS - 1 for p in prov]))
    assert all(np.isnan(v) for v in identity_trial_margin(np.array(outs)[: len(few)], few).values())


def test_either_shadow_counts_in_the_certification_log():
    """The log carries the all-32 shadow beside the central-11 one: an insensitive central-11 shadow
    no longer makes the test insensitive when the all-32 one reads the screen, and both are reported."""
    log = _synthetic_log("u0", 1, 120, seed=5, extra={"all32": (0.3, 0.3)})
    log.shadow = [0.3 * b + np.random.default_rng(6).normal(0, 0.3, b.shape) for b in log.body]   # body only
    res = log.certify("heldout", 1000)
    assert set(res["shadows"]) == {"central11", "all32"}
    flags = res["flags"]["y"]["shadows_read_screen"]
    assert flags == {"central11": False, "all32": True} and res["verdict"] != "TEST_INSENSITIVE", res["reason"]
    assert set(res["holm"]) >= {"y:central11:screen>null", "y:all32:screen>null"}
    with pytest.raises(ValueError, match="shadows"):
        CertificationLog.merge([log, _synthetic_log("u1", 1, 20, seed=7)])


# ---------------------------------------------------------------- review round 3
def test_each_day_reports_the_deployed_decoders_leak_transfer_and_the_report_check(baseline):
    """The screen-leak sweep shows the shadows' sensitivity; whether a display leak reaches the DEPLOYED
    decoder is a paired probe of that decoder (identical noise at leak 0 and 1), reported per day."""
    day = baseline[0]["per_user"][0]["days"][0]
    lt = day["leak_transfer"]
    assert lt["probe_leak"] == 1.0 and lt["n_windows"] > 0 and set(lt["gain"]) == {"x", "y"}
    assert np.isfinite(lt["gain"]["y"]) and (day["x_earned"] or lt["gain"]["x"] == 0.0)
    assert day["report_check"]["heldout"]["trials"] > 0 and day["report_check"]["heldout"]["dropped_by_report"] == 0
    assert day["margin_se"]["y"] > 0


def test_replays_do_not_consume_goal_quadrants_and_stay_on_the_block_geometry():
    """Review: a C2c replay popped a goal quadrant before the replay decision, so the rotated trials lost
    their balanced runs of four (|corr(sign dx, sign dy)| up to 0.5, the refusal threshold), and replays
    inside held-out blocks showed S-block (Fitts) trajectories.  Now the replay decision comes first and a
    held-out replay draws from the diagonal pool."""
    cfg = champion_config()
    sim = dataclasses.replace(BASE, identity_fraction=0.5)
    person = BodySubject(cfg.hardware, seed=3)
    person.set_day(1)
    path = np.linspace([0.0, 0.0], diagonal_targets(0.6)[0], 25)
    pools = {"diagonal": [Trajectory(diagonal_targets(0.6), 0, path)], "sighted": []}
    runner = ChampionRunner(person, cfg, sim, 1e9, pools)
    runner.sighted_task = dataclasses.replace(runner.sighted_task, trial_timeout_s=0.5)   # short trials
    block = next(b for b in session_plan_champion(3, 5) if b.kind == "heldout_mapping" and b.variant == "C2c")
    block = dataclasses.replace(block, duration_s=150.0, replay_fraction=0.25)
    import tempfile
    from neural_cursor.data_engine import SessionStore
    with tempfile.TemporaryDirectory() as root:
        run = runner.run_block(block, root, "c2c", 1, 7, {})
        events = SessionStore.read_events(run.run_dir + "/events.csv")
    starts = [p for _, kind, p in events if kind == "trial_start"]
    replays = [p for p in starts if p.get("replay_goal") is not None]
    rotated = [p for p in starts if p.get("replay_goal") is None]
    assert replays and len(rotated) >= 8
    diag = {tuple(np.round(t, 6)) for t in diagonal_targets(0.6)}
    assert all(tuple(np.round(p["replay_goal"], 6)) in diag for p in replays)       # diagonal pool
    signs = [tuple(np.sign(p["goal"])) for p in rotated]
    for i in range(0, len(signs) - len(signs) % 4, 4):                          # balanced runs of four
        assert len(set(signs[i:i + 4])) == 4, signs
    reports = [p for _, kind, p in events if kind == "body_report"]
    assert {r["trial"] for r in reports} >= {p["trial"] for p in replays}       # every replay is reported


def test_certification_is_null_calibrated_against_the_twin():
    """certify_user pairs every log with the same log of the screen_leak 0 twin; at screen leak 0 a run is
    its own null, so no shadow can read the screen beyond it (TEST_INSENSITIVE, never a nonspecific
    pass); without a null nothing is CERTIFIED."""
    log = _synthetic_log("u0", 1, 120, seed=5, extra={"all32": (0.3, 0.3)})
    res = {"days": [{"day": 1, "shadow_error": None}]}
    own = certify_user(dict(res, days=[dict(res["days"][0])]), {1: {"heldout": log, "replay": CertificationLog()}},
                       None, 1000)
    assert own["days"][0]["certification"]["heldout"]["flags"]["y"]["shadow_null_calibrated"] is False
    logs = {1: {"heldout": log, "replay": CertificationLog()}}
    selfnull = certify_user(dict(res, days=[dict(res["days"][0])]), logs, logs, 1000)
    c = selfnull["days"][0]["certification"]["heldout"]
    assert c["verdict"] == "TEST_INSENSITIVE" and not c["flags"]["y"]["shadow_reads_screen"], c["reason"]
    null = _synthetic_log("u0", 1, 120, seed=6, extra={"all32": (0.3, 0.0)})
    null.shadow = [0.3 * b + np.random.default_rng(7).normal(0, 0.3, b.shape) for b in null.body]
    twin = certify_user(dict(res, days=[dict(res["days"][0])]), logs,
                        {1: {"heldout": null, "replay": CertificationLog()}}, 1000)
    c = twin["days"][0]["certification"]["heldout"]
    assert c["flags"]["y"]["shadow_reads_screen"] and c["shadows_null"]["central11"]["y"]["rel_screen"] < 0.3


@pytest.mark.parametrize("args", [["--covert-leak", "0", "-1"], ["--dry-penalty", "0", "1.0"],
                                  ["--dry-penalty", "0", "0"], ["--covert-leak", "1", "1"]])
def test_cli_validates_every_swept_condition_before_running(args, capsys, monkeypatch):
    """Review: only the first --covert-leak and no --dry-penalty value was checked, so an hour-long sweep
    died on its last condition with a traceback and wrote nothing; duplicates silently overwrote."""
    monkeypatch.setattr(sc, "simulate", lambda *a, **k: pytest.fail("ran a condition before validating all"))
    with pytest.raises(SystemExit) as exc:
        main(["--days", "1", *args])
    assert exc.value.code == 2 and "error" in capsys.readouterr().err


def test_a_failing_condition_is_recorded_and_the_sweep_goes_on(monkeypatch):
    calls = []

    def fake(cond, cfg, workers):
        calls.append(cond.dry_penalty)
        if cond.dry_penalty > 0:
            raise ValueError("too few held-out trials")
        return {"config": dataclasses.asdict(cond), "ok": True}

    monkeypatch.setattr(sc, "simulate", fake)
    res = sweep(BASE, [0.0, 0.2])
    assert calls == [0.0, 0.2] and res["conditions"]["0"]["ok"]
    assert res["conditions"]["0.2"]["failed"] == "too few held-out trials"
    with pytest.raises(ValueError, match="repeat"):
        sweep(BASE, [0.0, 0.0])
