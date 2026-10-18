"""Certification under held-out mappings and pixel replay, on synthetic decoder outputs.

The key property: under rot±90 body intent and the on-screen direction are orthogonal, so a regression
of the output on both recovers which one a decoder follows.  A body reader is CERTIFIED when the shadow
positive control reads the screen; a screen reader is SCREEN_READING; a shadow that shows no screen
reading makes the test TEST_INSENSITIVE; replay separates private intent from the replayed direction.
Beyond that, each review finding has its test: cross-axis body coupling must not read as screen
reading (rotation-balanced units + the cross regressor), units with different gains and mapping mixes
are fitted separately and combined (with a per-unit veto), every axis must pass, the trained margin
binds the body criterion, and windows without a test-only source are refused.
"""
import json

import numpy as np
import pytest

from neural_cursor.certify import (MAX_BLOCK_CORR, MIN_BOOT, MIN_CLUSTERS, MIN_REPLAY_CLUSTERS, certify,
                                   certify_mapping, combine_units, confirmed_by_report, holm, replay_test)
from neural_cursor.mapping import (BODY_VECTORS, DIAGONAL_DIRECTIONS, HELDOUT_MAPPINGS, MAPPING_NAMES, Mapping,
                                   certification_regressors, diagonal_targets, quadrant_schedule, unit_directions)

N_TRIALS = 360          # held-out-mapping trials (two sessions): per-axis CIs must resolve ±10 % of GAIN
WIN_PER_TRIAL = 8       # overlapping 1-s windows per trial
GAIN = 0.6              # the decoder's trained margin in these simulations
TRIAL_SD = 0.15         # per-trial output offset (strategy, slow EEG state): what makes windows dependent
WINDOW_SD = 0.3         # per-window output noise


def _heldout_block(seed, mappings=("rot+90", "rot-90")):
    """Windows of closed-loop held-out-mapping trials: random targets, a cursor that drifts toward
    them, one mapping per trial.  Returns (body_intent, screen_dir_in_body, provenance)."""
    rng = np.random.default_rng(seed)
    body, screen, prov = [], [], []
    for t in range(N_TRIALS):
        m = Mapping(mappings[t % len(mappings)])
        a = rng.uniform(0, 2 * np.pi)
        target = 0.8 * np.array([np.cos(a), np.sin(a)])
        cursors = np.linspace(0.0, 0.6, WIN_PER_TRIAL)[:, None] * target + rng.normal(0, 0.05, (WIN_PER_TRIAL, 2))
        b, s = certification_regressors(np.tile(target, (WIN_PER_TRIAL, 1)), cursors, m)
        body.append(b)
        screen.append(s)
        prov += [{"session": f"s{t % 2}", "trial": t, "source": "H", "mapping": m.name}
                 for _ in range(WIN_PER_TRIAL)]
    return np.concatenate(body), np.concatenate(screen), prov


def _outputs(body, screen, prov, w_body, w_screen, seed, cross=(0.0, 0.0)):
    """Decoder output = w_body·body + w_screen·screen + cross-axis coupling + trial offset + window
    noise, clipped to [−1, 1].  ``cross`` = (x output per unit body_y, y output per unit body_x): an
    x model that reacts to feet/hands imagery, or a y model that reacts to one-hand imagery."""
    rng = np.random.default_rng(seed)
    trial = np.array([p["trial"] for p in prov])
    offset = rng.normal(0, TRIAL_SD, (trial.max() + 1, 2))[trial]
    out = w_body * body + w_screen * screen + offset + rng.normal(0, WINDOW_SD, body.shape)
    out += np.asarray(cross) * body[:, ::-1]
    return np.clip(out, -1.0, 1.0)


def _near_zero(ci, tol=0.08):
    """A CI inside ±tol: the β is bounded near 0.  (Asserting "contains 0" would fail for 5 % of seeds
    by construction; that size is checked separately in test_cluster_inference_keeps_its_size.)"""
    return -tol < ci[0] and ci[1] < tol


@pytest.fixture(scope="module")
def block():
    return _heldout_block(seed=1)


@pytest.fixture(scope="module")
def results(block):
    body, screen, prov = block
    fit = lambda wb, ws, seed: certify_mapping(_outputs(body, screen, prov, wb, ws, seed), body, screen, prov,
                                               margin=GAIN)
    return {"body_reader": fit(GAIN, 0.0, 2), "screen_reader": fit(0.0, GAIN, 3),
            "shadow_mixed": fit(0.3, 0.3, 4), "shadow_body_only": fit(GAIN, 0.0, 5), "noise": fit(0.0, 0.0, 6),
            "shadow_null": fit(0.3, 0.0, 44)}


def _null_of(result_kw, block_args, seed=44):
    """The no-display null of a shadow: the same shadow model (body gain 0.3) with no screen pathway, on
    the same geometry, through the same regression (what the simulation's screen_leak 0 twin gives)."""
    body, screen, prov = block_args
    return certify_mapping(_outputs(body, screen, prov, 0.3, 0.0, seed), body, screen, prov, **result_kw)


# ---------------------------------------------------------------- mapping geometry

def test_mappings_are_orthogonal_and_invertible():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 2))
    for name in MAPPING_NAMES:
        m = Mapping(name)
        assert np.allclose(m.M @ m.M.T, np.eye(2))
        assert np.allclose(m.screen_to_body(m.body_to_screen(v)), v)
        assert m.body_to_screen(v[0]).shape == (2,)
        assert m.heldout == (name in HELDOUT_MAPPINGS)
    assert "identity" not in HELDOUT_MAPPINGS
    with pytest.raises(ValueError):
        Mapping("rot+180")


def test_rotation_moves_feet_horizontally_and_mirror_flips_one_axis():
    feet, right = np.array(BODY_VECTORS["FEET"]), np.array(BODY_VECTORS["RIGHT"])
    assert np.allclose(Mapping("identity").body_to_screen(feet), [0, 1])
    assert np.allclose(Mapping("rot+90").body_to_screen(feet), [-1, 0])
    assert np.allclose(Mapping("rot-90").body_to_screen(feet), [1, 0])
    assert np.allclose(Mapping("mirror_x").body_to_screen(right), [-1, 0])
    assert np.allclose(Mapping("mirror_y").body_to_screen(feet), [0, -1])


def test_regressors_are_orthogonal_under_rotation_and_equal_under_identity():
    rng = np.random.default_rng(0)
    targets, cursors = rng.uniform(-1, 1, (40, 2)), rng.uniform(-1, 1, (40, 2))
    for name in ("rot+90", "rot-90"):
        b, s = certification_regressors(targets, cursors, Mapping(name))
        assert np.allclose(np.einsum("ij,ij->i", b, s), 0.0)
        assert np.allclose(np.linalg.norm(b, axis=1), 1.0)
        # the body intent, mapped to the screen, points at the target
        assert np.allclose(Mapping(name).body_to_screen(b), unit_directions(targets, cursors))
    b, s = certification_regressors(targets, cursors, Mapping("identity"))
    assert np.allclose(b, s)
    assert np.allclose(unit_directions([[0.2, 0.2]], [[0.2, 0.2]]), 0.0)   # on target: no direction


# ---------------------------------------------------------------- β estimates

def test_body_reader_has_body_beta_and_no_screen_beta(results):
    for key in ("x", "y", "pooled"):
        e = results["body_reader"][key]
        assert e["identifiable"]
        assert e["beta_body"] == pytest.approx(GAIN, abs=0.1)     # clipping to [−1, 1] shrinks it a little
        assert e["ci_body"][0] > 0.4
        assert _near_zero(e["ci_screen"])
        assert e["df"] == N_TRIALS - (5 if key == "pooled" else 4)      # t(G − k) reference


def test_screen_reader_has_screen_beta_and_no_body_beta(results, block):
    e = results["screen_reader"]["pooled"]
    assert e["ci_screen"][0] > 0.4
    assert _near_zero(e["ci_body"])
    assert e["rel_screen"] == pytest.approx(1.0, abs=0.1)     # supplied margin = GAIN
    body, screen, prov = block
    self_scaled = certify_mapping(_outputs(body, screen, prov, 0.0, GAIN, 3), body, screen, prov)["pooled"]
    assert self_scaled["rel_screen"] == pytest.approx(1.0, abs=0.1)   # fallback: β_body + β_screen


def test_supplied_margin_scales_relative_betas(block):
    body, screen, prov = block
    out = _outputs(body, screen, prov, GAIN, 0.0, 2)
    r = certify_mapping(out, body, screen, prov, margin=2 * GAIN)
    assert r["pooled"]["scale"] == 2 * GAIN
    assert r["pooled"]["rel_body"] == pytest.approx(0.5, abs=0.06)
    per_axis = certify_mapping(out, body, screen, prov, margin={"x": GAIN, "y": 2 * GAIN})
    assert per_axis["x"]["scale"] == GAIN and per_axis["y"]["scale"] == 2 * GAIN
    assert per_axis["pooled"]["scale"] == pytest.approx(1.5 * GAIN)


def test_trained_margin_makes_the_body_criterion_bind(block, results):
    """A decoder whose body gain collapses under the held-out mapping (0.10 against a trained margin of
    0.40) passes the body criterion on the self-estimated scale (β_body + β_screen ≈ β_body), and is
    INCONCLUSIVE once the trained margin is supplied: the self-scale makes the 70 % criterion vacuous."""
    body, screen, prov = block
    out = _outputs(body, screen, prov, 0.10, 0.0, 30)
    strong_shadow = lambda **kw: certify_mapping(_outputs(body, screen, prov, 0.3, 0.3, 4), body, screen, prov, **kw)
    collapsed = certify(certify_mapping(out, body, screen, prov, margin=0.40), strong_shadow(margin=0.40))
    assert collapsed["verdict"] == "INCONCLUSIVE", collapsed["reason"]
    assert not collapsed["flags"]["deployed_body_sufficient"]


def test_inference_clusters_trials_not_windows(block):
    """Repeating every window 4× inside its trial adds no information; a window-level interval would
    shrink by half, the trial-clustered (CR2) one must not."""
    body, screen, prov = block
    out = _outputs(body, screen, prov, GAIN, 0.0, 7)
    width = lambda r: np.diff(r["pooled"]["ci_body"])[0]
    once = certify_mapping(out, body, screen, prov)
    rep = lambda a: np.repeat(a, 4, axis=0)
    four = certify_mapping(rep(out), rep(body), rep(screen), [p for p in prov for _ in range(4)])
    assert four["pooled"]["beta_body"] == pytest.approx(once["pooled"]["beta_body"])
    assert width(four) == pytest.approx(width(once), rel=0.15)


def test_cluster_inference_keeps_its_size():
    """With no screen reading, the 95 % CI of β_screen excludes 0 in about 5 % of experiments (not the
    20–40 % a window-level interval gives with these trial offsets)."""
    n_rep = 40
    excl = 0
    for k in range(n_rep):
        body, screen, prov = _heldout_block(seed=1000 + k)
        out = _outputs(body, screen, prov, GAIN, 0.0, 2000 + k)
        lo, hi = certify_mapping(out, body, screen, prov, n_boot=MIN_BOOT)["pooled"]["ci_screen"]
        excl += not (lo < 0 < hi)
    assert excl / n_rep <= 0.125


def test_results_are_deterministic(block):
    """The single-unit fit is analytic (CR2, t(G − k)): the same windows give the same interval, and the
    legacy ``seed`` / ``n_boot`` arguments change nothing."""
    body, screen, prov = block
    out = _outputs(body, screen, prov, GAIN, 0.0, 8)
    a = certify_mapping(out, body, screen, prov)
    b = certify_mapping(out, body, screen, prov, seed=1, n_boot=5000)
    assert a["pooled"]["ci_body"] == b["pooled"]["ci_body"]


def test_mirror_and_single_rotation_identify_nothing_per_axis():
    """A mirror leaves body = ±screen on each axis; a single rotation leaves screen = ±(other axis's
    body).  Neither identifies β_screen per axis, so the default (every axis) verdict is
    TEST_INSENSITIVE; only the reported "pooled" fit (one β set over both axes) separates a mirror."""
    for mappings in (("mirror_x",), ("rot+90",)):
        body, screen, prov = _heldout_block(seed=9, mappings=mappings)
        out = _outputs(body, screen, prov, GAIN, 0.0, 10)
        r = certify_mapping(out, body, screen, prov, margin=GAIN)
        assert not r["x"]["identifiable"] and not r["y"]["identifiable"], mappings
        shadow = certify_mapping(_outputs(body, screen, prov, 0.0, GAIN, 11), body, screen, prov, margin=GAIN)
        assert certify(r, shadow)["verdict"] == "TEST_INSENSITIVE"
    body, screen, prov = _heldout_block(seed=9, mappings=("mirror_x",))
    r = certify_mapping(_outputs(body, screen, prov, GAIN, 0.0, 10), body, screen, prov, cross_axis=False)
    assert r["pooled"]["identifiable"] and r["pooled"]["ci_body"][0] > 0.4


def test_cross_axis_body_coupling_is_not_screen_reading():
    """Under rot+90 screen_y = body_x and screen_x = −body_y, so an x model that reacts to feet/hands
    imagery (and a y model that reacts to one-hand imagery) look like screen readers when only body
    and screen are regressed.  With both rotations in the unit and the cross regressor, the coupling
    goes to β_cross and β_screen stays bounded near 0 (the screen_leak = 0 null of the review)."""
    body, screen, prov = _heldout_block(seed=31)                                 # rot+90 / rot−90
    coupling = (-0.15, 0.10)
    out = _outputs(body, screen, prov, GAIN, 0.0, 32, cross=coupling)
    shadow = certify_mapping(_outputs(body, screen, prov, 0.3, 0.3, 33), body, screen, prov, margin=GAIN)
    r = certify_mapping(out, body, screen, prov, margin=GAIN)
    for k, a in enumerate(("x", "y")):
        assert r[a]["beta_cross"] == pytest.approx(coupling[k], abs=0.05)
        assert _near_zero(r[a]["ci_screen"], tol=0.05)
    assert certify(r, shadow, shadow_null=_null_of(dict(margin=GAIN), (body, screen, prov)))["verdict"] == "CERTIFIED"
    # The old two-regressor fit on one rotation calls the same decoder a screen reader.
    one, one_s, one_p = _heldout_block(seed=31, mappings=("rot+90",))
    old = certify_mapping(_outputs(one, one_s, one_p, GAIN, 0.0, 32, cross=coupling), one, one_s, one_p,
                          margin=GAIN, cross_axis=False)
    old_shadow = certify_mapping(_outputs(one, one_s, one_p, 0.3, 0.3, 33), one, one_s, one_p, margin=GAIN,
                                 cross_axis=False)
    assert certify(old, old_shadow)["verdict"] == "SCREEN_READING"


def test_screen_reading_on_one_axis_is_not_diluted_by_a_clean_axis(block):
    """x reads the screen (β_screen = 0.06 = 10 % of the margin… plus noise), y is clean: the pooled β
    shared over both axes halves it; the default verdict reads every axis and returns the worst."""
    body, screen, prov = block
    out = _outputs(body, screen, prov, GAIN, 0.0, 34)
    out[:, 0] += 0.12 * screen[:, 0]
    r = certify_mapping(out, body, screen, prov, margin=GAIN)
    shadow = certify_mapping(_outputs(body, screen, prov, 0.3, 0.3, 4), body, screen, prov, margin=GAIN)
    v = certify(r, shadow, shadow_null=_null_of(dict(margin=GAIN), block))
    assert v["verdict"] == "SCREEN_READING" and v["per_axis"]["x"]["verdict"] == "SCREEN_READING"
    assert v["per_axis"]["y"]["verdict"] == "CERTIFIED"
    # what the shared-β pooled entry does to it: about half the x estimate
    assert r["pooled"]["beta_screen"] < 0.7 * r["x"]["beta_screen"]


def _unit(seed, gain, w_screen, mappings, session, n_trials=200):
    rng = np.random.default_rng(seed)
    body, screen, prov = [], [], []
    for t in range(n_trials):
        m = Mapping(mappings[t % len(mappings)])
        a = rng.uniform(0, 2 * np.pi)
        u = np.array([[np.cos(a), np.sin(a)]] * WIN_PER_TRIAL)
        b, s = certification_regressors(u, np.zeros_like(u), m)                   # open loop: cursor at centre
        body.append(b)
        screen.append(s)
        prov += [{"session": session, "trial": t, "source": "H"} for _ in range(WIN_PER_TRIAL)]
    body, screen = np.concatenate(body), np.concatenate(screen)
    return _outputs(body, screen, prov, gain, w_screen, seed + 1), body, screen, prov


def test_units_with_different_gains_and_mappings_are_fitted_separately_and_vetoed():
    """The review's pooling case: a strong unit (gain 0.45, half its trials mirror_y, where body_y =
    −screen_y) and a weak unit (gain 0.10, rotations only), both with true β_screen = +0.04.  One
    pooled OLS mixes the gain difference into β_screen; per-unit fits find the strong unit's screen
    reading, and its own SCREEN_READING vetoes the combined verdict."""
    n = 400
    a = _unit(40, 0.45, 0.04, ("rot+90", "mirror_y", "rot-90", "mirror_y"), "A", n)
    b = _unit(42, 0.10, 0.04, ("rot+90", "rot-90"), "B", n)
    sa = _unit(40, 0.30, 0.30, ("rot+90", "mirror_y", "rot-90", "mirror_y"), "A", n)[0]
    sb = _unit(42, 0.30, 0.30, ("rot+90", "rot-90"), "B", n)[0]
    kw = dict(axes=("y",))
    one_fit = certify_mapping(*[np.concatenate([a[i], b[i]]) for i in range(3)], a[3] + b[3], cross_axis=False, **kw)
    assert one_fit["y"]["ci_screen"][0] < 0 < one_fit["y"]["ci_screen"][1]       # the pooled OLS misses it
    ra, rb = certify_mapping(*a, margin=0.45, **kw), certify_mapping(*b, margin=0.10, **kw)
    sha, shb = certify_mapping(sa, *a[1:], margin=0.45, **kw), certify_mapping(sb, *b[1:], margin=0.10, **kw)
    assert ra["y"]["ci_screen"][0] > 0                                           # unit A reads the screen
    dep = combine_units({"A": ra, "B": rb}, combination="random")
    sh = combine_units({"A": sha, "B": shb}, combination="random")
    v = certify(dep, sh)
    assert v["verdict"] == "SCREEN_READING" and v["unit_verdicts"]["A"] == "SCREEN_READING", v


def test_combination_pools_x_over_earners_and_random_effects_widen_the_ci():
    a = _unit(50, 0.4, 0.0, ("rot+90", "rot-90"), "A")
    b = _unit(52, 0.4, 0.0, ("rot+90", "rot-90"), "B")
    c = _unit(54, 0.4, 0.15, ("rot+90", "rot-90"), "C")          # a heterogeneous unit
    ra = certify_mapping(*a, margin=0.4)                          # x earner
    rb = certify_mapping(*b, margin=0.4, axes=("y",))             # y only
    rc = certify_mapping(*c, margin=0.4)
    fixed = combine_units({"A": ra, "B": rb, "C": rc}, combination="fixed")
    rand = combine_units({"A": ra, "B": rb, "C": rc}, combination="random")
    assert fixed["axes"] == ("x", "y") and fixed["x"]["n_units"] == 2 and fixed["y"]["n_units"] == 3
    assert rand["y"]["tau2_screen"] > 0
    width = lambda r: np.diff(r["y"]["ci_screen"])[0]
    assert width(rand) > width(fixed)
    with pytest.raises(ValueError, match="combination"):
        combine_units({"A": ra}, combination="mixed")


# ---------------------------------------------------------------- verdicts

def test_body_reader_is_certified_when_shadow_reads_screen(results):
    v = certify(results["body_reader"], results["shadow_mixed"], shadow_null=results["shadow_null"])
    assert v["verdict"] == "CERTIFIED", v["reason"]
    json.dumps({"flags": v["flags"], "per_axis": v["per_axis"]})     # reports are strict JSON
    assert v["flags"]["shadow_reads_screen"] and v["flags"]["deployed_screen_bounded"]
    assert v["flags"]["shadow_null_calibrated"]
    assert certify(results["body_reader"], results["shadow_mixed"], axis="y",
                   shadow_null=results["shadow_null"])["verdict"] == "CERTIFIED"


def test_screen_reading_deployed_decoder_is_flagged(results):
    v = certify(results["screen_reader"], results["shadow_mixed"])
    assert v["verdict"] == "SCREEN_READING"
    # a positive finding needs no positive control
    assert certify(results["screen_reader"], results["shadow_body_only"])["verdict"] == "SCREEN_READING"


def test_shadow_without_screen_reading_makes_test_insensitive(results):
    v = certify(results["body_reader"], results["shadow_body_only"])
    assert v["verdict"] == "TEST_INSENSITIVE"
    assert not v["flags"]["shadow_reads_screen"]
    assert certify(results["body_reader"], results["noise"])["verdict"] == "TEST_INSENSITIVE"


def test_decoder_without_signal_has_no_body_signal(results):
    assert certify(results["noise"], results["shadow_mixed"])["verdict"] == "NO_BODY_SIGNAL"


def test_mixed_reader_and_weak_body_gain_are_inconclusive_or_screen_reading(block, results):
    body, screen, prov = block
    leaky = certify_mapping(_outputs(body, screen, prov, GAIN, 0.12, 12), body, screen, prov)
    assert certify(leaky, results["shadow_mixed"])["verdict"] in ("SCREEN_READING", "INCONCLUSIVE")
    # body gain collapsed to 55 % of the trained margin under the mapping: the declared inconclusive zone
    # (every axis is now tested at α = 0.05, so ~5 % of seeds give a chance SCREEN_READING; this one
    # does not, and the screen test's size is checked in test_cluster_inference_keeps_its_size)
    weak = certify_mapping(_outputs(body, screen, prov, 0.55 * GAIN, 0.0, 15), body, screen, prov, margin=GAIN)
    assert certify(weak, results["shadow_mixed"])["verdict"] == "INCONCLUSIVE"


def test_y_only_decoder_is_certified_on_y(block, results):
    """x not earned: the decoder's x output is silent, so only y is tested (y is the primary axis)."""
    body, screen, prov = block
    out = _outputs(body, screen, prov, GAIN, 0.0, 24)
    out[:, 0] = 0.0
    r = certify_mapping(out, body, screen, prov, axes=("y",), margin=GAIN)
    shadow = certify_mapping(_outputs(body, screen, prov, 0.3, 0.3, 4), body, screen, prov, axes=("y",),
                             margin=GAIN)
    assert "x" not in r and r["pooled"]["beta_body"] == r["y"]["beta_body"]
    assert certify(r, shadow, shadow_null=_null_of(dict(axes=("y",), margin=GAIN), block))["verdict"] == "CERTIFIED"
    with pytest.raises(ValueError, match="not tested"):
        certify(r, shadow, axis="x")


def test_identity_block_cannot_certify(block):
    body, screen, prov = _heldout_block(seed=14, mappings=("identity",))
    out = _outputs(body, screen, prov, GAIN, 0.0, 15)
    r = certify_mapping(out, body, screen, prov)
    assert not r["pooled"]["identifiable"]
    assert certify(r, r)["verdict"] == "TEST_INSENSITIVE"


# ---------------------------------------------------------------- pixel replay

def _replay_block(seed, n_trials=N_TRIALS):
    """Replay trials: the person privately holds goal B, the screen replays a path toward goal A."""
    rng = np.random.default_rng(seed)
    a_b = rng.uniform(0, 2 * np.pi, n_trials)
    a_a = a_b + rng.uniform(np.pi / 3, 5 * np.pi / 3, n_trials)          # A differs from B
    unit = lambda a: np.stack([np.cos(a), np.sin(a)], axis=1)
    private = np.repeat(unit(a_b), WIN_PER_TRIAL, axis=0)
    replayed = Mapping("identity").screen_to_body(np.repeat(unit(a_a), WIN_PER_TRIAL, axis=0))
    prov = [{"session": "s0", "trial": t, "source": "R"} for t in range(n_trials) for _ in range(WIN_PER_TRIAL)]
    return private, replayed, prov


def test_replay_separates_private_intent_from_replayed_direction():
    private, replayed, prov = _replay_block(seed=16)
    body_reader = replay_test(_outputs(private, replayed, prov, GAIN, 0.0, 17), private, replayed, prov)
    cursor_reader = replay_test(_outputs(private, replayed, prov, 0.0, GAIN, 18), private, replayed, prov)
    shadow = replay_test(_outputs(private, replayed, prov, 0.3, 0.3, 19), private, replayed, prov)
    assert body_reader["regressors"]["body"] == "private_intent"
    assert body_reader["regressors"]["screen"] == "replayed_dir"
    e = body_reader["pooled"]
    assert e["ci_body"][0] > 0.4 and _near_zero(e["ci_screen"])
    e = cursor_reader["pooled"]
    assert e["ci_screen"][0] > 0.4 and _near_zero(e["ci_body"])
    # without an external margin the body criterion is vacuous: never CERTIFIED (test_external_margin_…)
    assert certify(body_reader, shadow)["verdict"] == "INCONCLUSIVE"
    external = replay_test(_outputs(private, replayed, prov, GAIN, 0.0, 17), private, replayed, prov, margin=GAIN)
    null = replay_test(_outputs(private, replayed, prov, 0.3, 0.0, 44), private, replayed, prov)
    assert certify(external, shadow, shadow_null=null)["verdict"] == "CERTIFIED"
    assert certify(cursor_reader, shadow)["verdict"] == "SCREEN_READING"


def test_replay_needs_enough_trials():
    """~10 replay trials a day undercover with a percentile cluster bootstrap: replay needs 30."""
    private, replayed, prov = _replay_block(seed=16, n_trials=MIN_REPLAY_CLUSTERS - 1)
    out = _outputs(private, replayed, prov, GAIN, 0.0, 17)
    with pytest.raises(ValueError, match="trials"):
        replay_test(out, private, replayed, prov)
    private, replayed, prov = _replay_block(seed=16, n_trials=MIN_REPLAY_CLUSTERS)
    assert replay_test(_outputs(private, replayed, prov, GAIN, 0.0, 17), private, replayed, prov)["pooled"]["identifiable"]


# ---------------------------------------------------------------- input contract

def test_refuses_training_blocks_and_mixed_tests(block, results):
    body, screen, prov = block
    out = _outputs(body, screen, prov, GAIN, 0.0, 20)
    for src in ("L1", "L2b", "S", "R", None, "missing"):
        bad = [dict(p, source=src) if i == 0 else p for i, p in enumerate(prov)]
        if src == "missing":
            del bad[0]["source"]                       # stripped provenance proves nothing about the window
        with pytest.raises(ValueError, match="source"):
            certify_mapping(out, body, screen, bad)
    with pytest.raises(ValueError, match="source"):
        replay_test(out, body, screen, prov)                            # "H" windows are not replay trials
    private, replayed, rprov = _replay_block(seed=21)
    replay = replay_test(_outputs(private, replayed, rprov, GAIN, 0.0, 22), private, replayed, rprov)
    with pytest.raises(ValueError, match="different tests"):
        certify(results["body_reader"], replay)
    # same number of windows and trials, different windows: the content hash tells them apart
    other_body, other_screen, other_prov = _heldout_block(seed=2)
    other = certify_mapping(_outputs(other_body, other_screen, other_prov, 0.3, 0.3, 4), other_body, other_screen,
                            other_prov, margin=GAIN)
    with pytest.raises(ValueError, match="different windows"):
        certify(results["body_reader"], other)


def test_refuses_too_few_draws_trials_or_bad_shapes(block):
    body, screen, prov = block
    out = _outputs(body, screen, prov, GAIN, 0.0, 23)
    few = WIN_PER_TRIAL * (MIN_CLUSTERS - 1)
    with pytest.raises(ValueError, match="trials"):
        certify_mapping(out[:few], body[:few], screen[:few], prov[:few])
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        certify_mapping(out[:, :1], body, screen, prov)
    with pytest.raises(ValueError, match="axes"):
        certify_mapping(out, body, screen, prov, axes=("z",))
    with pytest.raises(ValueError, match="margin"):
        certify_mapping(out, body, screen, prov, margin=0.0)
    with pytest.raises(ValueError, match="margin"):
        certify_mapping(out, body, screen, prov, margin={"y": GAIN})              # x tested, no x margin
    bad = out.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        certify_mapping(bad, body, screen, prov)


# ---------------------------------------------------------------- two positive controls, Holm, external margin
def test_either_shadow_makes_the_test_sensitive_and_both_are_reported(results):
    """The central-11 shadow can learn motor features instead of the screen (on y the up/down leak barely
    reaches the central strip); the all-32 shadow sees the posterior display signal.  The test is
    sensitive if EITHER shadow shows β_screen > 0; both are reported; neither → TEST_INSENSITIVE."""
    nulls = {"central11": results["shadow_null"], "all32": results["shadow_null"]}
    v = certify(results["body_reader"], {"central11": results["shadow_body_only"], "all32": results["shadow_mixed"]},
                shadow_null=nulls)
    assert v["verdict"] == "CERTIFIED", v["reason"]
    assert v["flags"]["shadows_read_screen"] == {"central11": False, "all32": True}
    assert set(v["shadows"]) == {"central11", "all32"} and v["shadow"] is v["shadows"]["central11"]
    assert "all32" in v["reason"]
    json.dumps({"flags": v["flags"], "per_axis": v["per_axis"]})
    none = certify(results["body_reader"], {"central11": results["shadow_body_only"], "all32": results["noise"]})
    assert none["verdict"] == "TEST_INSENSITIVE" and not none["flags"]["shadow_reads_screen"]
    with pytest.raises(ValueError, match="shadow"):
        certify(results["body_reader"], {})


def test_holm_step_down():
    out = holm({"a": 0.01, "b": 0.02, "c": 0.03, "d": float("nan")}, 0.05)
    assert [out[k]["rejected"] for k in "abc"] == [True, True, True] and not out["d"]["rejected"]
    assert [out[k]["p_holm"] for k in "abc"] == pytest.approx([0.03, 0.04, 0.04])     # 3p, 2p, p, monotone
    out = holm({"a": 0.02, "b": 0.02}, 0.025)                  # alone, each would pass at 0.025
    assert not out["a"]["rejected"] and not out["b"]["rejected"]


def test_the_verdict_family_is_holm_corrected(results):
    """Every decision behind a verdict (per axis: body > 0, screen below 10 %, each shadow's screen > 0)
    is one family at α/2 one-sided; the flags are its Holm decisions."""
    v = certify(results["body_reader"], {"central11": results["shadow_body_only"], "all32": results["shadow_mixed"]})
    tests = v["holm"]["tests"]
    assert set(tests) == {f"{a}:{t}" for a in ("x", "y") for t in (
        "deployed:body>0", "deployed:body>0.5margin", "deployed:screen<tol", "deployed:screen>-tol",
        "central11:screen>null", "all32:screen>null")}
    assert v["holm"]["alpha_one_sided"] == pytest.approx(0.025)
    for a in ("x", "y"):
        f = v["per_axis"][a]["flags"]
        assert f["deployed_body_significant"] == tests[f"{a}:deployed:body>0"]["rejected"]
        assert f["deployed_body_lower_bounded"] == tests[f"{a}:deployed:body>0.5margin"]["rejected"]
        assert f["deployed_screen_bounded"] == (tests[f"{a}:deployed:screen<tol"]["rejected"]
                                                and tests[f"{a}:deployed:screen>-tol"]["rejected"])
        assert f["shadows_read_screen"]["all32"] == tests[f"{a}:all32:screen>null"]["rejected"]
    ps = sorted(t["p"] for t in tests.values())
    assert all(t["p_holm"] >= t["p"] for t in tests.values())
    assert min(t["p_holm"] for t in tests.values()) == pytest.approx(min(1.0, len(ps) * ps[0]))


def test_only_an_external_margin_can_certify(block, results):
    """The self-estimated scale (β_body + β_screen from the same fit) makes the body criterion vacuous:
    a result without an external margin is never CERTIFIED, the same windows with one are."""
    body, screen, prov = block
    out = _outputs(body, screen, prov, GAIN, 0.0, 2)
    self_scaled = certify_mapping(out, body, screen, prov)
    assert self_scaled["margin_source"] == "self" and results["body_reader"]["margin_source"] == "external"
    v = certify(self_scaled, results["shadow_mixed"])
    assert v["verdict"] == "INCONCLUSIVE" and "external margin" in v["reason"]
    assert not v["flags"]["margin_external"]
    assert certify(results["body_reader"], results["shadow_mixed"],
                   shadow_null=results["shadow_null"])["verdict"] == "CERTIFIED"
    combined = combine_units({"a": self_scaled, "b": results["body_reader"]})
    assert combined["margin_source"] == "self"


# ---------------------------------------------------------------- held-out geometry and confounding
def _diagonal_block(seed, quadrant_of, n_trials=200, gain=GAIN, w_screen=0.0, y_only=True):
    """Open-loop held-out trials on diagonal targets, one session per rotation (a block), quadrants from
    ``quadrant_of(rng, n)``; a 1-D y person's output follows body_y only."""
    rng = np.random.default_rng(seed)
    body, screen, prov = [], [], []
    for b, name in enumerate(("rot+90", "rot-90")):
        q = quadrant_of(rng, n_trials // 2)
        for t, k in enumerate(q):
            u = np.tile(DIAGONAL_DIRECTIONS[k], (WIN_PER_TRIAL, 1))
            bb, ss = certification_regressors(u, np.zeros_like(u), Mapping(name))
            body.append(bb)
            screen.append(ss)
            prov += [{"session": f"blk{b}", "trial": t, "source": "H"} for _ in range(WIN_PER_TRIAL)]
    body, screen = np.concatenate(body), np.concatenate(screen)
    out = _outputs(body, screen, prov, gain, w_screen, seed + 1)
    if y_only:
        out[:, 0] = 0.0
    return out, body, screen, prov


def test_diagonal_targets_balance_the_signs_of_both_offsets():
    rng = np.random.default_rng(0)
    q = quadrant_schedule(43, rng)
    assert all(sorted(q[i:i + 4]) == [0, 1, 2, 3] for i in range(0, 40, 4))
    assert np.allclose(np.abs(diagonal_targets(0.8)), 0.8 / np.sqrt(2)) and len(set(map(tuple, np.sign(DIAGONAL_DIRECTIONS)))) == 4
    with pytest.raises(ValueError):
        diagonal_targets(0.0)
    # 1-D y under rot±90: the reachable (horizontal) offset is body_y; the vertical one is the screen
    # regressor.  Horizontal targets leave it 0 on every trial (β_screen unidentified); balanced diagonals
    # make it vary and uncorrelated with body_y.
    u_h = np.array([[1.0, 0.0], [-1.0, 0.0]] * 10)
    _, s_h = certification_regressors(u_h, np.zeros_like(u_h), Mapping("rot+90"))
    assert np.allclose(s_h[:, 1], 0.0)
    u_d = DIAGONAL_DIRECTIONS[quadrant_schedule(40, rng)]
    b_d, s_d = certification_regressors(u_d, np.zeros_like(u_d), Mapping("rot+90"))
    assert np.std(s_d[:, 1]) > 0.5 and abs(np.corrcoef(b_d[:, 1], s_d[:, 1])[0, 1]) < 0.1


def test_correlated_regressors_in_a_block_are_refused():
    """With mostly anti-diagonal targets the body and screen regressors of y correlate within each block
    (|r| > 0.5, opposite signs under the two rotations, so the unit as a whole looks balanced): such a
    unit is never certified.  Balanced diagonal targets give |r| ≈ 0 per block and certify."""
    skewed = lambda rng, n: np.where(rng.random(n) < 0.85, rng.choice([1, 3], n), rng.choice([0, 2], n))
    balanced = lambda rng, n: quadrant_schedule(n, rng)
    for quadrants, expect in ((skewed, "TEST_INSENSITIVE"), (balanced, "CERTIFIED")):
        out, body, screen, prov = _diagonal_block(60, quadrants)
        sh, *_ = _diagonal_block(60, quadrants, gain=0.3, w_screen=0.3)
        kw = dict(axes=("y",), margin=GAIN)
        dep, shadow = certify_mapping(out, body, screen, prov, **kw), certify_mapping(sh, body, screen, prov, **kw)
        null, *_ = _diagonal_block(60, quadrants, gain=0.3, w_screen=0.0)
        assert dep["y"]["identifiable"]
        assert set(dep["y"]["block_corr"]) == {"blk0", "blk1"}
        v = certify(dep, shadow, shadow_null=certify_mapping(null, body, screen, prov, **kw))
        assert v["verdict"] == expect, (v["reason"], dep["y"]["block_corr"])
        assert v["flags"]["regressors_confounded"] == (expect == "TEST_INSENSITIVE")
        if expect == "TEST_INSENSITIVE":
            assert dep["y"]["max_block_corr"] > MAX_BLOCK_CORR and abs(dep["y"]["regressor_corr"]) < MAX_BLOCK_CORR
            assert "correlate" in v["reason"]
            # a confounded unit is left out of a combination, and listed
            other = certify_mapping(*_diagonal_block(61, balanced), **kw)
            comb = combine_units({"skewed": dep, "balanced": other})
            assert comb["excluded_confounded"]["y"] == ["skewed"] and comb["y"]["n_units"] == 1


# ---------------------------------------------------------------- pooling people: person-then-trial bootstrap
def _person(seed, gain, w_screen, n_units=2, n_trials=60):
    units = {f"v{k}": _unit(seed + 10 * k, gain, w_screen, ("rot+90", "rot-90"), f"p{seed}_v{k}", n_trials)
             for k in range(n_units)}
    dep = {u: certify_mapping(*a, margin=gain) for u, a in units.items()}
    sha = {u: certify_mapping(_outputs(*a[1:], 0.2, 0.3, seed + 99), *a[1:], margin=gain) for u, a in units.items()}
    return combine_units(dep, combination="fixed"), combine_units(sha, combination="fixed"), dep


def test_people_are_pooled_by_a_person_then_trial_bootstrap():
    """Pooled over people, the interval must carry between-person variance: people are resampled, then
    each drawn person's trials.  The estimate is the equal-weight mean over people of each person's
    trial-weighted relative β; heterogeneous people widen the interval, and with few people the interval
    is a t with people − 1 df."""
    same = {f"p{i}": _person(70 + i, 0.4, 0.0) for i in range(4)}
    mixed = {f"p{i}": _person(70 + i, 0.4, (0.0, 0.0, 0.12, -0.12)[i]) for i in range(4)}
    for people in (same, mixed):
        dep = {p: d for p, (d, _, _) in people.items()}
        boot = combine_units(dep, combination="bootstrap", n_boot=MIN_BOOT, seed=5)
        e = boot["y"]
        assert boot["combination"] == "bootstrap" and e["n_units"] == 4 and e["df_screen"] == 3
        expect = np.mean([np.average([u["y"]["beta_screen"] / u["y"]["scale"] for u in leaves.values()],
                                     weights=[u["y"]["n_clusters"] for u in leaves.values()])
                          for _, _, leaves in people.values()])
        assert e["beta_screen"] == pytest.approx(expect)
        assert e["ci_screen"][0] < e["beta_screen"] < e["ci_screen"][1]
        again = combine_units(dep, combination="bootstrap", n_boot=MIN_BOOT, seed=5)
        assert again["y"]["ci_screen"] == e["ci_screen"]
    width = lambda people: np.diff(combine_units({p: d for p, (d, _, _) in people.items()},
                                                 combination="bootstrap", n_boot=MIN_BOOT)["y"]["ci_screen"])[0]
    assert width(mixed) > 1.5 * width(same)
    dep = {p: d for p, (d, _, _) in same.items()}
    sha = {p: s for p, (_, s, _) in same.items()}
    v = certify(combine_units(dep, combination="bootstrap", n_boot=MIN_BOOT),
                combine_units(sha, combination="bootstrap", n_boot=MIN_BOOT))
    assert v["verdict"] in ("CERTIFIED", "INCONCLUSIVE") and set(v["unit_verdicts"]) == set(same)
    with pytest.raises(ValueError, match="n_boot"):
        combine_units(dep, combination="bootstrap", n_boot=10)


def test_blocks_too_small_for_a_correlation_are_not_judged_alone():
    """Replay trials sit ~3 per sighted run: a correlation over 3 trials is mostly chance (it refused
    every replay unit in the full simulation), so only blocks of ≥ MIN_CLUSTERS trials are judged per
    block; the small ones still count through the unit's overall correlation."""
    private, replayed, prov = _replay_block(seed=16)
    tiny_runs = [dict(p, session=f"run{p['trial'] // 3}") for p in prov]
    r = replay_test(_outputs(private, replayed, tiny_runs, GAIN, 0.0, 17), private, replayed, tiny_runs, margin=GAIN)
    assert r["y"]["block_corr"] == {} and not r["y"]["confounded"] and np.isnan(r["y"]["max_block_corr"])


# ---------------------------------------------------------------- review round 3: validity findings
def _rot_diag_unit(n_trials, seed, w=5, trial_sd=0.3, window_sd=0.1, w_body=0.2, w_screen=0.0):
    """One held-out unit on the deployed geometry: rot+90 / rot−90 alternating, balanced diagonal
    quadrants, a trial random effect; a 1-D y decoder.  Returns (out, body, screen, prov)."""
    rng = np.random.default_rng(seed)
    q = quadrant_schedule(n_trials, rng)
    body, screen, prov = [], [], []
    for t in range(n_trials):
        m = Mapping("rot+90" if t % 2 == 0 else "rot-90")
        u = DIAGONAL_DIRECTIONS[q[t]]
        for k in range(w):
            body.append(m.screen_to_body(u))
            screen.append(u)
            prov.append({"source": "H", "session": m.name, "trial": t, "t": float(k)})
    body, screen = np.array(body), np.array(screen)
    trial = np.array([p["trial"] for p in prov])
    out = np.zeros_like(body)
    out[:, 1] = (w_body * body[:, 1] + w_screen * screen[:, 1] + rng.normal(0, trial_sd, n_trials)[trial]
                 + rng.normal(0, window_sd, len(body)))
    return out, body, screen, prov


def test_small_unit_intervals_keep_their_coverage():
    """Review: the percentile trial-cluster bootstrap's 95 % CI for β_screen missed a true 0 in 25 % of
    8-trial and 10 % of 12-trial units on this geometry.  The CR2 t(G − k) interval keeps its size (about
    5–6 % in 2,000 units per size; here 300 units, so the bound leaves room for Monte-Carlo error)."""
    for n_trials in (MIN_CLUSTERS, 12):
        miss = n = 0
        for r in range(300):
            out, body, screen, prov = _rot_diag_unit(n_trials, 7000 + 31 * n_trials + r)
            e = certify_mapping(out, body, screen, prov, axes=("y",), margin={"y": 0.2})["y"]
            if not e["identifiable"]:
                continue                          # a chance-collinear 8-trial schedule: no interval
            n += 1
            lo, hi = e["ci_screen"]
            miss += not lo <= 0.0 <= hi
        assert n >= 200 and miss / n <= 0.10, (n_trials, n, miss / n)


def test_negative_screen_dependence_is_screen_reading_not_a_pass():
    """Review: an output that follows the display with a NEGATIVE sign (−30 % of the margin) used to pass
    the one-sided bound 'β_screen < 10 %' more easily than a clean decoder.  The bound is two-sided
    (TOST): |β_screen| must be shown inside ±10 %, and a significant negative β_screen reads the display."""
    m, n_trials = 0.3, 400
    out, body, screen, prov = _rot_diag_unit(n_trials, 3, trial_sd=0.1, window_sd=0.05, w_body=0.9 * m,
                                             w_screen=-0.30 * m)
    sh, *_ = _rot_diag_unit(n_trials, 3, trial_sd=0.1, window_sd=0.05, w_body=0.9 * m, w_screen=0.5 * m)
    null, *_ = _rot_diag_unit(n_trials, 4, trial_sd=0.1, window_sd=0.05, w_body=0.9 * m)
    kw = dict(axes=("y",), margin={"y": m})
    d = certify_mapping(out, body, screen, prov, **kw)
    v = certify(d, certify_mapping(sh, body, screen, prov, **kw),
                shadow_null=certify_mapping(null, body, screen, prov, **kw))
    assert d["y"]["rel_ci_screen"][1] < 0
    assert v["verdict"] == "SCREEN_READING", v["reason"]
    assert v["flags"]["deployed_screen_sign"] == -1 and "anti-screen" in v["reason"]
    # the same decoder without the anti-screen term is certified on the same windows
    clean, *_ = _rot_diag_unit(n_trials, 3, trial_sd=0.1, window_sd=0.05, w_body=0.9 * m)
    ok = certify(certify_mapping(clean, body, screen, prov, **kw), certify_mapping(sh, body, screen, prov, **kw),
                 shadow_null=certify_mapping(null, body, screen, prov, **kw))
    assert ok["verdict"] == "CERTIFIED", ok["reason"]


def test_body_lower_limit_below_half_the_margin_is_inconclusive():
    """Review: CERTIFIED was issued with β_body = 0.76 of the margin and its lower CI at 0.475.  The design
    asks β_body ≥ 70 % with lower CI ≥ 50 % (F1b): 'β_body > 0.5·margin' is in the Holm family and must
    be shown.  The geometry (the review's): the y body regressor varies little (closed-loop-like), so
    β_body is imprecise while β_screen is tight; the noise is projected off the regressors so the point
    estimates are exactly the true 0.80 and 0."""
    rng = np.random.default_rng(1)
    n_trials, w, m = 1000, 4, 0.3
    prov, body, screen = [], [], []
    for t in range(n_trials):
        rot = 1 if t % 2 == 0 else -1
        sx, sy = rng.choice([-1, 1]), rng.choice([-1, 1])
        for k in range(w):
            body.append([rot * sy, 0.12 * sx])
            screen.append([sx, sy])
            prov.append({"source": "H", "session": f"b{rot}", "trial": t, "t": float(k)})
    body, screen = np.array(body), np.array(screen)
    X = np.column_stack([body[:, 1], screen[:, 1], body[:, 0], np.ones(len(body))])

    def exact(noise):
        return noise - X @ np.linalg.lstsq(X, noise, rcond=None)[0]

    noise = exact(rng.normal(0, 0.2, n_trials).repeat(w) + rng.normal(0, 0.05, len(body)))
    out, sh, null = (np.zeros_like(body) for _ in range(3))
    out[:, 1] = 0.80 * m * body[:, 1] + noise
    sh[:, 1] = out[:, 1] + 0.5 * m * screen[:, 1]
    null[:, 1] = 0.80 * m * body[:, 1] + exact(rng.normal(0, 0.2, n_trials).repeat(w))
    kw = dict(axes=("y",), margin={"y": m})
    d = certify_mapping(out, body, screen, prov, **kw)
    v = certify(d, certify_mapping(sh, body, screen, prov, **kw),
                shadow_null=certify_mapping(null, body, screen, prov, **kw))
    e = d["y"]
    assert e["rel_body"] == pytest.approx(0.80) and e["rel_ci_body"][0] < 0.50
    assert _near_zero(e["rel_ci_screen"], tol=0.1)
    assert v["verdict"] == "INCONCLUSIVE" and "F1b" in v["reason"], v["reason"]
    assert v["flags"]["deployed_body_sufficient"] and not v["flags"]["deployed_body_lower_bounded"]


def test_margin_uncertainty_enters_every_threshold(results, block):
    """Review: the external margin comes from 7–12 identity trials (CV 0.3–0.4) and was treated as exact,
    so a margin drawn high loosened the screen bound in proportion.  With ``margin_se`` the thresholds
    are tested as β − c·m with the margin's variance (Welch df): the same windows that certify with an
    exact margin are INCONCLUSIVE when the margin is that uncertain; tests against 0 do not move."""
    body, screen, prov = block
    out = _outputs(body, screen, prov, GAIN, 0.03, 2)
    exact = certify_mapping(out, body, screen, prov, margin=GAIN)
    noisy = certify_mapping(out, body, screen, prov, margin=GAIN, margin_se=0.38 * GAIN, margin_df=8)
    assert noisy["y"]["rel_body"] == exact["y"]["rel_body"]
    assert np.diff(noisy["y"]["rel_ci_body"])[0] > 2 * np.diff(exact["y"]["rel_ci_body"])[0]
    a = certify(exact, results["shadow_mixed"], shadow_null=results["shadow_null"])
    b = certify(noisy, results["shadow_mixed"], shadow_null=results["shadow_null"])
    assert a["verdict"] == "CERTIFIED", a["reason"]
    assert b["verdict"] == "INCONCLUSIVE", b["reason"]
    ta, tb = a["holm"]["tests"], b["holm"]["tests"]
    assert tb["y:deployed:body>0"]["p"] == pytest.approx(ta["y:deployed:body>0"]["p"])
    assert tb["y:deployed:body>0.5margin"]["p"] > ta["y:deployed:body>0.5margin"]["p"]
    with pytest.raises(ValueError, match="margin_se"):
        certify_mapping(out, body, screen, prov, margin_se=0.1)
    with pytest.raises(ValueError, match="margin_df"):
        certify_mapping(out, body, screen, prov, margin=GAIN, margin_se=0.1, margin_df=0.5)


def test_shadow_positive_control_is_null_calibrated(block):
    """Review: with no screen pathway at all a shadow can show β_screen > 0 (a nonspecific response to
    regressor misspecification), and the uncalibrated gate then passed.  Here both shadows carry the
    same nonspecific +0.15 'screen' term; only the second has a real display pathway on top.  Against
    its no-display null the first is not a positive control (TEST_INSENSITIVE); the second is.  Without
    a null the verdict is never CERTIFIED."""
    body, screen, prov = block
    dep = certify_mapping(_outputs(body, screen, prov, GAIN, 0.0, 2), body, screen, prov, margin=GAIN)
    nonspecific = certify_mapping(_outputs(body, screen, prov, 0.3, 0.15, 51), body, screen, prov, margin=GAIN)
    null = certify_mapping(_outputs(body, screen, prov, 0.3, 0.15, 52), body, screen, prov, margin=GAIN)
    real = certify_mapping(_outputs(body, screen, prov, 0.3, 0.45, 53), body, screen, prov, margin=GAIN)
    assert nonspecific["y"]["ci_screen"][0] > 0                      # it fires against 0 …
    v = certify(dep, nonspecific, shadow_null=null)
    assert v["verdict"] == "TEST_INSENSITIVE" and "null" in v["reason"]  # … not against its null
    assert certify(dep, real, shadow_null=null)["verdict"] == "CERTIFIED"
    uncal = certify(dep, real)
    assert uncal["verdict"] == "INCONCLUSIVE" and not uncal["flags"]["shadow_null_calibrated"]
    assert "null-calibrated" in uncal["reason"]
    # a null that did not test this axis cannot calibrate it: the shadow is not counted
    y_only_null = certify_mapping(_outputs(body, screen, prov, 0.3, 0.15, 52), body, screen, prov, margin=GAIN,
                                  axes=("y",))
    assert certify(dep, real, shadow_null=y_only_null)["per_axis"]["x"]["verdict"] == "TEST_INSENSITIVE"
    with pytest.raises(ValueError, match="shadow_null"):
        certify(dep, {"central11": real, "all32": real}, shadow_null={"central11": null})


def test_reports_that_do_not_confirm_the_intent_are_dropped():
    """Review: the C2b keyed report (and the post-replay report of the held goal) were recorded but never
    used, so a failed remap was scored as decoder behaviour.  A trial whose report points against its
    body intent is dropped; a diagonal intent is confirmed by either component class; a trial without a
    report (closed loop) is kept."""
    body = np.array([[0.0, 1.0]] * 2 + [[0.0, -1.0]] * 2 + [[0.7, 0.7]] * 2 + [[1.0, 0.0]] * 2)
    prov = [{"session": "s", "trial": t // 2, "body_report": r}
            for t, r in zip(range(8), [[0, 1]] * 2 + [[0, 1]] * 2 + [[1, 0]] * 2 + [None] * 2)]
    assert confirmed_by_report(prov, body).tolist() == [True, True, False, False, True, True, True, True]
    with pytest.raises(ValueError, match="one row"):
        confirmed_by_report(prov, body[:3])
