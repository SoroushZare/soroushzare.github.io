"""Certification under held-out mappings and pixel replay, on synthetic decoder outputs.

The key property: under rot±90 body intent and the on-screen direction are orthogonal, so a regression
of the output on both recovers which one a decoder follows.  A body reader is CERTIFIED when the shadow
positive control reads the screen; a screen reader is SCREEN_READING; a shadow that shows no screen
reading makes the test TEST_INSENSITIVE; replay separates private intent from the replayed direction.
"""
import numpy as np
import pytest

from neural_cursor.certify import (MIN_BOOT, MIN_CLUSTERS, certify, certify_mapping, replay_test)
from neural_cursor.mapping import (BODY_VECTORS, HELDOUT_MAPPINGS, MAPPING_NAMES, Mapping,
                                   certification_regressors, unit_directions)

N_TRIALS = 120          # held-out-mapping trials (two sessions)
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


def _outputs(body, screen, prov, w_body, w_screen, seed):
    """Decoder output = w_body·body + w_screen·screen + trial offset + window noise, clipped to [−1, 1]."""
    rng = np.random.default_rng(seed)
    trial = np.array([p["trial"] for p in prov])
    offset = rng.normal(0, TRIAL_SD, (trial.max() + 1, 2))[trial]
    out = w_body * body + w_screen * screen + offset + rng.normal(0, WINDOW_SD, body.shape)
    return np.clip(out, -1.0, 1.0)


def _near_zero(ci, tol=0.08):
    """A CI inside ±tol: the β is bounded near 0.  (Asserting "contains 0" would fail for 5 % of seeds
    by construction; that size is checked separately in test_cluster_bootstrap_keeps_its_size.)"""
    return -tol < ci[0] and ci[1] < tol


@pytest.fixture(scope="module")
def block():
    return _heldout_block(seed=1)


@pytest.fixture(scope="module")
def results(block):
    body, screen, prov = block
    fit = lambda wb, ws, seed: certify_mapping(_outputs(body, screen, prov, wb, ws, seed), body, screen, prov)
    return {"body_reader": fit(GAIN, 0.0, 2), "screen_reader": fit(0.0, GAIN, 3),
            "shadow_mixed": fit(0.3, 0.3, 4), "shadow_body_only": fit(GAIN, 0.0, 5), "noise": fit(0.0, 0.0, 6)}


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
        assert e["n_boot_valid"] >= MIN_BOOT


def test_screen_reader_has_screen_beta_and_no_body_beta(results):
    e = results["screen_reader"]["pooled"]
    assert e["ci_screen"][0] > 0.4
    assert _near_zero(e["ci_body"])
    assert e["rel_screen"] == pytest.approx(1.0, abs=0.1)     # self-scaled margin = β_body + β_screen


def test_supplied_margin_scales_relative_betas(block):
    body, screen, prov = block
    out = _outputs(body, screen, prov, GAIN, 0.0, 2)
    r = certify_mapping(out, body, screen, prov, margin=2 * GAIN)
    assert r["pooled"]["scale"] == 2 * GAIN
    assert r["pooled"]["rel_body"] == pytest.approx(0.5, abs=0.06)


def test_bootstrap_resamples_trials_not_windows(block):
    """Repeating every window 4× inside its trial adds no information; a window-level bootstrap would
    shrink the CI by half, the trial-cluster bootstrap must not."""
    body, screen, prov = block
    out = _outputs(body, screen, prov, GAIN, 0.0, 7)
    width = lambda r: np.diff(r["pooled"]["ci_body"])[0]
    once = certify_mapping(out, body, screen, prov)
    rep = lambda a: np.repeat(a, 4, axis=0)
    four = certify_mapping(rep(out), rep(body), rep(screen), [p for p in prov for _ in range(4)])
    assert four["pooled"]["beta_body"] == pytest.approx(once["pooled"]["beta_body"])
    assert width(four) == pytest.approx(width(once), rel=0.15)


def test_cluster_bootstrap_keeps_its_size():
    """With no screen reading, the 95 % CI of β_screen excludes 0 in about 5 % of experiments (not the
    20–40 % a window-level bootstrap gives with these trial offsets)."""
    n_rep = 40
    excl = 0
    for k in range(n_rep):
        body, screen, prov = _heldout_block(seed=1000 + k)
        out = _outputs(body, screen, prov, GAIN, 0.0, 2000 + k)
        lo, hi = certify_mapping(out, body, screen, prov, n_boot=MIN_BOOT)["pooled"]["ci_screen"]
        excl += not (lo < 0 < hi)
    assert excl / n_rep <= 0.125


def test_results_are_deterministic_and_seeded(block):
    body, screen, prov = block
    out = _outputs(body, screen, prov, GAIN, 0.0, 8)
    a = certify_mapping(out, body, screen, prov)
    b = certify_mapping(out, body, screen, prov)
    c = certify_mapping(out, body, screen, prov, seed=1)
    assert a["pooled"]["ci_body"] == b["pooled"]["ci_body"]
    assert a["pooled"]["ci_body"] != c["pooled"]["ci_body"]


def test_mirror_separates_only_when_axes_are_pooled():
    body, screen, prov = _heldout_block(seed=9, mappings=("mirror_x",))
    out = _outputs(body, screen, prov, GAIN, 0.0, 10)
    r = certify_mapping(out, body, screen, prov)
    assert not r["x"]["identifiable"] and not r["y"]["identifiable"]   # x: body = −screen; y: body = screen
    assert r["pooled"]["identifiable"]
    assert r["pooled"]["ci_body"][0] > 0.4 and _near_zero(r["pooled"]["ci_screen"])
    shadow = certify_mapping(_outputs(body, screen, prov, 0.0, GAIN, 11), body, screen, prov)
    assert certify(r, shadow, axis="y")["verdict"] == "TEST_INSENSITIVE"
    assert certify(r, shadow, axis="pooled")["verdict"] == "CERTIFIED"


# ---------------------------------------------------------------- verdicts

def test_body_reader_is_certified_when_shadow_reads_screen(results):
    v = certify(results["body_reader"], results["shadow_mixed"])
    assert v["verdict"] == "CERTIFIED", v["reason"]
    assert v["flags"]["shadow_reads_screen"] and v["flags"]["deployed_screen_bounded"]
    assert certify(results["body_reader"], results["shadow_mixed"], axis="y")["verdict"] == "CERTIFIED"


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
    weak = certify_mapping(_outputs(body, screen, prov, 0.55 * GAIN, 0.0, 13), body, screen, prov, margin=GAIN)
    assert certify(weak, results["shadow_mixed"])["verdict"] == "INCONCLUSIVE"


def test_y_only_decoder_is_certified_on_y(block, results):
    """x not earned: the decoder's x output is silent, so only y is tested (y is the primary axis)."""
    body, screen, prov = block
    out = _outputs(body, screen, prov, GAIN, 0.0, 24)
    out[:, 0] = 0.0
    r = certify_mapping(out, body, screen, prov, axes=("y",))
    shadow = certify_mapping(_outputs(body, screen, prov, 0.3, 0.3, 4), body, screen, prov, axes=("y",))
    assert "x" not in r and r["pooled"]["beta_body"] == r["y"]["beta_body"]
    assert certify(r, shadow, axis="y")["verdict"] == "CERTIFIED"
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
    assert body_reader["regressors"] == {"body": "private_intent", "screen": "replayed_dir"}
    e = body_reader["pooled"]
    assert e["ci_body"][0] > 0.4 and _near_zero(e["ci_screen"])
    e = cursor_reader["pooled"]
    assert e["ci_screen"][0] > 0.4 and _near_zero(e["ci_body"])
    assert certify(body_reader, shadow)["verdict"] == "CERTIFIED"
    assert certify(cursor_reader, shadow)["verdict"] == "SCREEN_READING"


# ---------------------------------------------------------------- input contract

def test_refuses_training_blocks_and_mixed_tests(block, results):
    body, screen, prov = block
    out = _outputs(body, screen, prov, GAIN, 0.0, 20)
    for src in ("L1", "L2b", "S", "R"):
        bad = [dict(p, source=src) if i == 0 else p for i, p in enumerate(prov)]
        with pytest.raises(ValueError, match="source"):
            certify_mapping(out, body, screen, bad)
    with pytest.raises(ValueError, match="source"):
        replay_test(out, body, screen, prov)                            # "H" windows are not replay trials
    private, replayed, rprov = _replay_block(seed=21)
    replay = replay_test(_outputs(private, replayed, rprov, GAIN, 0.0, 22), private, replayed, rprov)
    with pytest.raises(ValueError, match="different tests"):
        certify(results["body_reader"], replay)


def test_refuses_too_few_draws_trials_or_bad_shapes(block):
    body, screen, prov = block
    out = _outputs(body, screen, prov, GAIN, 0.0, 23)
    with pytest.raises(ValueError, match="n_boot"):
        certify_mapping(out, body, screen, prov, n_boot=MIN_BOOT - 1)
    few = WIN_PER_TRIAL * (MIN_CLUSTERS - 1)
    with pytest.raises(ValueError, match="trials"):
        certify_mapping(out[:few], body[:few], screen[:few], prov[:few])
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        certify_mapping(out[:, :1], body, screen, prov)
    with pytest.raises(ValueError, match="axes"):
        certify_mapping(out, body, screen, prov, axes=("z",))
    with pytest.raises(ValueError, match="margin"):
        certify_mapping(out, body, screen, prov, margin=0.0)
    bad = out.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        certify_mapping(bad, body, screen, prov)
