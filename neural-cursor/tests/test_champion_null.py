"""Null calibration of the champion certification (review findings: false SCREEN_READING and a firing
positive control at screen leak 0).

With ``screen_leak = 0`` the simulated EEG is exactly independent of the screen direction
(``sim_body``), so whatever the decoders do, a SCREEN_READING verdict there is a false positive of the
test itself, and a shadow "reading the screen" is a nonspecific response.  Both happened: under a single
rotation the screen regressor on one axis equals ± the other axis's body intent, so an x model's
reaction to feet/hands imagery read as screen reading; and an uncalibrated shadow showed β_screen 1.5×
the margin at Holm p 0.003.  This run uses the tiny configuration of ``test_simulate_champion.py`` (a
strong lateraliser, 10 fixation-probe minutes) with the screen pathway switched off; at leak 0 the run is
its own null, so its positive controls are calibrated against themselves.

That tiny run cannot reach the regime where the known false positive lives (review: at this scale the
deployed β_screen is never significant, and most seeds skip or give NO_BODY_SIGNAL), so the file also
calibrates certify's per-unit SCREEN_READING rate directly: many synthetic units with the simulation's
geometry (rot±90, balanced diagonals, 17–35 trials, trial random effects, a significant β_body and
residual cross-axis coupling) and no screen pathway must be flagged at most at about the nominal α; and
the committed default-scale screen_leak 0 result (3 people × 3 days) is checked: its positive controls,
calibrated against that run itself, cannot fire, and its per-unit flags stay within chance.
"""
import dataclasses
import json
import os

import numpy as np
import pytest
from scipy.stats import binom

from neural_cursor.certify import certify, certify_mapping
from neural_cursor.mapping import DIAGONAL_DIRECTIONS, Mapping, quadrant_schedule
from neural_cursor.simulate_champion import ChampionSimConfig, simulate

SLOTS = (("fixation_probe", 10.0), ("blind_closed_loop", 1.0), ("sighted", 6.0), ("heldout_mapping", 8.0))
NULL = ChampionSimConfig(days=1, hours=0.1, users=1, pop=2, seed=46, pop_minutes=1.0, rest_s=20.0,
                         eval_minutes=1.0, screen_leak=0.0, covert_leak=0.0, x_responder_prob=1.0,
                         x_depth_range=(0.8, 0.8), x_permutations=50, n_boot=1000, slot_minutes=SLOTS)


@pytest.fixture(scope="module")
def null_run():
    return simulate(NULL, workers=1)


def test_no_screen_reading_when_the_eeg_cannot_carry_the_screen(null_run):
    """At screen leak 0 the run is its own null: neither positive control can read the screen beyond it,
    so no axis passes (TEST_INSENSITIVE, or NO_BODY_SIGNAL when its body test already fails), and no
    axis reads the screen.  Since x is granted only on two consecutive passing grant tests, this one-day
    run has its first pass at most (``grant_pending``) and tests y alone; the cross-axis failure on x is
    calibrated on synthetic units below, and over 3 people × 3 days in the committed default-scale run."""
    day = null_run["per_user"][0]["days"][0]
    assert day["x_report"]["rule"] == "grant" and not day["x_earned"]
    cert = day["certification"]["heldout"]
    assert cert["axes"] == ["y"]
    assert cert["verdict"] != "SCREEN_READING" and "SCREEN_READING" not in cert["per_axis"].values()
    for axis in cert["axes"]:
        flags = cert["flags"][axis]
        assert flags["shadow_null_calibrated"]
        assert not (flags["deployed_screen_significant"] and not flags["deployed_screen_bounded"])
        # with nothing to read, neither positive control can fire ...
        assert not flags["shadow_reads_screen"] and not any(flags["shadows_read_screen"].values())
        # ... so no axis can pass: it is insensitive, or (Holm order) already fails on its body test.
        assert cert["per_axis"][axis] in ("TEST_INSENSITIVE", "NO_BODY_SIGNAL")
    assert cert["verdict"] in ("TEST_INSENSITIVE", "NO_BODY_SIGNAL")
    replay = day["certification"]["replay"]
    if "skipped" not in replay:                                   # the replay test, when it has its 30 trials
        assert replay["verdict"] in ("TEST_INSENSITIVE", "NO_BODY_SIGNAL")


def test_null_config_differs_from_the_leaky_one_only_in_the_leak_and_the_l1_minutes():
    from test_simulate_champion import BASE
    base_slots, null_slots = dict(BASE.slot_minutes), dict(NULL.slot_minutes)
    assert {s for s in base_slots if base_slots[s] != null_slots.get(s)} == {"fixation_probe"}
    assert set(base_slots) == set(null_slots) and null_slots["fixation_probe"] > base_slots["fixation_probe"]
    assert dataclasses.replace(BASE, screen_leak=0.0, slot_minutes=NULL.slot_minutes) == NULL


def _null_unit(seed, n_trials, coupling):
    """One held-out unit with no screen pathway: a 2-D decoder (β_body 0.25, cross-axis coupling,
    a trial random effect), rot±90 alternating, balanced diagonal quadrants, 5 windows per trial."""
    rng = np.random.default_rng(seed)
    q = quadrant_schedule(n_trials, rng)
    body, screen, prov = [], [], []
    for t in range(n_trials):
        m = Mapping("rot+90" if t % 2 == 0 else "rot-90")
        u = DIAGONAL_DIRECTIONS[q[t]]
        for k in range(5):
            body.append(m.screen_to_body(u))
            screen.append(u)
            prov.append({"source": "H", "session": m.name, "trial": t, "t": float(k)})
    body, screen = np.array(body), np.array(screen)
    trial = np.array([p["trial"] for p in prov])
    out = 0.25 * body + np.asarray(coupling) * body[:, ::-1] + rng.normal(0, 0.15, (n_trials, 2))[trial] \
        + rng.normal(0, 0.1, body.shape)
    return out, body, screen, prov


def test_per_unit_screen_reading_is_flagged_at_about_alpha_without_a_screen_pathway():
    """The per-unit SCREEN_READING flag is uncorrected by design (a positive finding), so its size must be
    the nominal α = 0.05 per unit-axis (both signs together) at the units' real size, with β_body
    significant and cross-axis coupling present — the regime the default-scale null run reached."""
    n_units, flagged, tested = 200, 0, 0
    for i in range(n_units):
        out, body, screen, prov = _null_unit(500 + i, 17 + i % 19, (0.08, -0.06))
        kw = dict(margin=0.25)
        d = certify_mapping(out, body, screen, prov, **kw)
        shadow = certify_mapping(out + 0.3 * screen, body, screen, prov, **kw)
        v = certify(d, shadow, shadow_null=d)
        for a in ("x", "y"):
            if v["per_axis"][a]["flags"].get("deployed_body_significant"):
                tested += 1
                flagged += v["per_axis"][a]["verdict"] == "SCREEN_READING"
    assert tested >= 0.9 * 2 * n_units
    assert flagged / tested <= binom.ppf(0.99, tested, 0.05) / tested + 0.01, (flagged, tested)


RESULT = os.path.join(os.path.dirname(__file__), "..", "results", "champion_d3_h1_u3_pop10_dry0_leak0_covert0.json")


@pytest.mark.skipif(not os.path.exists(RESULT), reason="default-scale screen_leak 0 result not committed")
def test_the_committed_default_scale_null_run():
    """3 people × 3 days at screen leak 0: the run is its own null, so no positive control fires anywhere
    (the old uncalibrated control fired at Holm p 0.003 here); the per-unit-axis SCREEN_READING flags
    stay within the binomial 99 % range of α = 0.05."""
    with open(RESULT) as f:
        cond = next(iter(json.load(f)["conditions"].values()))
    certs = [c for u in cond["per_user"] for r in u["days"] for c in r["certification"].values()
             if isinstance(c, dict) and "verdict" in c]
    certs += [c for u in cond["per_user"] for c in u["pooled_days"].values() if isinstance(c, dict) and "verdict" in c]
    certs += [c for c in cond["pooled_all_users"].values() if isinstance(c, dict) and "verdict" in c]
    assert certs
    for c in certs:
        for a, f in c["flags"].items():
            assert f["shadow_null_calibrated"] and not f.get("shadow_reads_screen"), (a, c["holm"])
        assert c["verdict"] != "CERTIFIED"
    per_unit = [v for u in cond["per_user"] for r in u["days"] for v in r["certification"]["heldout"].get(
        "per_axis", {}).values()]
    assert sum(v == "SCREEN_READING" for v in per_unit) <= binom.ppf(0.99, len(per_unit), 0.05)
