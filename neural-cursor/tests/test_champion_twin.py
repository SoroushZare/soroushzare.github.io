"""The shadow positive control against its no-display null, end to end (review finding: the positive
control fired with no screen pathway at all, so 'shadow β_screen > 0' could be passed by a nonspecific
response).

``simulate`` runs the same people a second time at screen_leak 0 — same seeds, noise streams and prior,
so the same decoder versions and test units — and each shadow must read the screen BEYOND its result
there.  This file runs the tiny leaky configuration of ``test_simulate_champion.py`` with its twin (two
worker processes: the run and its twin in parallel).
"""
import pytest

from neural_cursor.simulate_champion import simulate
from test_simulate_champion import BASE


@pytest.fixture(scope="module")
def run():
    return simulate(BASE, workers=2)


def test_the_twin_calibrates_every_certification_row(run):
    day = run["per_user"][0]["days"][0]
    for cert in (day["certification"]["heldout"], run["per_user"][0]["pooled_days"]["heldout"],
                 run["pooled_all_users"]["heldout"]):
        assert set(cert["shadows_null"]) == {"central11", "all32"}
        assert all(cert["flags"][a]["shadow_null_calibrated"] for a in cert["axes"])
        assert set(cert["holm"]) >= {"y:central11:screen>null", "y:all32:screen>null"}


def test_a_large_screen_leak_reads_beyond_the_null_on_y(run):
    """Screen leak 2: the all-32 shadow follows the display far beyond what the same shadow model does
    on the same people with no display pathway; the deployed decoder (trained on blind labels only) does
    not read the screen."""
    cert = run["per_user"][0]["days"][0]["certification"]["heldout"]
    y = cert["flags"]["y"]
    null = cert["shadows_null"]["all32"]["y"]["rel_screen"]
    assert y["shadows_read_screen"]["all32"] and cert["shadows"]["all32"]["y"]["rel_screen"] > null + 1.0
    assert not (y["deployed_screen_significant"] and not y["deployed_screen_bounded"])
    assert cert["per_axis"]["y"] != "TEST_INSENSITIVE"
