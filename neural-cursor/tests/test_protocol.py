"""Protocol: plans are deterministic and always keep their validity blocks; cues are balanced, never
run more than twice, are logged, and never put anything at a spatial position in cued_blank."""
from collections import Counter

import numpy as np
import pytest

from neural_cursor import data_engine
from neural_cursor.protocol import (
    BLOCK_KINDS, EVENT_CUE_OFFSET, FIXATION, MIN_CUED_FRACTION, N_DIRECTIONS, PROVENANCE_BLOCK, Block,
    CueScheduler, CuedTrialTiming, ZipfTargets, cued_goal, plan_summary, render_plan_markdown, session_plan,
    trial_start_payload,
)

DAYS = range(1, 61)
SEEDS = (0, 1, 7, 20261004)


def test_session_plan_is_deterministic_and_seed_dependent():
    for day in (1, 2, 5, 30):
        assert session_plan(day, 3) == session_plan(day, 3)
    assert any(session_plan(d, 3) != session_plan(d, 4) for d in (2, 5, 30))


@pytest.mark.parametrize("seed", SEEDS)
def test_every_session_keeps_validity_blocks(seed):
    for day in DAYS:
        plan = session_plan(day, seed)
        s = plan_summary(plan)
        assert s["cued_fraction"] >= MIN_CUED_FRACTION, (day, s)
        assert s["blank_feedback"] > 0, day
        assert plan[0].kind == "setup"
        assert plan[1].kind == "cued_blank"                         # fresh reference before any decoder
        for i, b in enumerate(plan):
            if b.kind == "blank_feedback":                          # probe paired with a visible-cursor block
                assert plan[i - 1].kind == "closed_loop"
        cues = {b.cue for b in plan if b.kind == "cued_blank"}
        if sum(b.kind == "cued_blank" for b in plan) >= 2:
            assert cues == {"auditory", "central_symbol"}


def test_time_shifts_from_cued_to_closed_loop():
    s1, s10 = plan_summary(session_plan(1, 0)), plan_summary(session_plan(10, 0))
    assert s1["cued_fraction"] >= 0.6
    assert s1["cued_blank"] > s1["closed_loop"] and s10["closed_loop"] > s10["cued_blank"]
    assert s1["free_use"] == 0 and s10["free_use"] > 0
    shares = [plan_summary(session_plan(d, 0))["cued_fraction"] for d in DAYS]
    assert all(a >= b for a, b in zip(shares, shares[1:]))
    # day 1 front-loads cued_blank: no decoder-driven block precedes the last cued_blank block
    plan = session_plan(1, 0)
    last_cued = max(i for i, b in enumerate(plan) if b.kind == "cued_blank")
    assert all(b.kind in ("setup", "rest", "cued_blank") for b in plan[:last_cued])


def test_cued_blank_has_no_spatial_screen_cue():
    for seed in SEEDS:
        for day in DAYS:
            for b in session_plan(day, seed):
                if b.kind == "cued_blank":
                    assert b.feedback == "none" and not b.feedback_visible
                    assert b.cue in ("auditory", "central_symbol")
                    assert all(not it.spatial and it.position == (0.0, 0.0) for it in b.screen_items())
    sched = CueScheduler(seed=5)
    for mode in ("auditory", "central_symbol"):
        for cue in sched.schedule(N_DIRECTIONS, mode=mode):
            items = cue.presentation(mode)["screen"]
            assert all(not it.spatial and it.position == (0.0, 0.0) for it in items)
    with pytest.raises(ValueError):
        cue.presentation("target")
    with pytest.raises(ValueError):
        Block("cued_blank", 60, cue="target")
    with pytest.raises(ValueError):
        Block("cued_blank", 60, feedback="cursor")
    assert Block("closed_loop", 60).feedback_visible and not Block("blank_feedback", 60).feedback_visible


def test_cue_symbol_is_gone_before_the_labelled_intent():
    """The central arrow maps one-to-one onto the label: it may be drawn only until cue offset, and
    during the intent phase (from trial_start) the screen holds the fixation mark alone."""
    timing = CuedTrialTiming()
    for cue_mode in ("auditory", "central_symbol"):
        b = Block("cued_blank", 240, cue=cue_mode)
        assert b.screen_items(phase="intent") == (FIXATION,)
        cue = CueScheduler(seed=0).next()
        pres = cue.presentation(cue_mode, timing)
        assert pres["duration_s"] == timing.cue_s and pres["after_offset"] == [FIXATION]
    assert any(it.name == "direction_symbol" for it in Block("cued_blank", 240, cue="central_symbol")
               .screen_items(phase="cue"))
    with pytest.raises(ValueError):
        Block("cued_blank", 240).screen_items(phase="iti")
    assert EVENT_CUE_OFFSET == data_engine.EVENT_CUE_OFFSET           # the labeler checks this event


def test_cue_scheduler_balanced_random_and_short_runs():
    sched = CueScheduler(seed=11)
    cues = sched.schedule(16 * 50)
    idx = np.array([c.index for c in cues])
    for k in range(16, len(idx) + 1, 16):                           # exact balance at block boundaries
        assert set(Counter(idx[:k].tolist()).values()) == {k // N_DIRECTIONS}
    for k in range(1, len(idx)):
        counts = np.bincount(idx[:k], minlength=N_DIRECTIONS)
        assert counts.max() - counts.min() <= 2
    run = 1
    for a, b in zip(idx, idx[1:]):
        run = run + 1 if a == b else 1
        assert run <= 2
    assert [c.index for c in CueScheduler(seed=11).schedule(64)] == idx[:64].tolist()
    assert [c.index for c in CueScheduler(seed=12).schedule(64)] != idx[:64].tolist()
    for c in cues[:N_DIRECTIONS]:
        assert np.isclose(np.hypot(*c.vector), 1.0)


def test_cue_scheduler_logs_every_cue():
    class Sink:
        def __init__(self):
            self.events = []

        def event(self, kind, payload, t):
            self.events.append((kind, payload, t))

    sink = Sink()
    sched = CueScheduler(seed=2, logger=sink)
    cues = [sched.next(t=1.5 * i, mode="auditory") for i in range(10)]
    assert [e["seq"] for e in sched.log] == list(range(10))
    assert [e["direction"] for e in sched.log] == [c.name for c in cues]
    assert [(k, p["index"], t) for k, p, t in sink.events] == [("cue", c.index, 1.5 * i)
                                                              for i, c in enumerate(cues)]


def test_payloads_match_data_engine_schema():
    assert {v for v in PROVENANCE_BLOCK.values() if v} == set(data_engine.BLOCK_KINDS)
    assert set(PROVENANCE_BLOCK) == set(BLOCK_KINDS)
    cue = CueScheduler(seed=0).next()
    p = trial_start_payload(Block("cued_blank", 240), trial=3, goal=cued_goal(cue))
    assert p["block"] == "cued_blank" and p["feedback_visible"] is False
    assert np.allclose(np.array(p["goal"]) / np.linalg.norm(p["goal"]), cue.vector)
    assert trial_start_payload(Block("free_use", 300), 0, None)["block"] == "free"
    with pytest.raises(ValueError):
        trial_start_payload(Block("rest", 60), 0, (0.5, 0.0))
    timing = CuedTrialTiming()
    assert timing.n_trials(Block("cued_blank", 240)) == int(240 // timing.trial_s)


def test_markdown_table_lists_minutes():
    plan = session_plan(4, 0)
    md = render_plan_markdown(plan, title="Day 4")
    rows = [ln for ln in md.splitlines() if ln.startswith("| ") and ln[2].isdigit()]
    assert len(rows) == len(plan)
    minutes = [float(r.split("|")[4]) for r in rows]
    assert np.allclose(minutes, [b.duration_s / 60 for b in plan])
    assert "Minutes" in md and "cued_blank" in md and "Total" in md


def test_zipf_targets_skewed_and_stable():
    z = ZipfTargets(seed=1)
    assert np.allclose(z.locations, ZipfTargets(seed=1).locations)
    draws = z.sample(5000, np.random.default_rng(0))
    counts = Counter(map(tuple, np.round(draws, 9)))
    top = counts[tuple(np.round(z.locations[0], 9))] / 5000
    assert abs(top - z.probabilities[0]) < 0.03 and z.probabilities[0] > 5 * z.probabilities[-1]
    assert np.all(np.linalg.norm(z.locations, axis=1) >= z.min_center_dist)
