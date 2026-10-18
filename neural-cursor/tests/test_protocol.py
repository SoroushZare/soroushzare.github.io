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


# ----------------------------------------------------------------------------------------------
# Champion protocol (body coordinates, certified purity)
# ----------------------------------------------------------------------------------------------
from neural_cursor import protocol as P                                    # noqa: E402
from neural_cursor.mapping import HELDOUT_MAPPINGS, Mapping               # noqa: E402

CHAMPION_DAYS = range(1, 41)


def _all_phase_items(block):
    return [it for phase in P.PHASES for it in block.screen_items(phase=phase)]


def test_champion_kinds_sources_and_validation():
    """Each champion kind has one fixed label source.  Test-only kinds are flagged, and impossible
    blocks cannot be built (a fixation probe with a target, a held-out block under the identity,
    replays in a blind or just-intend block, an uninstructed L1 block)."""
    expect = {"fixation_probe": ("L1", "none", False), "blind_closed_loop": ("L2b", "tone", False),
              "sighted": ("S", "cursor", True), "heldout_mapping": ("H", "cursor", True),
              "pixel_replay": ("R", "replay", True)}
    for kind, (src, fb, visible) in expect.items():
        b = Block(kind, 60)
        assert b.source == src and b.feedback == fb and b.feedback_visible is visible
        assert b.provenance_block == kind and b.is_champion
        assert b.is_test_only == (src in ("H", "R"))
    assert P.SOURCES_BY_KIND == data_engine.SOURCES_BY_BLOCK
    assert set(P.CHAMPION_BLOCK_KINDS) == set(data_engine.CHAMPION_BLOCKS)
    assert set(P.DEPLOYED_TRAINING_SOURCES) == set(data_engine.DEPLOYED_SOURCES)
    assert Block("heldout_mapping", 60).mapping in HELDOUT_MAPPINGS
    c2b = Block("heldout_mapping", 60, variant="C2b")
    assert c2b.feedback == "frozen" and c2b.feedback_visible and c2b.replay_fraction == 0.0
    assert Block("sighted", 60).replay_fraction == P.REPLAY_FRACTION == Block("heldout_mapping", 60).replay_fraction
    ji = Block("blind_closed_loop", 60, just_intend=True)
    assert ji.source == "J" and ji.is_test_only and ji.cue == "goal_word" and ji.instructed is False
    assert Block("sighted", 60, just_intend=True).source == "J"
    assert Block("sighted", 60).replay_block().source == "R"
    assert Block("heldout_mapping", 60, mapping="mirror_y").replay_block().mapping == "mirror_y"
    bad = [dict(kind="fixation_probe", cue="target"), dict(kind="fixation_probe", cue="body_word"),
           dict(kind="heldout_mapping", mapping="identity"), dict(kind="sighted", mapping="rot+90"),
           dict(kind="blind_closed_loop", replay_fraction=0.1), dict(kind="heldout_mapping", variant="C2b",
                                                                     replay_fraction=0.1),
           dict(kind="sighted", just_intend=True, replay_fraction=0.1), dict(kind="sighted", replay_fraction=0.5),
           dict(kind="fixation_probe", instructed=False), dict(kind="blind_closed_loop", instructed=False),
           dict(kind="blind_closed_loop", just_intend=True, cue="body_word"),
           dict(kind="heldout_mapping", variant="C3"), dict(kind="pixel_replay", just_intend=True),
           dict(kind="sighted", feedback="none"), dict(kind="cued_blank", mapping="identity"),
           dict(kind="closed_loop", just_intend=True), dict(kind="sighted", localiser=True)]
    for kw in bad:
        with pytest.raises(ValueError):
            Block(duration_s=60, **kw)
    assert Block("sighted", 60, instructed=False).instructed is False       # S may withdraw instruction
    assert Block("setup", 60, localiser=True).localiser


def test_fixation_and_blind_blocks_draw_nothing_at_a_trial_dependent_position():
    """L1 and L2b labels must not be identifiable from the display.  In every phase (cue, intent,
    outcome) and every cue mode they draw only items at fixation: the glyph during the cue phase and
    the outcome flash after the trial.  The test-only and sighted kinds do draw targets and a cursor."""
    blocks = [Block("fixation_probe", 60)] + [Block("blind_closed_loop", 60, cue=c)
                                              for c in ("mixed", "central_glyph", "body_word")]
    blocks.append(Block("blind_closed_loop", 60, just_intend=True))
    for b in blocks:
        for it in _all_phase_items(b):
            assert not it.spatial and it.position == (0.0, 0.0), (b.kind, b.cue, it)
        assert b.screen_items("intent") == (FIXATION,)                    # nothing but fixation while labelled
    assert P.BODY_GLYPH in Block("fixation_probe", 60).screen_items("cue")
    assert P.BODY_GLYPH not in Block("blind_closed_loop", 60, cue="body_word").screen_items()
    assert P.OUTCOME_FLASH in Block("blind_closed_loop", 60).screen_items("outcome")
    assert P.OUTCOME_FLASH not in Block("blind_closed_loop", 60).screen_items("intent")
    for kind in ("sighted", "heldout_mapping", "pixel_replay"):
        assert any(it.spatial and it.position is None for it in Block(kind, 60).screen_items())
    for b in (Block("heldout_mapping", 60), Block("pixel_replay", 60)):
        assert b.is_test_only
    with pytest.raises(ValueError):
        Block("fixation_probe", 60).screen_items(phase="iti")
    # the body cues themselves are presented at fixation or through the (centred) speaker only
    gm = P.glyph_map(3, 0)
    sched = P.BodyCueScheduler(seed=1, classes=P.L2B_CLASSES_XY, modes=P.BODY_CUE_MODES, glyphs=gm)
    for cue in sched.schedule(32):
        pres = cue.presentation(P.ClosedLoopTiming().cue_s)
        assert all(not it.spatial and it.position == (0.0, 0.0) for it in pres["screen"] + pres["after_offset"])
        if cue.mode == "body_word":
            assert pres["audio_pan"] == 0.0 and pres["audio_level_db"] == P.WORD_CUE_LEVEL_DB
        else:
            assert [g for g, _ in pres["glyph_sequence"]] == [gm[p] for p in cue.parts]


@pytest.mark.parametrize("cohort", ("deep", "broad"))
def test_champion_session_plan_structure(cohort):
    """Deterministic 85-min sessions in the proposal's order.  Every session keeps fixation probes,
    blind closed loop, at least one held-out-mapping block and replay trials.  L1 : L2b ≈ 43 : 57, the
    instruction is withdrawn only in S, after the cohort's session, and day 1 opens with the blind
    hint-free block before any instruction."""
    order = ["setup", "fixation_probe", "blind_closed_loop", "sighted", "heldout_mapping"]
    for seed in SEEDS:
        for day in CHAMPION_DAYS:
            plan = P.session_plan_champion(day, seed, cohort=cohort)
            assert plan == P.session_plan_champion(day, seed, cohort=cohort)
            s = P.champion_plan_summary(plan)
            assert s["total_s"] == 85 * 60
            assert s["source_s"]["L1"] == 15 * 60 and s["source_s"]["L2b"] == 20 * 60
            assert abs(s["l1_share"] - 0.43) < 0.01
            assert s["source_s"]["H"] > 0 and s["min_replay_trials"] >= 1
            assert all(m in HELDOUT_MAPPINGS for m in s["heldout_mappings"])
            assert any(m.startswith("rot") for m in s["heldout_mappings"])   # the primary test every session
            assert s["localiser"] == (day % 10 == 1)
            assert plan[0].kind == "setup"
            seen = [b.kind for b in plan if not b.just_intend]
            assert [k for i, k in enumerate(seen) if i == 0 or seen[i - 1] != k] == order
            for b in plan:
                if b.source in ("L1", "L2b", "H"):
                    assert b.instructed                                    # L1 and L2b stay cued
                if b.source == "S":
                    assert b.instructed == (day <= P.INSTRUCTION_KEPT_THROUGH[cohort])
                    assert b.replay_fraction == P.REPLAY_FRACTION
                if b.source == "J":
                    assert not b.instructed and b.mapping == "identity"
            variants = s["heldout_variants"]
            assert set(variants) == ({"C2b"} if day < P.CLOSED_LOOP_HELDOUT_FROM_DAY else {"C2b", "C2c"})
            ji = [b for b in plan if b.just_intend]
            assert {b.kind for b in ji} == {"blind_closed_loop", "sighted"}
            if day == 1:                                                    # hint-free: before any instruction
                assert plan[1].just_intend and plan[1].kind == "blind_closed_loop"
                first_instructed = min(i for i, b in enumerate(plan) if b.source in ("L1", "L2b", "S"))
                assert all(b.kind == "blind_closed_loop" and b.just_intend for b in plan[1:first_instructed])
            else:
                kinds = [b.kind for b in plan[-len(ji):]]
                assert all(a != b for a, b in zip(kinds, kinds[1:]))        # blind / sighted alternate
    assert any(P.session_plan_champion(d, 3) != P.session_plan_champion(d, 4) for d in (2, 5, 30))
    md = render_plan_markdown(P.session_plan_champion(5, 0), title="Champion day 5")
    assert "| L1 | identity |" in md and "heldout_mapping" in md and "Total 85 min" in md
    with pytest.raises(ValueError):
        P.session_plan_champion(1, 0, cohort="other")


def test_mirror_x_only_for_x_earners():
    """A mirror separates body from screen only on the axis it flips, so mirror_x is scheduled only for
    a person whose x axis is on."""
    for day in CHAMPION_DAYS:
        assert "mirror_x" not in P.champion_plan_summary(P.session_plan_champion(day, 0))["heldout_mappings"]
    used = {m for d in CHAMPION_DAYS
            for m in P.champion_plan_summary(P.session_plan_champion(d, 0, x_earned=True))["heldout_mappings"]}
    assert used == set(HELDOUT_MAPPINGS)


def test_glyph_map_is_redrawn_every_session_and_non_directional():
    maps = [P.glyph_map(d, 7) for d in CHAMPION_DAYS]
    for m in maps:
        assert set(m) == set(P.BODY_CLASSES) and len(set(m.values())) == 4
        assert set(m.values()) <= set(P.GLYPH_SET)
        assert not set(m.values()) & set("→↗↑↖←↙↓↘▲▼◀▶")                 # nothing that points
    assert maps[0] == P.glyph_map(1, 7)
    changed = sum(a != b for a, b in zip(maps, maps[1:]))
    assert changed >= len(maps) - 3                                        # (almost) always a new map


def test_body_cue_scheduler_balanced_logged_and_short_runs():
    sink = []

    class Sink:
        def event(self, kind, payload, t):
            sink.append((kind, payload, t))

    gm = P.glyph_map(2, 0)
    sched = P.BodyCueScheduler(seed=4, classes=P.L2B_CLASSES_XY, modes=P.BODY_CUE_MODES, glyphs=gm, logger=Sink())
    block = len(P.L2B_CLASSES_XY) * len(P.BODY_CUE_MODES) * 2
    cues = [sched.next(t=float(i)) for i in range(block * 10)]
    pairs = Counter((c.cls, c.mode) for c in cues)
    assert set(pairs.values()) == {20}                                     # every class × mode equally often
    run = 1
    for a, b in zip(cues, cues[1:]):
        run = run + 1 if a.cls == b.cls else 1
        assert run <= 2
    for c in cues:
        assert c.body == P.body_vector(c.cls) and np.abs(c.body).max() == 1
    assert P.body_vector("FEET+RIGHT") == (1, 1) and P.body_vector("HANDS") == (0, -1)
    assert [(k, p["body"], t) for k, p, t in sink] == [("cue", list(c.body), float(i)) for i, c in enumerate(cues)]
    l1 = P.BodyCueScheduler(seed=4, glyphs=gm).schedule(16)
    assert {c.cls for c in l1} == set(P.L1_CLASSES) and {c.mode for c in l1} == {"central_glyph"}
    y_only = P.BodyCueScheduler(seed=4, classes=P.L2B_CLASSES_Y, modes=("body_word",)).schedule(8)
    assert {c.body[0] for c in y_only} == {0} and {c.word for c in y_only} == {"feet", "both hands"}
    with pytest.raises(ValueError):
        P.BodyCueScheduler(seed=0)                                         # glyph cues need this session's map
    with pytest.raises(ValueError):
        P.body_vector("FEET+FEET")


def test_progress_tone_is_label_symmetric():
    """Pitch depends only on the distance decrease.  Mirror images of a trajectory, and rotations
    of it toward another class's goal, give exactly the same tone sequence.  Loudness and pan are
    constants."""
    tone = P.ProgressTone()
    rng = np.random.default_rng(0)
    start, goal = np.zeros(2), np.array([0.0, P.BLIND_GOAL_DISTANCE])              # FEET (up)
    steps = np.cumsum(rng.normal([0.0, 0.03], 0.04, (60, 2)), axis=0)
    traj = np.vstack([start, steps])
    ref = tone.sequence(P.distances_to_goal(traj, goal))
    transforms = [np.diag([-1.0, 1.0]), np.diag([1.0, -1.0]), Mapping("rot+90").M, Mapping("rot-90").M,
                  -np.eye(2)]
    for T in transforms:                                                   # mirror_x, mirror_y, rotations
        seq = tone.sequence(P.distances_to_goal(traj @ T.T, T @ goal))
        assert np.array_equal(seq, ref)
    # an x-class (RIGHT) trial with the same distance profile sounds exactly like the FEET trial
    right_goal = np.array([P.BLIND_GOAL_DISTANCE, 0.0])
    assert np.array_equal(tone.sequence(P.distances_to_goal(traj @ Mapping("rot-90").M.T, right_goal)), ref)
    # monotone in progress, saturating both ways, and nothing but distances goes in
    d0 = P.BLIND_GOAL_DISTANCE
    pitches = [tone.pitch_hz(d0, d) for d in np.linspace(2 * d0, 0.0, 50)]
    assert all(a <= b for a, b in zip(pitches, pitches[1:]))
    assert tone.pitch_hz(d0, d0) == tone.base_hz and tone.pitch_hz(d0, 0.0) == 2 * tone.base_hz
    assert tone.pitch_hz(d0, 10 * d0) == tone.base_hz * 2 ** tone.min_progress
    assert tone.pan == 0.0
    import inspect
    assert list(inspect.signature(tone.pitch_hz).parameters) == ["d0", "d"]


def test_replay_scheduler_exact_rare_and_never_first_or_adjacent():
    host = Block("sighted", 300, seed=11)
    flags = P.ReplayScheduler(host).schedule(200)
    assert flags == P.ReplayScheduler(Block("sighted", 300, seed=11)).schedule(200)
    assert flags != P.ReplayScheduler(Block("sighted", 300, seed=12)).schedule(200)
    assert not flags[0]
    assert not any(a and b for a, b in zip(flags, flags[1:]))
    for k in range(0, 200, 20):
        assert sum(flags[k:k + 20]) == 3                                   # round(0.15 × 20), every period
    with pytest.raises(ValueError):
        P.ReplayScheduler(Block("blind_closed_loop", 300))


def test_champion_payloads_parse_in_the_data_engine():
    """Payloads built by the protocol are accepted by ``data_engine.parse_trials``, with the right
    source, body cue and mapping.  The shared constants (lockout, label end, event names) agree."""
    assert P.CUE_LOCKOUT_S == data_engine.CUE_LOCKOUT_S and P.L1_LABEL_END_S == data_engine.L1_LABEL_END_S
    assert (P.EVENT_FIXATION_BREAK, P.EVENT_OUTCOME_FLASH, P.EVENT_BODY_REPORT) == (
        data_engine.EVENT_FIXATION_BREAK, data_engine.EVENT_OUTCOME_FLASH, data_engine.EVENT_BODY_REPORT)
    assert P.HELDOUT_VARIANTS == data_engine.HELDOUT_VARIANTS
    gm = P.glyph_map(4, 0)
    l1_cue = P.BodyCueScheduler(seed=0, glyphs=gm).next()
    l2_cue = P.BodyCueScheduler(seed=0, classes=P.L2B_CLASSES_XY, modes=P.BODY_CUE_MODES, glyphs=gm).next()
    word = CueScheduler(seed=0).next(mode="auditory")
    timing = P.FixationProbeTiming()
    events, t = [], 10.0
    plans = [(Block("fixation_probe", 300), l1_cue, {}), (Block("blind_closed_loop", 300), l2_cue, {}),
             (Block("blind_closed_loop", 300, just_intend=True), word, {}),
             (Block("sighted", 300), None, {"goal": (0.5, 0.2)}),
             (Block("sighted", 300), None, {"goal": (0.5, 0.2), "replay_goal": (-0.4, 0.3)}),
             (Block("heldout_mapping", 300, mapping="rot-90"), None, {"goal": (0.0, 0.6)})]
    for i, (block, cue, kw) in enumerate(plans):
        if cue is not None:
            body = list(cue.body) if isinstance(cue, P.BodyCue) else None
            events += [(t, "cue", {"seq": i, "body": body} if body else {"seq": i}),
                       (t + timing.cue_s, "cue_offset", {"seq": i})]
        p = P.champion_trial_start_payload(block, i, cue=cue, **kw)
        events += [(t + timing.label_start_s, "trial_start", p), (t + timing.label_end_s, "trial_end", {"trial": i})]
        t += timing.trial_s
    trials = data_engine.parse_trials("s", events)
    assert [tr.source for tr in trials] == ["L1", "L2b", "J", "S", "R", "H"]
    assert trials[0].info["body_cue"] == l1_cue.body and trials[0].goal is None
    assert trials[1].info["body_cue"] == l2_cue.body
    assert np.allclose(trials[1].goal, P.hidden_goal(l2_cue))              # hidden, encodes the cue
    assert trials[4].block == "pixel_replay" and trials[4].info["replay_goal"] == [-0.4, 0.3]
    assert trials[5].info["mapping"] == "rot-90" and trials[5].info["variant"] == "C2c"
    with pytest.raises(ValueError):
        P.champion_trial_start_payload(Block("fixation_probe", 60), 0, cue=l2_cue if len(l2_cue.parts) > 1
                                       else P.BodyCue(0, "FEET+LEFT", ("FEET", "LEFT"), (-1, 1), "central_glyph",
                                                      ("●", "■"), "feet and left hand"))
    with pytest.raises(ValueError):
        P.champion_trial_start_payload(Block("blind_closed_loop", 60, just_intend=True), 0, cue=l2_cue)
    with pytest.raises(ValueError):
        P.champion_trial_start_payload(Block("blind_closed_loop", 60), 0, cue=l2_cue, replay_goal=(0, 0))
    with pytest.raises(ValueError):
        P.champion_trial_start_payload(Block("sighted", 60), 0)            # the drawn goal is required
    with pytest.raises(ValueError):
        trial_start_payload(Block("sighted", 60), 0, (0.5, 0.5))           # champion blocks need the champion payload
    for block in (Block("fixation_probe", 60), Block("blind_closed_loop", 60), Block("sighted", 60),
                  Block("heldout_mapping", 60, variant="C2b")):
        assert P.trial_timing(block).n_trials(block) >= 1
