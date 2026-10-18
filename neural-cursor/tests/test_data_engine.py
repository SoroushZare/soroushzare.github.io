"""Data engine: event parsing, labels, provenance, pure-EEG data flow, weighted training, and the
champion/challenger promotion rule on synthetic RunLogger runs driven by a SimulatedEEG whose latent
intent is the goal direction."""
import os
import sys

from collections.abc import Mapping as AbcMapping

import numpy as np
import pytest

from neural_cursor.acquisition import SimulatedEEG
from neural_cursor.config import DEFAULT
from neural_cursor.data_engine import (RUN_FILES, ChampionChallenger, DataEngine, LabelledSet, Labeler,
                                       ModelRegistry, SessionStore, Trainer, WeightedVelocityDecoder,
                                       constant_ceiling, mean_cosine, personal_weight)
from neural_cursor.decoder import VelocityDecoder
from neural_cursor.logger import RunLogger

CFG = DEFAULT
FS = CFG.hardware.fs
TRIAL_S, ITI_S, SETTLE_S = 2.5, 0.5, 3.0
CLOSED_SPEED = 0.25            # screen units / s of the scripted closed-loop cursor
AUDITORS = ("eyetracker.f32", "emg.f32")


def make_run(root, tag, seed, n_cued, t0, n_closed=0, n_free=0):
    """One RunLogger run: rest, then cued_blank trials (no cursor rows), closed_loop trials (cursor
    logged at 10 Hz moving toward the goal), free trials (goal revealed only at trial_end), then a
    trial that never ends (a crash).  The simulated intent is always the true goal direction from the
    current cursor.  Auditor files are dropped next to the EEG to prove nothing reads them."""
    lg = RunLogger(root, CFG, tag=tag)
    sim = SimulatedEEG(CFG.hardware, seed=seed)
    rng = np.random.default_rng(seed)
    now = lambda: t0 + sim.n_sent / FS

    def emit(sec, intent):
        n = int(round(sec * FS))
        ts = now() + np.arange(n) / FS
        sim.set_intent(*intent)
        lg.eeg(sim.generate(n), ts)

    emit(SETTLE_S, (0, 0))
    lg.event("marker", {"note": "rest over"}, t=now())
    kinds = ["cued_blank"] * n_cued + ["closed_loop"] * n_closed + ["free"] * n_free
    for i, kind in enumerate(kinds):
        a = rng.uniform(0, 2 * np.pi)
        goal = 0.8 * np.array([np.cos(a), np.sin(a)])
        lg.event("trial_start", {"trial": i, "goal": None if kind == "free" else goal, "block": kind,
                                 "feedback_visible": kind != "cued_blank"}, t=now())
        if kind == "cued_blank":
            emit(TRIAL_S, goal / np.linalg.norm(goal))
        else:
            pos, dt = np.zeros(2), 0.1
            for _ in range(int(round(TRIAL_S / dt))):
                u = (goal - pos) / max(np.linalg.norm(goal - pos), 1e-9)
                lg.cursor(now(), pos, u, 0.5, -1, 0.0)
                emit(dt, u)
                pos = pos + CLOSED_SPEED * u * dt
        lg.event("trial_end", {"trial": i, "goal": goal if kind == "free" else None, "outcome": "timeout"},
                 t=now())
        emit(ITI_S, (0, 0))
    lg.event("trial_start", {"trial": 999, "goal": [0.5, 0.5], "block": "cued_blank",
                             "feedback_visible": False}, t=now())
    emit(1.5, (0.7, 0.7))
    lg.close()
    for name in AUDITORS:
        rng.standard_normal((2000, 4)).astype(np.float32).tofile(os.path.join(lg.dir, name))
    return lg.dir


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("runs"))
    a = make_run(root, "a", seed=0, n_cued=25, t0=1000.0, n_closed=4, n_free=2)
    b = make_run(root, "b", seed=50, n_cued=150, t0=9000.0)
    return root, a, b


def test_trials_labels_and_provenance(runs):
    root, a, _ = runs
    store = SessionStore(root, CFG.hardware.n_channels)
    assert store.list_runs() == sorted(store.list_runs()) and len(store.list_runs()) == 2
    sess = store.load(a)
    assert len(sess.trials) == 25 + 4 + 2                       # the never-ended trial is discarded
    ls = Labeler(CFG).label(sess)
    trials = {t.trial: t for t in sess.trials}
    assert len(ls) > 0 and ls.labels.shape == (len(ls), 2)
    assert np.allclose(np.linalg.norm(ls.labels, axis=1), 1.0)
    win = int(CFG.decoder.window_s * FS)
    assert all(w.shape == (win, CFG.hardware.n_channels) and w.dtype == np.float32 for w in ls.windows)
    seen_blocks = set()
    for lab, p in zip(ls.labels, ls.provenance):
        assert set(p) == {"block", "feedback_visible", "session", "t", "trial", "display_goal"}
        assert p["session"] == os.path.basename(a) and p["trial"] != 999
        tr = trials[p["trial"]]
        # the goal as drawn on screen (for the purity audit's display surrogate); cued goals never are
        assert (p["display_goal"] is None) if p["block"] == "cued_blank" else np.allclose(p["display_goal"], tr.goal)
        seen_blocks.add(p["block"])
        # window wholly inside its trial and after the filter settles: nothing outside trials
        assert tr.t_start + CFG.decoder.window_s - 1.0 / FS <= p["t"] <= tr.t_end + 1e-9
        assert p["t"] - CFG.decoder.window_s >= sess.ts[0] + 2.0 - 1.0 / FS
        assert p["feedback_visible"] == (p["block"] != "cued_blank")
        rows = sess.cursor[(sess.cursor[:, 0] <= p["t"]) & (sess.cursor[:, 0] >= tr.t_start)]
        if p["block"] == "cued_blank":                           # hidden, unlogged cursor: start_pos
            assert len(rows) == 0
            d = tr.goal
        else:                                                    # last logged cursor position
            d = tr.goal - rows[-1, 1:3]
        assert np.allclose(lab, d / np.linalg.norm(d), atol=1e-9)
    assert seen_blocks == {"cued_blank", "closed_loop", "free"}  # free: goal revealed at trial_end
    # hop 0.25 s: ends from window_s to TRIAL_S → 9 windows per cued trial
    n_cued = sum(p["block"] == "cued_blank" for p in ls.provenance)
    assert n_cued == 25 * (int((TRIAL_S - CFG.decoder.window_s) / 0.25) + 1)


def test_hidden_cursor_is_labelled_from_start_pos(tmp_path):
    """blank_feedback: the cursor moves (and is logged) but is not drawn, so the person can only aim
    from where it started; the label must be start_pos → goal, never the unseen cursor → goal."""
    lg = RunLogger(str(tmp_path), CFG, tag="hidden")
    sim = SimulatedEEG(CFG.hardware, seed=4)
    t0, start, goal = 500.0, np.array([0.1, -0.2]), np.array([0.7, 0.4])
    now = lambda: t0 + sim.n_sent / FS
    lg.eeg(sim.generate(int(SETTLE_S * FS)), t0 + np.arange(int(SETTLE_S * FS)) / FS)
    lg.event("trial_start", {"trial": 0, "goal": goal.tolist(), "block": "blank_feedback",
                             "feedback_visible": False, "start_pos": start.tolist()}, t=now())
    pos = start.copy()
    for _ in range(int(TRIAL_S / 0.1)):
        lg.cursor(now(), pos, np.zeros(2), 0.5, -1, 0.0)
        ts = now() + np.arange(FS // 10) / FS
        lg.eeg(sim.generate(FS // 10), ts)
        pos = pos + np.array([0.0, -0.05])                       # drifts away, unseen
    lg.event("trial_end", {"trial": 0, "outcome": "timeout"}, t=now())
    lg.close()
    ls = Labeler(CFG).label(SessionStore(str(tmp_path)).load(lg.dir))
    expect = (goal - start) / np.linalg.norm(goal - start)
    assert len(ls) == int((TRIAL_S - CFG.decoder.window_s) / 0.25) + 1
    assert all(p["block"] == "blank_feedback" and not p["feedback_visible"] for p in ls.provenance)
    assert np.allclose(ls.labels, expect, atol=1e-12)


def test_unknown_block_is_rejected(tmp_path):
    lg = RunLogger(str(tmp_path), CFG, tag="bad")
    lg.event("trial_start", {"goal": [0, 1], "block": "mystery", "feedback_visible": False}, t=1.0)
    lg.close()
    with pytest.raises(ValueError, match="mystery"):
        SessionStore(str(tmp_path)).load(lg.dir)


def test_only_eeg_reaches_the_trainer(runs, tmp_path):
    """Pure-EEG rule: the store opens only the four run files (an audit hook sees every open), the
    windows are invariant to the auditor files' content, and the trainer refuses non-EEG windows."""
    root, a, _ = runs
    opened = []
    recording = [True]
    sys.addaudithook(lambda ev, args: opened.append(str(args[0]))
                     if recording[0] and ev == "open" and isinstance(args[0], (str, bytes, os.PathLike)) else None)
    try:
        ls1 = Labeler(CFG).label(SessionStore(root).load(a))
    finally:
        recording[0] = False
    in_run = {os.path.basename(p) for p in opened if os.path.dirname(os.path.abspath(p)) == os.path.abspath(a)}
    assert in_run == set(RUN_FILES)
    for name in AUDITORS:                                        # rewrite auditors: nothing changes
        np.full((2000, 4), 1e6, np.float32).tofile(os.path.join(a, name))
    ls2 = Labeler(CFG).label(SessionStore(root).load(a))
    assert ls1.data_hash() == ls2.data_hash()
    gaze = [np.zeros((250, 4), np.float32)] * 10                 # an eye-tracker-shaped "window"
    bad = LabelledSet(gaze, np.tile([1.0, 0.0], (10, 1)),
                      [{"block": "cued_blank", "feedback_visible": False, "session": "x", "t": float(i),
                        "trial": i} for i in range(10)])
    with pytest.raises(ValueError, match="only EEG"):
        Trainer(CFG).fit(bad)
    with pytest.raises(ValueError, match="only EEG"):
        Trainer(CFG).fit(ls1, population=[(gaze, np.tile([1.0, 0.0], (10, 1)))])


def test_weighted_path_is_the_plain_estimator_and_weight_rule(runs):
    root, a, _ = runs
    ls = Labeler(CFG).label(SessionStore(root).load(a))
    plain = VelocityDecoder(CFG.decoder, FS).fit(ls.windows, ls.labels)
    # no population, no label corrections → all weights 1 and raw labels: exactly the plain fit
    weighted = Trainer(CFG, recenter_visible=False, max_free_share=1.0).fit(ls)
    assert isinstance(weighted, WeightedVelocityDecoder)
    assert np.allclose(plain.predict_many(ls.windows[:20]), weighted.predict_many(ls.windows[:20]), atol=1e-8)
    assert personal_weight(0) == 1.0 and personal_weight(300) == 1.0 and personal_weight(3000) == 10.0
    tr = Trainer(CFG)
    tr.fit(ls.take(range(100)), population=[(ls.windows[100:], ls.labels[100:])])
    n_pop = len(ls) - 100
    assert tr.last_fit_info["n_population"] == n_pop
    assert tr.last_fit_info["w_own"] == max(1.0, n_pop / 300)
    # the prior is a fixed 300-window pseudo-count once n_pop ≥ 300: its share ignores n_pop
    for n_own in (100, 300, 2000):
        shares = [n_pop / (n_pop + n_own * personal_weight(n_pop)) for n_pop in (300, 1500, 30000)]
        assert np.allclose(shares, 300 / (n_own + 300))


def _visible(ls, labels, block="closed_loop", session="s"):
    return LabelledSet(ls.windows, labels, [{"block": block, "feedback_visible": True, "session": session,
                                             "t": float(i), "trial": i // 10} for i in range(len(ls))])


def test_visible_labels_are_recentred_and_free_use_is_capped(runs):
    """Closed-loop labels point against the deployed decoder's bias (goal − where the bias parked the
    cursor).  Fitted raw, a decoder learns that bias in reverse even from EEG with no intent at all;
    re-centring per block removes it.  Hidden-cursor labels are left alone."""
    root, a, _ = runs
    ls = Labeler(CFG).label(SessionStore(root).load(a))
    rng = np.random.default_rng(3)
    bias = np.array([0.15, -0.6])                                 # display-driven mean label
    y = rng.normal(0, 0.5, (len(ls), 2)) + bias                   # independent of the EEG
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    vis = _visible(ls, y)
    raw = Trainer(CFG, recenter_visible=False).fit(vis).predict_many(ls.windows).mean(axis=0)
    fixed = Trainer(CFG).fit(vis).predict_many(ls.windows).mean(axis=0)
    assert np.dot(raw, bias) / np.linalg.norm(bias) > 0.3          # the bias is learned as an intercept
    assert np.linalg.norm(fixed) < 0.02                           # and is gone after re-centring
    hidden = LabelledSet(vis.windows, vis.labels, [dict(p, feedback_visible=False, block="blank_feedback")
                                                   for p in vis.provenance])
    assert np.array_equal(Trainer.recentred_labels(hidden), hidden.labels)
    two = LabelledSet.concat([_visible(ls.take(range(50)), y[:50], session="s1"),
                              _visible(ls.take(range(50, 100)), y[50:100], session="s2")])
    rc = Trainer.recentred_labels(two)
    assert np.allclose(rc[:50].mean(axis=0), 0) and np.allclose(rc[50:].mean(axis=0), 0)
    # free use: success-selected hindsight labels count for at most max_free_share of own weight
    free = LabelledSet.concat([LabelledSet(ls.windows[:20], ls.labels[:20],
                                           [dict(p, block="cued_blank") for p in ls.provenance[:20]]),
                               _visible(ls.take(range(20, 100)), ls.labels[20:100], block="free")])
    tr = Trainer(CFG)
    tr.fit(free)
    assert tr.last_fit_info["free_weight_share"] == pytest.approx(0.2)
    assert Trainer(CFG, max_free_share=0.9).free_weight(free) == 1.0  # below the cap: untouched


def test_more_data_is_promoted_shuffled_is_not_and_heldout_never_trained(runs, tmp_path):
    root, a, b = runs
    trained_keys = []

    class RecordingTrainer(Trainer):
        def fit(self, data, population=None):
            trained_keys.append(set(data.keys()))
            return super().fit(data, population)

    reg = ModelRegistry(str(tmp_path / "registry"))
    store = SessionStore(root)
    # run a has 25 cued trials, so its newest 20 % is 5 trials (the default minimum is 8)
    judge = lambda: ChampionChallenger(margin=0.02, min_heldout_trials=5)
    eng = DataEngine(store, Labeler(CFG), RecordingTrainer(CFG), reg, judge(), min_new_windows=100)
    n_a = eng.ingest(a)
    assert eng.ingest(a) == 0                                    # idempotent
    r1 = eng.maybe_retrain()
    assert r1["promoted"] and r1["champion_cosine"] == 0.0       # first champion must beat the null
    assert r1["challenger_cosine"] > r1["floor"] and r1["p_value"] <= 0.05
    h1 = eng.judge.heldout
    assert h1 and not (h1 & trained_keys[-1])
    assert all(k[0] == os.path.basename(a) for k in h1)
    cued_a = [k for k, p in zip(eng.data.keys(), eng.data.provenance) if p["block"] == "cued_blank"]
    assert set(cued_a[-len(h1):]) == h1                          # the most recent cued_blank windows
    assert len(h1) >= 0.2 * len(cued_a)

    assert eng.maybe_retrain()["promoted"] is False              # no new data: no retrain
    n_b = eng.ingest(b)
    assert n_b > 4 * n_a
    r2 = eng.maybe_retrain()
    assert r2["promoted"], r2                                    # more data beats the champion on H1 ∪ F
    assert r2["ci95"][0] > 0.02                                  # trial-level paired bootstrap bound
    # the test set is H1 plus the fresh cued trials of run b, which the champion never trained on
    assert r2["n_fresh"] > 0 and r2["n_heldout"] == len(h1) + r2["n_fresh"]
    assert not (h1 & trained_keys[-1])
    h2 = eng.judge.heldout
    assert not (h2 & trained_keys[-1])                           # new champion never saw the new frozen set
    assert all(k[0] == os.path.basename(b) for k in h2)

    # A challenger trained on shuffled labels (same windows, same exclusions) is not promoted.
    data = eng.data
    train = eng.judge.training_set(data, None)
    perm = np.random.default_rng(CFG.eval.seed).permutation(len(train))
    shuffled = LabelledSet(train.windows, train.labels[perm], train.provenance)
    verdict = eng.judge.challenge(eng.champion, Trainer(CFG).fit(shuffled), data)
    assert not verdict["promoted"]
    assert abs(verdict["challenger_cosine"]) < 0.15 < verdict["champion_cosine"]

    # Registry: append-only versions with the required metadata; champion log restores on restart.
    assert reg.versions() == [1, 2]
    meta = reg.meta(2)
    for k in ("n_windows", "provenance_mix", "data_hash", "metrics", "w_own", "n_population"):
        assert k in meta
    assert meta["provenance_mix"]["block"]["cued_blank"] > 0 and meta["promoted"]
    assert [h["version"] for h in reg.champion_history()] == [1, 2]
    eng2 = DataEngine(store, Labeler(CFG), Trainer(CFG), reg, judge(), min_new_windows=100)
    assert eng2.champion_version == 2 and eng2.judge.heldout == h2
    assert eng2.champion_sessions == eng.champion_sessions == {os.path.basename(a), os.path.basename(b)}
    eng2.ingest_all()                                            # served from the label cache
    assert eng2.data.data_hash() == data.data_hash()
    probe = data.take(range(5)).windows
    assert np.allclose(eng2.champion.predict_many(probe), eng.champion.predict_many(probe))


def test_mean_cosine_null_and_perfect():
    y = np.array([[1.0, 0.0], [0.0, -1.0]])
    assert mean_cosine(np.zeros((2, 2)), y) == 0.0
    assert mean_cosine(2 * y, y) == pytest.approx(1.0)
    assert mean_cosine(-y, y) == pytest.approx(-1.0)


class _Fixed:
    """A 'decoder' whose output for window i is row i of a table (windows carry their index)."""

    def __init__(self, table):
        self.table = np.asarray(table, float)

    def predict_many(self, windows):
        return self.table[[int(w[0, 0]) for w in windows]]


def _cued_set(rng, n_trials=12, per_trial=8):
    """Trial-constant cue labels from 8 directions, as cued_blank produces; tiny index windows."""
    ang = 2 * np.pi * rng.integers(0, 8, n_trials) / 8
    labels = np.repeat(np.stack([np.cos(ang), np.sin(ang)], 1), per_trial, axis=0)
    n = len(labels)
    prov = [{"block": "cued_blank", "feedback_visible": False, "session": "s", "t": float(i),
             "trial": i // per_trial} for i in range(n)]
    return LabelledSet([np.full((1, 1), i, np.float32) for i in range(n)], labels, prov)


def _trial_noise(rng, n_trials, per_trial, offset=(0.0, 0.0)):
    """No-signal output: a per-trial random direction (outputs drift slowly, like real decoders)
    plus a constant offset (drift-induced bias), never related to the label."""
    return np.repeat(rng.normal(0, 0.3, (n_trials, 2)), per_trial, axis=0) + np.asarray(offset)


def test_promotion_rule_is_calibrated_at_the_trial_level():
    """No-signal decoders must not be promoted more than ~5 % of the time.  The old rule (mean cosine
    beats the champion by 0.02) promoted a no-signal first champion whenever its constant offset
    happened to align with a small test set's mean label, and swapped equal decoders on noise."""
    n_seeds, n_trials, per_trial = 200, 12, 8
    first, swap, old_first, old_swap, power = [], [], [], [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        test = _cued_set(rng, n_trials, per_trial)
        keys = frozenset(test.keys())
        offset = rng.normal(0, 0.3, 2)
        null_a = _Fixed(_trial_noise(rng, n_trials, per_trial, offset))
        null_b = _Fixed(_trial_noise(rng, n_trials, per_trial, offset))
        judge = ChampionChallenger(seed=seed)
        v = judge.challenge(None, null_a, test, candidate=keys)
        first.append(v["promoted"])
        old_first.append(v["challenger_cosine"] > 0.02)
        judge.freeze(keys)
        v = judge.challenge(null_a, null_b, test)
        swap.append(v["promoted"])
        old_swap.append(v["challenger_cosine"] - v["champion_cosine"] > 0.02)
        good = _Fixed(test.labels + _trial_noise(rng, n_trials, per_trial))
        power.append(judge.challenge(null_a, good, test)["promoted"])
    assert np.mean(first) <= 0.07 and np.mean(swap) <= 0.07, (np.mean(first), np.mean(swap))
    assert np.mean(old_first) > 0.3 and np.mean(old_swap) > 0.2   # what the margin-only rule did
    assert np.mean(power) > 0.9                                   # a real improvement still wins
    assert constant_ceiling(np.tile([0.0, 1.0], (5, 1))) == pytest.approx(1.0)


def test_cue_offset_and_cursor_source_are_enforced(tmp_path):
    """A cued trial that starts while the cue may still be on screen, and visible-cursor trials whose
    cursor.csv is the loop's internal model (not what was displayed), are refused."""
    def run(tag, events):
        lg = RunLogger(str(tmp_path), CFG, tag=tag)
        sim = SimulatedEEG(CFG.hardware, seed=1)
        lg.eeg(sim.generate(6 * FS), 100.0 + np.arange(6 * FS) / FS)
        for t, kind, payload in events:
            lg.event(kind, payload, t=t)
        lg.close()
        return SessionStore(str(tmp_path)).load(lg.dir)

    cued = {"trial": 0, "goal": [0.6, 0.0], "block": "cued_blank", "feedback_visible": False}
    ok = run("ok", [(101.0, "cue", {"seq": 0}), (102.0, "cue_offset", {"seq": 0}),
                    (102.5, "trial_start", cued), (105.0, "trial_end", {"trial": 0})])
    assert len(ok.trials) == 1
    with pytest.raises(ValueError, match="cue_offset"):
        run("no_offset", [(101.0, "cue", {"seq": 0}), (102.5, "trial_start", cued),
                          (105.0, "trial_end", {"trial": 0})])
    visible = {"trial": 0, "goal": [0.6, 0.0], "block": "closed_loop", "feedback_visible": True}
    for source, refused in (("model", True), ("effector", False)):
        sess = run(source, [(100.5, "realtime_start", {"cursor_source": source}),
                            (102.5, "trial_start", visible), (105.0, "trial_end", {"trial": 0})])
        if refused:
            with pytest.raises(ValueError, match="cursor_source"):
                Labeler(CFG).label(sess)
        else:
            assert len(Labeler(CFG).label(sess)) > 0


# ----------------------------------------------------------------------------------------------
# Champion: label sources, body labels, and the deployed / shadow trainers
# ----------------------------------------------------------------------------------------------
from dataclasses import replace as _replace                                    # noqa: E402

from neural_cursor import data_engine as DE                                     # noqa: E402
from neural_cursor import protocol as P                                        # noqa: E402
from neural_cursor.body_decoder import BodyCoordinateDecoder                   # noqa: E402
from neural_cursor.data_engine import (BodyTrainer, ShadowBodyDecoder, ShadowTrainer, body_labels,  # noqa: E402
                                       parse_trials, refit_body_labels)
from neural_cursor.mapping import Mapping                                      # noqa: E402
from neural_cursor.protocol import Block                                       # noqa: E402

CFG1 = _replace(CFG, decoder=_replace(CFG.decoder, window_s=1.0))   # the body decoder's 1-s windows
LEGACY_KEYS = {"block", "feedback_visible", "session", "t", "trial", "display_goal"}
N_L1, N_L2B = 32, 24
L1_ABORTED_AT, L2B_ABORTED_AT = 5, 3                                # trial indices aborted by the protocol


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def make_champion_run(root, seed=3, t0=2000.0):
    """One champion session in miniature, logged through RunLogger.  It holds L1 fixation probes (one
    lost to a fixation break, each logged 0.5 s past its 4.0-s label end), L2b blind closed loop (one
    aborted, each followed by its outcome flash), a blind just-intend trial, S trials with one pixel
    replay, held-out C2c (rot+90) and C2b (mirror_y, keyed report) trials, and an original-protocol
    cued_blank trial.  The simulated intent is the body vector the person should produce."""
    lg = RunLogger(root, CFG1, tag="champion")
    sim = SimulatedEEG(CFG1.hardware, seed=seed)
    now = lambda: t0 + sim.n_sent / FS
    expect = {"cue_t": {}, "aborted": set(), "end": {}, "report": {}, "goal": {}}

    def emit(sec, intent=(0.0, 0.0)):
        n = int(round(sec * FS))
        ts = now() + np.arange(n) / FS                                       # before generate() advances the clock
        sim.set_intent(*np.asarray(intent, float))
        lg.eeg(sim.generate(n), ts)

    emit(SETTLE_S)
    gm = P.glyph_map(1, seed)
    fix, ct = P.FixationProbeTiming(), P.ClosedLoopTiming()
    tid = 0

    l1 = P.BodyCueScheduler(seed, glyphs=gm, logger=lg)
    for i in range(N_L1):
        expect["cue_t"][tid] = now()
        cue = l1.next(t=now())
        emit(fix.cue_s, cue.body)
        lg.event("cue_offset", {"seq": cue.seq}, t=now())
        emit(fix.label_start_s - fix.cue_s, cue.body)
        lg.event("trial_start", P.champion_trial_start_payload(Block("fixation_probe", 900), tid, cue=cue), t=now())
        if i == L1_ABORTED_AT:
            emit(1.2, cue.body)
            lg.event("fixation_break", {"trial": tid}, t=now())
            expect["aborted"].add(tid)
        emit(fix.label_end_s - fix.label_start_s + 0.5 - (1.2 if i == L1_ABORTED_AT else 0.0), cue.body)
        lg.event("trial_end", {"trial": tid}, t=now())
        emit(1.0)
        tid += 1

    l2 = P.BodyCueScheduler(seed + 1, classes=P.L2B_CLASSES_XY, modes=P.BODY_CUE_MODES, glyphs=gm, logger=lg)
    blind = Block("blind_closed_loop", 1200)
    for i in range(N_L2B):
        expect["cue_t"][tid] = now()
        cue = l2.next(t=now())
        emit(ct.cue_s, cue.body)
        lg.event("cue_offset", {"seq": cue.seq}, t=now())
        emit(ct.label_start_s - ct.cue_s, cue.body)                        # the offset lockout
        p = P.champion_trial_start_payload(blind, tid, cue=cue)
        lg.event("trial_start", p, t=now())
        pos = np.zeros(2)
        for _ in range(25):                                                  # the hidden cursor moves (logged)
            lg.cursor(now(), pos, np.zeros(2), 0.5, -1, 0.0)
            emit(0.1, cue.body)
            pos = pos + 0.02 * _unit(cue.body)
        aborted = i == L2B_ABORTED_AT
        lg.event("trial_end", {"trial": tid, "aborted": True} if aborted else {"trial": tid}, t=now())
        if aborted:
            expect["aborted"].add(tid)
        expect["end"][tid] = now()
        lg.event("outcome_flash", {"hit": False}, t=now())
        emit(ct.outcome_s + 0.5)
        tid += 1

    word = P.CueScheduler(seed=seed, logger=lg).next(t=now(), mode="auditory")  # blind just intend: goal word
    expect["cue_t"][tid] = now()
    emit(ct.cue_s, word.vector)
    lg.event("cue_offset", {"seq": word.seq}, t=now())
    emit(ct.label_start_s - ct.cue_s, word.vector)
    lg.event("trial_start", P.champion_trial_start_payload(Block("blind_closed_loop", 300, just_intend=True),
                                                           tid, cue=word), t=now())
    emit(2.5, word.vector)
    lg.event("trial_end", {"trial": tid}, t=now())
    emit(1.0)
    tid += 1

    rng = np.random.default_rng(seed)
    visible = ([(Block("sighted", 900), None, False)] * 6 + [(Block("sighted", 900), "replay", False)]
               + [(Block("heldout_mapping", 240, mapping="rot+90"), None, False)] * 3
               + [(Block("heldout_mapping", 240, mapping="mirror_y", variant="C2b"), None, True)] * 2)
    for k, (block, replay, frozen) in enumerate(visible):
        a = rng.uniform(0, 2 * np.pi)
        goal = 0.7 * np.array([np.cos(a), np.sin(a)])
        replay_goal = -goal if replay else None                              # the replay aims elsewhere
        emit(ct.label_start_s)                                               # target onset → trial_start
        lg.event("trial_start", P.champion_trial_start_payload(block, tid, goal=goal, replay_goal=replay_goal),
                 t=now())
        expect["goal"][tid] = goal
        M, pos = Mapping(block.mapping), np.zeros(2)
        for _ in range(25):
            lg.cursor(now(), pos, np.zeros(2), 0.5, -1, 0.0)
            emit(0.1, M.screen_to_body(_unit(goal - pos)))                   # body intent toward the goal
            if not frozen:
                pos = pos + 0.025 * _unit((replay_goal if replay else goal) - pos)
        lg.event("trial_end", {"trial": tid}, t=now())
        if frozen:
            cls = ("FEET", "RIGHT")[k % 2]
            lg.event("body_report", {"trial": tid, "class": cls}, t=now() + 0.5)
            expect["report"][tid] = list(P.body_vector(cls))
        emit(1.5)
        tid += 1

    lg.event("trial_start", {"trial": tid, "goal": [0.6, 0.0], "block": "cued_blank", "feedback_visible": False},
             t=now())
    emit(2.5, (1.0, 0.0))
    lg.event("trial_end", {"trial": tid}, t=now())
    emit(1.0)
    lg.close()
    return lg.dir, expect


@pytest.fixture(scope="module")
def champion(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("champion"))
    run, expect = make_champion_run(root)
    sess = SessionStore(root).load(run)
    return sess, Labeler(CFG1).label(sess), expect


def _by_source(ls, *sources):
    return ls.mask(np.array([p.get("source") in sources for p in ls.provenance], bool))


def test_champion_labels_sources_and_provenance(champion):
    """Every window carries its source.  L1/L2b labels are the cued body vector, and their windows
    lie ≥ 1.0 s after the cue (L1: ≤ 4.0 s after it) and before the outcome flash.  Aborted trials
    are gone.  Drawn-target windows keep the screen rule and expose the goal, mapping, displayed
    cursor, variant, report and replay goal.  Original-protocol windows keep their provenance."""
    sess, ls, ex = champion
    srcs = [p.get("source") for p in ls.provenance]
    assert set(srcs) == {"L1", "L2b", "J", "S", "R", "H", None}
    assert {p["trial"] for p in ls.provenance}.isdisjoint(ex["aborted"])
    assert len({p["trial"] for p in ls.provenance if p.get("source") == "L1"}) == N_L1 - 1
    assert len({p["trial"] for p in ls.provenance if p.get("source") == "L2b"}) == N_L2B - 1
    assert ls.mix()["source"]["L1"] > 0
    win = CFG1.decoder.window_s
    for lab, p in zip(ls.labels, ls.provenance):
        src = p.get("source")
        if src is None:
            assert set(p) == LEGACY_KEYS                                     # original blocks: unchanged
            continue
        if src == "H":
            assert p["mapping"] == {"C2c": "rot+90", "C2b": "mirror_y"}[p["variant"]]
        if src in ("L1", "L2b"):
            assert p["display_goal"] is None and "cursor_pos" not in p
            assert np.allclose(lab, _unit(p["body_label"]))
            cue_t = ex["cue_t"][p["trial"]]
            assert p["t"] - win >= cue_t + P.CUE_LOCKOUT_S - 1.0 / FS        # cue-evoked response excluded
            if src == "L1":
                assert sum(v != 0 for v in p["body_label"]) == 1
                assert p["t"] <= cue_t + P.L1_LABEL_END_S + 1e-6             # clipped at the label end
            else:
                assert p["t"] <= ex["end"][p["trial"]] + 1e-6                # before the outcome flash
        elif src == "J":
            assert p["display_goal"] is None and "body_label" not in p
        else:
            goal = ex["goal"][p["trial"]]
            assert np.allclose(p["display_goal"], goal) and p["feedback_visible"]
            assert np.allclose(lab, _unit(goal - np.array(p["cursor_pos"])), atol=1e-6)
            # the cursor at the window START (exogenous to the window's own output) is logged too; in
            # S / H the cursor moves toward the goal, so it is never nearer the goal than at the end
            start = np.array(p["cursor_pos_start"])
            if src != "R":
                assert np.linalg.norm(goal - start) >= np.linalg.norm(goal - np.array(p["cursor_pos"])) - 1e-9
            if src == "H":
                assert p["body_report"] == ex["report"].get(p["trial"])
            if src == "R":
                assert np.allclose(p["replay_goal"], -goal) and p["block"] == "pixel_replay"
    # both ends of the L1 window range are used: the first window starts at cue + 1.0 s and the
    # last one ends at cue + 4.0 s
    l1 = _by_source(ls, "L1")
    first = min(p["t"] - ex["cue_t"][p["trial"]] for p in l1.provenance)
    last = max(p["t"] - ex["cue_t"][p["trial"]] for p in l1.provenance)
    assert abs(first - (P.CUE_LOCKOUT_S + win)) < 2.0 / FS and abs(last - P.L1_LABEL_END_S) < 2.0 / FS
    # ReFIT body labels apply M⁻¹: under rot+90 a cursor-up label is a body-right (x +) intent
    h = _by_source(ls, "H")
    for u, b, p in zip(h.labels, refit_body_labels(h), h.provenance):
        assert np.allclose(b, Mapping(p["mapping"]).M.T @ u)
    assert np.allclose(refit_body_labels(_by_source(ls, "S")), _by_source(ls, "S").labels)  # identity


def test_champion_protocol_violations_raise():
    gm = P.glyph_map(1, 0)
    cue = P.BodyCueScheduler(0, glyphs=gm).next()
    l1 = P.champion_trial_start_payload(Block("fixation_probe", 300), 0, cue=cue)
    on = (0.0, "cue", {"seq": 0, "body": list(cue.body)})
    off = (0.5, "cue_offset", {"seq": 0})
    end = (4.0, "trial_end", {"trial": 0})
    assert len(parse_trials("s", [on, off, (1.0, "trial_start", l1), end])) == 1
    bad = {
        "after cue onset": [on, off, (0.9, "trial_start", l1), end],             # inside the 1.0-s lockout
        "after the cue ends": [on, (0.75, "cue_offset", {"seq": 0}), (1.0, "trial_start", l1), end],  # 0.25 s
        "no 'cue' event": [(1.0, "trial_start", l1), end],
        "cue_offset": [on, (1.0, "trial_start", l1), end],
        "differs": [(0.0, "cue", {"seq": 0, "body": [0, -1] if list(cue.body) != [0, -1] else [0, 1]}), off,
                    (1.0, "trial_start", l1), end],
        "one class": [on, off, (1.0, "trial_start", dict(l1, body_cue=[1, 1])), end],
        "source": [on, off, (1.0, "trial_start", dict(l1, source="S")), end],
        "feedback_visible must": [on, off, (1.0, "trial_start", dict(l1, feedback_visible=True)), end],
        "runs under": [(0.0, "trial_start", P.champion_trial_start_payload(Block("sighted", 60), 0, goal=(0.5, 0))
                                          | {"mapping": "rot+90"}), end],
        "held-out mapping": [(0.0, "trial_start", P.champion_trial_start_payload(
            Block("heldout_mapping", 60), 0, goal=(0.5, 0)) | {"mapping": "identity"}), end],
        "replay_goal": [(0.0, "trial_start", {k: v for k, v in P.champion_trial_start_payload(
            Block("sighted", 60), 0, goal=(0.5, 0), replay_goal=(0, 0.5)).items() if k != "replay_goal"}), end],
        "needs its goal": [(0.0, "trial_start", P.champion_trial_start_payload(Block("sighted", 60), 0, goal=(0.5, 0))
                                               | {"goal": None}), end],
        "outcome": [on, (1.0, "cue_offset", {}), (1.5, "trial_start", P.champion_trial_start_payload(
            Block("blind_closed_loop", 60), 0, cue=cue)), (2.0, "outcome_flash", {}), end],
        "keypress": [(0.0, "trial_start", P.champion_trial_start_payload(
            Block("heldout_mapping", 60, variant="C2b"), 0, goal=(0.5, 0))),
            (1.0, "body_report", {"trial": 0, "class": "FEET"}), end],
    }
    for match, events in bad.items():
        with pytest.raises(ValueError, match=match):
            parse_trials("s", events)
    # an aborted trial and a fixation break both discard; a break in a sighted trial does not
    assert parse_trials("s", [on, off, (1.0, "trial_start", l1), (4.0, "trial_end", {"trial": 0, "aborted": True})]) == []
    assert parse_trials("s", [on, off, (1.0, "trial_start", l1), (2.0, "fixation_break", {}), end]) == []
    s = P.champion_trial_start_payload(Block("sighted", 60), 0, goal=(0.5, 0))
    assert len(parse_trials("s", [(0.0, "trial_start", s), (1.0, "fixation_break", {}), end])) == 1


def test_deployed_trainer_uses_only_L1_L2b(champion):
    """The deployed body decoder's trainer refuses S, H, R and J windows, and original-protocol
    windows, whether alone or mixed in.  On the L1/L2b selection it trains, and the y axis it learns
    from cue-defined labels decodes held-out fixation probes.  The original Trainer never selects a
    champion window."""
    _, ls, _ = champion
    trainer = BodyTrainer(CFG1)
    for src in ("S", "H", "R", "J", None):
        part = ls.mask(np.array([p.get("source") == src for p in ls.provenance], bool))
        assert len(part) > 0
        with pytest.raises(ValueError, match="trains only on"):
            trainer.fit(part)
        with pytest.raises(ValueError, match="trains only on"):
            trainer.fit(LabelledSet.concat([BodyTrainer.select(ls), part]))
    sel = BodyTrainer.select(ls)
    assert {p["source"] for p in sel.provenance} == {"L1", "L2b"}
    l1_trials = sorted({p["trial"] for p in sel.provenance if p["source"] == "L1"})
    test_trials = set(l1_trials[len(l1_trials) // 2:])
    is_test = np.array([p["trial"] in test_trials for p in sel.provenance])
    seen = []

    class Spy(BodyCoordinateDecoder):
        def fit(self, windows, labels, provenance, session_references=None, **kw):
            seen.extend(p["source"] for p in provenance)
            assert all(np.shape(w) == (FS, CFG1.hardware.n_channels) for w in windows)   # only EEG
            return super().fit(windows, labels, provenance, session_references, **kw)

    import neural_cursor.data_engine as de
    original = de.BodyCoordinateDecoder
    de.BodyCoordinateDecoder = Spy
    try:
        dec = trainer.fit(sel.mask(~is_test))
    finally:
        de.BodyCoordinateDecoder = original
    assert set(seen) == {"L1", "L2b"} and isinstance(dec, BodyCoordinateDecoder)
    assert set(trainer.last_fit_info["train_mix"]["source"]) == {"L1", "L2b"}
    test = sel.mask(is_test)
    y = body_labels(test)[:, 1]
    out = dec.predict_many(test.windows)[:, 1]
    m = y != 0
    assert np.mean(np.sign(out[m]) == y[m]) > 0.8                           # feet vs hands from cue labels
    gaze = LabelledSet([np.zeros((FS, 4), np.float32)] * len(sel), sel.labels, sel.provenance)
    with pytest.raises(ValueError, match="only EEG"):
        trainer.fit(gaze)
    assert not any(p.get("source") for p in Trainer(CFG1).select(ls).provenance)


def test_shadow_trainer_uses_S_with_refit_body_labels(champion):
    """The shadow positive control trains only on S, with ReFIT labels in body coordinates given a sign
    per axis beyond sin 22.5°.  It is the deployed model class, takes the deployed decoder's x status,
    and cannot earn x on its own sighted data."""
    _, ls, _ = champion
    shadow = ShadowTrainer(CFG1)
    for src in ("L1", "L2b", "H", "R", "J"):
        with pytest.raises(ValueError, match="trains only on"):
            shadow.fit(_by_source(ls, src))
    s = ShadowTrainer.select(ls)
    assert {p["source"] for p in s.provenance} == {"S"}
    lab = shadow.axis_labels(s)
    u = refit_body_labels(s)
    assert np.array_equal(lab, np.where(np.abs(u) >= np.sin(np.pi / 8), np.sign(u), 0).astype(int))
    assert np.any(lab[:, 1] > 0) and np.any(lab[:, 1] < 0)
    dec = shadow.fit(s, x_enabled=True)
    assert isinstance(dec, ShadowBodyDecoder) and isinstance(dec, BodyCoordinateDecoder) and dec.x_enabled
    assert dec.predict(s.windows[0]).shape == (2,)
    assert shadow.last_fit_info["role"] == "shadow" and set(shadow.last_fit_info["train_mix"]["source"]) == {"S"}
    assert not ShadowTrainer(CFG1).fit(s).x_enabled
    with pytest.raises(ValueError):
        dec.earn_x(s.windows, lab, s.provenance)                            # x is earned on L1/L2b only
    with pytest.raises(ValueError, match="trains only on"):
        ShadowBodyDecoder(CFG1, FS).fit(_by_source(ls, "L1").windows, body_labels(_by_source(ls, "L1")),
                                        _by_source(ls, "L1").provenance)


# ----------------------------------------------------------------------------------------------
# Body metric (per-axis AUC on held-out L1 selection probes) and the deployed decoder end to end
# ----------------------------------------------------------------------------------------------
from neural_cursor.data_engine import BODY_METRIC, axis_outputs, body_auc, body_outputs   # noqa: E402

L1_BODY = ((0, 1), (0, -1), (-1, 0), (1, 0))                                   # FEET, HANDS, LEFT, RIGHT


class _FixedBody(BodyCoordinateDecoder):
    """A body decoder whose output for window i is row i of a table (windows carry their index)."""

    def __init__(self, table, x_enabled=False):
        super().__init__(CFG1, FS)
        self.table = np.asarray(table, float)
        self.x_enabled = x_enabled

    def predict_many(self, windows):
        return self.table[[int(w[0, 0]) for w in windows]]


class _GatedBody(_FixedBody):
    """A fixed body decoder whose idle gate, when armed, zeroes every other window."""

    def predict_many(self, windows):
        out = super().predict_many(windows).copy()
        if self._idle_model is not None:
            out[::2] = 0.0
        return out


def _probe_set(rng, n_trials=24, per_trial=5, session="s"):
    """L1 fixation probes, classes balanced in shuffled blocks of four, index windows."""
    order = np.concatenate([rng.permutation(4) for _ in range(n_trials // 4)])
    body = np.repeat(np.array(L1_BODY)[order], per_trial, axis=0)
    prov = [{"block": "fixation_probe", "feedback_visible": False, "session": session, "t": float(i),
             "trial": i // per_trial, "display_goal": None, "source": "L1", "body_label": list(map(int, b))}
            for i, b in enumerate(body)]
    return LabelledSet([np.full((1, 1), i, np.float32) for i in range(len(body))], body.astype(float), prov)


def _trial_body_noise(rng, n_trials, per_trial, offset=(0.0, 0.0)):
    return np.clip(np.repeat(rng.normal(0, 0.4, (n_trials, 2)), per_trial, axis=0) + np.asarray(offset), -1, 1)


def test_body_metric_is_calibrated_per_axis_on_L1_probes():
    """A body decoder is judged by per-axis AUC on L1 probes with whole-trial nulls and a class-
    stratified trial bootstrap: no-signal decoders are rarely promoted (first champion or swap), a real
    y improvement is; x counts only when a compared decoder drives it.  The score equals the plain
    window AUC and ignores an output's scale."""
    from sklearn.metrics import roc_auc_score
    n_seeds, n_trials, per_trial = 200, 40, 5
    first, swap, power = [], [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        test = _probe_set(rng, n_trials, per_trial)
        keys = frozenset(test.keys())
        offset = rng.normal(0, 0.3, 2)
        null_a = _FixedBody(_trial_body_noise(rng, n_trials, per_trial, offset))
        null_b = _FixedBody(_trial_body_noise(rng, n_trials, per_trial, offset))
        judge = ChampionChallenger(seed=seed)
        first.append(judge.challenge(None, null_a, test, candidate=keys)["promoted"])
        judge.freeze(keys)
        swap.append(judge.challenge(null_a, null_b, test)["promoted"])
        good = _FixedBody(np.clip(0.6 * test.labels + _trial_body_noise(rng, n_trials, per_trial), -1, 1))
        power.append(judge.challenge(null_a, good, test)["promoted"])
    assert np.mean(first) <= 0.07 and np.mean(swap) <= 0.07, (np.mean(first), np.mean(swap))
    assert np.mean(power) > 0.9, np.mean(power)

    rng = np.random.default_rng(7)
    test = _probe_set(rng)
    table = np.clip(test.labels + rng.normal(0, 0.8, test.labels.shape), -1, 1)
    dec = _FixedBody(table)
    y = test.labels[:, 1] != 0
    assert body_auc(dec, test)["y"] == pytest.approx(roc_auc_score(test.labels[y, 1] > 0, table[y, 1]))
    assert body_auc(_FixedBody(0.1 * table), test)["y"] == body_auc(dec, test)["y"]   # scale-free
    gated = _GatedBody(table)
    gated._idle_model = "armed"                                               # an idle gate zeroes windows
    assert np.all(body_outputs(gated, test)[::2] == 0)
    assert np.array_equal(axis_outputs(gated, test), body_outputs(dec, test))  # the metric ranks the axes
    assert body_auc(gated, test) == body_auc(dec, test)
    judge = ChampionChallenger()
    v = judge.challenge(None, dec, test, candidate=frozenset(test.keys()))
    assert v["metric"] == BODY_METRIC and v["axes"] == ["y"] and v["champion_auc"] == 0.5
    assert v["challenger_auc"] == pytest.approx(v["auc"]["challenger"]["y"]) and v["promoted"]
    vx = judge.challenge(None, _FixedBody(table, x_enabled=True), test, candidate=frozenset(test.keys()))
    assert vx["axes"] == ["x", "y"]
    assert vx["challenger_auc"] == pytest.approx(np.mean(list(vx["auc"]["challenger"].values())))
    nan_rows = table.copy()
    nan_rows[::3] = np.nan                                                    # invalid windows → zero command
    assert np.all(body_outputs(_FixedBody(nan_rows), test)[::3] == 0)
    mixed = LabelledSet.concat([test, LabelledSet(test.windows[:5], test.labels[:5],
                                                  [dict(p, source="L2b") for p in test.provenance[:5]])])
    mixed.provenance[-5:] = [dict(p, t=p["t"] + 1000.0) for p in mixed.provenance[-5:]]
    with pytest.raises(ValueError, match="L1 probes only"):
        judge.challenge(None, dec, mixed, candidate=frozenset(mixed.keys()))


def test_body_heldout_candidate_is_the_newest_L1_probes_with_both_y_classes():
    """H′ for a body decoder is the newest 20 % of L1 windows cut at a trial boundary (L2b and other
    windows are never selection probes); it needs ≥ min_class_trials FEET and HANDS probes."""
    rng = np.random.default_rng(3)
    l1 = _probe_set(rng, n_trials=60)
    l2b = LabelledSet(l1.windows[:50], l1.labels[:50],
                      [dict(p, source="L2b", block="blind_closed_loop", t=p["t"] + 1e4, trial=p["trial"] + 1000)
                       for p in l1.provenance[:50]])
    data = LabelledSet.concat([l1, l2b])
    judge = ChampionChallenger()
    h = judge.candidate_heldout(data, body=True)
    l1_keys = [k for k, p in zip(data.keys(), data.provenance) if p["source"] == "L1"]
    assert h == frozenset(l1_keys[-len(h):]) and len(h) >= 0.2 * len(l1_keys)
    assert judge.candidate_heldout(data) is None                             # no cued_blank windows at all
    only_x = data.mask(np.array([p["source"] == "L2b" or p["body_label"][1] == 0 for p in data.provenance]))
    assert judge.candidate_heldout(only_x, body=True) is None                # no FEET / HANDS probes held out


def make_body_run(root, tag, seed, n_l1, t0, n_l2b=8, n_s=4, gain=6.0):
    """A champion run of the deployed decoder's own data: L1 fixation probes, L2b blind closed loop
    (y classes), and sighted trials (S: never a deployed label).  The simulated person imagines the
    cued body vector; the sighted intent follows the drawn goal."""
    lg = RunLogger(root, CFG1, tag=tag)
    sim = SimulatedEEG(CFG1.hardware, seed=seed, intent_gain_uv=gain)
    now = lambda: t0 + sim.n_sent / FS

    def emit(sec, intent=(0.0, 0.0)):
        n = int(round(sec * FS))
        ts = now() + np.arange(n) / FS
        sim.set_intent(*np.asarray(intent, float))
        lg.eeg(sim.generate(n), ts)

    emit(SETTLE_S)
    gm, fix, ct = P.glyph_map(1, seed), P.FixationProbeTiming(), P.ClosedLoopTiming()
    tid = 0
    l1 = P.BodyCueScheduler(seed, glyphs=gm, logger=lg)
    for _ in range(n_l1):
        cue = l1.next(t=now())
        emit(fix.cue_s, cue.body)
        lg.event("cue_offset", {"seq": cue.seq}, t=now())
        emit(fix.label_start_s - fix.cue_s, cue.body)
        lg.event("trial_start", P.champion_trial_start_payload(Block("fixation_probe", 900), tid, cue=cue), t=now())
        emit(fix.label_end_s - fix.label_start_s, cue.body)
        lg.event("trial_end", {"trial": tid}, t=now())
        emit(fix.iti_s)
        tid += 1
    l2 = P.BodyCueScheduler(seed + 1, classes=P.L2B_CLASSES_Y, modes=P.BODY_CUE_MODES, glyphs=gm, logger=lg)
    for _ in range(n_l2b):
        cue = l2.next(t=now())
        emit(ct.cue_s, cue.body)
        lg.event("cue_offset", {"seq": cue.seq}, t=now())
        emit(ct.label_start_s - ct.cue_s, cue.body)
        lg.event("trial_start", P.champion_trial_start_payload(Block("blind_closed_loop", 900), tid, cue=cue),
                 t=now())
        emit(3.0, cue.body)
        lg.event("trial_end", {"trial": tid}, t=now())
        lg.event("outcome_flash", {"hit": False}, t=now())
        emit(ct.outcome_s + ct.iti_s)
        tid += 1
    rng = np.random.default_rng(seed)
    for _ in range(n_s):
        a = rng.uniform(0, 2 * np.pi)
        goal = 0.7 * np.array([np.cos(a), np.sin(a)])
        emit(ct.label_start_s)
        lg.event("trial_start", P.champion_trial_start_payload(Block("sighted", 900, replay_fraction=0.0), tid,
                                                               goal=goal), t=now())
        pos = np.zeros(2)
        for _ in range(20):
            lg.cursor(now(), pos, np.zeros(2), 0.5, -1, 0.0)
            emit(0.1, _unit(goal - pos))
            pos = pos + 0.02 * _unit(goal - pos)
        lg.event("trial_end", {"trial": tid}, t=now())
        emit(1.0)
        tid += 1
    lg.close()
    return lg.dir


class _ReadLog(AbcMapping):
    """A read-only provenance mapping that records every key any code path observes: item access and
    ``get`` (Mapping's ``get`` goes through ``__getitem__``), ``in``, ``items``/``values`` and plain
    iteration or ``keys`` (which hand out the names, so ``dict(p)``, ``{**p}``, copies and JSON dumps all
    count).  The old probe subclassed dict and logged only ``[]`` and ``get``, so a read through
    iteration bypassed it (review)."""
    reads = set()

    def __init__(self, base, **extra):
        self._d = dict(base, **extra)

    def __getitem__(self, k):
        _ReadLog.reads.add(k)
        return self._d[k]

    def __contains__(self, k):
        _ReadLog.reads.add(k)
        return k in self._d

    def __iter__(self):
        _ReadLog.reads.update(self._d)
        return iter(self._d)

    def __len__(self):
        return len(self._d)


def test_body_decoder_end_to_end_store_label_train_promote(tmp_path):
    """SessionStore → Labeler (1-s windows) → BodyTrainer → fit → promotion, on simulated runs.

    The first challenger must beat the null decoder on the newest L1 probes (body metric); later
    challengers are scored with the champion on the frozen probes plus fresh probes of new sessions,
    and a real improvement over a weak champion is promoted.  Only L1 / L2b windows of non-held-out
    trials reach the trainer, as (500, 32) EEG; S windows never do; the trainer reads no gaze field and
    refuses a drawn goal; the registry restores the body champion and its held-out probes."""
    root = str(tmp_path / "runs")
    a = make_body_run(root, "a", seed=0, n_l1=100, t0=1000.0)
    b = make_body_run(root, "b", seed=50, n_l1=100, t0=9000.0)
    seen = []

    class RecordingBodyTrainer(BodyTrainer):
        def fit(self, data, session_references=None, **kw):
            seen.append({"keys": set(data.keys()), "sources": {p["source"] for p in data.provenance},
                         "shapes": {np.shape(w) for w in data.windows}})
            return super().fit(data, session_references, **kw)

    reg = ModelRegistry(str(tmp_path / "registry"))
    store = SessionStore(root)
    labeler = Labeler(BodyCoordinateDecoder(CFG1, FS).loop_config(CFG), hop_s=0.5)
    eng = DataEngine(store, labeler, RecordingBodyTrainer(CFG1), reg, ChampionChallenger(), min_new_windows=100)
    assert eng.body
    eng.ingest(a)
    r1 = eng.maybe_retrain()
    assert r1["metric"] == BODY_METRIC and r1["axes"] == ["y"], r1
    assert r1["promoted"] and r1["champion_auc"] == 0.5 and r1["challenger_auc"] > r1["floor"], r1
    assert r1["p_value"] <= 0.05
    h1 = eng.judge.heldout
    l1_a = [k for k, p in zip(eng.data.keys(), eng.data.provenance) if p.get("source") == "L1"]
    assert h1 == frozenset(l1_a[-len(h1):])                                   # the newest selection probes
    assert not (h1 & seen[-1]["keys"]) and seen[-1]["sources"] == {"L1", "L2b"}
    assert seen[-1]["shapes"] == {(FS, CFG1.hardware.n_channels)}
    assert "S" in eng.data.mix()["source"]                                   # recorded, never trained on

    # A weak champion (trained on shuffled cue labels) loses to the next challenger on H1 ∪ fresh probes.
    train = BodyTrainer.select(eng.judge.training_set(eng.data, None))
    perm = np.random.default_rng(1).permutation(len(train))
    shuffled = LabelledSet(train.windows, train.labels,
                           [dict(p, body_label=train.provenance[j]["body_label"]) for p, j in zip(train.provenance, perm)])
    eng.champion = BodyTrainer(CFG1).fit(shuffled)
    eng.ingest(b)
    r2 = eng.maybe_retrain()
    assert r2["n_fresh"] > 0 and r2["n_heldout"] == len(h1) + r2["n_fresh"]
    assert r2["promoted"] and r2["ci95"][0] > r2["auc_margin"], r2
    assert r2["challenger_auc"] > r2["champion_auc"] + 0.1
    assert not (h1 & seen[-1]["keys"]) and not (eng.judge.heldout & seen[-1]["keys"])
    assert all(k[0] == os.path.basename(b) for k in eng.judge.heldout)
    assert seen[-1]["sources"] == {"L1", "L2b"}

    # The trainer reads no gaze field and refuses a window that records a drawn goal.
    sel = BodyTrainer.select(eng.data)
    _ReadLog.reads = set()
    logged = LabelledSet(sel.windows, sel.labels,
                         [_ReadLog(p, gaze=[0.1, 0.2], fixation_break=False) for p in sel.provenance])
    with_gaze = BodyTrainer(CFG1).fit(logged)
    assert _ReadLog.reads.isdisjoint({"gaze", "fixation_break"}), _ReadLog.reads
    # and, whatever path a future trainer reads through, its model does not depend on the gaze fields
    without = BodyTrainer(CFG1).fit(sel)
    np.testing.assert_array_equal(with_gaze.predict_many(sel.windows[:20]), without.predict_many(sel.windows[:20]))
    drawn = LabelledSet(sel.windows[:20], sel.labels[:20],
                        [dict(p, display_goal=[0.5, 0.0]) for p in sel.provenance[:20]])
    with pytest.raises(ValueError, match="drawn goal"):
        BodyTrainer(CFG1).fit(drawn)

    # Registry: the body champion and its frozen probes restore on restart.
    eng2 = DataEngine(store, labeler, BodyTrainer(CFG1), reg, ChampionChallenger(), min_new_windows=100)
    assert eng2.champion_version == eng.champion_version == 2 and eng2.judge.heldout == eng.judge.heldout
    assert isinstance(eng2.champion, BodyCoordinateDecoder)
    probe = sel.windows[:5]
    assert np.allclose(eng2.champion.predict_many(probe), eng.champion.predict_many(probe))
    with pytest.raises(ValueError, match="LabelledSet"):
        DataEngine(store, labeler, BodyTrainer(CFG1), reg, population=[(probe, np.zeros((5, 2)))])


def test_l2b_windows_end_by_cue_plus_4s_and_before_the_hidden_cursor_dwells(tmp_path):
    """DESIGN_PURE_EEG §4's L2b window rule (review: L2b windows ran to trial_end, up to 11.5 s after the
    cue, and 79 % ended after 4 s, so the deployed decoder trained mostly on decayed ERD).  Trial A's
    hidden cursor reaches the dwell radius 3.2 s after the cue: no window ends after that.  Trial B never
    gets there and runs to the 10-s timeout: its windows end ≤ 4.0 s after the cue."""
    lg = RunLogger(str(tmp_path), CFG1, tag="l2b")
    sim = SimulatedEEG(CFG1.hardware, seed=5)
    t0 = 500.0
    now = lambda: t0 + sim.n_sent / FS

    def emit(sec):
        n = int(round(sec * FS))
        ts = now() + np.arange(n) / FS                                       # before generate() advances the clock
        lg.eeg(sim.generate(n), ts)

    emit(SETTLE_S)
    ct, blind = P.ClosedLoopTiming(), Block("blind_closed_loop", 1200)
    sched = P.BodyCueScheduler(9, classes=P.L2B_CLASSES_Y, modes=("central_glyph",), glyphs=P.glyph_map(1, 9), logger=lg)
    cue_t, inside_t = {}, None
    for tid, reach in ((0, True), (1, False)):
        cue_t[tid] = now()
        cue = sched.next(t=now())
        emit(ct.cue_s)
        lg.event("cue_offset", {"seq": cue.seq}, t=now())
        emit(ct.label_start_s - ct.cue_s)
        lg.event("trial_start", P.champion_trial_start_payload(blind, tid, cue=cue), t=now())
        goal = np.array(P.hidden_goal(cue))
        for k in range(100):                                               # 10 s of control at 10 Hz
            frac = min(1.0, (k + 1) / 17) if reach else 0.0                  # A: at the goal after 1.7 s
            pos = frac * goal
            if reach and inside_t is None and np.linalg.norm(goal - pos) < CFG1.task.target_radius:
                inside_t = now()
            lg.cursor(now(), pos, np.zeros(2), 0.5, -1, 0.0)
            emit(0.1)
        lg.event("trial_end", {"trial": tid}, t=now())
        lg.event("outcome_flash", {"hit": reach}, t=now())
        emit(2.0)
    lg.close()
    ls = Labeler(CFG1, hop_s=0.25).label(SessionStore(str(tmp_path)).load(lg.dir))
    ends = {tid: [p["t"] for p in ls.provenance if p["trial"] == tid] for tid in (0, 1)}
    assert ends[0] and max(ends[0]) <= inside_t + 1e-6 and inside_t - cue_t[0] < 4.0
    assert ends[1] and max(ends[1]) - cue_t[1] <= DE.CUE_LABEL_END_S + 1e-6
    assert max(ends[1]) - cue_t[1] > DE.CUE_LABEL_END_S - 0.3                # the cap, not the trial, ends it
