"""Purity audit: a clean simulated user passes; an ocular leak, a display leak and feedback circularity
are each caught by the surrogate built for them; the permutation floor keeps its 5 % size on
autocorrelated data; label structure that needs no EEG is never credited as skill; auditor streams
never reach the EEG decoders."""
import numpy as np
import pytest

from neural_cursor.acquisition import SimulatedEEG
from neural_cursor.config import DEFAULT
from neural_cursor.data_engine import CHAMPION_TARGET_BLOCKS, TARGET_BLOCKS
from neural_cursor.purity import (CERTIFYING_BLOCKS, CERTIFYING_SOURCES, MS_LAMBDA, MS_MIN_DURATION_S,
                                  SPATIAL_BLOCKS, PurityAudit, _PermutationCV, auditor_features,
                                  circular_shift_permutations, displayed_velocity, is_certifying,
                                  microsaccade_features, microsaccades, slow_binned_means)
from neural_cursor.signal import CausalFilter, common_average
from sklearn.metrics import roc_auc_score

N_WINDOWS = 300
N_PERM = DEFAULT.eval.n_permutations   # the production null size; the solves are cheap enough
GAZE_FS = 500                          # eye-tracker rate of the simulated gaze segments (Hz)


def _unit(rng, n):
    a = rng.uniform(0, 2 * np.pi, n)
    return np.stack([np.cos(a), np.sin(a)], axis=1)


def _eeg(intents, seed, eye_gain_uv=0.0):
    """One filtered, CAR'd window per intent, generated back to back (time order = list order)."""
    hw = DEFAULT.hardware
    sim = SimulatedEEG(hw, seed=seed, eye_gain_uv=eye_gain_uv)
    filt = CausalFilter(hw, DEFAULT.pre)
    filt(sim.generate(hw.fs * 2))                                # settle the filter
    n_win = int(DEFAULT.decoder.window_s * hw.fs)
    out = []
    for v in intents:
        sim.set_intent(*v)
        out.append(common_average(filt(sim.generate(n_win))))
    return out


def _prov(n, blocks, step_s, goals=None):
    """Provenance in time order; target-drawing blocks get a displayed goal (default: unrelated to the
    labels, i.e. a screen that carries no label information)."""
    goals = np.random.default_rng(99).uniform(-0.8, 0.8, (n, 2)) if goals is None else goals
    out = []
    for i in range(n):
        b = blocks[i % len(blocks)]
        out.append({"block": b, "feedback_visible": b == "closed_loop", "session": "s0", "t": (i + 1) * step_s,
                    "display_goal": goals[i].tolist() if b in TARGET_BLOCKS else None})
    return out


def _wandering_log(rng, n, step_s):
    """A displayed cursor that wanders independently of the labels."""
    t = np.arange(0, n * step_s + 1, 0.1)
    return np.column_stack([t, np.cumsum(rng.normal(0, 0.02, (len(t), 2)), axis=0)])


def test_clean_user_passes():
    rng = np.random.default_rng(0)
    labels = _unit(rng, N_WINDOWS)
    windows = _eeg(labels, seed=1, eye_gain_uv=0.0)
    # a synthetic null for the screen: the displayed goal and cursor carry no label information
    prov = _prov(N_WINDOWS, ["closed_loop", "blank_feedback"], DEFAULT.decoder.window_s)
    log = _wandering_log(rng, N_WINDOWS, DEFAULT.decoder.window_s)
    res = PurityAudit(DEFAULT, n_permutations=N_PERM).run(windows, labels, prov, cursor_logs=log)
    r = res["reports"]
    assert res["metric"] == "cosine" and res["n"] == N_WINDOWS
    assert r["full"]["beats_floor"] and r["full"]["score"] > 0.2, r["full"]
    assert abs(r["intercept"]["score"]) < 0.1                     # centred labels: no EEG, no skill
    assert set(res["by_block"]) == {"closed_loop", "blank_feedback"}
    assert "score" in r["display"] and not r["display"]["beats_floor"], r["display"]
    assert "ci95" in res["feedback"]
    assert res["untested"]                                        # visual leakage stated as untested
    assert res["flags"] == ["PASS"], res


def test_ocular_leak_flagged_with_slow_features_and_auditors_stay_out_of_eeg():
    rng = np.random.default_rng(2)
    labels = _unit(rng, N_WINDOWS)
    windows = _eeg(labels, seed=3, eye_gain_uv=1.0)
    prov = _prov(N_WINDOWS, ["cued_blank"], DEFAULT.decoder.window_s)
    gaze = labels + rng.normal(0, 0.5, labels.shape)              # an eye tracker that follows intent
    audit = PurityAudit(DEFAULT, feature_set="bandpower+slow", n_permutations=N_PERM)
    res = audit.run(windows, labels, prov, auditors={"eye_tracker": gaze})
    assert "OCULAR_LEAK" in res["flags"], res
    assert res["reports"]["slow"]["beats_floor"]
    assert res["reports"]["auditor:eye_tracker"]["beats_floor"]
    assert "skipped" in res["reports"]["display"] and "skipped" in res["feedback"]
    assert not res["untested"]                                    # cued_blank: nothing spatial displayed
    # pure-EEG rule: adding an auditor changes no EEG decoder's score
    bare = audit.run(windows, labels, prov)
    for name in ("intercept", "full", "frontal", "slow", "occipital"):
        assert bare["reports"][name] == res["reports"][name]


def test_display_leak_flagged_when_labels_follow_displayed_velocity():
    rng = np.random.default_rng(4)
    step = 1.0                                                     # one window per displayed second
    labels = _unit(rng, N_WINDOWS)
    windows = _eeg(np.zeros((N_WINDOWS, 2)), seed=5)               # EEG carries no intent at all
    prov = _prov(N_WINDOWS, ["closed_loop"], step)
    # the cursor moved along the label direction during the second before each window end
    speed = 0.3
    pos = np.vstack([[0.0, 0.0], np.cumsum(labels * speed * step, axis=0)])
    log = np.column_stack([np.arange(N_WINDOWS + 1) * step, pos])
    res = PurityAudit(DEFAULT, n_permutations=N_PERM).run(windows, labels, prov, cursor_logs={"s0": log})
    assert res["flags"] == ["DISPLAY_LEAK"], res
    assert res["reports"]["display"]["score"] > 0.9
    assert not res["reports"]["full"]["beats_floor"]


def test_feedback_circularity_flagged_when_only_visible_windows_decode():
    n = 2 * N_WINDOWS                     # a between-condition difference needs more windows than a floor
    rng = np.random.default_rng(6)
    labels = _unit(rng, n)
    prov = _prov(n, ["closed_loop", "blank_feedback"], DEFAULT.decoder.window_s)
    # with the cursor hidden the brain carries no intent: the 'decoding' needed the feedback
    intents = np.where(np.array([p["feedback_visible"] for p in prov])[:, None], labels, 0.0)
    res = PurityAudit(DEFAULT, n_permutations=N_PERM).run(
        _eeg(intents, seed=7), labels, prov, cursor_logs=_wandering_log(rng, n, DEFAULT.decoder.window_s))
    fb = res["feedback"]
    assert fb["ci95"][0] > 0 and fb["visible"] > fb["blank"], fb
    assert "FEEDBACK_CIRCULAR" in res["flags"]


def test_binary_labels_use_auc():
    rng = np.random.default_rng(8)
    labels = _unit(rng, N_WINDOWS)
    windows = _eeg(labels, seed=9)
    res = PurityAudit(DEFAULT, n_permutations=200).run(
        windows, (labels[:, 0] > 0).astype(int), _prov(N_WINDOWS, ["cued_blank"], 0.5))
    assert res["metric"] == "auc"
    assert 0.0 <= res["reports"]["full"]["score"] <= 1.0 and res["reports"]["full"]["beats_floor"]


def test_helpers_and_input_validation():
    fs = DEFAULT.hardware.fs
    t = np.arange(fs // 2) / fs
    alpha = np.tile(np.sin(2 * np.pi * 10 * t)[:, None], (1, 32))
    assert np.allclose(slow_binned_means(alpha, fs), 0, atol=1e-9)   # 0.1 s bins null 10 Hz alpha
    log = np.array([[0.0, 0.0, 0.0], [2.0, 0.6, -0.2]])
    v = displayed_velocity(2.0, log)
    assert np.allclose(v[:5], 0.3) and np.allclose(v[5:], -0.1)
    with pytest.raises(ValueError):
        PurityAudit(DEFAULT, n_permutations=100)
    w = [np.zeros((250, 32))] * 20
    with pytest.raises(ValueError):
        PurityAudit(DEFAULT, n_permutations=200).run(w, np.zeros(20), _prov(20, ["cued_blank"], 0.5))


def test_ocular_flag_is_corrected_for_its_two_surrogates():
    """frontal and slow both test OCULAR_LEAK, so each needs p ≤ α / 2: a lone p = 0.04 that beats
    its p95 floor (a 1-in-10 fluke for two surrogates) is not a leak; p = 0.01 is."""
    audit = PurityAudit(DEFAULT, n_permutations=200)

    def rep(score, p):
        return {"score": score, "floor": 0.05, "p_value": p, "beats_floor": score > 0.05, "n": 300}

    base = {"full": rep(0.0, 0.5), "frontal": rep(0.0, 0.6), "display": {"skipped": "x"}}
    fluke = audit._flags(dict(base, slow=rep(0.1, 0.04)), {}, chance=0.0)
    leak = audit._flags(dict(base, slow=rep(0.1, 0.01)), {}, chance=0.0)
    assert fluke == ["PASS"] and leak == ["OCULAR_LEAK"]


def test_ocular_share_is_measured_above_the_no_eeg_baseline():
    """With non-centred labels an uninformative frontal surrogate scores ≈ the intercept-only model
    (0.3 here), which is ≥ 50 % of a full score of 0.4: measured above 0 the share test is met
    automatically; measured above the intercept-only score it is not (0.03 of 0.10 skill)."""
    audit = PurityAudit(DEFAULT, n_permutations=200)

    def rep(score, p):
        return {"score": score, "floor": score - 0.01, "p_value": p, "beats_floor": True, "n": 300}

    reps = {"full": rep(0.40, 0.001), "frontal": rep(0.33, 0.01), "slow": rep(0.0, 0.9),
            "display": {"skipped": "x"}}
    assert audit._flags(reps, {}, chance=0.0) == ["OCULAR_LEAK"]
    assert audit._flags(reps, {}, chance=0.30) == ["PASS"]


def _ar_features(rng, n, d, rho):
    x = np.zeros((n, d))
    e = rng.standard_normal((n, d))
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def test_permutation_floor_keeps_its_size_on_autocorrelated_nulls():
    """Features drifting slowly (AR(1), ρ = 0.98 per window) and labels that are constant within a
    trial or wander within it toward a per-trial goal, all independent of each other.  A window-level
    shuffle's null is too narrow and 'beats its floor' far above the nominal 5 %; the within-session
    circular shift stays at or below ~7 %."""
    n, per_trial, d, n_seeds, n_perm = 300, 15, 20, 150, 200
    prov = [{"session": f"s{i // 150}", "t": float(i), "trial": i // per_trial} for i in range(n)]
    rates = {}
    for kind in ("trial_constant", "closed_loop_like"):
        hits = {"shuffle": 0, "circular": 0}
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            goal = rng.uniform(0, 2 * np.pi, n // per_trial)
            ang = np.repeat(goal, per_trial)
            if kind == "closed_loop_like":                          # direction drifts within a trial
                ang = ang + rng.normal(0, 0.1, (n // per_trial, per_trial)).cumsum(axis=1).ravel()
            y = np.stack([np.cos(ang), np.sin(ang)], axis=1)
            X = _ar_features(rng, n, d, 0.98)
            nulls = {"shuffle": np.stack([rng.permutation(n) for _ in range(n_perm)]),
                     "circular": circular_shift_permutations(prov, n_perm, rng)}
            for name, perms in nulls.items():
                rep, _ = _PermutationCV(y, False, perms, DEFAULT.decoder.ridge_alpha, 5, 95.0).evaluate(X)
                hits[name] += rep["beats_floor"]
        rates[kind] = {k: v / n_seeds for k, v in hits.items()}
    for kind, r in rates.items():
        assert r["circular"] <= 0.07, rates
    assert max(r["shuffle"] for r in rates.values()) > 0.1, rates   # the old null fails this test


def test_circular_shifts_stay_within_sessions_and_away_from_identity():
    prov = [{"session": "a" if i < 40 else "b", "t": float(i % 40 if i < 40 else 100 - i)} for i in range(70)]
    perms = circular_shift_permutations(prov, 500, np.random.default_rng(0))
    a, b = np.arange(40), np.arange(40, 70)
    assert np.all(np.isin(perms[:, a], a)) and np.all(np.isin(perms[:, b], b))
    assert np.all(np.sort(perms, axis=1) == np.arange(70))          # each draw is a permutation
    order_b = b[np.argsort([prov[i]["t"] for i in b])]               # session b is in reverse t order
    pos = {int(w): k for k, w in enumerate(order_b)}
    shifts = np.array([pos[int(perms[r, order_b[0]])] for r in range(500)])
    assert shifts.min() >= 3 and shifts.max() <= 27                 # ⌈0.1 · 30⌉ = 3 away from identity


def test_display_surrogate_sees_the_goal_when_the_cursor_is_hidden():
    """blank_feedback draws the targets: the label (start → goal) is exactly the displayed goal's
    direction.  The display surrogate must see it (DISPLAY_LEAK), never treat 'cursor hidden' as
    'nothing displayed'; and a target block without a displayed goal cannot be audited."""
    assert set(SPATIAL_BLOCKS) == set(TARGET_BLOCKS + CHAMPION_TARGET_BLOCKS)   # renderer == labeler
    rng = np.random.default_rng(10)
    n = 200
    goals = 0.6 * _unit(rng, n)
    labels = goals / np.linalg.norm(goals, axis=1, keepdims=True)
    prov = _prov(n, ["blank_feedback"], DEFAULT.decoder.window_s, goals=goals)
    audit = PurityAudit(DEFAULT, n_permutations=200)
    windows = _eeg(np.zeros((n, 2)), seed=11)                       # EEG carries nothing
    res = audit.run(windows, labels, prov)                           # no cursor log needed: hidden cursor
    assert res["reports"]["display"]["score"] > 0.9 and "DISPLAY_LEAK" in res["flags"], res["flags"]
    assert not res["reports"]["full"]["beats_floor"]
    assert np.abs(audit._display_features(prov, None)).sum(axis=1).min() > 0
    with pytest.raises(ValueError, match="display_goal"):
        audit.run(windows, labels, [dict(p, display_goal=None) for p in prov])
    with pytest.raises(ValueError, match="cursor log"):            # a visible cursor needs its log
        audit.run(windows, labels, _prov(n, ["closed_loop"], DEFAULT.decoder.window_s, goals=goals))


def test_feedback_contrast_is_not_fooled_by_the_label_mean():
    """No EEG intent anywhere.  closed_loop labels point against a display-driven bias (non-zero
    mean), blank_feedback labels are centred: raw accuracy is higher on visible windows through the
    intercept alone, which the old contrast read as circularity.  Skill above the intercept-only
    model shows no difference.  Free-use windows are not part of the contrast."""
    rng = np.random.default_rng(13)
    n = 2 * N_WINDOWS
    blocks = ["closed_loop", "blank_feedback", "free"]
    prov = _prov(n, blocks, DEFAULT.decoder.window_s)
    vis = np.array([p["block"] == "closed_loop" for p in prov])
    raw = _unit(rng, n) + np.where(vis[:, None], np.array([0.15, -0.6]) * 2.5, 0.0)
    labels = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    res = PurityAudit(DEFAULT, n_permutations=200).run(
        _eeg(np.zeros((n, 2)), seed=14), labels, prov, cursor_logs=_wandering_log(rng, n, DEFAULT.decoder.window_s))
    fb = res["feedback"]
    assert fb["visible_raw"] - fb["blank_raw"] > 0.2                 # the intercept effect is real
    assert fb["ci95"][0] <= 0 and "FEEDBACK_CIRCULAR" not in res["flags"], fb
    assert fb["n_visible"] == vis.sum()                              # free use left out
    bb = res["by_block"]["closed_loop"]
    assert bb["intercept"] > 0.3 and bb["intercept"] >= bb["full"] - 0.05   # no-EEG baseline reported


def test_feedback_bootstrap_resamples_trials_not_windows():
    """Windows of one trial share its goal: with 12 trials of 25 identical-scoring windows per group
    and no true difference, a per-window bootstrap claims ~300 independent observations and calls a
    chance gap between trials circular; the trial-level bootstrap's CI covers zero."""
    rng = np.random.default_rng(12)
    n_trials, per_trial = 12, 25
    groups = []
    for block, visible in (("closed_loop", True), ("blank_feedback", False)):
        trial_scores = rng.normal(0.3, 0.15, n_trials)
        groups += [(block, visible, k, s) for k, s in enumerate(trial_scores) for _ in range(per_trial)]
    scores = np.array([g[3] for g in groups])
    with_trials = [{"block": b, "feedback_visible": v, "session": "s0", "trial": k} for b, v, k, _ in groups]
    no_trials = [{key: val for key, val in p.items() if key != "trial"} for p in with_trials]
    audit = PurityAudit(DEFAULT, n_permutations=200)
    clustered = audit._feedback_test(scores, with_trials, np.random.default_rng(0))
    naive = audit._feedback_test(scores, no_trials, np.random.default_rng(0))
    width = lambda fb: fb["ci95"][1] - fb["ci95"][0]
    assert clustered["n_visible_trials"] == clustered["n_blank_trials"] == n_trials
    assert width(clustered) > 3 * width(naive)
    assert clustered["ci95"][0] < 0 < clustered["ci95"][1]


def test_certification_is_scoped_to_blank_cue_windows_and_is_the_verdict():
    """A dataset mixing blank-cue windows (EEG carries the cued direction, nothing on screen) with
    cursor-visible windows (label = goal − displayed cursor, display-identifiable by design): the
    whole-set flags report DISPLAY_LEAK, but the verdict of record — computed on blank-cue windows
    only — is PASS, because there the display carries no label information and the EEG decodes."""
    cfg = DEFAULT
    rng = np.random.default_rng(3)
    sim = SimulatedEEG(cfg.hardware, seed=11, intent_gain_uv=8.0)
    filt = CausalFilter(cfg.hardware, cfg.pre)
    filt(sim.generate(cfg.hardware.fs * 2))
    fs, win = cfg.hardware.fs, int(cfg.decoder.window_s * cfg.hardware.fs)
    windows, labels, prov, cursor_rows = [], [], [], []
    t = 0.0
    for trial in range(120):
        ang = rng.uniform(0, 2 * np.pi)
        d = np.array([np.cos(ang), np.sin(ang)])
        visible = trial % 2 == 1
        sim.set_intent(*(d if not visible else np.zeros(2)))     # visible windows: no EEG intent at all
        for k in range(3):
            x = common_average(filt(sim.generate(win)))
            t += cfg.decoder.window_s
            windows.append(x)
            if visible:
                pos = rng.uniform(-0.5, 0.5, 2)
                goal = pos + 0.4 * d
                cursor_rows += [[t - cfg.decoder.window_s + j * 0.1, *pos] for j in range(int(cfg.decoder.window_s / 0.1) + 1)]
                labels.append(d)
                prov.append({"block": "closed_loop", "feedback_visible": True, "session": "s1", "t": t,
                             "trial": trial, "display_goal": goal.tolist()})
            else:
                labels.append(d)
                prov.append({"block": "cued_blank", "feedback_visible": False, "session": "s1", "t": t,
                             "trial": trial, "display_goal": None})
    audit = PurityAudit(cfg, n_permutations=200)
    res = audit.run(windows, np.array(labels), prov, cursor_logs=np.array(cursor_rows))
    assert "DISPLAY_LEAK" in res["flags"], res["flags"]
    cert = res["certification"]
    assert cert["n"] == 180 and cert["decodes"], cert
    assert res["verdict"] == ["PASS"], res["verdict"]


def test_certification_reports_no_cortical_evidence_and_gaze_tracking():
    """Blank-cue windows with no EEG intent but an eye tracker that follows the cue: the verdict says
    the EEG shows nothing and gaze tracks the label — a fixation failure to fix before any claim."""
    cfg = DEFAULT
    rng = np.random.default_rng(4)
    sim = SimulatedEEG(cfg.hardware, seed=12, intent_gain_uv=0.0)
    filt = CausalFilter(cfg.hardware, cfg.pre)
    filt(sim.generate(cfg.hardware.fs * 2))
    win = int(cfg.decoder.window_s * cfg.hardware.fs)
    windows, labels, prov, gaze = [], [], [], []
    for i in range(150):
        ang = rng.uniform(0, 2 * np.pi)
        d = np.array([np.cos(ang), np.sin(ang)])
        windows.append(common_average(filt(sim.generate(win))))
        labels.append(d)
        gaze.append(0.3 * d + 0.05 * rng.standard_normal(2))
        prov.append({"block": "cued_blank", "feedback_visible": False, "session": "s1",
                     "t": (i + 1) * cfg.decoder.window_s, "trial": i, "display_goal": None})
    res = PurityAudit(cfg, n_permutations=200).run(windows, np.array(labels), prov,
                                                   auditors={"eyetracker": np.array(gaze)})
    assert res["verdict"][0] == "NO_CORTICAL_EVIDENCE" and "GAZE_TRACKS_LABEL" in res["verdict"], res["verdict"]


# ----------------------------------------------------------------------------------------------
# Champion blank-cue windows, the deployed-decoder row and microsaccade auditors
# ----------------------------------------------------------------------------------------------
L1_DIRS = np.array([[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]])   # FEET, HANDS, LEFT, RIGHT
CHAMPION_PATTERN = (("fixation_probe", "L1"), ("blind_closed_loop", "L2b"), ("cued_blank", None),
                    ("blind_closed_loop", "J"), ("sighted", "S"))
"""Trial cycle of the mixed champion set: three certifying kinds, then blind just-intend and sighted."""
PER_TRIAL = 3


def _champion_set(n_trials, seed, intent=True):
    """Windows of a mixed champion session: trial-constant L1-style labels, the EEG carrying the label
    on every window (or none with ``intent=False``), provenance with ``source`` as the Labeler writes it."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(L1_DIRS[rng.integers(0, 4, n_trials)], PER_TRIAL, axis=0)
    prov = []
    for i in range(len(labels)):
        block, source = CHAMPION_PATTERN[(i // PER_TRIAL) % len(CHAMPION_PATTERN)]
        p = {"block": block, "feedback_visible": block == "sighted", "session": "s0",
             "t": (i + 1) * DEFAULT.decoder.window_s, "trial": i // PER_TRIAL,
             "display_goal": rng.uniform(-0.8, 0.8, 2).tolist() if block == "sighted" else None}
        if source is not None:
            p["source"] = source
        prov.append(p)
    windows = _eeg(labels if intent else np.zeros_like(labels), seed=seed + 1)
    return windows, labels, prov, _wandering_log(rng, len(labels), DEFAULT.decoder.window_s)


def test_certifying_windows_are_the_blank_cue_blocks_and_L1_L2b_sources():
    """fixation_probe and blind_closed_loop certify beside cued_blank, keyed on block or source; blind
    just-intend (J), drawn-target blocks and records whose block and source disagree do not."""
    assert set(CERTIFYING_BLOCKS) == {"cued_blank", "fixation_probe", "blind_closed_loop"}
    assert set(CERTIFYING_SOURCES) == {"L1", "L2b"}
    yes = [{"block": "cued_blank"}, {"block": "fixation_probe", "source": "L1"},
           {"block": "blind_closed_loop", "source": "L2b"}, {"source": "L1"}, {"source": "L2b"},
           {"block": "fixation_probe"}]
    no = [{"block": "blind_closed_loop", "source": "J"}, {"block": "sighted", "source": "S"},
          {"block": "heldout_mapping", "source": "H"}, {"block": "closed_loop"}, {"block": "blank_feedback"},
          {"block": "sighted", "source": "L1"}, {"source": "J"}, {}]
    assert all(is_certifying(p) for p in yes) and not any(is_certifying(p) for p in no)


def test_deployed_row_scores_the_shipped_decoder_on_the_same_null():
    """A deployed decoder's out-of-sample predictions are scored beside the ridge "full" row against the
    same label permutations, on the whole set and on the certifying windows only (L1, L2b, cued_blank:
    J and S are left out).  An informative deployed decoder beats its floor; an uninformative one
    (with some invalid NaN windows, scored as the loop's zero command) yields DEPLOYED_NO_EVIDENCE
    even though the bandpower ridge decodes.  Nothing but EEG windows reaches the decoder, and the
    deployed row changes no surrogate."""
    windows, labels, prov, log = _champion_set(80, seed=21)
    n = len(labels)
    rng = np.random.default_rng(22)
    good = labels + rng.normal(0, 0.8, labels.shape)
    seen = set()

    def useless(ws):
        seen.update(np.shape(w) for w in ws)
        out = rng.normal(0, 0.5, (len(ws), 2))
        out[::7] = np.nan
        return out

    audit = PurityAudit(DEFAULT, n_permutations=200)
    res = audit.run(windows, labels, prov, cursor_logs=log, deployed=good)
    bad = audit.run(windows, labels, prov, cursor_logs=log, deployed=useless)
    n_cert = sum(is_certifying(p) for p in prov)
    assert n_cert == 3 * n // 5 and res["certification"]["n"] == n_cert
    cert = res["certification"]
    assert res["reports"]["deployed"]["beats_floor"] and cert["reports"]["deployed"]["beats_floor"]
    assert cert["deployed_decodes"] and "DEPLOYED_NO_EVIDENCE" not in res["verdict"], res["verdict"]
    assert all(cert["reports"][s]["share_of_deployed"] is not None for s in ("frontal", "slow", "occipital"))
    assert bad["certification"]["decodes"] and bad["certification"]["deployed_decodes"] is False
    assert "DEPLOYED_NO_EVIDENCE" in bad["verdict"] and "NO_CORTICAL_EVIDENCE" not in bad["verdict"]
    assert bad["reports"]["deployed"]["n_zero_rows"] >= len(range(0, n, 7))
    assert seen == {(int(DEFAULT.decoder.window_s * DEFAULT.hardware.fs), DEFAULT.hardware.n_channels)}
    for name in ("intercept", "full", "frontal", "slow", "occipital", "display"):
        assert {k: v for k, v in res["reports"][name].items() if k != "share_of_deployed"} == \
               {k: v for k, v in bad["reports"][name].items() if k != "share_of_deployed"}
    with pytest.raises(ValueError, match="deployed predictions"):
        audit.run(windows, labels, prov, cursor_logs=log, deployed=good[:, :1])


def test_fixed_prediction_floor_keeps_its_size_and_matches_auc():
    """The deployed row's null (fixed predictions against circularly shifted labels) keeps the size of
    the surrogates' floor when predictions are unrelated to trial-constant labels, even when they drift
    (AR(1), ρ = 0.9 per window): about 5–8 % at 60 trials, the small excess of a p95 read off a few
    dozen effectively independent shifts.  On binary labels its score is the plain AUC."""
    n, per_trial, n_seeds = 300, 5, 200
    prov = [{"session": "s0", "t": float(i), "trial": i // per_trial} for i in range(n)]
    hits = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        y = np.repeat(L1_DIRS[rng.integers(0, 4, n // per_trial)], per_trial, axis=0)
        pred = _ar_features(rng, n, 2, 0.9)
        cv = _PermutationCV(y, False, circular_shift_permutations(prov, 200, rng), 1.0, 5, 95.0)
        hits += cv.evaluate_fixed(pred)["beats_floor"]
    assert hits / n_seeds <= 0.09, hits / n_seeds
    rng = np.random.default_rng(1)
    yb = (rng.random(n) > 0.5).astype(float)
    score = yb + rng.normal(0, 1.0, n)
    cv = _PermutationCV(yb, True, circular_shift_permutations(prov, 200, rng), 1.0, 5, 95.0)
    assert cv.evaluate_fixed(score)["score"] == pytest.approx(roc_auc_score(yb, score))


def _gaze_segment(rng, direction, n_events, fs=GAZE_FS, n=None):
    """One fixation segment (x, y in degrees) as long as an EEG window: tracker noise on a slow random
    walk, plus ``n_events`` 0.2–0.4° microsaccades (12 ms ramps, ≈ 25°/s) along ``direction``, each followed by a slow 0.5-s
    drift back — the mean position stays within the fixation window."""
    n = n or int(DEFAULT.decoder.window_s * fs)
    xy = np.cumsum(rng.normal(0, 0.003, (n, 2)), axis=0) + rng.normal(0, 0.003, (n, 2))
    t = np.arange(n)
    for t0 in rng.choice(np.arange(30, n - 40, 20), n_events, replace=False):
        ramp = np.clip((t - t0) / 6.0, 0, 1) - np.clip((t - t0 - 6) / 250.0, 0, 1)
        xy += rng.uniform(0.2, 0.4) * ramp[:, None] * np.asarray(direction, float)
    return xy


def test_microsaccades_are_detected_with_their_direction():
    """Velocity-threshold events (λ = 6 median SDs, ≥ 6 ms): injected microsaccades are found with their
    direction, tracker noise and a one-sample spike are not events, and track loss (NaN) is tolerated."""
    rng = np.random.default_rng(30)
    assert MS_LAMBDA == 6.0 and MS_MIN_DURATION_S == pytest.approx(0.006)
    for k in range(40):
        a = rng.uniform(0, 2 * np.pi)
        d = np.array([np.cos(a), np.sin(a)])
        f = microsaccade_features(_gaze_segment(rng, d, 2), GAZE_FS)
        assert f[0] == pytest.approx(2 / DEFAULT.decoder.window_s)          # 2 events per segment
        assert f[1:] @ d > 0.95, (f, d)
    quiet = [microsaccade_features(_gaze_segment(rng, (0, 0), 0), GAZE_FS)[0] for _ in range(40)]
    assert max(quiet) == 0.0
    spike = _gaze_segment(rng, (0, 0), 0)
    spike[len(spike) // 2] += 0.5                                            # a one-sample tracker glitch
    assert len(microsaccades(spike, GAZE_FS)) == 0
    lost = _gaze_segment(rng, (1, 0), 1)
    lost[:20] = np.nan                                                      # track loss before the event
    assert len(microsaccades(lost, GAZE_FS)) >= 1
    assert np.array_equal(microsaccade_features(np.zeros((3, 2)), GAZE_FS), np.zeros(3))


def test_microsaccade_auditor_flags_gaze_tracking_on_fixation_probes():
    """Fixation probes with no EEG intent, fixation held (mean gaze within a fraction of a degree), but
    microsaccades biased toward the cued direction: with the gaze stream's rate given, the auditor's
    microsaccade direction decodes the label and the verdict says NO_CORTICAL_EVIDENCE and
    GAZE_TRACKS_LABEL.  A gaze rate for an auditor that was not given, or for featurised gaze, raises."""
    rng = np.random.default_rng(31)
    n_trials = 60
    labels = np.repeat(L1_DIRS[rng.integers(0, 4, n_trials)], PER_TRIAL, axis=0)
    prov = [{"block": "fixation_probe", "source": "L1", "feedback_visible": False, "session": "s0",
             "t": (i + 1) * DEFAULT.decoder.window_s, "trial": i // PER_TRIAL, "display_goal": None}
            for i in range(len(labels))]
    gaze = []
    for d in labels:
        biased = rng.random() < 0.8
        a = rng.uniform(0, 2 * np.pi)
        gaze.append(_gaze_segment(rng, d if biased else (np.cos(a), np.sin(a)), 1))
    gaze = np.stack(gaze)
    assert np.abs(gaze.mean(axis=1)).max() < 0.5                            # inside the fixation window
    windows = _eeg(np.zeros_like(labels), seed=32)
    audit = PurityAudit(DEFAULT, n_permutations=200)
    res = audit.run(windows, labels, prov, auditors={"eyetracker": gaze}, gaze_fs={"eyetracker": GAZE_FS})
    rep = res["certification"]["reports"]["auditor:eyetracker"]
    assert rep["beats_floor"], rep
    assert res["verdict"][0] == "NO_CORTICAL_EVIDENCE" and "GAZE_TRACKS_LABEL" in res["verdict"], res["verdict"]
    rate = res["reports"]["auditor:eyetracker"]["microsaccade_rate_hz"]
    assert rate == pytest.approx(1 / DEFAULT.decoder.window_s, rel=0.1)     # one event per segment
    feats = auditor_features(gaze, len(gaze), GAZE_FS)
    assert feats.shape[1] == 3 * 2 + 3
    assert np.mean(np.sum(feats[:, -2:] * labels, axis=1)) > 0.5            # direction follows the cue
    with pytest.raises(ValueError, match="not given"):
        audit.run(windows, labels, prov, auditors={"eyetracker": gaze}, gaze_fs={"tracker": GAZE_FS})
    with pytest.raises(ValueError, match="raw gaze segments"):
        auditor_features(gaze.mean(axis=1), len(gaze), GAZE_FS)
