"""Body-imagery simulator (INTERFACES_CHAMPION item J).

The champion's certification is rehearsed on this simulator, so the properties it must have are the
ones the certification relies on: the stream does not depend on chunking (real-time and offline see
one signal); with ``screen_leak = 0`` the screen direction cannot reach the EEG at all (otherwise a
"clean" deployed decoder could fail certification for the simulator's sake); with a leak and a rotated
mapping the screen-locked and body-locked components land on different regressors (otherwise the
shadow positive control could never fire); and the dry penalty weakens decodability monotonically
(otherwise the dry-penalty sweep would not be a sweep)."""
import numpy as np
import pytest
from scipy import signal as sps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neural_cursor.body_decoder import BodyCoordinateDecoder
from neural_cursor.config import DEFAULT
from neural_cursor.mapping import BODY_VECTORS, Mapping
from neural_cursor.signal import common_average
from neural_cursor.sim_body import (ERD_PEAK_S, ERD_SUSTAIN, X_DEPTH_RANGE, X_RESPONDER_PROB, Y_DEPTH_RANGE,
                                    BodyPopulation, BodySubject, erd_envelope)
from neural_cursor.sim_population import PopulationSim

HW = DEFAULT.hardware
FS = HW.fs
CH = list(HW.channels)
CLASSES = ("FEET", "HANDS", "RIGHT", "LEFT")


def _sizes(total, rng):
    out = []
    while sum(out) < total:
        out.append(int(rng.integers(1, 60)))
    out[-1] -= sum(out) - total
    return out


def _log_power(x, chans, band):
    """Log band power of the average of ``chans``: a simple spatial filter that keeps a generator's
    coherent rhythm and averages down the channel-independent background."""
    sos = sps.butter(4, band, "bandpass", fs=FS, output="sos")
    y = sps.sosfiltfilt(sos, x[:, [CH.index(c) for c in chans]].mean(axis=1))
    return float(np.log(np.mean(y ** 2)))


def _trials(sub, schedule, rest_s=1.0, task_s=3.0):
    """One epoch per (body vector, screen direction | None): rest with nothing drawn, then the cue and
    (if given) the target appear together.  Returns the task-period EEG of every trial."""
    out = []
    for body, screen in schedule:
        sub.set_screen_direction(None)
        sub.cue(0.0, 0.0)
        sub.generate(int(rest_s * FS))
        sub.set_screen_direction(screen)
        sub.cue(*body)
        out.append(sub.generate(int(task_s * FS)))
    return out


def _ols_t(y, regressors):
    """OLS of y on [1, regressors]; t statistics of the regressors (classical SEs)."""
    X = np.column_stack([np.ones(len(y))] + list(regressors))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return (beta / se)[1:]


# ------------------------------------------------------------------------------------------------
def test_generate_is_chunk_invariant_with_every_pathway_on():
    kw = dict(seed=17, drift=0.05, eye_gain_uv=1.0, screen_leak=0.8, covert_leak=0.3,
              session_drift_sd=0.5, x_responder=True)
    n1, n2 = 3 * FS + 7, 2 * FS + 11

    def run(chunked):
        sub = BodySubject(HW, **kw)
        sub.set_day(2)
        sub.begin_block(2, 1)
        sub.set_gaze(0.3, -0.2)
        sub.set_screen_direction((0.6, -0.8))
        sub.cue(1.0, 0.5)
        parts = []
        for n, seed in ((n1, 0), (n2, 1)):
            sizes = _sizes(n, np.random.default_rng(seed)) if chunked else [n]
            parts += [sub.generate(k) for k in sizes]
            sub.set_intent(-0.4, 1.0)                      # state change at the same sample in both runs
            sub.set_screen_direction((-1.0, 0.0))
        return np.concatenate(parts)

    whole, pieces = run(False), run(True)
    assert whole.shape == (n1 + n2, 32) and whole.dtype == np.float32
    assert np.allclose(whole, pieces, rtol=0, atol=1e-4)


def test_start_rebases_without_changing_the_signal():
    a, b = BodySubject(HW, seed=4, session_drift_sd=0.5), BodySubject(HW, seed=4, session_drift_sd=0.5)
    for s in (a, b):
        s.cue(0.0, 1.0)
    xa = np.concatenate([a.generate(700), a.generate(900)])
    first = b.generate(700)
    b.start()                                              # real-time clock restart mid-episode
    xb = np.concatenate([first, b.generate(900)])
    assert np.allclose(xa, xb, rtol=0, atol=1e-4)


def test_without_screen_leak_nothing_depends_on_the_screen_direction():
    """Exactly equal EEG for any sequence of drawn / hidden targets when screen_leak = 0 — even with
    covert (body-locked) attention on, because covert attention follows the body, not the screen."""
    rng = np.random.default_rng(3)
    directions = [None, (1.0, 0.0), (0.0, -1.0), None] + [tuple(rng.uniform(-1, 1, 2)) for _ in range(6)]

    def run(leak, screen_seq):
        sub = BodySubject(HW, seed=8, screen_leak=leak, covert_leak=0.5, eye_gain_uv=0.0)
        sub.begin_block(1, 0)
        out = []
        for k, d in enumerate(screen_seq):
            sub.set_screen_direction(d)
            if k % 3 == 0:
                sub.cue(*BODY_VECTORS[CLASSES[k % 4]])
            out.append(sub.generate(FS // 2))
        return np.concatenate(out)

    base = run(0.0, [None] * len(directions))
    assert np.array_equal(run(0.0, directions), base)
    assert np.array_equal(run(0.0, directions[::-1]), base)
    assert not np.allclose(run(0.5, directions), base)


def test_screen_and_body_components_separate_under_a_rotated_mapping():
    """rot+90: feet/hands move the cursor sideways and right/left hand move it up/down, so on each axis
    the body and screen regressors are orthogonal across trials.  With a leak, posterior alpha and the
    frontal gaze potential follow the SCREEN while the C3/C4 mu/beta lateralisation follows the BODY.
    The central CP sites (inside the central-11 montage) pick up the screen signal too — the leak a
    sighted-label shadow decoder can learn; it is isolated by pairing each trial with the same person
    and noise at zero leak.  Without a leak no screen dependence remains anywhere."""
    rot = Mapping("rot+90")
    rng = np.random.default_rng(11)
    bodies = [np.array(BODY_VECTORS[c]) for c in rng.permutation(np.repeat(CLASSES, 32))]
    screens = [rot.body_to_screen(b) for b in bodies]
    (bx, by), (sx, sy) = np.array(bodies).T, np.array(screens).T
    assert abs(np.corrcoef(bx, sx)[0, 1]) < 1e-12 and abs(np.corrcoef(by, sy)[0, 1]) < 1e-12

    def features(leak):
        sub = BodySubject(HW, seed=21, screen_leak=leak, x_responder=True, y_depth=0.35, x_depth=0.5)
        f = {"posterior": [], "motor": [], "hgaze": [], "vgaze": [], "central_cp": []}
        for x in _trials(sub, list(zip(bodies, screens))):
            w = x[FS:3 * FS]                                    # 1–3 s after the cue
            f["posterior"].append(_log_power(w, ["PO7", "PO3"], (8, 12))
                                  - _log_power(w, ["PO8", "PO4"], (8, 12)))
            f["motor"].append(_log_power(w, ["C3", "FC5", "FC1"], (8, 30))
                              - _log_power(w, ["C4", "FC6", "FC2"], (8, 30)))
            f["hgaze"].append(float(np.mean(w[:, CH.index("F8")] - w[:, CH.index("F7")])))
            f["vgaze"].append(float(np.mean(w[:, [CH.index("FP1"), CH.index("FP2")]])))
            f["central_cp"].append(_log_power(w, ["CP5", "CP1"], (8, 12))
                                   - _log_power(w, ["CP6", "CP2"], (8, 12)))
        return {k: np.array(v) for k, v in f.items()}

    leak, clean = features(1.0), features(0.0)
    t_body, t_screen = _ols_t(leak["posterior"], [bx, sx])
    assert t_screen < -6 and abs(t_body) < 3                    # attend right → left posterior alpha falls
    t_body, t_screen = _ols_t(leak["hgaze"], [bx, sx])
    assert t_screen > 6 and abs(t_body) < 3                     # gaze right → F8 − F7 positive
    t_body, t_screen = _ols_t(leak["vgaze"], [by, sy])
    assert t_screen > 6 and abs(t_body) < 3                     # gaze up → FP positive
    t_body, t_screen = _ols_t(leak["motor"], [bx, sx])
    assert t_body < -4 and abs(t_screen) < 3                    # right-hand imagery → C3 mu/beta falls
    t_body, t_screen = _ols_t(leak["central_cp"] - clean["central_cp"], [bx, sx])
    assert t_screen < -6 and abs(t_body) < 3                    # the central montage sees the screen leak

    # without a leak the screen regressor has nothing to find (exact independence is tested above),
    # and the leak never touches the hand areas
    assert np.array_equal(clean["motor"], leak["motor"])


def _feet_hands(dry, seed=5, n_per_class=40):
    sub = BodySubject(HW, seed=seed, dry_penalty=dry, y_depth=0.35)
    rng = np.random.default_rng(seed)
    labels = rng.permutation(np.repeat([0, 1], n_per_class))           # 1 = feet, 0 = hands
    epochs = _trials(sub, [(BODY_VECTORS["FEET" if c else "HANDS"], None) for c in labels])
    return [x[FS:2 * FS] for x in epochs], labels                        # the 1–2 s ERD peak window


def test_dry_penalty_lowers_decodability_monotonically():
    """The dry penalty is a d′ loss: the vertex-vs-hand-area ERD contrast's d′ falls strictly with it,
    roughly in proportion, and so does a cross-validated tangent-space decoder's AUC."""
    penalties = (0.0, 0.2, 0.4, 0.6, 0.7)
    dprime, auc = [], {}
    for p in penalties:
        windows, y = _feet_hands(p)
        f = np.array([0.5 * (_log_power(w, ["C3", "FC5"], (8, 30)) + _log_power(w, ["C4", "FC6"], (8, 30)))
                      - _log_power(w, ["Cz"], (8, 30)) for w in windows])     # hand areas minus vertex
        a, b = f[y == 0], f[y == 1]
        dprime.append((b.mean() - a.mean()) / np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))))
        if p in (0.0, 0.4, 0.7):
            dec = BodyCoordinateDecoder(DEFAULT, FS)
            good = np.ones(HW.n_channels, bool)
            F = np.array([dec.features(common_average(w, good)) for w in windows])
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            auc[p] = cross_val_score(clf, F, y, cv=cv, scoring="roc_auc").mean()
    assert all(d1 > d2 for d1, d2 in zip(dprime, dprime[1:])), dprime
    assert dprime[0] > 1.0 and dprime[-1] < 0.5 * dprime[0]
    assert auc[0.0] > auc[0.4] > auc[0.7] and auc[0.0] > 0.7, auc


def test_erd_time_course_is_transient():
    tau = np.linspace(-1, 6, 7001)
    e = erd_envelope(tau)
    assert np.all(e[tau < 0] == 0) and np.isclose(e.max(), 1.0)
    assert 1.0 <= tau[np.argmax(e)] <= 2.0 and np.isclose(tau[np.argmax(e)], ERD_PEAK_S, atol=1e-3)
    assert np.all(np.diff(e[(tau >= 0) & (tau <= ERD_PEAK_S)]) >= 0)    # rises to the peak
    assert np.all(np.diff(e[tau >= ERD_PEAK_S]) <= 0)                    # then only decays
    assert ERD_SUSTAIN < e[-1] < ERD_SUSTAIN + 0.05                       # toward a sustained floor
    win = [erd_envelope(np.linspace(a, a + 1, 501)).mean() for a in (0, 1, 2, 3)]
    assert win[1] == max(win) and win[1] > win[2] > win[3] > 0.5 * win[1]
    # in the EEG: vertex mu/beta power after a feet cue is lowest in the 1–2 s window
    sub = BodySubject(HW, seed=2, y_depth=0.5)
    epochs = _trials(sub, [(BODY_VECTORS["FEET"], None)] * 30, task_s=4.0)
    p = [np.mean([_log_power(x[a * FS:(a + 1) * FS], ["Cz"], (8, 30)) for x in epochs]) for a in range(4)]
    assert np.argmin(p) == 1 and p[0] > p[1] < p[3]


def test_population_draws_responders_and_keeps_subjects_fixed():
    pop = BodyPopulation(500, seed=7)
    frac = np.mean([b.x_responder for b in pop.body])
    assert X_RESPONDER_PROB == 0.2 and 0.15 < frac < 0.25
    assert all(Y_DEPTH_RANGE[0] <= b.y_depth <= Y_DEPTH_RANGE[1] for b in pop.body)
    assert all(X_DEPTH_RANGE[0] <= b.x_depth <= X_DEPTH_RANGE[1] for b in pop.body)
    # extends the population simulator: same inherited people, and subject i fixed under any size or condition
    assert pop.params[:5] == PopulationSim(5, seed=7).params
    small = BodyPopulation(3, seed=7, dry_penalty=0.5, screen_leak=1.0)
    assert small.body == pop.body[:3]
    s = small.subject(2)
    assert isinstance(s, BodySubject) and s.dry_penalty == 0.5 and s.screen_leak == 1.0
    assert s.heterogeneity == pop.params[2].heterogeneity and s.x_responder == pop.body[2].x_responder
    assert set(pop.describe([0])[0]) >= {"heterogeneity", "y_depth", "x_depth", "x_responder"}
    with pytest.raises(ValueError):
        BodyPopulation(2, seed=1, x_responder_prob=1.5)


def test_non_responders_have_no_lateralised_x_signal():
    def lateralisation_t(responder):
        sub = BodySubject(HW, seed=15, x_responder=responder, x_depth=0.45)
        labels = np.random.default_rng(15).permutation(np.repeat([1.0, -1.0], 32))   # right = +1
        epochs = _trials(sub, [((v, 0.0), None) for v in labels])
        f = np.array([_log_power(x[FS:3 * FS], ["C3", "FC5", "FC1"], (8, 30))
                      - _log_power(x[FS:3 * FS], ["C4", "FC6", "FC2"], (8, 30)) for x in epochs])
        return _ols_t(f, [labels])[0]

    assert lateralisation_t(True) < -4
    assert abs(lateralisation_t(False)) < 3


def test_blocks_are_independent_of_earlier_blocks_and_parameters_validated():
    a, b = BodySubject(HW, seed=9), BodySubject(HW, seed=9)
    a.cue(1.0, 0.0)
    a.set_screen_direction((0.0, 1.0))
    a.generate(1234)                                           # an earlier block, mid-episode
    out = []
    for s in (a, b):
        s.set_day(2)
        s.begin_block(2, 3)
        s.cue(0.0, -1.0)
        out.append(s.generate(800))
    assert np.array_equal(out[0], out[1])
    for bad in (dict(dry_penalty=1.0), dict(dry_penalty=-0.1), dict(screen_leak=-1.0), dict(erd_sustain=1.5)):
        with pytest.raises(ValueError):
            BodySubject(HW, seed=0, **bad)
