"""BodyCoordinateDecoder tests on a minimal local SMR simulator.

The simulator is deliberately small and stated here, so each test's premise is visible: three
mu (8–13 Hz) + beta (16–24 Hz) sources under C3, Cz and C4, mixed into neighbouring channels, plus
white sensor noise on all 32 channels.  Imagery attenuates source power (ERD): FEET → Cz, HANDS →
C3 and C4; LEFT → C4 and RIGHT → C3 for a *lateralised* subject, but both hand cues → the same
bilateral ERD for a subject without a lateralised signal (the x axis then carries nothing).
"""
import time

import numpy as np
import pytest
from scipy import signal as sps
from sklearn.metrics import roc_auc_score

from neural_cursor.body_decoder import CENTRAL11, IDLE_SOURCE, BodyCoordinateDecoder
from neural_cursor.config import DEFAULT
from neural_cursor.effectors import NullEffector
from neural_cursor.realtime import RealTimeLoop
from neural_cursor.signal import common_average
from neural_cursor.sim_body import BodySubject
from neural_cursor.simulate_engine import OnlinePreprocessor

FS = DEFAULT.hardware.fs
N_WIN = FS                      # 1-s windows
CH = list(DEFAULT.hardware.channels)
CLASSES = {"FEET": (0, 1), "HANDS": (0, -1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
# source → {channel: mixing weight}: a crude volume-conduction footprint over the central strip
SOURCES = {
    "C3": {"C3": 1.0, "FC5": 0.5, "FC1": 0.4, "CP5": 0.5, "CP1": 0.4, "T7": 0.3, "Cz": 0.2},
    "Cz": {"Cz": 1.0, "FC1": 0.5, "FC2": 0.5, "CP1": 0.5, "CP2": 0.5, "C3": 0.2, "C4": 0.2, "Fz": 0.3, "Pz": 0.3},
    "C4": {"C4": 1.0, "FC6": 0.5, "FC2": 0.4, "CP6": 0.5, "CP2": 0.4, "T8": 0.3, "Cz": 0.2},
}
SOURCE_UV = 8.0                 # rhythm amplitude at rest
NOISE_UV = 6.0                  # white sensor noise
ERD = 0.65                      # imagery keeps 65 % of the source amplitude (≈ −3.7 dB power)
JITTER = 0.35                   # log-normal SD of per-window source amplitude (non-stationarity)
TRIALS_PER_CLASS = 16
WINDOWS_PER_TRIAL = 3           # 1.0–4.0 s after the cue, non-overlapping 1-s windows
BANDS = ((8.0, 13.0), (16.0, 24.0))
_SOS = [sps.butter(4, b, "bandpass", fs=FS, output="sos") for b in BANDS]
_MIX = np.zeros((len(SOURCES), len(CH)))
for _k, _foot in enumerate(SOURCES.values()):
    for _c, _w in _foot.items():
        _MIX[_k, CH.index(_c)] = _w


def _erd(cls: str, lateralised: bool) -> np.ndarray:
    """Amplitude factor per source (C3, Cz, C4) for a cue class."""
    f = np.ones(3)
    if cls == "FEET":
        f[1] = ERD
    elif cls == "HANDS" or not lateralised:
        f[[0, 2]] = ERD
    else:
        f[2 if cls == "LEFT" else 0] = ERD
    return f


def _rhythm(rng, n):
    """(n, 3) band-limited mu + beta source activity, unit RMS per band; 2n samples generated so the
    filters' onset transient is discarded."""
    out = np.zeros((n, 3))
    for sos in _SOS:
        z = sps.sosfilt(sos, rng.standard_normal((2 * n, 3)), axis=0)[n:]
        out += z / z.std(axis=0)
    return out / np.sqrt(len(_SOS))


def window(rng, factors, gains=None):
    """One (N_WIN, 32) µV window with source amplitude ``factors`` (3,) and optional channel gains."""
    amp = SOURCE_UV * factors * np.exp(JITTER * rng.standard_normal(3))
    x = (_rhythm(rng, N_WIN) * amp) @ _MIX + NOISE_UV * rng.standard_normal((N_WIN, len(CH)))
    return x * gains if gains is not None else x


def session(seed, session_id, lateralised, source="L1", classes=tuple(CLASSES), gains=None):
    """Cued trials in random order → (windows, body labels (n, 2), provenance)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.repeat(classes, TRIALS_PER_CLASS))
    ws, ys, prov = [], [], []
    for trial, cls in enumerate(order):
        for k in range(WINDOWS_PER_TRIAL):
            ws.append(window(rng, _erd(cls, lateralised), gains))
            ys.append(CLASSES[cls])
            prov.append({"source": source, "session": session_id, "trial": trial, "t": 6.0 * trial + 1.0 + k})
    return ws, np.array(ys), prov


def rest(seed, n=40, gains=None):
    rng = np.random.default_rng(seed)
    return [window(rng, np.ones(3), gains) for _ in range(n)]


def trial_permutation_floor(scores, labels, trials, n_perm=1000, seed=0):
    """p95 of the AUC when the labels are permuted across trials (windows of a trial share a label)."""
    rng = np.random.default_rng(seed)
    keys, inv = np.unique(trials, return_inverse=True)
    trial_label = np.zeros(len(keys), bool)
    trial_label[inv] = labels
    null = [roc_auc_score(rng.permutation(trial_label)[inv], scores) for _ in range(n_perm)]
    return float(np.percentile(null, 95))


# earn_x's null at 100 circular shifts (p95 resolution 0.01) keeps each call to ~5 s; the production
# default is 200.
N_PERM_TEST = 100


def make_decoder(**kw):
    return BodyCoordinateDecoder(DEFAULT, FS, **kw)


@pytest.fixture(scope="module")
def lateral():
    """A lateralised subject: training session 0, test session 1."""
    return session(1, 0, True), session(2, 1, True)


@pytest.fixture(scope="module")
def fitted(lateral):
    (ws, ys, prov), _ = lateral
    return make_decoder().fit(ws, ys, prov)


def test_y_decodes_feet_vs_hands_above_floor_on_a_new_session(fitted, lateral):
    _, (ws, ys, prov) = lateral
    m = ys[:, 1] != 0
    out = fitted.predict_many([w for w, k in zip(ws, m) if k])
    labels = ys[m, 1] > 0
    trials = np.array([p["trial"] for p, k in zip(prov, m) if k])
    auc = roc_auc_score(labels, out[:, 1])
    floor = trial_permutation_floor(out[:, 1], labels, trials)
    assert auc > floor and auc > 0.75, (auc, floor)


def test_predict_returns_two_bounded_values_and_x_is_zero_until_earned(fitted, lateral):
    _, (ws, _, _) = lateral
    for w in ws[:10]:
        v = fitted.predict(w)
        assert v.shape == (2,) and np.isfinite(v).all() and np.all(np.abs(v) <= 1)
        assert v[0] == 0.0                 # fitted, but x not earned yet
    assert fitted.predict_many(ws[:10]).shape == (10, 2)


def test_predict_validates_the_window():
    with pytest.raises(ValueError, match="fs"):
        BodyCoordinateDecoder(DEFAULT, FS // 2)                # window and band built for the wrong rate
    dec = make_decoder()
    with pytest.raises(ValueError, match="shape"):
        dec.predict(np.zeros((N_WIN, len(CENTRAL11))))        # central channels only: not a raw window
    with pytest.raises(ValueError, match="loop_config"):
        dec.predict(np.zeros((N_WIN // 2, 32)))               # 0.5 s: the loop's default window
    with pytest.raises(RuntimeError, match="not fitted"):
        dec.predict(np.random.default_rng(0).standard_normal((N_WIN, 32)))


@pytest.mark.parametrize("source", ["S", "H", "R", None])
def test_fit_and_earn_x_refuse_display_identifiable_or_test_provenance(lateral, source):
    (ws, ys, prov), _ = lateral
    tainted = [dict(p) for p in prov]
    tainted[7]["source"] = source                             # one bad window poisons the whole call
    if source is None:
        del tainted[7]["source"]
    dec = make_decoder()
    with pytest.raises(ValueError, match="trains only on"):
        dec.fit(ws, ys, tainted)
    with pytest.raises(ValueError, match="trains only on"):
        dec.earn_x(ws, ys, tainted)
    assert dec._models["y"] is None and not dec.x_enabled


def test_invalid_windows_give_nan_not_a_saturated_command(fitted, lateral):
    """A flat-lined amplifier, a near-zero window, a dead last 0.25 s, a non-finite or gated central channel:
    each would push the tangent features to extremes (or crash sklearn) and the output to ±1; the decoder
    returns NaN instead, which the real-time loop turns into a counted zero command."""
    _, (ws, _, _) = lateral
    good = ws[0]
    dead_end = good.copy()
    dead_end[-N_WIN // 4:] = 0.0                                  # the amplifier flat-lined 0.25 s ago
    nan_cz = good.copy()
    nan_cz[10, CH.index("Cz")] = np.nan
    nan_other = good.copy()
    nan_other[10, CH.index("PO8")] = np.nan                       # outside the montage: irrelevant
    for bad in (np.zeros((N_WIN, 32)), good * 1e-6, good * 1e4, dead_end, nan_cz):
        assert np.isnan(fitted.predict(bad)).all()
    assert np.isfinite(fitted.predict(nan_other)).all()
    gated = np.ones(32, bool)
    gated[CH.index("C3")] = False
    assert np.isnan(fitted.predict(good, good=gated)).all()
    gated = np.ones(32, bool)
    gated[CH.index("Oz")] = False
    assert np.isfinite(fitted.predict(good, good=gated)).all()
    out = fitted.predict_many([good, nan_cz, good])                # one bad window does not abort the call
    assert np.isfinite(out[[0, 2]]).all() and np.isnan(out[1]).all()
    assert np.isfinite(fitted.predict_many(ws[:50])).all()        # real windows are never refused


def test_fit_accepts_blind_closed_loop_windows():
    ws, ys, prov = session(3, 0, True, source="L2b")
    assert make_decoder().fit(ws, ys, prov).predict(ws[0]).shape == (2,)


def _two_sessions(lateralised, seed):
    a, b = session(seed, 0, lateralised), session(seed + 1, 1, lateralised)
    return a[0] + b[0], np.concatenate([a[1], b[1]]), a[2] + b[2]


def test_x_stays_disabled_without_a_lateralised_signal():
    ws, ys, prov = _two_sessions(False, 10)
    dec = make_decoder(n_permutations=N_PERM_TEST).fit(ws, ys, prov)
    rep = dec.earn_x(ws, ys, prov)
    assert not rep["enabled"] and not dec.x_enabled
    assert rep["auc"] < max(rep["floor"], rep["auc_min"])
    assert np.all(dec.predict_many(ws[:20])[:, 0] == 0.0)


ALPHA_SITES = {"left": {"P3": 1.0, "PO7": 1.0, "CP5": 0.6, "CP1": 0.4},
               "right": {"P4": 1.0, "PO8": 1.0, "CP6": 0.6, "CP2": 0.4}}
ALPHA_UV = 10.0                 # posterior alpha amplitude
ATTENTION_DEPTH = 0.3           # covert attention to one side lowers contralateral alpha by 30 %
_ALPHA_SOS = sps.butter(4, (8.0, 12.0), "bandpass", fs=FS, output="sos")
_ALPHA_MIX = np.zeros((2, len(CH)))
for _k, _side in enumerate(("left", "right")):
    for _c, _w in ALPHA_SITES[_side].items():
        _ALPHA_MIX[_k, CH.index(_c)] = _w


def attention_session(seed, session_id):
    """An x NON-responder (bilateral hand ERD) who covertly attends toward the cued side: posterior
    alpha lateralises and spreads onto CP5/CP1 and CP6/CP2, so the central-11 x model can decode it."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.repeat(tuple(CLASSES), TRIALS_PER_CLASS))
    ws, ys, prov = [], [], []
    for trial, cls in enumerate(order):
        for k in range(WINDOWS_PER_TRIAL):
            f = np.ones(2)
            if cls in ("RIGHT", "LEFT"):
                f[0 if cls == "RIGHT" else 1] = 1.0 - ATTENTION_DEPTH        # attend right: LEFT alpha falls
            z = sps.sosfilt(_ALPHA_SOS, rng.standard_normal((2 * N_WIN, 2)), axis=0)[N_WIN:]
            alpha = (z / z.std(axis=0)) * ALPHA_UV * f * np.exp(0.2 * rng.standard_normal(2))
            ws.append(window(rng, _erd(cls, False)) + alpha @ _ALPHA_MIX)
            ys.append(CLASSES[cls])
            prov.append({"source": "L1", "session": session_id, "trial": trial, "t": 6.0 * trial + 1.0 + k})
    return ws, np.array(ys), prov


def test_x_carried_by_posterior_attention_is_not_earned():
    """earn_x was a decodability gate only: a non-responder whose x labels drive covert attention would
    earn an attention axis through the label-pure path.  The purity rule (x skill must survive
    regressing out posterior alpha lateralisation) refuses it, and a motor x keeps all of its skill."""
    a, b = attention_session(70, 0), attention_session(71, 1)
    ws, ys, prov = a[0] + b[0], np.concatenate([a[1], b[1]]), a[2] + b[2]
    dec = make_decoder(n_permutations=50).fit(ws, ys, prov)
    rep = dec.earn_x(ws, ys, prov)
    assert rep["auc"] > max(rep["floor"], 0.60)                    # the old gate would have granted it
    assert not rep["enabled"] and "posterior" in rep["reason"] and rep["purity_retained"] < 0.70


def _weak_lateraliser(erd):
    """Two sessions of a person whose one-hand ERD keeps ``erd`` of the contralateral amplitude."""
    ws, ys, prov = [], [], []
    for sid, seed in ((0, 60), (1, 61)):
        rng = np.random.default_rng(seed)
        order = rng.permutation(np.repeat(tuple(CLASSES), TRIALS_PER_CLASS))
        for trial, cls in enumerate(order):
            f = _erd(cls, True) if cls in ("FEET", "HANDS") else np.where(_erd(cls, True) < 1, erd, 1.0)
            for k in range(WINDOWS_PER_TRIAL):
                ws.append(window(rng, f))
                ys.append(CLASSES[cls])
                prov.append({"source": "L1", "session": sid, "trial": trial, "t": 6.0 * trial + 1.0 + k})
    return ws, np.array(ys), prov


def test_earned_x_is_kept_unless_a_separate_test_rejects_it():
    """Once earned, x stays while it is above its chance floor (and motor), even below the grant
    threshold, so sampling noise around the threshold does not toggle the control scheme daily; it is
    revoked when it falls to the floor."""
    ws, ys, prov = _weak_lateraliser(0.85)
    dec = make_decoder(n_permutations=50, x_auc_min=0.65).fit(ws, ys, prov)
    grant = dec.earn_x(ws, ys, prov)
    keep = dec.earn_x(ws, ys, prov, previously_enabled=True)
    assert grant["floor"] < grant["auc"] < 0.65 and not grant["enabled"]
    assert keep["enabled"] and keep["rule"] == "keep" and dec.x_enabled
    ws, ys, prov = _weak_lateraliser(0.9)
    revoked = make_decoder(n_permutations=50).fit(ws, ys, prov).earn_x(ws, ys, prov, previously_enabled=True)
    assert revoked["auc"] <= revoked["floor"] and not revoked["enabled"]


def test_x_is_judged_on_a_fixed_window_population():
    """Only the fixation probes (L1) enter the x statistic by default: adding many blind closed-loop
    windows in which the lateralised ERD has decayed must not change it (it did: x flipped daily)."""
    ws, ys, prov = _two_sessions(True, 20)
    l2b_w, l2b_y, l2b_p = session(80, 2, False, source="L2b", classes=("LEFT", "RIGHT"))
    dec = make_decoder(n_permutations=50).fit(ws, ys, prov)
    only_l1 = dec.earn_x(ws, ys, prov)
    mixed = dec.earn_x(ws + l2b_w, np.concatenate([ys, l2b_y]), prov + l2b_p)
    assert mixed["auc"] == only_l1["auc"] and mixed["n_windows"] == only_l1["n_windows"]
    widened = dec.earn_x(ws + l2b_w, np.concatenate([ys, l2b_y]), prov + l2b_p, sources=("L1", "L2b"))
    assert widened["n_windows"] > only_l1["n_windows"] and widened["auc"] < only_l1["auc"]


def test_x_is_earned_with_a_lateralised_signal_and_then_decodes_right_vs_left():
    """A grant needs two consecutive passing grant tests (the test is repeated after every retrain, so a
    single 5 % pass is not enough): the first pass is pending, the confirming one grants."""
    ws, ys, prov = _two_sessions(True, 20)
    dec = make_decoder(n_permutations=N_PERM_TEST).fit(ws, ys, prov)
    first = dec.earn_x(ws, ys, prov)
    assert first["passed"] and first["grant_pending"] and not first["enabled"] and not dec.x_enabled
    rep = dec.earn_x(ws, ys, prov, previous_grant_passed=first["passed"])
    assert rep["enabled"] and dec.x_enabled and dec.x_report is rep and not rep["grant_pending"]
    assert rep["auc"] > rep["floor"] and rep["auc"] > 0.60 and rep["n_windows"] == 2 * 2 * 16 * 3
    assert rep["purity_retained"] > 0.9                            # motor x: posterior alpha explains none
    tw, ty, _ = session(30, 2, True, classes=("LEFT", "RIGHT"))
    out = dec.predict_many(tw)
    assert roc_auc_score(ty[:, 0] > 0, out[:, 0]) > 0.65
    # earning is deterministic (fixed permutation seed)
    assert make_decoder(n_permutations=N_PERM_TEST).earn_x(ws, ys, prov, previous_grant_passed=True) == rep


def test_rest_recentring_with_the_identity_reference_changes_nothing(lateral):
    (ws, ys, prov), (tw, _, _) = lateral
    plain = make_decoder().fit(ws, ys, prov)
    eye = np.eye(len(CENTRAL11))
    explicit = make_decoder().set_reference(reference=eye).fit(ws, ys, prov, session_references={0: eye})
    np.testing.assert_allclose(explicit.predict_many(tw[:30]), plain.predict_many(tw[:30]), atol=1e-12)
    np.testing.assert_allclose(explicit.features(tw[0]), plain.features(tw[0]), atol=1e-12)


def test_rest_recentring_absorbs_a_channel_gain_change_between_sessions(lateral):
    """A cap re-fit that changes electrode gains shifts every covariance; re-centring each session on
    its own rest moves both sessions' rest to the identity, so the trained y boundary still applies."""
    (ws, ys, prov), _ = lateral
    gains = np.ones(len(CH))
    gains[[CH.index(c) for c in ("Cz", "FC1", "FC2", "CP1", "CP2")]] = 2.5     # vertex electrodes
    tw, ty, _ = session(40, 1, True, classes=("FEET", "HANDS"), gains=gains)
    labels = ty[:, 1] > 0

    raw = make_decoder().fit(ws, ys, prov)
    acc_raw = np.mean((raw.predict_many(tw)[:, 1] > 0) == labels)

    dec = make_decoder()
    dec.fit(ws, ys, prov, session_references={0: dec.reference_from(rest(41))})
    dec.set_reference(rest(42, gains=gains))
    acc_rc = np.mean((dec.predict_many(tw)[:, 1] > 0) == labels)
    assert acc_rc > 0.75 and acc_rc > acc_raw + 0.15, (acc_rc, acc_raw)


def test_set_reference_validates_its_input():
    dec = make_decoder()
    with pytest.raises(ValueError, match="positive-definite"):
        dec.set_reference(reference=-np.eye(len(CENTRAL11)))
    with pytest.raises(ValueError, match="not both"):
        dec.set_reference(rest(0, n=2), reference=np.eye(len(CENTRAL11)))


# ---------------------------------------------------------------- local reference
def test_local_reference_cancels_what_the_global_car_carries_into_the_montage(fitted, lateral):
    """The real-time chain re-references to the mean of ALL good channels, so a blink, a saccadic spike
    or frontal EMG on FP1/FP2/F7/F8 enters every central channel at 1/32 of its size.  The decoder's
    central-11 re-reference cancels that term exactly: features and output are unchanged, whereas a
    decoder without it decodes a different window."""
    _, (ws, _, _) = lateral
    rng = np.random.default_rng(11)
    frontal = np.zeros((N_WIN, 32))
    frontal[:, [CH.index(c) for c in ("FP1", "FP2")]] = 150.0 * np.cumsum(rng.standard_normal((N_WIN, 1)), 0) / 20
    frontal[:, [CH.index(c) for c in ("F7", "F8")]] = 40.0 * rng.standard_normal((N_WIN, 2))     # broadband EMG
    good = np.ones(32, bool)
    clean, leaky = common_average(ws[0], good), common_average(ws[0] + frontal, good)
    central = [CH.index(c) for c in CENTRAL11]
    assert np.abs(leaky[:, central] - clean[:, central]).max() > 5.0          # CAR carried it in
    np.testing.assert_allclose(fitted.features(leaky), fitted.features(clean), atol=1e-8)
    np.testing.assert_array_equal(fitted.covariance(leaky).shape, (len(CENTRAL11) - 1,) * 2)
    np.testing.assert_allclose(fitted.predict(leaky), fitted.predict(clean), atol=1e-10)
    plain = make_decoder(local_reference=False)
    assert np.abs(plain.features(leaky) - plain.features(clean)).max() > 1e-2


def test_a_reference_in_channel_coordinates_is_mapped_into_the_local_basis(lateral):
    """``set_reference`` takes the rest covariance either as ``reference_from`` returns it or in channel
    coordinates (k × k), which the local reference maps by its basis: the same re-centring either way."""
    (ws, _, _), _ = lateral
    dec = make_decoder()
    rest_windows = rest(43)
    reduced = dec.reference_from(rest_windows)
    channel = np.mean([np.cov(w[:, [CH.index(c) for c in CENTRAL11]], rowvar=False) for w in rest_windows], 0)
    assert reduced.shape == (len(CENTRAL11) - 1,) * 2
    a = make_decoder().set_reference(reference=dec._basis.T @ channel @ dec._basis).features(ws[0])
    b = make_decoder().set_reference(reference=channel).features(ws[0])
    np.testing.assert_allclose(a, b, atol=1e-10)


# ---------------------------------------------------------------- idle gate
def _idle_prov(n, session=0):
    """Rest windows interleaved in time with the session's probes (the ITI after each trial), as the
    protocol records them: the gate's contiguous folds then each hold both classes."""
    return [{"source": IDLE_SOURCE, "session": session, "t": 6.0 * (i % (len(CLASSES) * TRIALS_PER_CLASS)) + 4.5}
            for i in range(n)]


def test_fit_refuses_rest_windows_and_the_gate_refuses_display_or_test_windows(lateral):
    (ws, ys, prov), _ = lateral
    rest_windows = rest(44, n=10)
    with pytest.raises(ValueError, match="trains only on"):
        make_decoder().fit(ws[:10] + rest_windows, np.vstack([ys[:10], np.zeros((10, 2), int)]),
                           prov[:10] + _idle_prov(10))
    dec = make_decoder().fit(ws, ys, prov)
    for source in ("S", "H", "R", "J", None):
        bad = _idle_prov(10)
        bad[3] = dict(bad[3], source=source)
        with pytest.raises(ValueError, match="idle gate trains only on"):
            dec.fit_idle(rest_windows, bad)
    with pytest.raises(ValueError, match="call fit first"):
        make_decoder().fit_idle(rest_windows, _idle_prov(10))


def _strong_responder(seed, session_id, erd=0.35):
    """Feet/hands probes of a person whose imagery keeps only ``erd`` of the source amplitude (power
    ÷ 8): rest and intent then differ clearly in absolute power, which a 1-s window can place."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.repeat(("FEET", "HANDS"), TRIALS_PER_CLASS))
    ws, ys, prov = [], [], []
    for trial, cls in enumerate(order):
        f = np.where(_erd(cls, False) < 1, erd, 1.0)
        for k in range(WINDOWS_PER_TRIAL):
            ws.append(window(rng, f))
            ys.append(CLASSES[cls])
            prov.append({"source": "L1", "session": session_id, "trial": trial, "t": 6.0 * trial + 1.0 + k})
    return ws, np.array(ys), prov


def test_an_armed_gate_zeroes_rest_exactly_and_passes_imagery():
    """Rest (no ERD) against the decoder's own training windows: the gate is armed when its out-of-fold
    AUC reaches 0.70, and then a rest window's output is exactly (0, 0) whenever p(idle) > 0.7; every
    window it lets through keeps the axis models' output unchanged."""
    ws, ys, prov = _strong_responder(48, 0)
    tw, _, _ = _strong_responder(49, 1)
    dec = make_decoder().fit(ws, ys, prov)
    ungated = dec.predict_many(tw)
    rep = dec.fit_idle(rest(45, n=64), _idle_prov(64))
    assert rep["armed"] and rep["cv_auc"] >= 0.70 and rep["n_intent"] == len(ws), rep
    test_rest = rest(46, n=60)
    rest_out, intent_out = dec.predict_many(test_rest), dec.predict_many(tw)
    zero = np.all(rest_out == 0.0, axis=1)
    np.testing.assert_array_equal(zero, dec.idle_probability(test_rest) > dec.idle_threshold)
    assert zero.mean() > 0.6 and np.all(intent_out[:, 0] == 0.0)
    passed = ~np.all(intent_out == 0.0, axis=1)
    assert passed.mean() > 0.8
    np.testing.assert_array_equal(intent_out[passed], ungated[passed])           # the gate only zeroes
    np.testing.assert_array_equal(dec.ungated().predict_many(tw), ungated)


def test_an_unplaceable_gate_is_not_armed_and_changes_nothing(lateral):
    """'Rest' windows drawn from the same imagery mixture as the training windows cannot be told apart:
    out-of-fold AUC ≈ 0.5, the gate stays unarmed, and the output is exactly the ungated one."""
    (ws, ys, prov), (tw, _, _) = lateral
    dec = make_decoder().fit(ws, ys, prov)
    before = dec.predict_many(tw[:40])
    lookalike, _, _ = session(47, 9, True)
    rep = dec.fit_idle(lookalike[:96], _idle_prov(96))
    assert not rep["armed"] and rep["cv_auc"] < 0.70, rep
    np.testing.assert_array_equal(dec.predict_many(tw[:40]), before)
    assert np.all(dec.idle_probability(tw[:5]) == 0.0)


# ---------------------------------------------------------------- in the real-time loop
@pytest.fixture(scope="module")
def loop_setup():
    """A decoder trained on windows from the loop's own preprocessing chain (``OnlinePreprocessor``:
    causal filter → gate → CAR) of a simulated SMR person (``sim_body.BodySubject``, strong feet/hands
    ERD), and that person as the live source."""
    dec = make_decoder()
    cfg = dec.loop_config(DEFAULT)
    person = BodySubject(cfg.hardware, seed=3, y_depth=0.4, x_depth=0.35, x_responder=False)
    person.set_day(1)
    person.begin_block(1, 0)
    pre = OnlinePreprocessor(cfg)
    pre.push(person.generate(2 * FS))
    ws, ys, prov = [], [], []
    for trial in range(40):
        vy = 1 if trial % 2 else -1
        person.cue(0, vy)
        for k in range(8):                                          # 0.5-s hops, windows 1.5–4 s after the cue
            pre.push(person.generate(FS // 2))
            if k >= 3:
                ws.append(pre.window())
                ys.append((0, vy))
                prov.append({"source": "L1", "session": "cal", "trial": trial, "t": 6.0 * trial + 0.5 * k})
        person.cue(0, 0)
        pre.push(person.generate(2 * FS))
    return dec.fit(ws, np.array(ys), prov), cfg, person


def _live(cfg, decoder, person, block, vy, source=None, seconds=3.0):
    person.begin_block(1, block)
    person.start()
    person.cue(0, vy)
    loop = RealTimeLoop(source or person, decoder, NullEffector(), cfg, warmup_s=1.0, sample_clock=time.time)
    return loop, loop.run(seconds)


def test_runs_in_the_real_time_loop_and_follows_the_imagined_body_part(loop_setup):
    """The loop hands the decoder (n_win, 32) windows of the length it was trained on, and its output
    follows the imagery: feet push y up, both hands down, x stays 0 (not earned), within the latency
    budget."""
    dec, cfg, person = loop_setup
    seen = set()

    class Spy:
        def predict(self, w, good=None):
            seen.add(w.shape)
            return dec.predict(w, good=good)

    feet, st = _live(cfg, Spy(), person, 1, +1)
    hands, _ = _live(cfg, dec, person, 2, -1)
    assert seen == {(N_WIN, 32)} and st["n_updates"] > 20 and st["n_invalid"] == 0
    assert feet.decoded[:, 1].mean() > hands.decoded[:, 1].mean() + 0.2
    assert np.all(feet.decoded[:, 0] == 0.0)
    assert st["latency_ms"]["p95"] < 50.0, st["latency_ms"]


class _Fault:
    """A source that turns faulty ``after_s`` seconds after it is created: a flat line at 250 µV on
    every channel (headbox unplugged), NaN on Cz (one electrode dropped, driver marks it), or C3 held at
    a constant 12.3 µV (one dead electrode, driver does NOT mark it)."""

    def __init__(self, inner, kind, after_s=1.5):
        self.inner, self.kind, self.after = inner, kind, after_s
        self.t0 = time.time()

    def pull(self):
        x, ts = self.inner.pull()
        if len(ts) and ts[-1] - self.t0 > self.after:
            x = np.full_like(x, 250.0) if self.kind == "flat" else x.copy()
            if self.kind == "nan_cz":
                x[:, CH.index("Cz")] = np.nan
            if self.kind == "dead_c3":
                x[:, CH.index("C3")] = 12.3
        return x, ts


@pytest.mark.parametrize("kind", ["flat", "nan_cz"])
def test_a_flat_or_dropped_central_channel_stops_the_cursor(loop_setup, kind):
    """Samples keep arriving, so the stale guard never fires.  Without a validity check a flat line drove
    y to −1.000 on every update and a dropped Cz (held at its last value) read as FEET; now the
    decoder refuses those windows and the loop sends counted zero commands."""
    dec, cfg, person = loop_setup
    person.begin_block(1, 9)
    person.start()
    person.cue(0, 0)
    loop = RealTimeLoop(_Fault(person, kind), dec, NullEffector(), cfg, warmup_s=1.0, sample_clock=time.time)
    st = loop.run(3.0)                                          # the fault starts ~0.5 s into the run
    assert st["n_invalid"] >= 0.6 * (st["n_invalid"] + st["n_updates"]), st
    assert np.all(np.abs(loop.decoded[:, 1]) < 0.999)           # no saturated command was ever sent


def test_a_dead_central_electrode_is_gated_and_stops_the_cursor(loop_setup):
    """Review: a constant-valued C3 passed every validity check (the decoder's dead-segment and power
    checks pool all montage dimensions) and drove y to −0.55 at rest.  The amplitude gate's low-RMS rule
    flags it, the decoder refuses windows with a bad central channel, and the loop sends counted zeros."""
    dec, cfg, person = loop_setup
    person.begin_block(1, 10)
    person.start()
    person.cue(0, 0)
    loop = RealTimeLoop(_Fault(person, "dead_c3", after_s=1.2), dec, NullEffector(), cfg, warmup_s=1.0,
                        sample_clock=time.time)
    st = loop.run(4.0)
    n = st["n_invalid"] + st["n_updates"]
    assert st["n_invalid"] >= 0.5 * n, st
    tail = int(1.0 * cfg.task.control_hz)                       # the last second: the fault long established
    assert np.all(loop.decoded[-tail:] == 0.0) or st["n_invalid"] >= 0.6 * n


def test_the_gate_flags_a_dead_channel_but_not_an_all_flat_window():
    from neural_cursor.signal import AmplitudeGate
    gate = AmplitudeGate(DEFAULT.hardware, DEFAULT.pre)
    rng = np.random.default_rng(0)
    w = rng.normal(0, 15.0, (N_WIN, len(CH)))
    assert gate.update(w).all()
    w[:, CH.index("C3")] = 0.0
    good = gate.update(w)
    assert not good[CH.index("C3")] and good.sum() == len(CH) - 1
    assert AmplitudeGate(DEFAULT.hardware, DEFAULT.pre).update(np.zeros((N_WIN, len(CH)))).all()


# ---------------------------------------------------------------- review round 3
def test_set_reference_without_arguments_returns_to_the_identity(lateral):
    """``set_reference()`` is the documented way back to no re-centring; it used to install a k × k
    identity (11) where the local-reference features are k − 1 (10), so every later predict crashed."""
    (ws, ys, prov), (tw, _, _) = lateral
    fresh = make_decoder().fit(ws, ys, prov)
    dec = make_decoder().fit(ws, ys, prov)
    dec.set_reference(rest_windows=rest(5))
    assert not np.allclose(dec.predict_many(tw[:10]), fresh.predict_many(tw[:10]))
    dec.set_reference()
    np.testing.assert_allclose(dec.predict_many(tw[:10]), fresh.predict_many(tw[:10]), atol=1e-12)


def test_ungated_copy_never_replaces_the_deployed_models(lateral):
    """``ungated`` used to share the model table, so fitting the copy rewrote the deployed decoder."""
    (ws, ys, prov), (tw, _, _) = lateral
    dec = make_decoder().fit(ws, ys, prov)
    before = dec.predict_many(tw[:10])
    copy_ = dec.ungated()
    assert copy_._models is not dec._models
    other = session(77, 5, False)
    copy_.fit(*other)
    np.testing.assert_array_equal(dec.predict_many(tw[:10]), before)


def test_x_model_trains_only_on_the_windows_x_was_earned_on(lateral):
    """``x_mask``: the x model that drives the cursor is the recipe ``earn_x`` certified (own L1), not
    one diluted by other windows (the review found 'earned' x below its own pool floor)."""
    (ws, ys, prov), (tw, ty, _) = lateral
    noise_w, noise_y, noise_p = session(90, 3, False)                       # a non-lateralised "prior"
    allw, ally, allp = ws + noise_w, np.concatenate([ys, noise_y]), prov + noise_p
    mask = np.r_[np.ones(len(ws), bool), np.zeros(len(noise_w), bool)]
    masked = make_decoder().fit(allw, ally, allp, x_mask=mask)
    alone = make_decoder().fit(ws, ys, prov)
    masked.x_enabled = alone.x_enabled = True
    xs = lambda d: d.ungated()._models["x"].decision_function(np.stack([d.features(w) for w in tw[:20]]))
    np.testing.assert_allclose(xs(masked), xs(alone), rtol=1e-6, atol=1e-8)
    with pytest.raises(ValueError, match="x_mask"):
        make_decoder().fit(ws, ys, prov, x_mask=np.ones(3, bool))


def test_neutral_one_hand_windows_keep_y_still_during_lateral_imagery(lateral):
    """Trained only on feet/hands, y reads one-hand ERD (on a hand generator) as HANDS, so an x earner
    imagining LEFT drives the cursor down.  ``neutral_one_hand`` enters one-hand windows as y = 0
    (both classes at half weight): their mean y output moves to ≈ 0 while feet/hands stay separated."""
    (ws, ys, prov), (tw, ty, _) = lateral
    plain = make_decoder().fit(ws, ys, prov)
    neutral = make_decoder().fit(ws, ys, prov, neutral_one_hand=True)
    one_hand = ty[:, 1] == 0
    y_one = lambda d: float(np.mean(d.predict_many([w for w, k in zip(tw, one_hand) if k])[:, 1]))
    assert y_one(plain) < -0.15 and abs(y_one(neutral)) < 0.5 * abs(y_one(plain))
    fh = ~one_hand
    auc = lambda d: roc_auc_score(ty[fh, 1] > 0, d.predict_many([w for w, k in zip(tw, fh) if k])[:, 1])
    assert auc(neutral) > 0.85 and neutral.neutral_one_hand and not plain.neutral_one_hand
