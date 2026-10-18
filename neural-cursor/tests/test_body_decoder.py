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

from neural_cursor.acquisition import SimulatedEEG
from neural_cursor.body_decoder import CENTRAL11, BodyCoordinateDecoder
from neural_cursor.config import DEFAULT
from neural_cursor.effectors import NullEffector
from neural_cursor.realtime import RealTimeLoop

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


def test_x_is_earned_with_a_lateralised_signal_and_then_decodes_right_vs_left():
    ws, ys, prov = _two_sessions(True, 20)
    dec = make_decoder(n_permutations=N_PERM_TEST).fit(ws, ys, prov)
    rep = dec.earn_x(ws, ys, prov)
    assert rep["enabled"] and dec.x_enabled and dec.x_report is rep
    assert rep["auc"] > rep["floor"] and rep["auc"] > 0.60 and rep["n_windows"] == 2 * 2 * 16 * 3
    tw, ty, _ = session(30, 2, True, classes=("LEFT", "RIGHT"))
    out = dec.predict_many(tw)
    assert roc_auc_score(ty[:, 0] > 0, out[:, 0]) > 0.65
    # earning is deterministic (fixed permutation seed)
    assert make_decoder(n_permutations=N_PERM_TEST).earn_x(ws, ys, prov) == rep


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


def test_runs_in_the_real_time_loop_on_32_channel_windows(fitted):
    """The loop's chain (filter → gate → CAR → predict) hands the decoder (n_win, 32) windows; with
    ``loop_config`` the window is the decoder's 1 s and every output is a bounded (2,) command."""
    seen = set()

    class Spy:
        def predict(self, w):
            seen.add(w.shape)
            return fitted.predict(w)

    cfg = fitted.loop_config(DEFAULT)
    sim = SimulatedEEG(cfg.hardware, seed=5)
    sim.start()
    loop = RealTimeLoop(sim, Spy(), NullEffector(), cfg, warmup_s=1.2, sample_clock=time.time)
    stats = loop.run(1.0)
    assert seen == {(N_WIN, 32)}
    assert stats["n_updates"] > 0 and loop.decoded.shape == (stats["n_updates"], 2)
    assert np.all(np.abs(loop.decoded) <= 1) and np.all(loop.decoded[:, 0] == 0.0)
