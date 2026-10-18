"""Real-data experiments on ds004362 (109 subjects).  Every number in DESIGN.md §real-data comes from here.

    python -I -m realdata.experiments features   --cache C --feat F          # covariances + slow features
    python -I -m realdata.experiments within     --feat F --out R            # E1 within-subject + floors
    python -I -m realdata.experiments confound   --feat F --out R            # E3 eyes/vision vs motor cortex
    python -I -m realdata.experiments scaling    --feat F --out R            # E2 cross-subject scaling (Riemannian)
    python -I -m realdata.experiments fewshot    --feat F --out R            # E2b pooled prior + k own trials
    python -I -m realdata.experiments online     --feat F --out R            # E6 collect-retrain-adapt replay
    python -I -m realdata.experiments transfer   --feat F --out R            # E4 executed→imagined, E5 idle

Conventions
- Montage of record = the 32 channels of the g.tec dry cap that exist in this 64-ch gel recording.
- SMR features: 8–30 Hz, 0.5–3.5 s after target onset, OAS covariance → Riemannian tangent space → L2
  logistic regression (C = 1).  Slow features: 1–4 Hz, mean amplitude in eight 0.5 s bins over 0–4 s.
- Calibration-free alignment (RA): each subject's covariances re-centred on the Riemannian mean of its
  own REST epochs (T0, unlabelled), so no target label is ever used for alignment.
- Floors are the null's 95th percentile at matched n (label permutation through the same CV), never 0.5.
- Fixed seeds; the cross-subject test set is fixed across every rung of every ladder.
Technical exclusion only (sampling rate ≠ 160 Hz or < 20 trials); never on performance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FS = 160
T0 = -1.0                         # epoch start in the cache
GTEC32 = ["Fp1", "Fp2", "Af3", "Af4", "F7", "F3", "Fz", "F4", "F8", "Fc5", "Fc1", "Fc2", "Fc6", "T7", "C3", "Cz",
          "C4", "T8", "Cp5", "Cp1", "Cp2", "Cp6", "P7", "P3", "Pz", "P4", "P8", "Po7", "Po3", "Po4", "Po8", "Oz"]
SUBSETS = {
    "ocular_frontal": ["Fp1", "Fpz", "Fp2", "Af7", "Af8", "F7", "F8"],
    "sensorimotor": ["Fc3", "Fc4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "Cp3", "Cp4"],
    "occipital": ["Po7", "Po3", "Poz", "Po4", "Po8", "O1", "Oz", "O2"],
    "gtec32": GTEC32,
    "all64": None,
}
TASKS = ["imag_LR", "exec_LR", "imag_FF", "exec_FF"]
SEED = 20261018


def _sl(t0, t1):
    return slice(int(round((t0 - T0) * FS)), int(round((t1 - T0) * FS)))


def bandpass(x, lo, hi):
    sos = butter(4, [lo, hi], "bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def covs(x):
    from pyriemann.estimation import Covariances
    return Covariances(estimator="oas").fit_transform(x.astype(np.float64))


def slow_feats(x):
    """1–4 Hz, mean amplitude in 8 bins of 0.5 s over 0–4 s → (n, ch*8).  The ocular/visual path."""
    xs = bandpass(x, 1.0, 4.0)[:, :, _sl(0.0, 4.0)]
    n, c, t = xs.shape
    return xs.reshape(n, c, 8, t // 8).mean(-1).reshape(n, c * 8)


# --------------------------------------------------------------------------------------- features
def _features_one(sub, cache, feat):
    out = os.path.join(feat, f"sub-{sub:03d}.npz")
    if os.path.exists(out):
        return sub
    d = np.load(os.path.join(cache, f"sub-{sub:03d}.npz"), allow_pickle=False)
    names = [str(n) for n in d["ch_names"]]
    low = {n.lower(): i for i, n in enumerate(names)}
    res = {"ch_names": np.array(names), "sfreq": d["sfreq"]}
    g = [low[c.lower()] for c in GTEC32]
    if "X_rest" in d:
        xr = bandpass(d["X_rest"], 8, 30)[:, :, _sl(0.5, 3.5)]
        res["Crest32"] = covs(xr[:, g])
        res["Crest64"] = covs(xr)
    for t in TASKS:
        if f"X_{t}" not in d:
            continue
        x = d[f"X_{t}"]
        xb = bandpass(x, 8, 30)[:, :, _sl(0.5, 3.5)]
        res[f"C32_{t}"] = covs(xb[:, g])
        res[f"y_{t}"] = d[f"y_{t}"]
        res[f"o_{t}"] = d[f"o_{t}"]
        for name, chans in SUBSETS.items():
            idx = list(range(len(names))) if chans is None else [low[c.lower()] for c in chans]
            if name != "gtec32":
                res[f"Csub_{name}_{t}"] = covs(xb[:, idx]) if name != "all64" else covs(xb)
            res[f"S_{name}_{t}"] = slow_feats(x[:, idx])
        # raw 32-ch 0.5–3.5 s, 4–40 Hz, for the deep model (float16 to keep it small)
        res[f"R32_{t}"] = bandpass(x[:, g], 4, 40)[:, :, _sl(0.5, 3.5)].astype(np.float16)
    if "X_rest" in d:
        res["Rrest32"] = bandpass(d["X_rest"][:, g], 4, 40)[:, :, _sl(0.5, 3.5)].astype(np.float16)
    np.savez(out, **res)
    return sub


def cmd_features(a):
    os.makedirs(a.feat, exist_ok=True)
    subs = sorted(int(f[4:7]) for f in os.listdir(a.cache) if f.startswith("sub-"))
    Parallel(n_jobs=a.jobs, verbose=0)(delayed(_features_one)(s, a.cache, a.feat) for s in subs)
    print("features for", len(subs), "subjects")


def load_feat(feat, sub):
    return dict(np.load(os.path.join(feat, f"sub-{sub:03d}.npz"), allow_pickle=False))


def usable(feat, task, min_trials=20):
    subs = []
    for f in sorted(os.listdir(feat)):
        if not f.startswith("sub-"):
            continue
        d = np.load(os.path.join(feat, f), allow_pickle=False)
        if list(d["sfreq"]) == [160.0] and f"y_{task}" in d.files and len(d[f"y_{task}"]) >= min_trials and "Crest32" in d.files:
            subs.append(int(f[4:7]))
    return subs


# --------------------------------------------------------------------------------------- models
def ts_lr():
    from pyriemann.tangentspace import TangentSpace
    return make_pipeline(TangentSpace(metric="riemann"), LogisticRegression(C=1.0, max_iter=2000))


def slow_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


def cv_acc(model_fn, X, y, seeds=(0, 1, 2), folds=5):
    accs = []
    for s in seeds:
        skf = StratifiedKFold(folds, shuffle=True, random_state=SEED + s)
        pred = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            m = model_fn().fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        accs.append(np.mean(pred == y))
    return float(np.mean(accs))


def recentre(C, ref):
    from pyriemann.utils.base import invsqrtm
    w = invsqrtm(ref)
    return w @ C @ w


def rest_ref(d, n=32):
    from pyriemann.utils.mean import mean_riemann
    return mean_riemann(d[f"Crest{n}"])


def summarize(vals, floor):
    v = np.asarray(vals)
    lo, hi = np.percentile([np.mean(np.random.default_rng(SEED + i).choice(v, len(v))) for i in range(2000)], [2.5, 97.5])
    return dict(n=len(v), mean=float(v.mean()), ci95=[float(lo), float(hi)], median=float(np.median(v)),
                floor_p95=floor, frac_above_floor=float(np.mean(v > floor)) if floor is not None else None)


def perm_floor(model_fn, Xs, ys, n_perm, rng):
    """p95 of CV accuracy under label permutation, pooled over the given subjects (matched n)."""
    null = []
    for X, y in zip(Xs, ys):
        for _ in range(n_perm):
            null.append(cv_acc(model_fn, X, rng.permutation(y), seeds=(0,)))
    return float(np.percentile(null, 95)), len(null)


# --------------------------------------------------------------------------------------- E1 within
def _within_one(sub, feat):
    d = load_feat(feat, sub)
    out = {}
    for t in TASKS:
        if f"y_{t}" in d:
            out[t] = cv_acc(ts_lr, d[f"C32_{t}"], d[f"y_{t}"])
    return sub, out


def cmd_within(a):
    subs = usable(a.feat, "imag_LR")
    res = Parallel(n_jobs=a.jobs)(delayed(_within_one)(s, a.feat) for s in subs)
    rng = np.random.default_rng(SEED)
    fl_subs = rng.choice(subs, 12, replace=False)
    Xs = [load_feat(a.feat, s)["C32_imag_LR"] for s in fl_subs]
    ys = [load_feat(a.feat, s)["y_imag_LR"] for s in fl_subs]
    floor, nnull = perm_floor(ts_lr, Xs, ys, 25, rng)
    table = {t: summarize([r[1][t] for r in res if t in r[1]], floor) for t in TASKS}
    out = dict(experiment="E1 within-subject CV, 32-ch g.tec montage, SMR tangent space", subjects=len(subs),
               floor_note=f"label-permutation p95 through the same 5-fold CV, {nnull} null draws on 12 subjects, n≈45",
               per_task=table, per_subject={int(s): r for s, r in res})
    _write(a.out, "E1_within", out)


# --------------------------------------------------------------------------------------- E3 confound
def _confound_one(sub, feat):
    d = load_feat(feat, sub)
    out = {}
    for t in ["imag_LR", "exec_LR", "imag_FF"]:
        if f"y_{t}" not in d:
            continue
        y = d[f"y_{t}"]
        for name in SUBSETS:
            C = d[f"C32_{t}"] if name == "gtec32" else d[f"Csub_{name}_{t}"]
            out[f"{t}|{name}|smr"] = cv_acc(ts_lr, C, y)
            out[f"{t}|{name}|slow"] = cv_acc(slow_lr, d[f"S_{name}_{t}"], y)
    return sub, out


def cmd_confound(a):
    subs = usable(a.feat, "imag_LR")
    res = Parallel(n_jobs=a.jobs)(delayed(_confound_one)(s, a.feat) for s in subs)
    rng = np.random.default_rng(SEED + 1)
    fl_subs = rng.choice(subs, 8, replace=False)
    floor_slow, _ = perm_floor(slow_lr, [load_feat(a.feat, s)["S_ocular_frontal_imag_LR"] for s in fl_subs],
                               [load_feat(a.feat, s)["y_imag_LR"] for s in fl_subs], 20, rng)
    keys = sorted(res[0][1].keys())
    table = {k: summarize([r[1][k] for r in res if k in r[1]], floor_slow) for k in keys}
    out = dict(experiment="E3 confound audit: which channels and which features decode the displayed target side",
               subjects=len(subs), floor_note="label-permutation p95, slow-feature pipeline, 160 null draws on 8 subjects",
               table=table, per_subject={int(s): r for s, r in res})
    _write(a.out, "E3_confound", out)


# --------------------------------------------------------------------------------------- E2 scaling
def _aligned(feat, subs, task, align=True):
    Cs, ys, ids = [], [], []
    for s in subs:
        d = load_feat(feat, s)
        C = d[f"C32_{task}"]
        if align:
            C = recentre(C, rest_ref(d))
        Cs.append(C)
        ys.append(d[f"y_{task}"])
        ids.append(np.full(len(C), s))
    return Cs, ys, ids


def split_subjects(subs, n_test=25):
    rng = np.random.default_rng(SEED)
    test = sorted(rng.choice(subs, n_test, replace=False).tolist())
    pool = [s for s in subs if s not in test]
    return test, pool


def _scaling_rung(N, rep, pool, Cs_pool, ys_pool, Cs_test, ys_test, ref_identity):
    rng = np.random.default_rng(SEED + 1000 * N + rep)
    pick = rng.choice(len(pool), N, replace=False)
    X = np.concatenate([Cs_pool[i] for i in pick])
    y = np.concatenate([ys_pool[i] for i in pick])
    from pyriemann.tangentspace import TangentSpace
    m = make_pipeline(TangentSpace(metric="riemann", tsupdate=False), LogisticRegression(C=1.0, max_iter=3000))
    m.fit(X, y)
    accs = [float(np.mean(m.predict(C) == yy)) for C, yy in zip(Cs_test, ys_test)]
    return N, rep, float(np.mean(accs)), accs


def cmd_scaling(a):
    task = a.task
    subs = usable(a.feat, task)
    test, pool = split_subjects(subs)
    out = dict(experiment=f"E2 cross-subject zero-shot scaling, {task}, 32-ch, TS+LR",
               test_subjects=test, pool_size=len(pool), rungs={})
    for align in (True, False):
        Cs_pool, ys_pool, _ = _aligned(a.feat, pool, task, align)
        Cs_test, ys_test, _ = _aligned(a.feat, test, task, align)
        Ns = [n for n in (1, 2, 4, 8, 16, 32, 48, 64, len(pool)) if n <= len(pool)]
        jobs = [(N, r) for N in Ns for r in range(8 if N < 16 else (4 if N < len(pool) else 1))]
        res = Parallel(n_jobs=a.jobs)(delayed(_scaling_rung)(N, r, pool, Cs_pool, ys_pool, Cs_test, ys_test, align) for N, r in jobs)
        tag = "RA_rest_aligned" if align else "unaligned"
        rung = {}
        for N in Ns:
            v = [r[2] for r in res if r[0] == N]
            per_test = np.mean([r[3] for r in res if r[0] == N], axis=0)
            rung[str(N)] = dict(mean=float(np.mean(v)), sd_over_draws=float(np.std(v)), draws=len(v),
                                per_test_subject_mean=float(per_test.mean()),
                                frac_test_subjects_above_0p64=float(np.mean(per_test > 0.644)))
        out["rungs"][tag] = rung
        print(tag, {k: round(v["mean"], 3) for k, v in rung.items()}, flush=True)
    out["floor_note"] = ("zero-shot accuracy on 25 fixed held-out subjects × ~45 trials; per-subject binomial p95 at n=45 is "
                         "0.644; for the 25-subject mean the label-permutation p95 is ≈0.52")
    _write(a.out, f"E2_scaling_{task}", out)


# --------------------------------------------------------------------------------------- E2b few-shot
def cmd_fewshot(a):
    task = a.task
    subs = usable(a.feat, task)
    test, pool = split_subjects(subs)
    Cs_pool, ys_pool, _ = _aligned(a.feat, pool, task, True)
    Xp, yp = np.concatenate(Cs_pool), np.concatenate(ys_pool)
    ks = [0, 4, 8, 16, 24, 30]
    from pyriemann.tangentspace import TangentSpace

    def one(s):
        d = load_feat(a.feat, s)
        C = recentre(d[f"C32_{task}"], rest_ref(d))
        y = d[f"y_{task}"]
        o = np.lexsort((np.arange(len(y)), d[f"o_{task}"]))        # recording order
        C, y = C[o], y[o]
        te = slice(len(y) - 15, len(y))                              # fixed test: the last 15 trials
        r = {}
        for k in ks:
            Xk, yk = C[:k], y[:k]
            # pooled prior + own trials up-weighted
            w = np.r_[np.ones(len(yp)), np.full(k, max(1.0, len(yp) / 300.0))]
            m = make_pipeline(TangentSpace(metric="riemann", tsupdate=False), LogisticRegression(C=1.0, max_iter=3000))
            m.fit(np.concatenate([Xp, Xk]) if k else Xp, np.r_[yp, yk] if k else yp,
                  logisticregression__sample_weight=w if k else None)
            r[f"pooled+own_{k}"] = float(np.mean(m.predict(C[te]) == y[te]))
            if k >= 4 and len(set(yk)) == 2:
                m2 = ts_lr().fit(Xk, yk)
                r[f"own_only_{k}"] = float(np.mean(m2.predict(C[te]) == y[te]))
            else:
                r[f"own_only_{k}"] = 0.5
        return s, r

    res = Parallel(n_jobs=a.jobs)(delayed(one)(s) for s in test)
    keys = list(res[0][1].keys())
    out = dict(experiment=f"E2b pooled population prior + k own labelled trials vs own-only, {task}; fixed test = last 15 trials",
               test_subjects=test, table={k: summarize([r[1][k] for r in res], None) for k in keys},
               note="own_only with k<4 or one class is reported as 0.5 (cannot be trained)")
    _write(a.out, f"E2b_fewshot_{task}", out)


# --------------------------------------------------------------------------------------- E6 online
def cmd_online(a):
    """Collect → retrain → adapt, replayed trial by trial in recording order (predict, then learn)."""
    task = a.task
    subs = usable(a.feat, task)
    test, pool = split_subjects(subs)
    Cs_pool, ys_pool, _ = _aligned(a.feat, pool, task, True)
    Xp, yp = np.concatenate(Cs_pool), np.concatenate(ys_pool)
    from pyriemann.tangentspace import TangentSpace
    prior = make_pipeline(TangentSpace(metric="riemann", tsupdate=False), LogisticRegression(C=1.0, max_iter=3000)).fit(Xp, yp)

    def one(s):
        d = load_feat(a.feat, s)
        C = recentre(d[f"C32_{task}"], rest_ref(d))
        y = d[f"y_{task}"]
        o = np.lexsort((np.arange(len(y)), d[f"o_{task}"]))
        C, y = C[o], y[o]
        hits = {"population_frozen": [], "own_from_scratch": [], "population_then_adapt": []}
        for t in range(len(y)):
            hits["population_frozen"].append(int(prior.predict(C[t:t + 1])[0] == y[t]))
            if t >= 6 and len(set(y[:t])) == 2:
                hits["own_from_scratch"].append(int(ts_lr().fit(C[:t], y[:t]).predict(C[t:t + 1])[0] == y[t]))
            else:
                hits["own_from_scratch"].append(int(np.random.default_rng(SEED + t + s).integers(0, 2) == y[t]))
            if t >= 2:
                w = np.r_[np.ones(len(yp)), np.full(t, max(1.0, len(yp) / 300.0))]
                m = make_pipeline(TangentSpace(metric="riemann", tsupdate=False), LogisticRegression(C=1.0, max_iter=3000))
                m.fit(np.concatenate([Xp, C[:t]]), np.r_[yp, y[:t]], logisticregression__sample_weight=w)
                hits["population_then_adapt"].append(int(m.predict(C[t:t + 1])[0] == y[t]))
            else:
                hits["population_then_adapt"].append(hits["population_frozen"][-1])
        return s, hits

    res = Parallel(n_jobs=a.jobs)(delayed(one)(s) for s in test)
    blocks = {}
    for k in res[0][1]:
        M = np.array([r[1][k][:45] for r in res if len(r[1][k]) >= 45], float)
        blocks[k] = [float(M[:, i:i + 9].mean()) for i in range(0, 45, 9)]
    out = dict(experiment=f"E6 sequential predict-then-learn replay, {task}, 25 held-out subjects, blocks of 9 trials",
               test_subjects=test, block_accuracy=blocks,
               note="own_from_scratch guesses at random until it has 6 trials of both classes")
    _write(a.out, f"E6_online_{task}", out)


# --------------------------------------------------------------------------------------- E4/E5
def _transfer_one(sub, feat):
    d = load_feat(feat, sub)
    r = {}
    if "y_exec_LR" in d and "y_imag_LR" in d:
        m = ts_lr().fit(d["C32_exec_LR"], d["y_exec_LR"])
        r["exec_to_imag_LR"] = float(np.mean(m.predict(d["C32_imag_LR"]) == d["y_imag_LR"]))
        m = ts_lr().fit(d["C32_imag_LR"], d["y_imag_LR"])
        r["imag_to_exec_LR"] = float(np.mean(m.predict(d["C32_exec_LR"]) == d["y_exec_LR"]))
        X = np.concatenate([d["C32_imag_LR"], d["C32_exec_LR"]])
        y = np.r_[d["y_imag_LR"], d["y_exec_LR"]]
        isimag = np.r_[np.ones(len(d["y_imag_LR"])), np.zeros(len(d["y_exec_LR"]))].astype(bool)
        # imagined test accuracy when executed trials are added to training (5-fold over imagined)
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        Xi, yi = d["C32_imag_LR"], d["y_imag_LR"]
        pred = np.zeros(len(yi))
        for tr, te in skf.split(Xi, yi):
            m = ts_lr().fit(np.concatenate([Xi[tr], d["C32_exec_LR"]]), np.r_[yi[tr], d["y_exec_LR"]])
            pred[te] = m.predict(Xi[te])
        r["imag_LR_cv_plus_exec_data"] = float(np.mean(pred == yi))
    if "Crest32" in d and "C32_imag_LR" in d:
        Xr = d["Crest32"]
        Xt = d["C32_imag_LR"]
        n = min(len(Xr), len(Xt))
        X = np.concatenate([Xr[:n], Xt[:n]])
        y = np.r_[np.zeros(n), np.ones(n)].astype(int)
        r["idle_vs_imagine"] = cv_acc(ts_lr, X, y)
    return sub, r


def cmd_transfer(a):
    subs = usable(a.feat, "imag_LR")
    res = Parallel(n_jobs=a.jobs)(delayed(_transfer_one)(s, a.feat) for s in subs)
    keys = sorted({k for _, r in res for k in r})
    out = dict(experiment="E4 executed↔imagined transfer and E5 idle-vs-imagine detection, 32-ch TS+LR",
               subjects=len(subs), table={k: summarize([r[k] for _, r in res if k in r], None) for k in keys})
    _write(a.out, "E4E5_transfer_idle", out)


def _write(out_dir, name, obj):
    os.makedirs(out_dir, exist_ok=True)
    obj["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)
    print("wrote", os.path.join(out_dir, name + ".json"), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--cache")
    ap.add_argument("--feat", required=True)
    ap.add_argument("--out", default="results/realdata")
    ap.add_argument("--task", default="imag_LR")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    {"features": cmd_features, "within": cmd_within, "confound": cmd_confound, "scaling": cmd_scaling,
     "fewshot": cmd_fewshot, "online": cmd_online, "transfer": cmd_transfer}[a.cmd](a)
