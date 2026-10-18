"""Independent replication: does an eye proxy out-decode motor cortex on ds004362, cross-subject?

The tournament's population finalist reported that slow frontal activity plus derived EOG decodes the
imagined left/right label better than 8–30 Hz sensorimotor rhythms when pooled across people.  This
script re-derives it from the raw epoch cache with code written separately (no shared functions with
that analysis), on the same fixed 25 held-out subjects as E2.

Features
- eye proxy: 1–8 Hz zero-phase band, mean amplitude in 16 bins of 0.25 s over 0–4 s, on Fp1, Fpz, Fp2,
  Af7, Af8, F7, F8 plus HEOG = F8 − F7 and VEOG = mean(Fp1, Fp2)  (9 + … = 10 signals × 16 bins)
- motor: 8–30 Hz, 0.5–3.5 s, OAS covariance on 14 sensorimotor/parietal channels, re-centred on the
  person's own rest covariance, tangent space
Both: standardised, L2 logistic regression, trained on N pooled people, tested zero-shot.

    realdata/run.sh eye_proxy --cache C --out results/realdata
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from realdata.experiments import SEED, split_subjects, usable, _write

FS, T0 = 160, -1.0
EYE = ["Fp1", "Fpz", "Fp2", "Af7", "Af8", "F7", "F8"]
MOTOR = ["Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "C3", "C1", "Cz", "C2", "C4", "Cp3", "Cpz", "Cp4", "Pz"]


def _sl(a, b):
    return slice(int(round((a - T0) * FS)), int(round((b - T0) * FS)))


def _bp(x, lo, hi):
    return sosfiltfilt(butter(4, [lo, hi], "bandpass", fs=FS, output="sos"), x, axis=-1)


def subject_features(cache, sub, task):
    d = np.load(os.path.join(cache, f"sub-{sub:03d}.npz"), allow_pickle=False)
    names = [str(n).lower() for n in d["ch_names"]]
    ix = lambda chans: [names.index(c.lower()) for c in chans]
    X, y = d[f"X_{task}"].astype(np.float64), d[f"y_{task}"].astype(int)
    e = _bp(X[:, ix(EYE)], 1.0, 8.0)
    heog = e[:, EYE.index("F8")] - e[:, EYE.index("F7")]
    veog = 0.5 * (e[:, EYE.index("Fp1")] + e[:, EYE.index("Fp2")])
    e = np.concatenate([e, heog[:, None], veog[:, None]], axis=1)[:, :, _sl(0.0, 4.0)]
    n, c, t = e.shape
    eye = e.reshape(n, c, 16, t // 16).mean(-1).reshape(n, -1)
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.utils.mean import mean_riemann
    m = _bp(X[:, ix(MOTOR)], 8.0, 30.0)[:, :, _sl(0.5, 3.5)]
    C = Covariances("oas").fit_transform(m)
    rest = Covariances("oas").fit_transform(_bp(d["X_rest"][:, ix(MOTOR)].astype(np.float64), 8.0, 30.0)[:, :, _sl(0.5, 3.5)])
    ref = mean_riemann(rest)
    w, v = np.linalg.eigh(ref)
    W = (v / np.sqrt(w)) @ v.T
    C = W @ C @ W
    motor = TangentSpace(metric="riemann").fit(np.eye(len(MOTOR))[None].repeat(2, 0)).transform(C)
    return eye, motor, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--feat", required=True, help="feature dir (only used to pick usable subjects and the split)")
    ap.add_argument("--out", default="results/realdata")
    ap.add_argument("--task", default="imag_LR")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    subs = usable(a.feat, a.task)
    test, pool = split_subjects(subs)
    feats = dict(zip(subs, Parallel(n_jobs=a.jobs)(delayed(subject_features)(a.cache, s, a.task) for s in subs)))
    out = dict(experiment=f"eye proxy vs motor cortex, cross-subject zero-shot, {a.task}", test_subjects=test,
               eye_channels=EYE + ["HEOG=F8-F7", "VEOG=mean(Fp1,Fp2)"], motor_channels=MOTOR, rungs={})
    for N in (8, 16, 32, 64, len(pool)):
        res = {"eye": [], "motor": [], "both": []}
        for r in range(4 if N < len(pool) else 1):
            pick = np.random.default_rng(SEED + 31 * N + r).choice(pool, N, replace=False)
            for k, j in (("eye", 0), ("motor", 1)):
                Xtr = np.concatenate([feats[s][j] for s in pick]); ytr = np.concatenate([feats[s][2] for s in pick])
                m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(Xtr, ytr)
                res[k].append(np.mean([np.mean(m.predict(feats[s][j]) == feats[s][2]) for s in test]))
            Xtr = np.concatenate([np.hstack([feats[s][0], feats[s][1]]) for s in pick]); ytr = np.concatenate([feats[s][2] for s in pick])
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(Xtr, ytr)
            res["both"].append(np.mean([np.mean(m.predict(np.hstack([feats[s][0], feats[s][1]])) == feats[s][2]) for s in test]))
        out["rungs"][str(N)] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v)), draws=len(v)) for k, v in res.items()}
        print(N, {k: round(float(np.mean(v)), 3) for k, v in res.items()}, flush=True)
    _write(a.out, f"E7_eye_proxy_{a.task}", out)


if __name__ == "__main__":
    main()
