"""E2-deep: does a general learned model (EEGNet-8,2) improve with more people?  Zero-shot to 25 fixed
held-out subjects, Euclidean alignment from each subject's own REST epochs (no target labels).

    realdata/run.sh deep --feat F --out results/realdata --task imag_LR

Hyperparameters of record (fixed in advance, not tuned on the test subjects):
  input 32 ch × 480 samples (0.5–3.5 s, 4–40 Hz, 160 Hz); EEGNet F1=8, D=2, F2=16, kernel 64,
  dropout 0.25; AdamW lr 1e-3, weight decay 1e-2, batch 64, 60 epochs, no early stopping;
  seeds 0 and 1 per rung; torch threads 4.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from realdata.experiments import SEED, load_feat, split_subjects, usable, _write


class EEGNet(nn.Module):
    def __init__(self, n_ch=32, n_t=480, F1=8, D=2, F2=16, k=64, p=0.25, n_cls=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(p),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(p), nn.Flatten())
        with torch.no_grad():
            n_out = self.net(torch.zeros(1, 1, n_ch, n_t)).shape[1]
        self.fc = nn.Linear(n_out, n_cls)

    def forward(self, x):
        return self.fc(self.net(x.unsqueeze(1)))


def ea(X, rest):
    """Euclidean alignment: whiten by the subject's own rest covariance, then unit scale."""
    R = np.mean([r @ r.T / r.shape[1] for r in rest.astype(np.float64)], axis=0)
    w, v = np.linalg.eigh(R)
    W = (v * (1.0 / np.sqrt(np.clip(w, 1e-10, None)))) @ v.T
    return np.einsum("ij,njt->nit", W, X.astype(np.float64)).astype(np.float32)


def load_aligned(feat, subs, task):
    Xs, ys = [], []
    for s in subs:
        d = load_feat(feat, s)
        Xs.append(ea(d[f"R32_{task}"], d["Rrest32"]))
        ys.append(d[f"y_{task}"].astype(np.int64))
    return Xs, ys


def train_eval(Xtr, ytr, Xtests, ytests, seed, epochs=60):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = EEGNet(n_ch=Xtr.shape[1], n_t=Xtr.shape[2])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss()
    X, y = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            lossf(m(X[b]), y[b]).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        return [float((m(torch.from_numpy(Xt)).argmax(1).numpy() == yt).mean()) for Xt, yt in zip(Xtests, ytests)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--out", default="results/realdata")
    ap.add_argument("--task", default="imag_LR")
    ap.add_argument("--rungs", default="4,8,16,32,64,all")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(4)
    subs = usable(a.feat, a.task)
    test, pool = split_subjects(subs)
    Xp, yp = load_aligned(a.feat, pool, a.task)
    Xt, yt = load_aligned(a.feat, test, a.task)
    out = dict(experiment=f"E2-deep EEGNet zero-shot scaling, {a.task}, 32-ch, rest-EA", test_subjects=test,
               pool_size=len(pool), rungs={})
    for r in a.rungs.split(","):
        N = len(pool) if r == "all" else int(r)
        vals, per = [], []
        for sd in range(a.seeds if N < len(pool) else max(1, a.seeds)):
            rng = np.random.default_rng(SEED + 7 * N + sd)
            pick = rng.choice(len(pool), N, replace=False)
            t0 = time.time()
            accs = train_eval(np.concatenate([Xp[i] for i in pick]), np.concatenate([yp[i] for i in pick]), Xt, yt, SEED + sd)
            vals.append(float(np.mean(accs)))
            per.append(accs)
            print(f"N={N} seed={sd} zero-shot={vals[-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)
        out["rungs"][str(N)] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals)), draws=len(vals),
                                    frac_test_subjects_above_0p64=float(np.mean(np.mean(per, axis=0) > 0.644)))
        _write(a.out, f"E2deep_scaling_{a.task}", out)


if __name__ == "__main__":
    main()
