"""Resource ladder: decoder quality vs training volume with the TEST SET HELD FIXED (rule 52).

Reports evidence-decoder AUC and velocity-decoder r as a function of the number of labelled
segments, for a fixed held-out set drawn first.  This is the in-simulation version of the scaling
curve the real protocol pre-registers; in the simulator the signal is stationary, so the curve
saturates at the decoder's noise ceiling, not at a physiological one.

Run:  python -m neural_cursor.scaling_ladder --users 4 --errp 6
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from .config import DEFAULT
from .decoder import EvidenceDecoder, VelocityDecoder
from .evaluation import pearson_r
from .simulate import calibration
from .simulate_composite import SimUser2, evidence_calibration

LADDER = (40, 80, 160, 320, 640)


def run(users: int, gain: float, errp: float, seed: int):
    rng = np.random.default_rng(seed)
    auc = {k: [] for k in LADDER}
    rr = {k: [] for k in LADDER}
    for u in range(users):
        user = SimUser2(DEFAULT, seed + 100 * u, gain, 0.0, errp, 0.0)
        ews, eys = evidence_calibration(user, 160, rng)                 # fixed test set, drawn first
        vws, vys = calibration(user, 120, rng)
        tews, teys = evidence_calibration(user, max(LADDER), rng)      # training pool
        tvws, tvys = calibration(user, max(LADDER), rng)
        for k in LADDER:
            e = EvidenceDecoder(DEFAULT.decoder).fit(tews[:k], teys[:k])
            auc[k].append(roc_auc_score(eys, e.predict_many(ews)))
            v = VelocityDecoder(DEFAULT.decoder, DEFAULT.hardware.fs).fit(tvws[:k], tvys[:k])
            p = v.predict_many(vws)
            rr[k].append(np.mean([pearson_r(p[:, j], vys[:, j]) for j in range(2)]))
    return {k: dict(evidence_auc=float(np.mean(auc[k])), evidence_auc_sd=float(np.std(auc[k])),
                    velocity_r=float(np.mean(rr[k])), velocity_r_sd=float(np.std(rr[k]))) for k in LADDER}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--gain", type=float, default=8.0)
    ap.add_argument("--errp", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=DEFAULT.eval.seed)
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()
    res = run(a.users, a.gain, a.errp, a.seed)
    print("| labelled segments | evidence AUC (sd) | velocity r (sd) |\n|---|---|---|")
    for k, v in res.items():
        print(f"| {k} | {v['evidence_auc']:.3f} ({v['evidence_auc_sd']:.3f}) | {v['velocity_r']:.3f} ({v['velocity_r_sd']:.3f}) |")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=2)
