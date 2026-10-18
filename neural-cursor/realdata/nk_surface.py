"""E8: people × trials-per-person scaling surface on ds004362 (INTERFACES_CHAMPION item L).

Question.  For a population prior on the champion's central montage, does the next labelled trial buy
more when it comes from a NEW person or from one MORE trial of a person already in the pool?  E2 varied
only the number of people (all ~45 trials each) and E2b/E6 only the own-trial count of the test person;
neither separates breadth from depth of the training pool.  The proposal (§3c) leaves the people axis to
existing data, and the skeptic corrections (C4) say the trials-per-person axis is untested beyond 45
trials — so this measures both inside the only range real data cover.

Design (all pre-declared here, nothing tuned on the result)
- Features: the champion's central-11 montage (FC5 FC1 FC2 FC6 C3 Cz C4 CP5 CP1 CP2 CP6), 8–30 Hz,
  0.5–3.5 s after target onset, OAS covariance, re-centred on the Riemannian mean of the person's own
  REST (T0) covariances, Riemannian tangent space at the training mean, L2 logistic regression (C = 1).
  Exactly the deployed BodyCoordinateDecoder's per-axis pipeline, so the surface speaks to that decoder.
- Training pool: N people in {2, 4, 8, 16, 32, 64, 81} drawn from the 81 non-test subjects, each giving
  their FIRST k trials in recording order, k in {10, 20, 30, 45} (a person with fewer trials gives all).
  First-k, not random-k, because that is how a calibration session accrues data.
- Draws: 4 random people-draws per N (1 at N = 81, the whole pool).  A draw's people are shared by every
  k, so the k-effect is estimated on paired pools; the seed per (N, draw) equals E2's, so the k = 45
  column is E2's people-draws re-run on the central-11 montage.
- Test: zero-shot on E2's fixed 25 held-out subjects (experiments.split_subjects), all their trials.
- Tasks: imag_FF (both fists vs both feet → the champion's y axis) and imag_LR (→ the x axis).
- Model of the surface: acc = a + b·log2(N) + c·log2(k), OLS on the 28 cell means.  b and c are the
  gains per doubling; since doubling N and doubling k both double the total trial count, b − c is the
  equal-budget answer to "people or trials-per-person".  95% CIs: bootstrap over test subjects (the
  declared resampling unit), plus a wider one that also resamples people-draws within each N.  A
  log2(N)·log2(k) interaction fit is reported as a check on the additive form, and per-cell residuals
  show where the log-linear form breaks (E2 saturates at 32–48 people).
- Robustness (added after the first run showed the largest residuals at N = 2): the same fit without
  the N = 2 row, reported beside the declared fit and labelled post hoc, never replacing it.

Every input is EEG; no eye, EMG or behavioural channel is read.

    realdata/run.sh nk_surface --cache CACHE --feat FEAT --nkfeat NKFEAT --out results/realdata
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from realdata.experiments import (SEED, _sl, _write, bandpass, covs, load_feat, recentre, rest_ref,
                                  split_subjects, usable)

CENTRAL11 = ["Fc5", "Fc1", "Fc2", "Fc6", "C3", "Cz", "C4", "Cp5", "Cp1", "Cp2", "Cp6"]   # dataset spelling
BAND_HZ = (8.0, 30.0)
WINDOW_S = (0.5, 3.5)
TASKS = ("imag_FF", "imag_LR")
NS = (2, 4, 8, 16, 32, 64, 81)
KS = (10, 20, 30, 45)
DRAWS = 4                  # people-draws per N below the full pool (the full pool has exactly one)
LR_C = 1.0                 # same L2 strength as E2 and the deployed decoder
LR_MAX_ITER = 3000
N_BOOT = 2000
ROBUST_MIN_N = 4           # post hoc robustness fit drops the N = 2 row (largest residuals in the first run)
ALPHA_FLOOR = 0.05         # per-subject binomial floor: one-sided p < 0.05 against chance at that n
# Recorded in every feature file; a file with a different spec is recomputed, never silently reused.
FEATURE_SPEC = f"central11={','.join(CENTRAL11)};band={BAND_HZ};win={WINDOW_S};cov=oas;tasks={','.join(TASKS)}"


# --------------------------------------------------------------------------------------- features
def subject_features(cache: str, nkfeat: str, sub: int) -> int:
    """Central-11 task and rest covariances for one subject, cached as nkfeat/sub-XXX.npz.

    The file keeps experiments.load_feat's layout (C11_<task>, y_<task>, o_<task>, Crest11) so that
    load_feat and rest_ref(d, n=11) read it directly.  Written to a temp name then renamed, so an
    interrupted run never leaves a truncated file that a later run would trust.
    """
    out = os.path.join(nkfeat, f"sub-{sub:03d}.npz")
    if os.path.exists(out):
        with np.load(out, allow_pickle=False) as d:
            if "spec" in d.files and str(d["spec"]) == FEATURE_SPEC:
                return sub
    d = np.load(os.path.join(cache, f"sub-{sub:03d}.npz"), allow_pickle=False)
    low = {str(n).lower(): i for i, n in enumerate(d["ch_names"])}
    idx = [low[c.lower()] for c in CENTRAL11]
    win = _sl(*WINDOW_S)
    res = {"spec": np.array(FEATURE_SPEC), "ch_names": np.array(CENTRAL11), "sfreq": d["sfreq"]}
    if "X_rest" in d.files:
        res["Crest11"] = covs(bandpass(d["X_rest"][:, idx], *BAND_HZ)[:, :, win])
    for t in TASKS:
        if f"X_{t}" in d.files:
            res[f"C11_{t}"] = covs(bandpass(d[f"X_{t}"][:, idx], *BAND_HZ)[:, :, win])
            res[f"y_{t}"] = d[f"y_{t}"]
            res[f"o_{t}"] = d[f"o_{t}"]
    tmp = os.path.join(nkfeat, f".sub-{sub:03d}.tmp.npz")
    np.savez(tmp, **res)
    os.replace(tmp, out)
    return sub


def load_subjects(nkfeat: str, subs: list[int], task: str):
    """Rest-re-centred central-11 covariances and labels per subject, sorted into recording order.

    Recording order = run index, then position within the run (stable sort), so "first k trials" is
    what a calibration session would have produced after k cues.
    """
    Cs, ys = [], []
    for s in subs:
        d = load_feat(nkfeat, s)
        order = np.lexsort((np.arange(len(d[f"y_{task}"])), d[f"o_{task}"]))
        Cs.append(recentre(d[f"C11_{task}"], rest_ref(d, n=11))[order])
        ys.append(d[f"y_{task}"].astype(int)[order])
    return Cs, ys


# --------------------------------------------------------------------------------------- surface
def _model():
    from pyriemann.tangentspace import TangentSpace
    # tsupdate=False: test people are mapped at the TRAINING mean, as a deployed prior would be.
    return make_pipeline(TangentSpace(metric="riemann", tsupdate=False),
                         LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER))


def _draw_people(N: int, draw: int, pool_size: int) -> np.ndarray:
    """Same seed formula as experiments._scaling_rung, so k = 45 reproduces E2's people-draws."""
    return np.random.default_rng(SEED + 1000 * N + draw).choice(pool_size, N, replace=False)


def run_draw(N, draw, ks, Cs_pool, ys_pool, Cs_test, ys_test):
    """One people-draw, every k: returns {k: (per-test-subject accuracies, trials used)}."""
    pick = _draw_people(N, draw, len(Cs_pool))
    out = {}
    for k in ks:
        X = np.concatenate([Cs_pool[i][:k] for i in pick])
        y = np.concatenate([ys_pool[i][:k] for i in pick])
        m = _model().fit(X, y)
        out[k] = ([float(np.mean(m.predict(C) == yy)) for C, yy in zip(Cs_test, ys_test)], int(len(y)))
    return N, draw, out


def _design(Ns, ks, interaction=False):
    lN, lk = np.meshgrid(np.log2(Ns), np.log2(ks), indexing="ij")
    cols = [np.ones(lN.size), lN.ravel(), lk.ravel()]
    if interaction:
        cols.append((lN * lk).ravel())
    return np.column_stack(cols)


def _ols(D, cell_means):
    return np.linalg.lstsq(D, cell_means.ravel(), rcond=None)[0]


def fit_surface(acc, Ns, ks, n_boot=N_BOOT):
    """Log-linear surface fit with bootstrap CIs.

    acc[N][k] is an array (draws, n_test) of accuracies.  The point estimate is OLS on the cell means
    (the design is balanced, so this equals OLS on every subject × cell point).  Two bootstraps:
    'subjects' resamples the 25 test subjects (the declared CI); 'subjects_and_draws' also resamples
    people-draws within each N, with one resample per N shared by every k because draws are paired
    across k.  P(b > c) is the bootstrap share in which people beat trials-per-person per doubling.
    """
    n_test = acc[Ns[0]][ks[0]].shape[1]
    D, Di = _design(Ns, ks), _design(Ns, ks, interaction=True)
    per_subj = np.stack([[acc[N][k].mean(0) for k in ks] for N in Ns])        # (nN, nk, n_test)
    means = per_subj.mean(-1)
    coef, coef_i = _ols(D, means), _ols(Di, means)
    resid = means.ravel() - D @ coef
    r2 = 1.0 - resid.var() / means.ravel().var()

    rng = np.random.default_rng(SEED + 8)
    boot = {"subjects": [], "subjects_and_draws": []}
    boot_i = []
    for _ in range(n_boot):
        s = rng.integers(0, n_test, n_test)
        m = per_subj[:, :, s].mean(-1)
        boot["subjects"].append(_ols(D, m))
        boot_i.append(_ols(Di, m))
        md = np.empty_like(m)
        for i, N in enumerate(Ns):
            nd = acc[N][ks[0]].shape[0]
            dr = rng.integers(0, nd, nd)
            for j, k in enumerate(ks):
                md[i, j] = acc[N][k][dr][:, s].mean()
        boot["subjects_and_draws"].append(_ols(D, md))

    def ci(v):
        return [float(x) for x in np.percentile(v, [2.5, 97.5])]

    res = dict(model="acc = a + b*log2(N) + c*log2(k)",
               a=float(coef[0]), b_per_doubling_people=float(coef[1]), c_per_doubling_trials=float(coef[2]),
               b_minus_c=float(coef[1] - coef[2]), r2_on_cell_means=float(r2),
               max_abs_residual=float(np.abs(resid).max()), ci95={})
    for name, B in boot.items():
        B = np.asarray(B)
        res["ci95"][name] = dict(a=ci(B[:, 0]), b=ci(B[:, 1]), c=ci(B[:, 2]), b_minus_c=ci(B[:, 1] - B[:, 2]),
                                 p_b_gt_c=float(np.mean(B[:, 1] > B[:, 2])))
    Bi = np.asarray(boot_i)
    res["interaction_check"] = dict(model="acc = a + b*log2(N) + c*log2(k) + d*log2(N)*log2(k)",
                                    a=float(coef_i[0]), b=float(coef_i[1]), c=float(coef_i[2]), d=float(coef_i[3]),
                                    d_ci95_subjects=ci(Bi[:, 3]))
    lo, hi = res["ci95"]["subjects"]["b_minus_c"]
    res["verdict"] = ("PEOPLE buy more per doubling" if lo > 0 else
                      "TRIALS-PER-PERSON buy more per doubling" if hi < 0 else
                      "INDISTINGUISHABLE: the b - c CI contains 0")
    return res, means, (D @ coef).reshape(means.shape)


def surface(nkfeat: str, test: list[int], pool: list[int], task: str, Ns, ks, draws, jobs, n_boot):
    """Every (N, draw) pool fitted at every k, scored zero-shot on the fixed test set, then fitted.

    Parallel over (N, draw) rather than over cells, so the k sweep for one draw reuses its people.
    """
    Cs_pool, ys_pool = load_subjects(nkfeat, pool, task)
    Cs_test, ys_test = load_subjects(nkfeat, test, task)
    Ns = [N for N in Ns if N <= len(pool)]
    jobs_ = [(N, r) for N in Ns for r in range(1 if N == len(pool) else draws)]
    res = Parallel(n_jobs=jobs)(delayed(run_draw)(N, r, ks, Cs_pool, ys_pool, Cs_test, ys_test)
                                for N, r in jobs_)
    acc = {N: {k: np.array([o[k][0] for n, _, o in res if n == N]) for k in ks} for N in Ns}
    used = {N: {k: [o[k][1] for n, _, o in res if n == N] for k in ks} for N in Ns}
    fit, means, fitted = fit_surface(acc, Ns, ks, n_boot)
    robust_Ns = [N for N in Ns if N >= ROBUST_MIN_N]
    fit_robust = fit_surface(acc, robust_Ns, ks, n_boot)[0]

    n_trials = np.array([len(y) for y in ys_test])
    floor = binom.ppf(1 - ALPHA_FLOOR, n_trials, 0.5) / n_trials                 # per-subject, at its own n
    cells = {}
    for i, N in enumerate(Ns):
        for j, k in enumerate(ks):
            a = acc[N][k]
            per_test = a.mean(0)
            cells[f"N{N}_k{k}"] = dict(
                N=N, k=k, draws=int(a.shape[0]), mean_train_trials=float(np.mean(used[N][k])),
                mean=float(means[i, j]), sd_over_draws=float(a.mean(1).std()), fitted=float(fitted[i, j]),
                residual=float(means[i, j] - fitted[i, j]),
                frac_test_subjects_above_binomial_floor=float(np.mean(per_test > floor)),
                per_test_subject=[round(float(v), 4) for v in per_test])
    return dict(
        experiment=f"E8 people × trials-per-person scaling surface, {task}, central-11, rest-recentred TS+LR",
        task=task, channels=CENTRAL11, band_hz=list(BAND_HZ), window_s=list(WINDOW_S),
        Ns=Ns, ks=list(ks), draws_per_N=draws, test_subjects=test, pool_size=len(pool),
        trial_order="first k trials in recording order (run, then position in run)",
        fit=fit,
        fit_post_hoc_without_N2=dict(Ns=robust_Ns, note=("post hoc robustness check: the N = 2 cells carry the "
                                                         "largest residuals; the declared fit above is the result"),
                                     **fit_robust),
        cells=cells,
        floor_note=(f"per-test-subject binomial floor (one-sided p < {ALPHA_FLOOR} vs 0.5 at the subject's own n; "
                    "0.644 at n = 45); the 25-subject mean's label-permutation p95 is about 0.52 (E2)"),
        reading=("b and c are accuracy gained per doubling of people and of trials per person; doubling either "
                 "doubles the total labelled trials, so b - c compares them at equal budget. Bootstrap over the "
                 "25 test subjects is the declared CI; subjects_and_draws also carries people-draw variance."))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True, help="raw epoch cache from realdata.load (64 ch, -1..4 s, 160 Hz)")
    ap.add_argument("--feat", required=True, help="E-series feature dir: only picks usable subjects and the fixed split")
    ap.add_argument("--nkfeat", required=True, help="where the central-11 covariances are cached")
    ap.add_argument("--out", default="results/realdata")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--Ns", default=",".join(map(str, NS)))
    ap.add_argument("--ks", default=",".join(map(str, KS)))
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.nkfeat, exist_ok=True)
    tasks = a.tasks.split(",")
    subs_by_task = {t: usable(a.feat, t) for t in tasks}
    all_subs = sorted(set().union(*subs_by_task.values()))
    Parallel(n_jobs=a.jobs)(delayed(subject_features)(a.cache, a.nkfeat, s) for s in all_subs)
    print("central-11 features for", len(all_subs), "subjects", flush=True)
    for t in tasks:
        test, pool = split_subjects(subs_by_task[t])
        out = surface(a.nkfeat, test, pool, t, [int(x) for x in a.Ns.split(",")],
                      [int(x) for x in a.ks.split(",")], a.draws, a.jobs, a.boot)
        f = out["fit"]
        print(t, f"b={f['b_per_doubling_people']:+.4f} c={f['c_per_doubling_trials']:+.4f}",
              f"b-c CI={f['ci95']['subjects']['b_minus_c']}", f["verdict"], flush=True)
        _write(a.out, f"E8_nk_surface_{t}", out)


if __name__ == "__main__":
    main()
