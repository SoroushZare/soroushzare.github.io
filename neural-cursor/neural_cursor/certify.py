"""Closed-loop certification: does the deployed decoder follow the body, or the screen?

Why a regression on body, screen and cross-axis regressors
----------------------------------------------------------
Under the training mapping (identity) a decoder that reads body-part intent and one that reads the
screen — gaze toward the target, posterior alpha lateralised toward it, visual responses to the moving
cursor — make the same prediction on every window: the person wants the cursor to go up, imagines
feet, and the target is above the cursor.  Accuracy under the identity can never tell them apart.
Under a held-out mapping ``M`` (``mapping.HELDOUT_MAPPINGS``) the person must produce body velocity
``M⁻¹ (target − cursor)`` while the screen still shows ``target − cursor``; under rot±90 the two are
orthogonal on every trial.  Regressing the decoder's output (body coordinates) per axis a on

    output_a = β_body · body_a + β_screen · screen_a + β_cross · body_ā + c_a + ε       (ā: the other axis)

splits its direction-following into the part that tracks the body, the part that tracks the screen and
the part that is CROSS-AXIS BODY COUPLING (e.g. an x model that reacts a little to feet/hands imagery
through an individual topography).  The cross term is not optional: under rot+90, screen_y = body_x and
screen_x = −body_y, so without it any cross-axis coupling lands in β_screen and reads as screen reading
with no screen pathway at all.  Under rot−90 the signs flip (screen_y = −body_x).  Only data that contain
both rotations separate the two: the rotation-symmetric part is β_screen, the antisymmetric part is
β_cross.  A test unit with a single rotation therefore leaves β_screen unidentified (``identifiable``
False → TEST_INSENSITIVE), which is why the protocol runs rot+90 and rot−90 in every session.  A mirror
flips one axis and leaves body = ±screen on it, so mirror windows identify nothing per axis.

Why the regressors must also be uncorrelated block by block
-----------------------------------------------------------
Identifiability only asks that the regressors are not collinear; a design in which body_a and screen_a
are still strongly correlated (e.g. a 1-D y person under rot±90 with targets placed so that the
reachable horizontal offset predicts the vertical one) leaves β_screen to be separated by a model the
data barely constrain, and any non-linearity of the body response then leaks into it.  Held-out blocks
therefore use diagonal targets with balanced signs (``mapping.diagonal_targets``,
``mapping.quadrant_schedule``), and every entry reports the body–screen correlation per block
(``block_corr``; a block is a provenance "session", i.e. one run, judged when it holds at least
``MIN_CLUSTERS`` trials — fewer give a chance correlation).  An entry whose correlation exceeds
``MAX_BLOCK_CORR`` (0.5) in any block, or overall, is ``confounded`` and refused outright: a non-linear
body response (the output's clip to ±1, a saturating logistic) leaks into β_screen with EITHER sign
there, so neither a pass nor a SCREEN_READING is read from it, the per-unit veto skips it, and
``combine_units`` leaves it out of the combination.

Pixel replay (``replay_test``) is the same regression with the person's private, pre-committed intent as
the body regressor and the replayed cursor direction as the screen regressor.

Why cluster-robust CR2 standard errors with t(G − k), and a minimum number of trials
------------------------------------------------------------------------------------
Windows of one trial overlap and share the trial's direction, the person's strategy and slow EEG state,
so they are not independent: a window-level interval is too narrow by roughly √(windows per trial).
Inference is therefore clustered on whole trials ``(session, trial)``.  It used to be a percentile
trial-cluster bootstrap, which undercovers badly with few clusters: on rot±90 diagonal geometry with a
true β_screen of 0 its 95 % interval missed 0 in 25 % of units of 8 trials and 10 % of 12 (review
finding).  A single unit here is one decoder version with 17–35 held-out trials, so small-G behaviour
is the normal case.  Each fit now uses the Bell–McCaffrey CR2 cluster-robust covariance (each trial's
residuals are rescaled by (I − H_gg)^{-1/2}, which undoes the shrinkage of in-sample residuals) with a
t reference on G − k degrees of freedom (G trials, k coefficients).  On the same geometry its 95 %
interval misses 0 in 5.5–6.3 % of units at 8, 12, 16, 20 and 30 trials (2,000 units each;
``tests/test_certify.py`` keeps a coverage regression).  ``MIN_CLUSTERS`` (8) trials remain the floor
below which no interval is attempted (G − k would leave ≤ 3 df); replay keeps ``MIN_REPLAY_CLUSTERS``
(30) because it has only ~10 trials a day and is pooled over days by design, not tested per day.

Why per-unit fits, combined, instead of one pooled OLS
------------------------------------------------------
Decoder gain differs a lot between people and days (and the held-out mapping mix differs too), so one
common β_body over several units leaves (β_unit − β_mean)·body in the residual, which correlates with
the screen regressor and biases β_screen in a direction set by which unit got which mapping.  Each unit
(a person's decoder version, or a person) is therefore fitted on its own (its own slopes and
intercepts), expressed relative to its own margin, and the per-unit estimates are combined
(``combine_units``): inverse-variance (fixed effect) over the days of one person — the claim is about
those decoders; over people either random effects (DerSimonian–Laird τ², Hartung–Knapp–Sidik–Jonkman t
interval) or a person-then-trial bootstrap ("bootstrap": people are resampled, then each drawn person's
trials, and the equal-weight mean of the people's relative βs is recomputed), whose interval carries
between-person variance by construction.  Any unit that alone reads the screen vetoes the combined
verdict (``certify``); because a study has many units, that per-unit test is Bonferroni-corrected over
every unit and axis, so the veto does not turn the family's chance findings into a false SCREEN_READING.

Why the screen bound is two-sided (TOST), and the body bound has a lower limit
-----------------------------------------------------------------------------
"Screen reading bounded below 10 % of the margin" is a statement about |β_screen|: a decoder whose output
follows the display with a NEGATIVE sign still carries display information to the cursor (the simulator
produces leaks of both signs: a decoder trained on L1 alone picked up −13 % and −23 % of its motor
slope from the display in two of six people).  A one-sided "β_screen < 10 %" passes such a decoder more
easily than a clean one.  The bound is therefore an equivalence test (two one-sided tests, TOST): both
β_screen < +10 % and β_screen > −10 % of the margin must be shown, and a β_screen significantly
different from 0 in EITHER direction that is not bounded inside ±10 % is SCREEN_READING.  The body
criterion is the design's "β_body ≥ 70 % of the margin with lower CI ≥ 50 %" (DESIGN_PURE_EEG §5, F1b):
the point estimate must reach ``body_pass`` and β_body > ``body_lower`` · margin must be shown.

Why every axis must pass, and why Holm
--------------------------------------
``certify`` reads the verdict on every tested axis separately and returns the worst: one β_screen shared
over x and y would let a clean y axis dilute screen reading on x.  The "pooled" entry (both axes, one
β set) is still reported, for continuity, but never decides.  Each verdict rests on a family of
one-sided tests — per tested axis: deployed β_body > 0, deployed β_body > ``body_lower`` · margin, the
two halves of the screen equivalence test, and β_screen above its null for EACH shadow.  "Either shadow
reads the screen" is a union of tests and the verdict claims all of the family at once, so the family is
Holm-corrected (``holm``) at α/2 one-sided (one test alone then decides exactly as its 95 % CI would).
Detecting screen reading in the deployed decoder is NOT in the family: it is a negative finding, and
correcting it would make a leak harder to report.

Why positive controls, two of them, and why they are null-calibrated
--------------------------------------------------------------------
"β_screen ≈ 0" is evidence of purity only if the same test, on the same blocks, *does* detect screen
reading when it is there.  The shadow decoders (trained on sighted ReFIT labels, never deployed) are
that positive control.  The central-11 shadow is the deployed model with display-identifiable labels;
it differs from the deployed decoder in label provenance AND in training data — it trains on sighted
(S) windows only, about a third as many as the deployed decoder's own L1/L2b windows, and without the
population prior — so its sensitivity is not that of a matched-n twin.  It can only learn what reaches
the central strip: in 8–30 Hz central features an up/down screen leak is weak, so on y it often learns
the motor signal instead.  The all-32-channel shadow sees the whole posterior and frontal display
signal.  Either counts.

A shadow's β_screen is not 0 just because there is no display pathway: in closed loop and in replay the
regressors correlate with time in the trial (corr(τ, screen_y) = 0.27 in replay), and a shadow that
learned the motor signal on ReFIT labels responds to that misspecification.  In the committed
screen_leak 0 simulation a central-11 shadow showed β_screen 1.5× the margin (Holm p 0.003) with no
screen pathway at all.  "Shadow β_screen > 0" can therefore be passed by a nonspecific response.  The
positive control is calibrated against a NULL: ``shadow_null`` holds each shadow's result on data with
no display pathway (the simulation's screen_leak 0 twin of the same people, days and noise streams;
real data need a surrogate, see INTERFACES_CHAMPION), and a shadow counts as reading the screen only if
its relative β_screen is shown ABOVE its null's (a Welch test on the difference, the null's own
uncertainty included).  Without ``shadow_null`` the test runs against 0, is flagged uncalibrated, and
the verdict is never CERTIFIED.  The shadows' sensitivity is that of a large leak; the bound on a small
one is the deployed TOST, which must lie inside ±``SCREEN_TOL`` of the margin.

Scale: an external margin, with its uncertainty
-----------------------------------------------
Thresholds are fractions of the decoder's margin: its output slope on body intent under the identity,
measured on identity trials interleaved with the held-out ones (same blocks, same shift: moving eyes, a
visible cursor) and never on the windows being certified.  Pass it as ``margin`` (one value, or one per
axis).  Without it the margin is estimated as β_body + β_screen from the same fit; that self-scale makes
the 70 % body criterion vacuous (bounding β_screen below 10 % already puts β_body above 90 %), so such a
result (``margin_source`` "self") is never CERTIFIED: it is only for synthetic checks.  The external
margin is itself an estimate from a handful of identity trials (CV 0.3–0.4 at 7–12 trials), and a
margin drawn high loosens the screen bound in proportion.  ``margin_se`` / ``margin_df`` (its standard
error and degrees of freedom, e.g. from a CR2 fit over the identity trials) make every threshold test a
test of a ratio: "β_screen < tol · m" is tested as β_screen − tol·m < 0 with variance
se²(β_screen) + tol²·se²(m) and Welch–Satterthwaite degrees of freedom (the identity trials are other
trials than the held-out ones, so the two estimates are independent).  Relative βs carry delta-method
SEs (for combining units) and Fieller intervals (reported: the set of ratios those linear tests do not
reject, so an interval excluding 0 or ±10 % agrees with the verdict).  The tests against 0 do not depend
on the margin.

Pure-EEG rule: this module sees only decoder outputs and task geometry, never a signal of any kind.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

from .data_engine import trial_clusters

AXES: Tuple[str, ...] = ("x", "y")
N_BOOT = 2000               # person-then-trial bootstrap draws (``combine_units``): percentiles stable to ~0.005
MIN_BOOT = 1000             # fewer draws make the bootstrap's own SD noisy
ALPHA = 0.05                # two-sided: 95 % CIs; one-sided tests run at ALPHA / 2
SEED = 20261018             # fixes every bootstrap draw, so a verdict is reproducible from the logged outputs
MIN_CLUSTERS = 8            # below this a CR2 t interval has ≤ 3 df: no interval is attempted
MIN_REPLAY_CLUSTERS = 30    # replay has ~10 trials a day: it is pooled over days, never tested per day
MAX_REGRESSOR_CORR = 0.98   # above this multiple correlation of a regressor with the others it is not
                            # separable (VIF > 25): a single rotation (screen = ±cross), the identity
                            # and a mirror's unflipped axis (screen = body) land here
MAX_BLOCK_CORR = 0.5        # |corr(body, screen)| above this in any block of ≥ MIN_CLUSTERS trials (or
                            # overall): confounded, never certified (module docstring); balanced
                            # diagonal targets give ≈ 0
MAX_COND = 1e10             # bootstrap draws whose normal equations are this ill-conditioned are dropped
MIN_LEVERAGE_GAP = 1e-8     # eigenvalues of I − H_gg below this are clipped before the CR2 inverse root
SCREEN_TOL = 0.10           # |β_screen| must be shown < 10 % of the margin (proposal P2/P3, two-sided)
BODY_PASS = 0.70            # β_body must reach 70 % of the margin (proposal P2); 50–70 % is inconclusive
BODY_LOWER = 0.50           # …and β_body > 50 % of the margin must be shown (DESIGN §5 "lower CI ≥ 50 %", F1b)

HELDOUT_SOURCES: Tuple[str, ...] = ("H",)   # provenance "source" of held-out-mapping test blocks
REPLAY_SOURCES: Tuple[str, ...] = ("R",)    # provenance "source" of pixel-replay test trials

VERDICTS: Tuple[str, ...] = ("SCREEN_READING", "NO_BODY_SIGNAL", "TEST_INSENSITIVE", "INCONCLUSIVE",
                             "CERTIFIED")
"""All verdicts, worst first: ``certify(axis="all")`` returns the first of these that any axis gets."""
COMBINATIONS: Tuple[str, ...] = ("fixed", "random", "bootstrap")

Margin = Union[None, float, Mapping[str, float]]
_COEF = {"body": 0, "screen": 1, "cross": 2}   # column of each β in the regressor matrix


def _validate(outputs, body, screen, provenance, allowed_sources: Tuple[str, ...], axes: Sequence[str],
              min_clusters: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out = np.asarray(outputs, float)
    b = np.asarray(body, float)
    s = np.asarray(screen, float)
    for name, a in (("outputs", out), ("body regressor", b), ("screen regressor", s)):
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError(f"{name} must be (n, 2) in body coordinates, got {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} contains non-finite values")
    if not (len(out) == len(b) == len(s) == len(provenance)):
        raise ValueError("outputs, both regressors and provenance must have one row per window")
    # A missing source counts as refused: provenance without it cannot show the window is test-only.
    bad = sorted({str(p.get("source")) for p in provenance if p.get("source") not in allowed_sources})
    if bad:
        raise ValueError(f"windows with source {bad} cannot enter this test (allowed: {allowed_sources}): "
                         "only test-only blocks are evidence, and training blocks are seen by the decoder")
    if not axes or any(a not in AXES for a in axes) or len(set(axes)) != len(axes):
        raise ValueError(f"axes must be a non-empty subset of {AXES}, got {tuple(axes)}")
    cl = trial_clusters(provenance)
    if cl.size == 0 or cl.max() + 1 < min_clusters:
        raise ValueError(f"need >= {min_clusters} trials for cluster-robust inference, "
                         f"got {0 if cl.size == 0 else cl.max() + 1}")
    return out, b, s, cl


def window_key(provenance: Sequence[dict], body: np.ndarray, screen: np.ndarray) -> str:
    """Short hash of the windows' identities ((session, trial, t)) and both regressors: two results
    with the same key were computed on the same windows with the same geometry."""
    h = hashlib.sha256()
    h.update(json.dumps([(str(p["session"]), int(p["trial"]), None if p.get("t") is None else float(p["t"]))
                         for p in provenance]).encode())
    h.update(np.ascontiguousarray(body, np.float64).tobytes())
    h.update(np.ascontiguousarray(screen, np.float64).tobytes())
    return h.hexdigest()[:16]


def _regressor_corr(body_cols: Sequence[np.ndarray], screen_cols: Sequence[np.ndarray]) -> float:
    """corr(body, screen) after removing each axis's mean (what the per-axis intercepts absorb); NaN if
    either regressor is constant."""
    b = np.concatenate([c - c.mean() for c in body_cols])
    s = np.concatenate([c - c.mean() for c in screen_cols])
    nb, ns = np.linalg.norm(b), np.linalg.norm(s)
    if nb < 1e-12 or ns < 1e-12:
        return float("nan")
    return float(b @ s / (nb * ns))


def _block_corr(body: np.ndarray, screen: np.ndarray, blocks: np.ndarray, cl: np.ndarray,
                axes: Sequence[int]) -> Dict[str, float]:
    """``_regressor_corr`` within each block (one run) of at least ``MIN_CLUSTERS`` trials: a confound
    the whole-unit correlation can hide when blocks of opposite sign are mixed.  A block with fewer trials
    (e.g. the ~3 replay trials of one sighted run) has a correlation that is mostly chance; it is left
    out here and counts only through the unit's overall correlation."""
    out = {}
    for b in np.unique(blocks):
        m = blocks == b
        if len(np.unique(cl[m])) >= MIN_CLUSTERS:
            out[str(b)] = _regressor_corr([body[m, a] for a in axes], [screen[m, a] for a in axes])
    return out


def _collinearity(X: np.ndarray, n_reg: int) -> float:
    """Largest multiple correlation of one of the first ``n_reg`` columns (the regressors) with all the
    other columns (the other regressors and the per-axis intercepts).  1 (or NaN for a constant
    regressor) means some β is not identified."""
    worst = 0.0
    for j in range(n_reg):
        others = np.delete(X, j, axis=1)
        coef, *_ = np.linalg.lstsq(others, X[:, j], rcond=None)
        resid = X[:, j] - others @ coef
        intercepts = X[:, n_reg:]
        mean_fit = intercepts @ np.linalg.lstsq(intercepts, X[:, j], rcond=None)[0]
        tss = float(np.sum((X[:, j] - mean_fit) ** 2))
        if tss < 1e-12:
            return float("nan")
        worst = max(worst, float(np.sqrt(max(0.0, 1.0 - np.sum(resid ** 2) / tss))))
    return worst


def _cluster_stats(X: np.ndarray, y: np.ndarray, cl: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-trial sufficient statistics XᵀX (n_clusters, p, p) and Xᵀy (n_clusters, p): what the
    person-then-trial bootstrap of ``combine_units`` refits from."""
    xtx = np.zeros((n_clusters, X.shape[1], X.shape[1]))
    np.add.at(xtx, cl, X[:, :, None] * X[:, None, :])
    xty = np.zeros((n_clusters, X.shape[1]))
    np.add.at(xty, cl, X * y[:, None])
    return xtx, xty


def cluster_robust_ols(X: np.ndarray, y: np.ndarray, cl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and their Bell–McCaffrey CR2 cluster-robust covariance (module docstring);
    ``cl`` holds each row's cluster (trial) id.  Use t(G − k) degrees of freedom with it."""
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ (X.T @ y)
    resid = y - X @ beta
    order = np.argsort(cl, kind="stable")
    bounds = np.flatnonzero(np.diff(cl[order])) + 1
    meat = np.zeros_like(bread)
    for rows in np.split(order, bounds):
        Xg = X[rows]
        w, V = np.linalg.eigh(np.eye(len(rows)) - Xg @ bread @ Xg.T)
        adj = (V * np.maximum(w, MIN_LEVERAGE_GAP) ** -0.5) @ V.T
        score = Xg.T @ (adj @ resid[rows])
        meat += np.outer(score, score)
    return beta, bread @ meat @ bread


def _resample_counts(rng: np.random.Generator, n_draws: int, n_clusters: int) -> np.ndarray:
    """(n_draws, n_clusters) multiplicities of whole trials drawn with replacement."""
    idx = rng.integers(0, n_clusters, (n_draws, n_clusters))
    return np.bincount((idx + n_clusters * np.arange(n_draws)[:, None]).ravel(),
                       minlength=n_draws * n_clusters).reshape(n_draws, n_clusters).astype(float)


def _solve_draws(xtx: np.ndarray, xty: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """OLS coefficients for each row of trial weights ``counts`` (draws, trials); NaN rows where the
    re-weighted normal equations are too ill-conditioned to solve."""
    A = np.einsum("bc,cij->bij", counts, xtx)
    v = np.einsum("bc,ci->bi", counts, xty)
    out = np.full((len(counts), xtx.shape[1]), np.nan)
    ok = np.linalg.cond(A) < MAX_COND
    if ok.any():
        out[ok] = np.linalg.solve(A[ok], v[ok][..., None])[..., 0]
    return out


def welch(terms: Sequence[Tuple[float, Optional[float]]]) -> Tuple[float, Optional[float]]:
    """Total variance of independent estimates and its Welch–Satterthwaite degrees of freedom.

    ``terms`` are (variance, df) pairs; df None means a normal (infinite-df) component.  Returns
    (variance, df), df None when every non-zero component is normal."""
    var = float(sum(v for v, _ in terms))
    finite = [(v, d) for v, d in terms if v > 0 and d is not None and np.isfinite(d)]
    if not finite or var <= 0:
        return var, None
    return var, float(var ** 2 / sum(v ** 2 / d for v, d in finite))


def _t_quantile(q: float, df: Optional[float]) -> float:
    return float(stats.t.ppf(q, df) if df else stats.norm.ppf(q))


def _per_axis(value, axes: Sequence[str], name: str) -> Optional[Dict[str, float]]:
    """``value`` (None, a float or {axis: float}) as {axis: float} over ``axes``; None stays None."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        if any(a not in value for a in axes):
            raise ValueError(f"{name} needs a value for every tested axis {tuple(axes)}, got {dict(value)}")
        return {a: None if value[a] is None else float(value[a]) for a in axes}
    return {a: float(value) for a in axes}


def _entry(out: np.ndarray, body: np.ndarray, screen: np.ndarray, cl: np.ndarray, blocks: np.ndarray,
           axes: Sequence[int], alpha: float, scale: Optional[float], scale_se: float,
           scale_df: Optional[float], cross_axis: bool) -> dict:
    """One regression over the given axes (one axis, or several pooled with an intercept per axis)."""
    n, k = len(out), len(axes)
    n_reg = 3 if cross_axis else 2
    X = np.zeros((n * k, n_reg + k))
    for j, a in enumerate(axes):
        rows = slice(j * n, (j + 1) * n)
        X[rows, 0] = body[:, a]
        X[rows, 1] = screen[:, a]
        if cross_axis:
            X[rows, 2] = body[:, 1 - a]               # the other axis's body intent, same sign convention
        X[rows, n_reg + j] = 1.0
    corr = _regressor_corr([body[:, a] for a in axes], [screen[:, a] for a in axes])
    block_corr = _block_corr(body, screen, blocks, cl, axes)
    finite = [abs(c) for c in block_corr.values() if np.isfinite(c)]
    max_block = float(max(finite)) if finite else float("nan")
    collin = _collinearity(X, n_reg)
    n_clusters = int(cl.max()) + 1
    df = n_clusters - X.shape[1]
    entry = {"n_windows": n, "n_rows": n * k, "n_clusters": n_clusters, "regressor_corr": corr,
             "block_corr": block_corr, "max_block_corr": max_block,
             "confounded": bool((np.isfinite(corr) and abs(corr) > MAX_BLOCK_CORR)
                                or (np.isfinite(max_block) and max_block > MAX_BLOCK_CORR)),
             "collinearity": collin, "cross_axis": cross_axis,
             "identifiable": bool(np.isfinite(collin) and collin <= MAX_REGRESSOR_CORR and df >= 1)}
    nan, nan2 = float("nan"), (float("nan"), float("nan"))
    if not entry["identifiable"]:
        entry.update(df=None, beta_body=nan, beta_screen=nan, beta_cross=nan, ci_body=nan2, ci_screen=nan2,
                     ci_cross=nan2, se_body=nan, se_screen=nan, se_cross=nan, intercept=[nan] * k,
                     scale=nan if scale is None else scale, scale_se=scale_se, scale_df=scale_df,
                     rel_body=nan, rel_screen=nan, rel_ci_body=nan2, rel_ci_screen=nan2, rel_se_body=nan,
                     rel_se_screen=nan, rel_df_body=None, rel_df_screen=None, cluster_stats=None)
        return entry
    y = np.concatenate([out[:, a] for a in axes])
    cl_rows = np.tile(cl, k)
    beta, cov = cluster_robust_ols(X, y, cl_rows)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t = _t_quantile(1 - alpha / 2, df)
    if scale is None:                                      # self-scale: a synthetic-check ruler, no SE
        scale, scale_se, scale_df = float(beta[0] + beta[1]), 0.0, None
    entry.update(df=df, beta_body=float(beta[0]), beta_screen=float(beta[1]),
                 beta_cross=float(beta[2]) if cross_axis else nan,
                 se_body=float(se[0]), se_screen=float(se[1]), se_cross=float(se[2]) if cross_axis else nan,
                 ci_body=(float(beta[0] - t * se[0]), float(beta[0] + t * se[0])),
                 ci_screen=(float(beta[1] - t * se[1]), float(beta[1] + t * se[1])),
                 ci_cross=(float(beta[2] - t * se[2]), float(beta[2] + t * se[2])) if cross_axis else nan2,
                 intercept=[float(c) for c in beta[n_reg:]], scale=scale, scale_se=scale_se, scale_df=scale_df,
                 cluster_stats=dict(zip(("xtx", "xty"), _cluster_stats(X, y, cl_rows, n_clusters))))
    for key in ("body", "screen"):
        b, s = entry[f"beta_{key}"], entry[f"se_{key}"]
        if scale > 0:
            # delta-method SE of b / m (independent estimates of b and m) for combining units, and a
            # Fieller interval for reporting: it holds every ratio r whose linear test b − r·m = 0 the
            # threshold tests (``_p_value``) would not reject, so a reported interval that excludes 0 or
            # ±10 % says exactly what the verdict says (a delta interval can include 0 while β alone is
            # significant, because it charges the margin's error to the sign question too).
            var, rdf = welch([(s ** 2 / scale ** 2, df), (b ** 2 * scale_se ** 2 / scale ** 4, scale_df)])
            rse, rel = float(np.sqrt(var)), b / scale
            q = _t_quantile(1 - alpha / 2, rdf if scale_se > 0 else df)
            entry.update({f"rel_{key}": float(rel), f"rel_se_{key}": rse, f"rel_df_{key}": rdf,
                          f"rel_ci_{key}": _fieller(b, s, scale, scale_se, q)})
        else:
            entry.update({f"rel_{key}": nan, f"rel_se_{key}": nan, f"rel_df_{key}": None,
                          f"rel_ci_{key}": nan2})
    return entry


def _fieller(b: float, se_b: float, m: float, se_m: float, q: float) -> Tuple[float, float]:
    """Fieller interval for b / m (independent estimates): every r with (b − r·m)² ≤ q²·(se_b² + r²·se_m²).
    Unbounded (±inf) when the margin itself is not clearly positive (m² ≤ q²·se_m²)."""
    a = m * m - q * q * se_m * se_m
    if a <= 0:
        return float("-inf"), float("inf")
    root = float(np.sqrt(b * b * q * q * se_m * se_m + a * q * q * se_b * se_b))
    return float((b * m - root) / a), float((b * m + root) / a)


def _check_margin(margin: Margin, margin_se, margin_df, axes: Sequence[str]) -> None:
    if margin is None:
        if margin_se is not None or margin_df is not None:
            raise ValueError("margin_se / margin_df need an external margin")
        return
    values = _per_axis(margin, axes, "margin")
    if not all(np.isfinite(v) and v > 0 for v in values.values()):
        raise ValueError(f"margin must be a positive slope (per axis), got {margin}")
    ses = _per_axis(margin_se, axes, "margin_se")
    if ses is not None and not all(v is not None and np.isfinite(v) and v >= 0 for v in ses.values()):
        raise ValueError(f"margin_se must be finite and >= 0, got {margin_se}")
    dfs = _per_axis(margin_df, axes, "margin_df")
    if dfs is not None and not all(v is None or v >= 1 for v in dfs.values()):
        raise ValueError(f"margin_df must be >= 1 (None: a normal reference), got {margin_df}")


def _scale_for(margin: Margin, margin_se, margin_df, axes: Sequence[str]) -> Tuple[Optional[float], float,
                                                                                    Optional[float]]:
    """(ruler, its SE, its df) of an entry over ``axes``: the axis's margin, or the mean of the tested
    axes' margins for the pooled entry; (None, 0, None) when no margin was supplied."""
    if margin is None:
        return None, 0.0, None
    m = _per_axis(margin, axes, "margin")
    se = _per_axis(margin_se, axes, "margin_se") or {a: 0.0 for a in axes}
    dfs = _per_axis(margin_df, axes, "margin_df")
    var = float(np.sum([se[a] ** 2 for a in axes])) / len(axes) ** 2
    _, df = welch([(se[a] ** 2 / len(axes) ** 2, None if dfs is None else dfs[a]) for a in axes])
    return float(np.mean([m[a] for a in axes])), float(np.sqrt(var)), df


def _body_screen_regression(test: str, regressor_names: Dict[str, str], outputs, body, screen,
                            provenance: Sequence[dict], allowed_sources: Tuple[str, ...],
                            axes: Sequence[str], margin: Margin, margin_se, margin_df, alpha: float,
                            min_clusters: int, cross_axis: bool) -> dict:
    out, b, s, cl = _validate(outputs, body, screen, provenance, allowed_sources, axes, min_clusters)
    _check_margin(margin, margin_se, margin_df, axes)
    blocks = np.array([str(p["session"]) for p in provenance])
    ax = [AXES.index(a) for a in axes]
    result = {"test": test, "regressors": dict(regressor_names), "axes": tuple(axes), "alpha": alpha,
              "inference": "CR2 cluster-robust, t(G - k)",
              "margin": dict(margin) if isinstance(margin, Mapping) else margin,
              "margin_se": dict(margin_se) if isinstance(margin_se, Mapping) else margin_se,
              "margin_df": dict(margin_df) if isinstance(margin_df, Mapping) else margin_df,
              "margin_source": "self" if margin is None else "external",
              "window_key": window_key(provenance, b, s)}
    for a in ax:
        result[AXES[a]] = _entry(out, b, s, cl, blocks, [a], alpha,
                                 *_scale_for(margin, margin_se, margin_df, [AXES[a]]), cross_axis)
    result["pooled"] = _entry(out, b, s, cl, blocks, ax, alpha, *_scale_for(margin, margin_se, margin_df, axes),
                              cross_axis)
    return result


def certify_mapping(decoder_outputs_body, body_intent, screen_dir_in_body, provenance: Sequence[dict], *,
                    axes: Sequence[str] = AXES, margin: Margin = None, n_boot: int = N_BOOT,
                    alpha: float = ALPHA, seed: int = SEED, min_clusters: int = MIN_CLUSTERS,
                    cross_axis: bool = True, margin_se: Margin = None, margin_df: Margin = None) -> dict:
    """β_body, β_screen (and β_cross) of a frozen decoder on held-out-mapping windows (proposal P2).

    decoder_outputs_body  (n, 2) decoder output in body coordinates, one row per window.
    body_intent           (n, 2) ``M⁻¹ (target − cursor)`` unit vectors (``mapping.certification_regressors``).
    screen_dir_in_body    (n, 2) ``M_train⁻¹ (target − cursor)``: what a screen reader would output.
    provenance            one dict per window with "session" (a block run: the unit of ``block_corr``)
                          and "trial" (the clusters) and "source", which must be "H" (a missing source is
                          refused too): no window the decoder could have trained on is evidence.
    axes                  axes to test; pass ("y",) when x is not earned (a silent x output carries no
                          information).  y is the primary axis.
    margin                the decoder's EXTERNAL margin: one positive slope, or {axis: slope}, measured on
                          interleaved identity trials, never on these windows; None estimates β_body +
                          β_screen (``margin_source`` "self": never certified; module docstring).
    margin_se, margin_df  the margin's standard error and degrees of freedom (one value or per axis):
                          every threshold test then accounts for the margin's own sampling error
                          (module docstring).  Omitted: the margin is treated as exact.
    cross_axis            include the other axis's body intent as a regressor (module docstring); only
                          for checks of the old two-regressor model should this be False.
    n_boot, seed          accepted for callers of the earlier bootstrap interface; the single-unit fit is
                          analytic (CR2, t(G − k)) and does not draw.

    Returns a dict with one entry per axis and "pooled" (all ``axes`` stacked, an intercept per axis,
    one shared β set; reported, never decisive): β_body, β_screen, β_cross with CR2 SEs (``se_*``), 95 %
    t intervals on ``df`` = G − k degrees of freedom, both main βs also relative to the margin
    (``rel_*``, ``rel_se_*``, ``rel_df_*``: delta method with the margin's SE; ``rel_ci_*``: Fieller), the
    regressor correlation overall and per block (``block_corr``, ``max_block_corr``, ``confounded``),
    the largest multiple correlation among the regressors (``collinearity``), whether the βs are
    identified at all ("identifiable"), per-trial sufficient statistics (for the person-then-trial
    bootstrap), plus ``window_key`` (``certify`` checks it) and ``margin_source``.
    """
    return _body_screen_regression("heldout_mapping", {"body": "body_intent", "screen": "screen_dir_in_body",
                                                       "cross": "other_axis_body_intent"},
                                   decoder_outputs_body, body_intent, screen_dir_in_body, provenance,
                                   HELDOUT_SOURCES, axes, margin, margin_se, margin_df, alpha, min_clusters,
                                   cross_axis)


def replay_test(outputs, private_intent, replayed_dir, provenance: Sequence[dict], *,
                axes: Sequence[str] = AXES, margin: Margin = None, n_boot: int = N_BOOT,
                alpha: float = ALPHA, seed: int = SEED, min_clusters: int = MIN_REPLAY_CLUSTERS,
                cross_axis: bool = True, margin_se: Margin = None, margin_df: Margin = None) -> dict:
    """Pixel replay (proposal P3): the screen silently replays a past trajectory toward goal A while the
    person holds a pre-committed goal B.  Same regression and output as ``certify_mapping``, with
    ``private_intent`` (body velocity toward B, in the decoder's frame) as the body regressor and
    ``replayed_dir`` (the replayed cursor's direction, mapped into body coordinates through the training
    mapping) as the screen regressor.  A decoder that reads the moving cursor follows A, not B.
    ``provenance`` "source" must be "R".  At least ``MIN_REPLAY_CLUSTERS`` trials (module docstring)."""
    return _body_screen_regression("pixel_replay", {"body": "private_intent", "screen": "replayed_dir",
                                                    "cross": "other_axis_private_intent"},
                                   outputs, private_intent, replayed_dir, provenance, REPLAY_SOURCES,
                                   axes, margin, margin_se, margin_df, alpha, min_clusters, cross_axis)


def confirmed_by_report(provenance: Sequence[dict], body_intent: np.ndarray) -> np.ndarray:
    """(n,) bool: windows of trials whose keyed report confirms the required body intent.

    A trial with a ``body_report`` (the C2b keyed report, the post-replay report of the held goal) is
    kept only if the reported class points along the body intent of its first window (positive dot
    product: for a diagonal intent either component class confirms it).  A trial that reported another
    class failed to remap or switched goal, so β_screen / β_replay on it would measure that behaviour,
    not the decoder (DESIGN_PURE_EEG §5 P2/P3, F2).  Trials without a report (closed-loop C2c) are kept."""
    body = np.asarray(body_intent, float).reshape(-1, 2)
    if len(body) != len(provenance):
        raise ValueError("body_intent and provenance must have one row per window")
    first: Dict[Tuple[str, int], int] = {}
    keep = np.ones(len(provenance), bool)
    for i, p in enumerate(provenance):
        key = (str(p["session"]), int(p["trial"]))
        j = first.setdefault(key, i)
        report = p.get("body_report")
        if report is not None:
            keep[i] = float(np.dot(np.asarray(report, float), body[j])) > 0
    return keep


# ----------------------------------------------------------------------------------------------
# Combining test units
# ----------------------------------------------------------------------------------------------
def _usable(entry: dict) -> bool:
    """A unit's axis entry can enter a combination: identified, on a positive scale, not confounded."""
    scale = entry.get("scale")
    return bool(entry.get("identifiable") and not entry.get("confounded") and scale is not None
                and np.isfinite(scale) and scale > 0)


def _unit_estimates(entry: dict, key: str) -> Optional[Tuple[float, float, Tuple[float, float], Optional[float]]]:
    """(relative β, its SE, its CI, its df) of one unit's axis entry, or None when it cannot enter."""
    if not _usable(entry):
        return None
    est, se = entry[f"rel_{key}"], entry[f"rel_se_{key}"]
    if not (np.isfinite(est) and np.isfinite(se) and se > 0):
        return None
    lo, hi = entry[f"rel_ci_{key}"]
    return float(est), float(se), (float(lo), float(hi)), entry.get(f"rel_df_{key}")


def _pool(est: np.ndarray, se: np.ndarray, single: Tuple[Tuple[float, float], Optional[float]], combination: str,
          alpha: float) -> Tuple[float, Tuple[float, float], float, float, Optional[float]]:
    """Inverse-variance (fixed) or random-effects (DL τ², HKSJ t interval) combination of per-unit
    estimates.  Returns (estimate, CI, SE, τ², t degrees of freedom or None for a normal interval).
    One unit keeps its own interval and degrees of freedom."""
    if len(est) == 1:
        return float(est[0]), single[0], float(se[0]), 0.0, single[1]
    w = 1.0 / se ** 2
    mu_fe = float(np.sum(w * est) / np.sum(w))
    if combination == "fixed":
        s = float(np.sqrt(1.0 / np.sum(w)))
        z = stats.norm.ppf(1 - alpha / 2)
        return mu_fe, (mu_fe - z * s, mu_fe + z * s), s, 0.0, None
    k = len(est)
    q = float(np.sum(w * (est - mu_fe) ** 2))
    tau2 = max(0.0, (q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    ws = 1.0 / (se ** 2 + tau2)
    mu = float(np.sum(ws * est) / np.sum(ws))
    # HKSJ variance, never below the DL variance (the modification that keeps it from collapsing
    # when the few units happen to agree); t with k − 1 df, as few-unit meta-analysis requires.
    var = max(float(np.sum(ws * (est - mu) ** 2) / ((k - 1) * np.sum(ws))), 1.0 / float(np.sum(ws)))
    s = float(np.sqrt(var))
    t = stats.t.ppf(1 - alpha / 2, k - 1)
    return mu, (mu - t * s, mu + t * s), s, float(tau2), k - 1


def _leaves(result: dict) -> List[dict]:
    """The single-unit results under a (possibly nested) combined result."""
    if "units" not in result:
        return [result]
    return [leaf for _, r in sorted(result["units"].items()) for leaf in _leaves(r)]


def _person_bootstrap(units: Mapping[str, dict], axis: str, ok: List[str], alpha: float, n_boot: int,
                      seed: int) -> Dict[str, tuple]:
    """Person-then-trial bootstrap of the equal-weight mean over people of each person's relative β
    (the trial-weighted mean over that person's single-unit fits), for "body" and "screen".

    Each replicate draws len(ok) people with replacement, then, for every drawn person and each of
    their units, that unit's trials with replacement (an independent resample per appearance), refits
    each unit's OLS from its per-trial sufficient statistics and divides by a draw of the unit's margin
    (its estimate plus its SE times a t(df) draw: a parametric resample of the identity trials, which are
    not in the per-trial statistics), and recomputes the mean.  Percentile intervals over 2–3 people
    undercover badly, so the interval is t_{k−1} with the bootstrap SD inflated by √(k/(k−1)) (the
    bootstrap variance of a mean of k is (k−1)/k of the unbiased one): the HKSJ analogue.  Returns
    {key: (estimate, CI, SE, df, valid replicates)}."""
    rng = np.random.default_rng(seed)
    k = len(ok)
    leaves = {u: [e[axis] for e in _leaves(units[u]) if axis in e.get("axes", ()) and _usable(e[axis])
                  and e[axis].get("cluster_stats") is not None] for u in ok}
    empty = sorted(u for u in ok if not leaves[u])
    if empty:
        raise ValueError(f"units {empty} hold no single-unit fit with per-trial statistics on axis {axis!r}: "
                         "the person-then-trial bootstrap resamples their trials")

    def person_value(leaf_betas: List[np.ndarray], scales: List[np.ndarray], u: str) -> np.ndarray:
        w = np.array([e["n_clusters"] for e in leaves[u]], float)
        rel = np.stack([b / m[:, None] for b, m in zip(leaf_betas, scales)])       # (leaves, draws, 2)
        return np.tensordot(w / w.sum(), rel, axes=1)

    point = np.mean([person_value([np.array([[e["beta_body"], e["beta_screen"]]]) for e in leaves[u]],
                                  [np.array([e["scale"]]) for e in leaves[u]], u)[0] for u in ok], axis=0)
    draws_people = rng.integers(0, k, (n_boot, k))
    values = np.full((n_boot, k, 2), np.nan)
    for i, u in enumerate(ok):
        slots = np.argwhere(draws_people == i)                    # (replicate, position) of each appearance
        if not len(slots):
            continue
        betas, scales = [], []
        for e in leaves[u]:
            st = e["cluster_stats"]
            counts = _resample_counts(rng, len(slots), st["xtx"].shape[0])
            betas.append(_solve_draws(st["xtx"], st["xty"], counts)[:, :2])
            noise = (rng.standard_t(e["scale_df"], len(slots)) if e.get("scale_df")
                     else rng.standard_normal(len(slots)))
            m = e["scale"] + e.get("scale_se", 0.0) * noise
            scales.append(np.where(m > 0, m, np.nan))             # a non-positive margin draw is no ruler
        values[slots[:, 0], slots[:, 1]] = person_value(betas, scales, u)
    reps = values.mean(axis=1)
    reps = reps[np.isfinite(reps).all(axis=1)]
    out = {}
    t = stats.t.ppf(1 - alpha / 2, k - 1)
    for j, key in enumerate(("body", "screen")):
        se = float(reps[:, j].std(ddof=1) * np.sqrt(k / (k - 1))) if len(reps) > 1 else float("nan")
        est = float(point[j])
        out[key] = (est, (est - t * se, est + t * se), se, k - 1, int(len(reps)))
    return out


def combine_units(units: Mapping[str, dict], *, combination: str = "fixed", alpha: float = ALPHA,
                  n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Combine per-unit results of ONE test (``certify_mapping`` / ``replay_test`` results, or results
    of this function) into one result in units of each unit's margin (scale 1).

    ``units`` maps a unit id (a person's decoder version, a person) to its result; each unit was fitted
    on its own windows with its own slopes and intercepts, so differing gains or mapping mixes cannot
    bias another unit's β_screen.  Per axis, every unit that tested that axis, is identifiable on it and
    is not confounded enters (x therefore pools over x earners only, y over everyone); the others are
    listed in ``excluded`` (``excluded_confounded`` for the confounded ones).  Each unit enters with its
    relative β and its delta-method SE (margin uncertainty included).  ``combination``:
      "fixed"      inverse variance: the claim is about these units, e.g. the days of one person;
      "random"     DL τ² with a HKSJ t interval: a claim about the population the units come from;
      "bootstrap"  person-then-trial bootstrap (``n_boot`` replicates, ``seed``): units are people,
                   whose results may themselves be combinations of single-unit fits; the estimate is
                   the equal-weight mean over people (module docstring).
    The result has the entry keys ``certify`` reads (relative βs: ``scale`` 1, ``scale_se`` 0), plus
    ``units`` (the inputs, for the per-unit veto), ``n_units``, ``tau2_body`` / ``tau2_screen``,
    ``df_body`` / ``df_screen`` (t degrees of freedom of each interval, None for a normal one) and
    ``margin_source`` ("external" only if every unit's is).
    """
    if combination not in COMBINATIONS:
        raise ValueError(f"combination must be one of {COMBINATIONS}, got {combination!r}")
    if not units:
        raise ValueError("no units to combine")
    tests = {r["test"] for r in units.values()}
    if len(tests) != 1:
        raise ValueError(f"units come from different tests {sorted(tests)}")
    if combination == "bootstrap" and n_boot < MIN_BOOT:
        raise ValueError(f"n_boot must be >= {MIN_BOOT}, got {n_boot}")
    axes = tuple(a for a in AXES if any(a in r.get("axes", ()) for r in units.values()))
    any_r = next(iter(units.values()))
    sources = {r.get("margin_source", "self") for r in units.values()}
    out: dict = {"test": any_r["test"], "regressors": any_r["regressors"], "axes": axes,
                 "combination": combination, "alpha": alpha, "units": dict(units), "excluded": {},
                 "excluded_confounded": {}, "margin_source": "external" if sources == {"external"} else "self",
                 "window_key": hashlib.sha256("|".join(f"{u}:{r['window_key']}" for u, r in sorted(units.items()))
                                               .encode()).hexdigest()[:16]}
    nan, nan2 = float("nan"), (float("nan"), float("nan"))
    for a in axes:
        tested = {u: r[a] for u, r in units.items() if a in r.get("axes", ())}
        body = {u: _unit_estimates(e, "body") for u, e in tested.items()}
        screen = {u: _unit_estimates(e, "screen") for u, e in tested.items()}
        ok = [u for u in tested if body[u] is not None and screen[u] is not None]
        out["excluded"][a] = sorted(u for u in tested if u not in ok)
        out["excluded_confounded"][a] = sorted(u for u in tested if tested[u].get("confounded"))
        blocks = [tested[u]["max_block_corr"] for u in ok if np.isfinite(tested[u].get("max_block_corr", nan))]
        entry = {"n_units": len(ok), "n_windows": int(sum(tested[u]["n_windows"] for u in ok)),
                 "n_rows": int(sum(tested[u]["n_rows"] for u in ok)),
                 "n_clusters": int(sum(tested[u]["n_clusters"] for u in ok)),
                 "regressor_corr": nan, "collinearity": nan, "identifiable": bool(ok), "scale": 1.0,
                 "scale_se": 0.0, "scale_df": None,
                 "max_block_corr": float(max(blocks)) if blocks else nan, "confounded": False,
                 "beta_cross": nan, "ci_cross": nan2, "intercept": None, "n_boot_valid": None,
                 "cluster_stats": None, "df": None}
        boot = (_person_bootstrap(units, a, ok, alpha, n_boot, seed)
                if combination == "bootstrap" and len(ok) > 1 else None)
        for key, est in (("body", body), ("screen", screen)):
            if not ok:
                mu, ci, se, tau2, df = nan, nan2, nan, nan, None
            elif boot is not None:
                mu, ci, se, df, entry["n_boot_valid"] = boot[key]
                tau2 = nan
            else:
                e = np.array([est[u][0] for u in ok])
                s = np.array([est[u][1] for u in ok])
                mu, ci, se, tau2, df = _pool(e, s, (est[ok[0]][2], est[ok[0]][3]), combination, alpha)
            entry.update({f"beta_{key}": mu, f"ci_{key}": ci, f"se_{key}": se, f"tau2_{key}": tau2,
                          f"df_{key}": df, f"rel_{key}": mu, f"rel_ci_{key}": ci, f"rel_se_{key}": se,
                          f"rel_df_{key}": df})
        out[a] = entry
    return out


# ----------------------------------------------------------------------------------------------
# One-sided tests and the Holm family
# ----------------------------------------------------------------------------------------------
def _p_value(entry: dict, key: str, c: float, above: bool, c_se: float = 0.0,
             c_df: Optional[float] = None) -> float:
    """One-sided p-value of H0 β_key ≤ c·m (``above``: small p means β > c·m) or H0 β_key ≥ c·m, with
    ``c`` in margin units (m = the entry's scale; 1 for combined entries).

    The statistic is (β − c·m) / √(se²(β) + c²·se²(m) + m²·c_se²): the margin's sampling error enters
    any test against a non-zero threshold, and a threshold that is itself an estimate (a shadow's null,
    with SE ``c_se`` in margin units) adds its own.  Its reference is t with Welch–Satterthwaite degrees
    of freedom from the components' df (β: the entry's ``df_<key>`` or ``df``; normal where None)."""
    if not entry.get("identifiable") or not np.isfinite(c) or not np.isfinite(c_se):
        return float("nan")
    est, se = entry[f"beta_{key}"], entry[f"se_{key}"]
    m, m_se = entry.get("scale", 1.0), entry.get("scale_se", 0.0) or 0.0
    if not (np.isfinite(est) and np.isfinite(se) and se > 0 and np.isfinite(m)):
        return float("nan")
    df = entry.get(f"df_{key}", entry.get("df"))
    var, wdf = welch([(se ** 2, df), ((c * m_se) ** 2, entry.get("scale_df")), ((m * c_se) ** 2, c_df)])
    z = (est - c * m) / np.sqrt(var)
    dist = stats.t(wdf) if wdf else stats.norm
    return float(dist.sf(z) if above else dist.cdf(z))


def holm(pvalues: Mapping[str, float], alpha: float) -> Dict[str, dict]:
    """Holm's step-down correction: {name: {"p", "p_holm", "rejected"}} at family-wise level ``alpha``.
    NaN p-values (tests that could not run) are never rejected and do not count toward the family."""
    finite = sorted(((p, name) for name, p in pvalues.items() if np.isfinite(p)))
    m = len(finite)
    out = {name: {"p": float(p), "p_holm": float("nan"), "rejected": False}
           for name, p in pvalues.items() if not np.isfinite(p)}
    running = 0.0
    for i, (p, name) in enumerate(finite):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = {"p": float(p), "p_holm": float(running), "rejected": bool(running <= alpha)}
    return out


def _screen_tests(entry: dict, screen_tol: float, alpha1: float) -> Tuple[bool, int, bool]:
    """(significant, sign, bounded) of an entry's β_screen at one-sided level ``alpha1`` per side:
    significant = different from 0 in either direction (sign +1 / −1, 0 if not), bounded = both halves
    of the ±``screen_tol`` equivalence test pass."""
    pos = _p_value(entry, "screen", 0.0, above=True) < alpha1
    neg = _p_value(entry, "screen", 0.0, above=False) < alpha1
    bounded = (_p_value(entry, "screen", screen_tol, above=False) < alpha1
               and _p_value(entry, "screen", -screen_tol, above=True) < alpha1)
    return bool(pos or neg), (1 if pos else -1 if neg else 0), bool(bounded)


# ----------------------------------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------------------------------
def _shadow_map(shadow, what: str = "shadow") -> Dict[str, dict]:
    """``shadow`` as {name: result}: one result (named "shadow") or a mapping of named results."""
    if isinstance(shadow, Mapping) and "test" in shadow:
        return {"shadow": shadow}
    if not isinstance(shadow, Mapping) or not shadow or not all(isinstance(r, Mapping) and "test" in r
                                                                for r in shadow.values()):
        raise ValueError(f"{what} must be a certification result or a non-empty {{name: result}} mapping")
    return dict(shadow)


def _null_threshold(null: Optional[dict], axis: str) -> Tuple[float, float, Optional[float]]:
    """(relative β_screen, its SE, its df) of a shadow's null result on ``axis``; (0, 0, None) without a
    null (an uncalibrated test against 0), NaN when the null did not test this axis or this unit."""
    if null is None:
        return 0.0, 0.0, None
    e = null.get(axis)
    if e is None or not e.get("identifiable") or not np.isfinite(e.get("rel_screen", float("nan"))):
        return float("nan"), float("nan"), None
    return float(e["rel_screen"]), float(e["rel_se_screen"]), e.get("rel_df_screen")


def _family(deployed: dict, shadows: Dict[str, dict], nulls: Dict[str, Optional[dict]], axes: Sequence[str],
            screen_tol: float, body_lower: float) -> Dict[str, float]:
    """p-values of the verdict family (module docstring) over ``axes``."""
    p = {}
    for a in axes:
        d = deployed[a]
        scaled = np.isfinite(d["scale"]) and d["scale"] > 0
        p[f"{a}:deployed:body>0"] = _p_value(d, "body", 0.0, above=True)
        p[f"{a}:deployed:body>{body_lower:g}margin"] = (_p_value(d, "body", body_lower, above=True)
                                                        if scaled else float("nan"))
        p[f"{a}:deployed:screen<tol"] = _p_value(d, "screen", screen_tol, above=False) if scaled else float("nan")
        p[f"{a}:deployed:screen>-tol"] = _p_value(d, "screen", -screen_tol, above=True) if scaled else float("nan")
        for name, s in shadows.items():
            c, c_se, c_df = _null_threshold(nulls.get(name), a)
            p[f"{a}:{name}:screen>null"] = _p_value(s[a], "screen", c, above=True, c_se=c_se, c_df=c_df)
    return p


def _verdict_on(d: dict, shadows: Dict[str, dict], axis: str, screen_tol: float, body_pass: float,
                body_lower: float, tests: Dict[str, dict], alpha1: float, margin_external: bool,
                calibrated: bool) -> dict:
    """Verdict from one axis entry of the deployed (``d``) and of each shadow ({name: entry}), with the
    Holm decisions ``tests`` of the family it belongs to."""
    for s in shadows.values():
        if d["n_rows"] != s["n_rows"] or d["n_clusters"] != s["n_clusters"]:
            raise ValueError("deployed and shadow were scored on different windows")
    identifiable = bool(d["identifiable"] and all(s["identifiable"] for s in shadows.values()))
    flags = {"identifiable": identifiable, "margin_external": bool(margin_external),
             "shadow_null_calibrated": bool(calibrated)}
    if identifiable:
        reads = {n: bool(tests[f"{axis}:{n}:screen>null"]["rejected"]) for n in shadows}
        significant, sign, _ = _screen_tests(d, screen_tol, alpha1)
        flags.update(                       # plain bools: the flags are written to JSON reports
            deployed_screen_significant=significant, deployed_screen_sign=sign,
            deployed_screen_bounded=bool(tests[f"{axis}:deployed:screen<tol"]["rejected"]
                                         and tests[f"{axis}:deployed:screen>-tol"]["rejected"]),
            deployed_body_significant=bool(tests[f"{axis}:deployed:body>0"]["rejected"]),
            deployed_body_sufficient=bool(d["scale"] > 0 and d["beta_body"] >= body_pass * d["scale"]),
            deployed_body_lower_bounded=bool(tests[f"{axis}:deployed:body>{body_lower:g}margin"]["rejected"]),
            shadow_reads_screen=any(reads.values()), shadows_read_screen=reads,
            regressors_confounded=bool(d.get("confounded", False)),
            max_block_corr=float(d.get("max_block_corr", float("nan"))))
    if not identifiable:
        verdict, reason = "TEST_INSENSITIVE", (f"β_screen is not separable from the other regressors on axis "
                                               f"{axis!r} (collinearity > {MAX_REGRESSOR_CORR}); rot+90 and "
                                               "rot−90 are both needed in every test unit")
    elif flags["regressors_confounded"]:
        verdict, reason = "TEST_INSENSITIVE", (f"body and screen regressors correlate (max |r| per block "
                                               f"{flags['max_block_corr']:.2f} > {MAX_BLOCK_CORR}): refused; "
                                               "use balanced diagonal targets")
    elif flags["deployed_screen_significant"] and not flags["deployed_screen_bounded"]:
        side = "above" if flags["deployed_screen_sign"] > 0 else "below (an anti-screen dependence on the display)"
        verdict, reason = "SCREEN_READING", (f"deployed β_screen is {side} 0 and not shown inside "
                                             f"±{screen_tol:.0%} of the margin")
    elif not flags["deployed_body_significant"]:
        verdict, reason = "NO_BODY_SIGNAL", "deployed β_body is not shown above 0 (Holm family)"
    elif not flags["shadow_reads_screen"]:
        null = "its no-display null" if calibrated else "0 (uncalibrated: no shadow_null)"
        verdict, reason = "TEST_INSENSITIVE", (f"no shadow positive control ({', '.join(shadows)}) shows "
                                               f"β_screen above {null}")
    elif not flags["deployed_screen_bounded"]:
        verdict, reason = "INCONCLUSIVE", (f"deployed β_screen is not shown inside ±{screen_tol:.0%} of the "
                                           "margin: screen reading is neither shown nor excluded")
    elif not flags["deployed_body_sufficient"]:
        verdict, reason = "INCONCLUSIVE", f"deployed β_body is below {body_pass:.0%} of the margin"
    elif not flags["deployed_body_lower_bounded"]:
        verdict, reason = "INCONCLUSIVE", (f"deployed β_body is not shown above {body_lower:.0%} of the margin "
                                           "(F1b: its lower confidence limit is below it)")
    elif not margin_external:
        verdict, reason = "INCONCLUSIVE", ("no external margin: the self-estimated scale (β_body + β_screen) "
                                           "makes the body criterion vacuous; pass the identity-trial margin")
    elif not calibrated:
        verdict, reason = "INCONCLUSIVE", ("the shadow positive control is not null-calibrated (no shadow_null): "
                                           "a nonspecific shadow response could pass it")
    else:
        sensitive = [n for n, r in flags["shadows_read_screen"].items() if r]
        verdict, reason = "CERTIFIED", ("follows body intent, |β_screen| bounded inside "
                                        f"±{screen_tol:.0%} of the margin, and the shadow(s) {sensitive} read "
                                        "the screen beyond their no-display null")
    if "n_units" in d:                      # a combined entry: say what the interval is over
        reason += f" ({d['n_units']} unit(s) combined)"
    return {"verdict": verdict, "reason": reason, "axis": axis, "flags": flags, "deployed": d,
            "shadow": next(iter(shadows.values())), "shadows": dict(shadows)}


def certify(deployed: dict, shadow, *, axis: str = "all", screen_tol: float = SCREEN_TOL,
            body_pass: float = BODY_PASS, body_lower: float = BODY_LOWER, shadow_null=None) -> dict:
    """Verdict for the deployed decoder from ``certify_mapping`` / ``replay_test`` / ``combine_units``
    results of the deployed decoder and of the sighted-label shadow decoder(s) on the SAME windows (same
    ``window_key``).  ``shadow`` is one result or {name: result} (e.g. the central-11 and the all-32
    shadow); the test counts as sensitive if EITHER shows β_screen above its null, and every shadow is
    reported.  ``shadow_null`` holds each shadow's result of the same test on data without a display
    pathway (same structure as ``shadow``; for combined results its ``units`` calibrate the per-unit
    verdicts): the positive control must exceed it (module docstring).  Without it the shadows are
    tested against 0, flagged uncalibrated, and nothing is CERTIFIED.

    Per axis, in order (the first that applies; "shown" = rejected in the Holm-corrected family of
    one-sided tests at α/2, module docstring):
      TEST_INSENSITIVE  β_screen is not identified on these windows (e.g. one rotation only, a mirror,
                        the identity), so it means nothing;
      TEST_INSENSITIVE  the body and screen regressors are confounded (|r| > ``MAX_BLOCK_CORR`` in a
                        block or overall): refused, since β_screen is biased there with either sign;
      SCREEN_READING    deployed β_screen differs from 0 in either direction (one-sided α/2 per side,
                        uncorrected) and is not shown inside ±``screen_tol`` · margin — a positive
                        finding, which needs no positive control;
      NO_BODY_SIGNAL    deployed β_body is not shown above 0;
      TEST_INSENSITIVE  no shadow's β_screen is shown above its null (0 without ``shadow_null``): the test
                        did not detect the leak the positive controls are built to have;
      INCONCLUSIVE      the proposal's declared zone: deployed β_screen not shown inside ±``screen_tol``
                        · margin, or β_body < ``body_pass`` · margin, or β_body not shown above
                        ``body_lower`` · margin (F1b), or no external margin, or no ``shadow_null``;
      CERTIFIED         all of the above passed.
    ``axis`` "all" (default) reads every tested axis (one Holm family over them) and returns the worst
    verdict (``VERDICTS`` order; ties: y, the primary axis, is reported); a single axis name, or
    "pooled" (one β set over both axes, never the basis of a claim), reads that entry only.  For combined
    results every unit is also certified on its own (``unit_verdicts``), and a unit whose β_screen
    differs from 0 and is not bounded inside ±``screen_tol`` · margin at the Bonferroni level over all
    units and axes vetoes the verdict to SCREEN_READING (``vetoed_by``).
    Returns {"verdict", "reason", "axis", "flags", "deployed", "shadow" (the first shadow's entry),
    "shadows" ({name: entry}), "holm"}, plus "per_axis" for "all" and "unit_verdicts" / "vetoed_by" for
    combined results.
    """
    shadows = _shadow_map(shadow)
    for name, s in shadows.items():
        if deployed.get("test") != s.get("test"):
            raise ValueError(f"deployed ({deployed.get('test')}) and shadow {name!r} ({s.get('test')}) results "
                             "come from different tests; the positive control must run on the same windows")
        if deployed.get("window_key") != s.get("window_key"):
            raise ValueError(f"deployed and shadow {name!r} were scored on different windows (window_key differs)")
    nulls: Dict[str, Optional[dict]] = {n: None for n in shadows}
    if shadow_null is not None:
        given = _shadow_map(shadow_null, "shadow_null")
        if set(given) != set(shadows):
            raise ValueError(f"shadow_null names {sorted(given)}, the shadows are {sorted(shadows)}")
        for name, r in given.items():
            if r.get("test") != deployed.get("test"):
                raise ValueError(f"shadow_null {name!r} comes from test {r.get('test')!r}, not {deployed.get('test')!r}")
        nulls = dict(given)
    if axis == "all":
        axes = tuple(deployed.get("axes", ()))
        for name, s in shadows.items():
            if axes != tuple(s.get("axes", ())):
                raise ValueError(f"deployed tested {axes}, shadow {name!r} {tuple(s.get('axes', ()))}")
    else:
        if axis not in deployed or any(axis not in s for s in shadows.values()):
            raise ValueError(f"axis {axis!r} was not tested (available: {deployed.get('axes')}, 'all'"
                             f"{', pooled' if 'pooled' in deployed else ''})")
        axes = (axis,)
    alpha1 = float(deployed.get("alpha", ALPHA)) / 2
    tests = holm(_family(deployed, shadows, nulls, axes, screen_tol, body_lower), alpha1)
    external = deployed.get("margin_source") == "external"
    calibrated = shadow_null is not None
    per = {a: _verdict_on(deployed[a], {n: s[a] for n, s in shadows.items()}, a, screen_tol, body_pass,
                          body_lower, tests, alpha1, external, calibrated)
           for a in reversed(axes)}
    if axis == "all":
        res = dict(min(per.values(), key=lambda v: VERDICTS.index(v["verdict"])))
        res["reason"] = f"axis {res['axis']}: {res['reason']}"
        res["axis"] = "all"
        res["per_axis"] = {a: {"verdict": v["verdict"], "reason": v["reason"], "flags": v["flags"]}
                           for a, v in per.items()}
    else:
        res = per[axis]
    res["holm"] = {"alpha_one_sided": alpha1, "tests": tests}
    if "units" in deployed:
        for name, s in shadows.items():
            if set(deployed["units"]) != set(s.get("units", {})):
                raise ValueError(f"deployed and shadow {name!r} were combined over different units")

        def unit_null(u: str):
            if shadow_null is None:
                return None
            # a null that lacks this unit leaves its shadows uncalibratable: NaN thresholds, never sensitive
            return {n: (nulls[n].get("units", {}).get(u) or {"test": deployed["test"]}) for n in shadows}

        unit_v = {u: certify(deployed["units"][u], {n: s["units"][u] for n, s in shadows.items()},
                             axis="all", screen_tol=screen_tol, body_pass=body_pass, body_lower=body_lower,
                             shadow_null=unit_null(u))
                  for u in sorted(deployed["units"])}
        res["unit_verdicts"] = {u: v["verdict"] for u, v in unit_v.items()}
        leaves = _leaf_entries(deployed)
        alpha_leaf = float(deployed.get("alpha", ALPHA)) / (2 * max(1, len(leaves)))
        reading = []
        for u, a, e in leaves:
            if e["identifiable"] and not e.get("confounded") and e["scale"] > 0:
                significant, _, bounded = _screen_tests(e, screen_tol, alpha_leaf)
                if significant and not bounded:
                    reading.append(u)
        reading = sorted(set(reading))
        res["vetoed_by"] = reading
        if reading and res["verdict"] != "SCREEN_READING":
            res["verdict"] = "SCREEN_READING"
            res["reason"] = (f"unit(s) {reading} read the screen on their own (β_screen ≠ 0 at the "
                             f"Bonferroni level α/{len(leaves)} over every unit and axis, and not bounded "
                             f"inside ±{screen_tol:.0%} of the margin)")
    return res


def _leaf_entries(result: dict, prefix: str = "") -> list:
    """(unit path, axis, entry) of every single-unit axis entry under a combined result: the tests the
    per-unit veto runs, and so the size of its Bonferroni family."""
    out = []
    for u, r in sorted(result["units"].items()):
        path = f"{prefix}{u}"
        if "units" in r:
            out += _leaf_entries(r, path + "/")
        else:
            out += [(path, a, r[a]) for a in r["axes"]]
    return out
