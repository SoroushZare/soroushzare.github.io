"""Closed-loop certification: does the deployed decoder follow the body, or the screen?

Why a regression on two regressors
----------------------------------
Under the training mapping (identity) a decoder that reads body-part intent and one that reads the
screen — gaze toward the target, posterior alpha lateralised toward it, visual responses to the moving
cursor — make the same prediction on every window: the person wants the cursor to go up, imagines
feet, and the target is above the cursor.  Accuracy under the identity can never tell them apart.
Under a held-out mapping ``M`` (``mapping.HELDOUT_MAPPINGS``) the person must produce body velocity
``M⁻¹ (target − cursor)`` while the screen still shows ``target − cursor``; under rot±90 the two are
orthogonal on every trial.  Regressing the decoder's output (body coordinates) on both,

    output_a = β_body · body_intent_a + β_screen · screen_dir_in_body_a + c_a + ε      (axis a),

splits its direction-following into the part that tracks the body and the part that tracks the screen.
A body-reading decoder has β_screen ≈ 0; a screen-reading one has β_body ≈ 0 and β_screen ≈ its
trained margin.  Pixel replay (``replay_test``) is the same regression with the person's private,
pre-committed intent as the body regressor and the replayed cursor direction as the screen regressor.

Why a trial-cluster bootstrap
-----------------------------
Windows of one trial overlap and share the trial's direction, the person's strategy and slow EEG state,
so they are not independent: a window-level CI is too narrow by roughly √(windows per trial).  Whole
trials ``(session, trial)`` are resampled with replacement (≥ ``MIN_BOOT`` draws, fixed seed), the OLS is
refitted on each draw from per-trial sufficient statistics, and the percentile interval is reported.

Why a positive control
----------------------
"β_screen ≈ 0" is evidence of purity only if the same test, on the same blocks, *does* detect screen
reading when it is there.  The shadow decoder (trained on sighted ReFIT labels, never deployed) is that
positive control: unless its β_screen is significantly > 0 the test is declared insensitive
(``TEST_INSENSITIVE``), not passed.

Scale.  Thresholds are fractions of the decoder's trained margin (its output slope on its own label
under the identity).  When the caller does not supply it, the margin is estimated as β_body + β_screen
from the same fit: under the identity both regressors equal the label, so that sum is exactly the
slope the decoder would show there if its total gain carries over.  The margin is a ruler, so its own
uncertainty is not propagated; every significance decision ("CI > 0") uses raw, scale-free CIs.

Pure-EEG rule: this module sees only decoder outputs and task geometry, never a signal of any kind.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .data_engine import trial_clusters

AXES: Tuple[str, ...] = ("x", "y")
N_BOOT = 2000               # bootstrap draws: the 2.5/97.5 % percentiles are stable to ~0.005 at this n
MIN_BOOT = 1000             # fewer draws make the tails of the percentile CI themselves noisy
ALPHA = 0.05                # two-sided: 95 % CIs
SEED = 20261018             # fixes every draw, so a verdict is reproducible from the logged outputs
MIN_CLUSTERS = 8            # a cluster bootstrap over fewer trials gives CIs that are too narrow
MAX_REGRESSOR_CORR = 0.98   # above this |corr(body, screen)| the two readings are not separable (VIF > 25):
                            # the identity (corr 1) and a mirror's unflipped axis (corr 1) land here
MAX_COND = 1e10             # bootstrap draws whose normal equations are this ill-conditioned are dropped
SCREEN_TOL = 0.10           # β_screen must be shown < 10 % of the margin (proposal P2/P3)
BODY_PASS = 0.70            # β_body must reach 70 % of the margin (proposal P2); 50–70 % is inconclusive

HELDOUT_SOURCES: Tuple[str, ...] = ("H",)   # provenance "source" of held-out-mapping test blocks
REPLAY_SOURCES: Tuple[str, ...] = ("R",)    # provenance "source" of pixel-replay test trials

VERDICTS: Tuple[str, ...] = ("CERTIFIED", "SCREEN_READING", "NO_BODY_SIGNAL", "TEST_INSENSITIVE",
                             "INCONCLUSIVE")


def _validate(outputs, body, screen, provenance, allowed_sources: Tuple[str, ...],
              axes: Sequence[str], n_boot: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    bad = sorted({p["source"] for p in provenance if "source" in p and p["source"] not in allowed_sources})
    if bad:
        raise ValueError(f"windows with source {bad} cannot enter this test (allowed: {allowed_sources}): "
                         "only test-only blocks are evidence, and training blocks are seen by the decoder")
    if not axes or any(a not in AXES for a in axes) or len(set(axes)) != len(axes):
        raise ValueError(f"axes must be a non-empty subset of {AXES}, got {tuple(axes)}")
    if n_boot < MIN_BOOT:
        raise ValueError(f"n_boot must be >= {MIN_BOOT}, got {n_boot}")
    cl = trial_clusters(provenance)
    if cl.size == 0 or cl.max() + 1 < MIN_CLUSTERS:
        raise ValueError(f"need >= {MIN_CLUSTERS} trials for a cluster bootstrap, "
                         f"got {0 if cl.size == 0 else cl.max() + 1}")
    return out, b, s, cl


def _regressor_corr(body_cols: Sequence[np.ndarray], screen_cols: Sequence[np.ndarray]) -> float:
    """corr(body, screen) after removing each axis's mean (what the per-axis intercepts absorb); NaN if
    either regressor is constant, which makes its β unidentifiable."""
    b = np.concatenate([c - c.mean() for c in body_cols])
    s = np.concatenate([c - c.mean() for c in screen_cols])
    nb, ns = np.linalg.norm(b), np.linalg.norm(s)
    if nb < 1e-12 or ns < 1e-12:
        return float("nan")
    return float(b @ s / (nb * ns))


def _cluster_ols(X: np.ndarray, y: np.ndarray, cl: np.ndarray, n_clusters: int, counts: np.ndarray,
                 alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """OLS coefficients and their trial-cluster percentile CIs.

    Each draw re-weights whole trials by how often they were drawn (``counts``, shape (draws, trials)),
    so the per-trial sufficient statistics XᵀX and Xᵀy are summed once and every draw is one small
    solve.  Returns (β, ci_lo, ci_hi, valid draws)."""
    xtx = np.zeros((n_clusters, X.shape[1], X.shape[1]))
    np.add.at(xtx, cl, X[:, :, None] * X[:, None, :])
    xty = np.zeros((n_clusters, X.shape[1]))
    np.add.at(xty, cl, X * y[:, None])
    beta = np.linalg.solve(xtx.sum(axis=0), xty.sum(axis=0))
    A = np.einsum("bc,cij->bij", counts, xtx)
    v = np.einsum("bc,ci->bi", counts, xty)
    ok = np.linalg.cond(A) < MAX_COND
    if not ok.any():
        nan = np.full(X.shape[1], np.nan)
        return beta, nan, nan, 0
    draws = np.linalg.solve(A[ok], v[ok][..., None])[..., 0]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return beta, lo, hi, int(ok.sum())


def _entry(out: np.ndarray, body: np.ndarray, screen: np.ndarray, cl: np.ndarray, axes: Sequence[int],
           counts: np.ndarray, alpha: float, margin: Optional[float]) -> dict:
    """One regression over the given axes (one axis, or several pooled with an intercept per axis)."""
    n = len(out)
    corr = _regressor_corr([body[:, a] for a in axes], [screen[:, a] for a in axes])
    entry = {"n_windows": n, "n_rows": n * len(axes), "n_clusters": int(cl.max()) + 1, "regressor_corr": corr,
             "identifiable": bool(np.isfinite(corr) and abs(corr) <= MAX_REGRESSOR_CORR)}
    nan2 = (float("nan"), float("nan"))
    if not entry["identifiable"]:
        entry.update(beta_body=float("nan"), beta_screen=float("nan"), ci_body=nan2, ci_screen=nan2,
                     intercept=[float("nan")] * len(axes), n_boot_valid=0, scale=float("nan"),
                     rel_body=float("nan"), rel_screen=float("nan"), rel_ci_body=nan2, rel_ci_screen=nan2)
        return entry
    k = len(axes)
    X = np.zeros((n * k, 2 + k))
    for j, a in enumerate(axes):
        rows = slice(j * n, (j + 1) * n)
        X[rows, 0] = body[:, a]
        X[rows, 1] = screen[:, a]
        X[rows, 2 + j] = 1.0
    y = np.concatenate([out[:, a] for a in axes])
    beta, lo, hi, n_valid = _cluster_ols(X, y, np.tile(cl, k), int(cl.max()) + 1, counts, alpha)
    scale = float(margin) if margin is not None else float(beta[0] + beta[1])
    rel = (lambda v: float(v) / scale) if scale > 0 else (lambda v: float("nan"))
    entry.update(beta_body=float(beta[0]), beta_screen=float(beta[1]),
                 ci_body=(float(lo[0]), float(hi[0])), ci_screen=(float(lo[1]), float(hi[1])),
                 intercept=[float(c) for c in beta[2:]], n_boot_valid=n_valid, scale=scale,
                 rel_body=rel(beta[0]), rel_screen=rel(beta[1]),
                 rel_ci_body=(rel(lo[0]), rel(hi[0])), rel_ci_screen=(rel(lo[1]), rel(hi[1])))
    return entry


def _body_screen_regression(test: str, regressor_names: Dict[str, str], outputs, body, screen,
                            provenance: Sequence[dict], allowed_sources: Tuple[str, ...],
                            axes: Sequence[str], margin: Optional[float], n_boot: int, alpha: float,
                            seed: int) -> dict:
    out, b, s, cl = _validate(outputs, body, screen, provenance, allowed_sources, axes, n_boot)
    if margin is not None and not margin > 0:
        raise ValueError(f"margin must be a positive slope, got {margin}")
    n_clusters = int(cl.max()) + 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, (n_boot, n_clusters))
    counts = np.bincount((idx + n_clusters * np.arange(n_boot)[:, None]).ravel(),
                         minlength=n_boot * n_clusters).reshape(n_boot, n_clusters).astype(float)
    # The same resampled trials serve every axis and the pooled fit, so they are mutually consistent.
    ax = [AXES.index(a) for a in axes]
    result = {"test": test, "regressors": dict(regressor_names), "axes": tuple(axes), "n_boot": n_boot,
              "alpha": alpha, "seed": seed, "margin": margin}
    for a in ax:
        result[AXES[a]] = _entry(out, b, s, cl, [a], counts, alpha, margin)
    result["pooled"] = _entry(out, b, s, cl, ax, counts, alpha, margin)
    return result


def certify_mapping(decoder_outputs_body, body_intent, screen_dir_in_body, provenance: Sequence[dict], *,
                    axes: Sequence[str] = AXES, margin: Optional[float] = None, n_boot: int = N_BOOT,
                    alpha: float = ALPHA, seed: int = SEED) -> dict:
    """β_body and β_screen of a frozen decoder on held-out-mapping windows (proposal P2, C2c).

    decoder_outputs_body  (n, 2) decoder output in body coordinates, one row per window.
    body_intent           (n, 2) ``M⁻¹ (target − cursor)`` unit vectors (``mapping.certification_regressors``).
    screen_dir_in_body    (n, 2) ``M_train⁻¹ (target − cursor)``: what a screen reader would output.
    provenance            one dict per window with "session" and "trial" (the bootstrap clusters); a
                          "source" key, when present, must be "H" — no window the decoder could have
                          trained on is evidence.
    axes                  axes to test; pass ("y",) when x is not earned (a silent x output would only
                          dilute the pooled fit).  y is the primary axis.
    margin                the decoder's trained margin (its output slope on its own label under the
                          identity); None estimates it as β_body + β_screen (see the module docstring).

    Returns a dict with one entry per axis and "pooled" (all ``axes`` stacked, an intercept per axis):
    β_body, β_screen, their 95 % trial-cluster bootstrap CIs, both also relative to the margin, the
    regressor correlation and whether the two readings are separable at all ("identifiable": a mirror's
    unflipped axis, or the identity, is not).
    """
    return _body_screen_regression("heldout_mapping", {"body": "body_intent", "screen": "screen_dir_in_body"},
                                   decoder_outputs_body, body_intent, screen_dir_in_body, provenance,
                                   HELDOUT_SOURCES, axes, margin, n_boot, alpha, seed)


def replay_test(outputs, private_intent, replayed_dir, provenance: Sequence[dict], *,
                axes: Sequence[str] = AXES, margin: Optional[float] = None, n_boot: int = N_BOOT,
                alpha: float = ALPHA, seed: int = SEED) -> dict:
    """Pixel replay (proposal P3): the screen silently replays a past trajectory toward goal A while the
    person holds a pre-committed goal B.  Same regression and output as ``certify_mapping``, with
    ``private_intent`` (body velocity toward B, in the decoder's frame) as the body regressor and
    ``replayed_dir`` (the replayed cursor's direction, mapped into body coordinates through the training
    mapping) as the screen regressor.  A decoder that reads the moving cursor follows A, not B.
    ``provenance`` "source", when present, must be "R"."""
    return _body_screen_regression("pixel_replay", {"body": "private_intent", "screen": "replayed_dir"},
                                   outputs, private_intent, replayed_dir, provenance, REPLAY_SOURCES,
                                   axes, margin, n_boot, alpha, seed)


def certify(deployed: dict, shadow: dict, *, axis: str = "pooled", screen_tol: float = SCREEN_TOL,
            body_pass: float = BODY_PASS) -> dict:
    """Verdict for the deployed decoder from ``certify_mapping`` / ``replay_test`` results of the deployed
    decoder and of the sighted-label shadow decoder on the SAME windows.

    In order (the first that applies):
      TEST_INSENSITIVE  the two regressors are not separable on these windows (e.g. a mirror's unflipped
                        axis), so neither β means anything;
      SCREEN_READING    deployed β_screen CI lies above 0 and is not shown below ``screen_tol`` · margin —
                        a positive finding, which needs no positive control;
      NO_BODY_SIGNAL    deployed β_body CI does not lie above 0;
      TEST_INSENSITIVE  shadow β_screen CI does not lie above 0: the test did not detect the leak that
                        the positive control is built to have, so "no screen reading" is uninformative;
      INCONCLUSIVE      the proposal's declared zone: the deployed β_screen CI reaches ``screen_tol`` ·
                        margin (e.g. contains 0 but is too wide to bound), or β_body < ``body_pass`` ·
                        margin (only binding with a supplied margin: with the self-estimated one,
                        bounding β_screen below 10 % already puts β_body above 90 %);
      CERTIFIED         deployed β_body CI > 0, β_screen bounded below ``screen_tol`` · margin, and the
                        shadow's β_screen CI > 0.
    Returns {"verdict", "reason", "axis", "flags", "deployed", "shadow"} (the latter two are the axis
    entries the verdict was read from).
    """
    if deployed.get("test") != shadow.get("test"):
        raise ValueError(f"deployed ({deployed.get('test')}) and shadow ({shadow.get('test')}) results come "
                         "from different tests; the positive control must run on the same windows")
    if axis not in deployed or axis not in shadow:
        raise ValueError(f"axis {axis!r} was not tested (available: {deployed.get('axes')} and 'pooled')")
    d, s = deployed[axis], shadow[axis]
    if d["n_rows"] != s["n_rows"] or d["n_clusters"] != s["n_clusters"]:
        raise ValueError("deployed and shadow were scored on different windows")
    identifiable = d["identifiable"] and s["identifiable"]
    flags = {"identifiable": identifiable}
    if identifiable:
        bounded = d["scale"] > 0 and d["ci_screen"][1] < screen_tol * d["scale"]
        flags.update(
            deployed_screen_significant=d["ci_screen"][0] > 0,
            deployed_screen_bounded=bounded,
            deployed_body_significant=d["ci_body"][0] > 0,
            deployed_body_sufficient=d["scale"] > 0 and d["beta_body"] >= body_pass * d["scale"],
            shadow_reads_screen=s["ci_screen"][0] > 0)
    if not identifiable:
        verdict, reason = "TEST_INSENSITIVE", (f"body and screen regressors are collinear on axis {axis!r} "
                                               f"(|corr| > {MAX_REGRESSOR_CORR}); pool axes or use rot±90")
    elif flags["deployed_screen_significant"] and not flags["deployed_screen_bounded"]:
        verdict, reason = "SCREEN_READING", "deployed β_screen CI is above 0 and not below the tolerance"
    elif not flags["deployed_body_significant"]:
        verdict, reason = "NO_BODY_SIGNAL", "deployed β_body CI does not exclude 0"
    elif not flags["shadow_reads_screen"]:
        verdict, reason = "TEST_INSENSITIVE", "the shadow positive control shows no screen reading"
    elif not flags["deployed_screen_bounded"]:
        verdict, reason = "INCONCLUSIVE", (f"deployed β_screen CI reaches {screen_tol:.0%} of the margin: "
                                           "screen reading is neither shown nor excluded")
    elif not flags["deployed_body_sufficient"]:
        verdict, reason = "INCONCLUSIVE", f"deployed β_body is below {body_pass:.0%} of the trained margin"
    else:
        verdict, reason = "CERTIFIED", ("follows body intent, screen reading bounded below "
                                        f"{screen_tol:.0%} of the margin, and the shadow shows the test "
                                        "detects screen reading")
    return {"verdict": verdict, "reason": reason, "axis": axis, "flags": flags, "deployed": d, "shadow": s}
