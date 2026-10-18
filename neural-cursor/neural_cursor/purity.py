"""Purity audit: is the decoder reading the brain, or the eyes, the screen, or its own feedback?

A non-invasive cursor decoder can score well for reasons that have nothing to do with cortical intent:

* **Ocular leak** — gaze follows intent, and the corneo-retinal dipole puts a *signed*, slow potential
  on the frontal channels.  A decoder that reads it is an eye tracker built out of EEG electrodes.
* **Display leak** — in any block that draws the targets (closed_loop, blank_feedback, free use) the
  label (cursor or start → goal) is a function of what the screen shows: the goal's position, and with
  a visible cursor its position and motion.  Any signal that follows the screen (visual evoked
  responses, gaze toward the target) will then look like "decoding".
* **Feedback circularity** — a decoder that only works while the user watches the cursor may be
  decoding the response to the feedback rather than the intent that should produce it.
* **Label structure without EEG** — visible-cursor labels point against the deployed decoder's bias
  (the cursor sits where the bias pushed it), so a predictor with no EEG at all — the training mean
  label — scores well above 0 there.  Every score is reported next to that intercept-only score, and
  "skill" means skill above it.

Each check is a *surrogate decoder* scored by the same cross-validation as the real one and compared
with its own permutation floor at the same n, so "beats its floor" always means the same thing: the
observed score exceeds the null's 95th percentile (``cfg.eval.floor_percentile``).

The null keeps the data's dependence structure.  Labels are constant within a trial (hidden cursor) or
change smoothly (visible cursor), and band-power features drift slowly; shuffling labels window by
window breaks that autocorrelation, the null comes out too narrow, and surrogates with no information
"beat their floor" 10–60 % of the time instead of 5 %.  Labels are therefore circularly shifted within
each session (``circular_shift_permutations``): every null draw keeps the label sequence's own
autocorrelation and per-session distribution and only breaks its alignment with the features.

Scope: visual-cortex responses to the display are NOT separately tested.  The occipital surrogate is
reported, but occipital channels also carry alpha-lateralised spatial attention — intent — so its
share of the full score cannot tell a visual leak from intent.  Only cued_blank windows, during which
nothing spatial is drawn, can certify decoding; the result's ``untested`` list says so whenever an
audit contains target-display windows.

Pure-EEG rule: auditor streams (eye tracker, EMG) passed to :meth:`PurityAudit.run` are used *only* by
the auditor-only surrogate in this module.  They are never mixed into the EEG decoders' features, and
nothing here returns a model that could be deployed.
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import rankdata
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .config import Config
from .decoder import VelocityDecoder
from .features import slow_shift
from .protocol import BLOCK_KINDS as PROTOCOL_KINDS, PROVENANCE_BLOCK, Block

FRONTAL = ("FP1", "FP2", "AF3", "AF4", "F7", "F8")   # where gaze-position EOG is largest
OCCIPITAL = ("PO7", "PO3", "PO4", "PO8", "Oz")       # where screen-driven visual responses are largest

N_SPLITS = 5                # folds of the out-of-fold predictions every surrogate is scored on
MIN_PERMUTATIONS = 200      # below this the p95 of the null is itself too noisy to be a floor
MIN_SHIFT_FRAC = 0.1        # circular shifts within 10 % of a session's length (either way) are excluded:
                            # they leave autocorrelated labels nearly aligned with their own windows
SLOW_BIN_S = 0.1            # boxcar width of the slow-potential bins (see slow_binned_means)
DISPLAY_HISTORY_S = 1.0     # how much displayed cursor motion before a window end the screen surrogate sees
DISPLAY_BINS = 5            # 0.2 s velocity bins: keeps the direction changes within that second
OCULAR_FRACTION = 0.5       # a frontal/slow surrogate reaching this share of the full decoder is a leak
OCULAR_SURROGATES = ("frontal", "slow")   # either can carry the ocular leak; one flag covers both
MIN_GROUP_N = 20            # fewest windows per feedback condition for a bootstrap CI worth reporting
_EPS = 1e-12

SPATIAL_BLOCKS: Tuple[str, ...] = tuple(sorted(
    PROVENANCE_BLOCK[k] for k in PROTOCOL_KINDS
    if PROVENANCE_BLOCK[k] is not None and any(it.spatial for it in Block(k, 1.0).screen_items())))
"""Provenance blocks whose screen shows something at a trial-dependent position (the targets), derived
from the renderer's contract ``protocol.Block.screen_items`` so the two cannot disagree."""

CursorLogs = Union[np.ndarray, Mapping[str, np.ndarray]]


def slow_binned_means(window: np.ndarray, fs: int, bin_s: float = SLOW_BIN_S) -> np.ndarray:
    """Per channel, means over consecutive ``bin_s`` bins ending at the window's last sample.

    A boxcar mean is a low-pass with its −3 dB point near 0.44 / bin_s (≈ 4.4 Hz for 0.1 s), so the
    features carry the signed ≈1–4 Hz potential that band power discards.  The lower edge is not
    filtered here: a 0.5 s window cannot resolve 1 Hz, and the upstream causal 0.5 Hz high-pass already
    removes DC drift.  The 0.1 s width is chosen because a boxcar has exact nulls at multiples of
    1 / width, i.e. at 10 Hz alpha, 20 Hz beta and 50 Hz mains: oscillatory power that genuinely
    carries intent cannot alias into this "slow-only" surrogate and make it look like a leak.
    """
    k = max(1, int(round(bin_s * fs)))
    n_bins = len(window) // k
    if n_bins == 0:
        raise ValueError(f"window of {len(window)} samples is shorter than one {bin_s} s slow bin")
    tail = window[len(window) - n_bins * k:]           # bins aligned to the newest sample
    return tail.reshape(n_bins, k, -1).mean(axis=1).ravel()


def displayed_velocity(t_end: float, log: np.ndarray, history_s: float = DISPLAY_HISTORY_S,
                       n_bins: int = DISPLAY_BINS) -> np.ndarray:
    """Mean on-screen cursor velocity in ``n_bins`` equal bins over ``[t_end − history_s, t_end]``.

    ``log`` is (m, ≥3) with columns t, x, y sorted by t (cursor.csv's first three columns).  Positions
    at the bin edges are linearly interpolated, so an irregular control clock is handled; before the
    log starts the position is held, i.e. nothing moved on screen.  Returns (2 · n_bins,): all x
    velocities, then all y velocities.
    """
    edges = t_end - history_s + np.arange(n_bins + 1) * (history_s / n_bins)
    x = np.interp(edges, log[:, 0], log[:, 1])
    y = np.interp(edges, log[:, 0], log[:, 2])
    return np.concatenate([np.diff(x), np.diff(y)]) / (history_s / n_bins)


def auditor_features(stream: np.ndarray, n: int) -> np.ndarray:
    """Per-window features of an auditor stream.

    (n, d) is taken as already-featurised (e.g. mean gaze per window).  (n, n_samples, d) raw
    segments become mean (gaze position), signed late-minus-early shift (gaze movement) and log SD
    (EMG power) per auditor channel — the three ways eyes and muscles typically carry intent.
    """
    a = np.asarray(stream, dtype=np.float64)
    if a.shape[0] != n:
        raise ValueError(f"auditor stream has {a.shape[0]} windows, EEG has {n}")
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        shift = np.stack([slow_shift(seg) for seg in a])
        return np.concatenate([a.mean(axis=1), shift, np.log(a.std(axis=1) + _EPS)], axis=1)
    raise ValueError(f"auditor stream must be (n, d) or (n, n_samples, d), got shape {a.shape}")


def circular_shift_permutations(provenance: Sequence[dict], n_permutations: int, rng: np.random.Generator,
                                min_shift_frac: float = MIN_SHIFT_FRAC) -> np.ndarray:
    """(n_permutations, n) index arrays: within each session (in ``t`` order), labels rotated by a
    random shift in [m, len − m], m = ⌈min_shift_frac · len⌉.

    Why a rotation and not a shuffle: a shuffle assumes windows are exchangeable, but labels are
    trial-constant or smooth and EEG features drift, so a shuffle's null is too narrow (see the module
    docstring).  A rotation keeps each series' autocorrelation and only breaks their alignment; for
    i.i.d. trial labels a rotation by whole trials is an exact permutation of trials.  Shifts are kept
    within a session so the null also keeps every session's (block's) own label distribution — a
    decoder that merely tells blocks apart cannot beat it.  A session too short to rotate stays aligned
    in every draw, which only widens the null (conservative).
    """
    n = len(provenance)
    perms = np.tile(np.arange(n), (n_permutations, 1))
    sessions: Dict[object, List[int]] = {}
    for i, p in enumerate(provenance):
        sessions.setdefault(p.get("session"), []).append(i)
    for idx in sessions.values():
        order = np.array(sorted(idx, key=lambda i: (float(provenance[i].get("t", i)), i)))
        m = len(order)
        lo = max(1, math.ceil(min_shift_frac * m))
        if m - lo < lo:
            continue
        shift = rng.integers(lo, m - lo + 1, n_permutations)
        perms[:, order] = order[(np.arange(m)[None] + shift[:, None]) % m]
    return perms


def _cluster_sums(values: np.ndarray, provenance: Sequence[dict], mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-trial sum and count of ``values`` over the windows selected by ``mask``.  A trial is
    ``(session, trial)`` from the provenance; a window with no ``trial`` key is its own cluster."""
    keys: Dict[object, int] = {}
    cluster = np.empty(int(mask.sum()), dtype=int)
    for j, i in enumerate(np.flatnonzero(mask)):
        p = provenance[i]
        key = (p.get("session"), p["trial"]) if "trial" in p else ("window", int(i))
        cluster[j] = keys.setdefault(key, len(keys))
    sums = np.bincount(cluster, weights=values[mask], minlength=len(keys))
    counts = np.bincount(cluster, minlength=len(keys)).astype(np.float64)
    return sums, counts


class _PermutationCV:
    """Out-of-fold ridge scores for the true labels and every permuted copy, at matched n.

    Ridge regression is linear in its targets and the scaler never sees labels, so the true labels and
    all P permuted label sets are fitted as one multi-output problem: one Cholesky factorisation per
    fold, folded into a label-free (n_test × n_train) hat matrix, instead of P + 1 fits, with exactly
    the same folds, features and regulariser as the observed score.  The closed form equals sklearn's
    ``Ridge(alpha)`` with intercept (X is centred by the scaler, the intercept is each target's training
    mean); it is written out so the label-free hat matrix is built once per fold and then applied to
    all P + 1 label sets as a single matrix product.  ``X = None`` is the intercept-only model (the
    training-fold mean label, no EEG): the baseline every "skill" is measured above.
    """

    def __init__(self, labels: np.ndarray, binary: bool, perms: np.ndarray, alpha: float,
                 n_splits: int, percentile: float):
        self.binary = binary
        self.alpha = alpha
        self.percentile = percentile
        y = labels.reshape(len(labels), -1).astype(np.float64)
        self.k = y.shape[1]
        stacked = np.concatenate([y[None], y[perms]], axis=0)        # (P + 1, n, k)
        self.y = np.transpose(stacked, (1, 0, 2))                      # (n, P + 1, k)
        # Contiguous folds, not shuffled ones: consecutive windows overlap (hop < window) and share a
        # trial, so a shuffled split would score near-duplicates of its own training windows.
        self.folds = list(KFold(n_splits=n_splits, shuffle=False).split(np.arange(len(labels))))

    def _oof(self, X: Optional[np.ndarray]) -> np.ndarray:
        n, draws = self.y.shape[0], self.y.shape[1]
        Y = self.y.reshape(n, draws * self.k)
        out = np.empty_like(Y)
        # Matrices here are a few hundred wide: multithreaded BLAS only adds synchronisation and
        # competes with a realtime loop sharing the machine, so the solves run single-threaded.
        with threadpool_limits(limits=1, user_api="blas"):
            for tr, te in self.folds:
                out[te] = Y[tr].mean(axis=0) if X is None else self._fold(X, Y, tr, te)
        return out.reshape(n, draws, self.k)

    def _fold(self, X: np.ndarray, Y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
        scaler = StandardScaler().fit(X[tr])
        xtr, xte = scaler.transform(X[tr]), scaler.transform(X[te])
        y_mean = Y[tr].mean(axis=0)
        gram = xtr.T @ xtr + self.alpha * np.eye(xtr.shape[1])
        hat = xte @ cho_solve(cho_factor(gram), xtr.T)           # (n_test, n_train), label-free
        return hat @ (Y[tr] - y_mean) + y_mean

    def evaluate(self, X: Optional[np.ndarray]) -> Tuple[Dict, np.ndarray]:
        """Returns the report dict and the per-window score of the true labels (for the feedback test
        and the per-block breakdown).  ``X = None`` scores the intercept-only model."""
        pred = self._oof(None if X is None else np.asarray(X, dtype=np.float64))
        if self.binary:
            p, pos = pred[:, :, 0], self.y[:, :, 0] > 0.5
            n1 = pos.sum(axis=0)
            n0 = len(pos) - n1
            ranks = rankdata(p, axis=0)
            scores = ((ranks * pos).sum(axis=0) - n1 * (n1 + 1) / 2) / np.maximum(n1 * n0, 1)  # Mann–Whitney AUC
            per_window = ((p[:, 0] > 0.5) == pos[:, 0]).astype(np.float64)
        else:
            p = np.clip(pred, -1, 1)               # the deployed VelocityDecoder clips; score what ships
            num = np.sum(p * self.y, axis=2)
            den = np.linalg.norm(p, axis=2) * np.linalg.norm(self.y, axis=2) + _EPS
            cos = num / den
            scores = cos.mean(axis=0)
            per_window = cos[:, 0]
        observed, null = float(scores[0]), scores[1:]
        floor = float(np.percentile(null, self.percentile))
        report = {
            "score": observed,
            "floor": floor,
            "p_value": float((np.sum(null >= observed) + 1) / (len(null) + 1)),
            "beats_floor": bool(observed > floor),
            "n": int(len(per_window)),
        }
        return report, per_window


class PurityAudit:
    """Surrogate-decoder audit of a labelled window set.  See the module docstring for the rationale.

    The EEG surrogates reuse :class:`VelocityDecoder`'s own feature function on a channel subset, and
    the same scaler + ridge (``cfg.decoder.ridge_alpha``) for every surrogate, so a difference between
    the full decoder and a surrogate is a difference of *inputs*, never of model capacity.
    """

    def __init__(self, cfg: Config, fs: Optional[int] = None, feature_set: str = "bandpower",
                 n_permutations: Optional[int] = None, n_splits: int = N_SPLITS,
                 seed: Optional[int] = None, ocular_fraction: float = OCULAR_FRACTION,
                 min_group_n: int = MIN_GROUP_N):
        self.cfg = cfg
        self.fs = int(fs or cfg.hardware.fs)
        self.feature_set = feature_set
        self.n_permutations = int(n_permutations or cfg.eval.n_permutations)
        if self.n_permutations < MIN_PERMUTATIONS:
            raise ValueError(f"n_permutations must be ≥ {MIN_PERMUTATIONS}, got {self.n_permutations}")
        self.n_splits = n_splits
        self.seed = cfg.eval.seed if seed is None else seed
        self.ocular_fraction = ocular_fraction
        self.min_group_n = min_group_n
        names = list(cfg.hardware.channels)
        self.frontal = self._mask(names, FRONTAL)
        self.occipital = self._mask(names, OCCIPITAL)

    @staticmethod
    def _mask(names: List[str], subset: Sequence[str]) -> np.ndarray:
        missing = [c for c in subset if c not in names]
        if missing:
            raise ValueError(f"channels {missing} are not in cfg.hardware.channels")
        return np.array([c in subset for c in names])

    def _eeg_features(self, windows: List[np.ndarray], mask: Optional[np.ndarray]) -> np.ndarray:
        dec = VelocityDecoder(self.cfg.decoder, self.fs, channels=mask, feature_set=self.feature_set)
        return np.stack([dec.features(w) for w in windows])

    @staticmethod
    def _displays_spatial(p: dict) -> bool:
        return p.get("block") in SPATIAL_BLOCKS or bool(p.get("feedback_visible", False))

    def _display_features(self, provenance: Sequence[dict], cursor_logs: Optional[CursorLogs]) -> np.ndarray:
        """Everything spatial on screen per window: the displayed goal (blocks that draw targets;
        provenance ``display_goal``), and with a visible cursor its position at the window end and its
        velocity over the previous second (``cursor_logs``).  Zero only where nothing spatial was
        drawn (cued_blank).  Missing inputs for a displayed window raise: an untested display must not
        read as a clean one."""
        logs: Dict[object, np.ndarray] = {}
        if cursor_logs is not None:
            def by_time(log) -> np.ndarray:
                arr = np.asarray(log, dtype=np.float64)
                return arr[np.argsort(arr[:, 0])]
            if isinstance(cursor_logs, Mapping):
                logs = {s: by_time(v) for s, v in cursor_logs.items()}
            else:
                only = by_time(cursor_logs)
                logs = {p.get("session"): only for p in provenance}
        out = np.zeros((len(provenance), 4 + 2 * DISPLAY_BINS))
        for i, p in enumerate(provenance):
            if p.get("block") in SPATIAL_BLOCKS:
                if p.get("display_goal") is None:
                    raise ValueError(f"window {i}: block {p.get('block')!r} draws the targets but provenance "
                                     f"has no 'display_goal'; the display surrogate cannot see the goal")
                out[i, :2] = p["display_goal"]
            if p.get("feedback_visible", False):
                if p.get("session") not in logs:
                    raise ValueError(f"window {i}: the cursor was visible but there is no cursor log for "
                                     f"session {p.get('session')!r}")
                log, t = logs[p["session"]], float(p["t"])
                out[i, 2:4] = np.interp(t, log[:, 0], log[:, 1]), np.interp(t, log[:, 0], log[:, 2])
                out[i, 4:] = displayed_velocity(t, log)
        return out

    def run(self, windows: List[np.ndarray], labels: np.ndarray, provenance: Sequence[dict],
            auditors: Optional[Dict[str, np.ndarray]] = None,
            cursor_logs: Optional[CursorLogs] = None) -> dict:
        """Audit one window set.

        windows     list of (n_samples, n_channels) EEG windows, in time order, as the decoder sees them
        labels      (n, k) continuous targets (scored by mean cosine) or (n,) binary (scored by AUC)
        provenance  per window {"block", "feedback_visible", "session", "t", "trial", "display_goal"}
                    (``data_engine.Labeler``).  "trial" is optional and makes the feedback bootstrap
                    resample whole trials; "display_goal" is required for blocks that draw targets.
        auditors    optional {name: (n, d) features or (n, n_samples, d) segments} — eye tracker, EMG
        cursor_logs (m, ≥3) [t, x, y] or {session: that array}; required when any window had a
                    visible cursor

        Returns {"metric", "n", "n_permutations", "floor_percentile", "null", "reports", "by_block",
        "feedback", "untested", "flags"}.  Every report has score, floor, p_value, beats_floor and n,
        or "skipped" with the reason.  ``reports["intercept"]`` is the no-EEG baseline.
        """
        n = len(windows)
        labels, binary = self._check(windows, labels, provenance)
        rng = np.random.default_rng(self.seed)
        perms = circular_shift_permutations(provenance, self.n_permutations, rng)
        cv = _PermutationCV(labels, binary, perms, self.cfg.decoder.ridge_alpha, self.n_splits,
                            self.cfg.eval.floor_percentile)
        reports: Dict[str, dict] = {}

        reports["intercept"], per_window_icpt = cv.evaluate(None)
        reports["full"], per_window = cv.evaluate(self._eeg_features(windows, None))
        reports["frontal"], _ = cv.evaluate(self._eeg_features(windows, self.frontal))
        reports["slow"], _ = cv.evaluate(np.stack([slow_binned_means(w, self.fs) for w in windows]))
        reports["occipital"], _ = cv.evaluate(self._eeg_features(windows, self.occipital))
        reports["frontal"]["channels"] = list(FRONTAL)
        reports["occipital"]["channels"] = list(OCCIPITAL)

        spatial = [self._displays_spatial(p) for p in provenance]
        if not any(spatial):
            reports["display"] = {"skipped": "nothing spatial was displayed (no targets, no cursor)"}
        else:
            reports["display"], _ = cv.evaluate(self._display_features(provenance, cursor_logs))

        for name, stream in (auditors or {}).items():
            reports[f"auditor:{name}"], _ = cv.evaluate(auditor_features(stream, n))

        untested = []
        if any(spatial):
            untested.append("visual-cortex responses to the display on target-display windows: occipital "
                            "channels also carry alpha-lateralised attention (intent), so the occipital "
                            "surrogate cannot separate a visual leak from intent; only cued_blank windows "
                            "certify decoding")
        feedback = self._feedback_test(per_window, provenance, rng, baseline=per_window_icpt)
        return {
            "metric": "auc" if binary else "cosine",
            "n": n,
            "n_permutations": self.n_permutations,
            "floor_percentile": self.cfg.eval.floor_percentile,
            "null": "labels circularly shifted within each session",
            "feature_set": self.feature_set,
            "reports": reports,
            "by_block": self._by_block(per_window, per_window_icpt, provenance),
            "feedback": feedback,
            "untested": untested,
            "flags": self._flags(reports, feedback, chance=reports["intercept"]["score"]),
        }

    @staticmethod
    def _by_block(full: np.ndarray, icpt: np.ndarray, provenance: Sequence[dict]) -> Dict[str, dict]:
        """Mean per-window score of the full decoder and of the intercept-only (no-EEG) model, per
        block: a visible-block score is only meaningful next to what no EEG at all achieves there."""
        blocks = np.array([str(p.get("block")) for p in provenance])
        return {b: {"n": int((blocks == b).sum()), "full": float(full[blocks == b].mean()),
                    "intercept": float(icpt[blocks == b].mean())} for b in sorted(set(blocks))}

    def _check(self, windows, labels, provenance) -> Tuple[np.ndarray, bool]:
        n = len(windows)
        if n < 2 * self.n_splits:
            raise ValueError(f"need at least {2 * self.n_splits} windows, got {n}")
        if len(provenance) != n:
            raise ValueError(f"{len(provenance)} provenance records for {n} windows")
        n_ch = self.cfg.hardware.n_channels
        bad = [i for i, w in enumerate(windows) if np.ndim(w) != 2 or np.shape(w)[1] != n_ch]
        if bad:
            raise ValueError(f"windows {bad[:5]} are not (n_samples, {n_ch}) EEG")
        y = np.asarray(labels, dtype=np.float64)
        if len(y) != n:
            raise ValueError(f"{len(y)} labels for {n} windows")
        if y.ndim == 2:
            return y, False
        values = np.unique(y)
        if y.ndim != 1 or len(values) != 2:
            raise ValueError("labels must be (n, k) continuous or (n,) with exactly two classes; "
                             "pass a 1-D continuous target as (n, 1)")
        return (y == values[1]).astype(np.float64), True

    def _feedback_test(self, per_window: np.ndarray, provenance: Sequence[dict],
                       rng: np.random.Generator, baseline: Optional[np.ndarray] = None) -> dict:
        """closed_loop vs blank_feedback skill of the full decoder's out-of-fold predictions.

        Matched contrast: the two groups are the closed_loop block and the blank_feedback probe, which
        run the same task (Fitts targets, the protocol places each probe straight after a closed_loop
        block) and differ only in whether the cursor is drawn.  Free use is left out (Zipf targets,
        success-selected hindsight labels), and so is cued_blank (a different task).

        Skill is the per-window score minus ``baseline``, the intercept-only (no-EEG) model's score on
        the same window: the two groups have very different label means (visible-cursor labels point
        against the decoder's bias), so raw accuracy differs between them even with no EEG at all, and
        that intercept effect would otherwise fire — or hide — FEEDBACK_CIRCULAR.

        The CI is a stratified *cluster* bootstrap of the difference: each group's trials (provenance
        ``(session, trial)``; a window without ``trial`` is its own cluster) are resampled within the
        group, and a resample's skill is the window-weighted mean over its trials.  Windows of one
        trial share a goal (and, with the cursor hidden, an identical label) and overlap in time, so
        resampling single windows treats a few dozen trials as hundreds of independent observations;
        the CI comes out too narrow and FEEDBACK_CIRCULAR fires on chance differences between trials.
        """
        base = np.zeros_like(per_window) if baseline is None else np.asarray(baseline, float)
        skill = per_window - base
        vis = np.array([p.get("block") == "closed_loop" and bool(p.get("feedback_visible", False))
                        for p in provenance])
        blank = np.array([p.get("block") == "blank_feedback" and not p.get("feedback_visible", False)
                          for p in provenance])
        a, b = skill[vis], skill[blank]
        if len(a) < self.min_group_n or len(b) < self.min_group_n:
            return {"skipped": f"need ≥ {self.min_group_n} closed_loop and blank_feedback windows,"
                               f" have {len(a)} and {len(b)}"}
        sums_a, counts_a = _cluster_sums(skill, provenance, vis)
        sums_b, counts_b = _cluster_sums(skill, provenance, blank)
        if min(len(sums_a), len(sums_b)) < 2:
            return {"skipped": f"need ≥ 2 trials per feedback condition, have {len(sums_a)} and {len(sums_b)}"}
        draws = self.cfg.eval.n_bootstrap

        def resampled_means(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
            idx = rng.integers(0, len(sums), (draws, len(sums)))
            return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)

        diffs = resampled_means(sums_a, counts_a) - resampled_means(sums_b, counts_b)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return {"visible": float(a.mean()), "blank": float(b.mean()),
                "difference": float(a.mean() - b.mean()), "ci95": [float(lo), float(hi)],
                "visible_raw": float(per_window[vis].mean()), "blank_raw": float(per_window[blank].mean()),
                "intercept_visible": float(base[vis].mean()), "intercept_blank": float(base[blank].mean()),
                "n_visible": int(len(a)), "n_blank": int(len(b)),
                "n_visible_trials": int(len(sums_a)), "n_blank_trials": int(len(sums_b))}

    def _flags(self, reports: Dict[str, dict], feedback: dict, chance: float) -> List[str]:
        """Shares of the full score are taken above ``chance`` — the intercept-only (no-EEG) score in
        ``run`` — so "half the decoder's skill" means half its distance from what the label structure
        alone achieves.  With chance = 0 a surrogate carrying no information already reaches ≥ 50 % of
        the full score through its intercept whenever labels are not centred, and the share condition
        would be met automatically.

        OCULAR_LEAK is one flag tested by two surrogates, so each must also pass a Bonferroni-corrected
        permutation p-value (α / 2 with α = 1 − floor_percentile / 100).  Without it a clean recording
        is flagged ~10 % of the time — twice the stated 5 % — and whenever the full decoder is at
        chance the share condition cannot catch such a fluke."""
        full_skill = reports["full"]["score"] - chance
        alpha = (1.0 - self.cfg.eval.floor_percentile / 100.0) / len(OCULAR_SURROGATES)
        flags = []
        for name in OCULAR_SURROGATES:
            r = reports[name]
            if (r["beats_floor"] and r["p_value"] <= alpha
                    and r["score"] - chance >= self.ocular_fraction * full_skill):
                flags.append("OCULAR_LEAK")
                break
        if reports["display"].get("beats_floor", False):
            flags.append("DISPLAY_LEAK")
        if "ci95" in feedback and feedback["ci95"][0] > 0:
            flags.append("FEEDBACK_CIRCULAR")
        return flags or ["PASS"]
