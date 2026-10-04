"""Endpoints, floors and surrogates.

Primary endpoint for a cursor system is *controller utility*, not decoder AUC:

* Fitts throughput (ISO 9241-9 effective form): TP = mean over conditions of IDe / MT, bits/s,
  with IDe = log2(De / We + 1), We = 4.133 × SD(endpoint error along the task axis).
* Time-to-target and success rate within the timeout.
* Wolpaw ITR for the discrete layer (bits/selection × selections/min).

Every statistic comes with a floor computed at the matching sample size (the null's 95th percentile)
and a bootstrap CI; the caller is expected to report the non-neural surrogate on the same trials.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

from .task import TrialOutcome


def fitts_throughput(outcomes: Sequence[TrialOutcome]) -> float:
    """Effective throughput in bits/s over (distance, width) conditions, ISO 9241-9 style.

    Computed on successful trials only (failures are reported through the success rate), because a
    trial that never moved has a tiny endpoint spread and would otherwise *inflate* IDe.  NaN if a
    condition has fewer than two successes.
    """
    by_cond: Dict[tuple, List[TrialOutcome]] = {}
    for o in outcomes:
        if o.success:
            by_cond.setdefault((round(o.distance, 3), round(o.width, 3)), []).append(o)
    tps = []
    for (D, W), os_ in by_cond.items():
        if len(os_) < 2:
            continue
        err = np.array([o.endpoint_error for o in os_])
        mt = np.array([o.movement_time_s for o in os_])
        We = 4.133 * max(err.std(ddof=1) if len(err) > 1 else err.mean(), 1e-3)
        De = D   # effective distance ≈ nominal when the start is the centre
        IDe = np.log2(De / We + 1)
        tps.append(IDe / mt.mean())
    return float(np.mean(tps)) if tps else float("nan")


def wolpaw_itr_bits_per_trial(n_classes: int, accuracy: float) -> float:
    p = float(np.clip(accuracy, 1e-6, 1 - 1e-6))
    n = n_classes
    if n < 2:
        return 0.0
    b = np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1))
    return float(max(b, 0.0))


def success_rate(outcomes: Sequence[TrialOutcome]) -> float:
    return float(np.mean([o.success for o in outcomes]))


def time_to_target(outcomes: Sequence[TrialOutcome]) -> float:
    ok = [o.movement_time_s for o in outcomes if o.success]
    return float(np.mean(ok)) if ok else float("nan")


@dataclass
class Floor:
    observed: float
    floor_p95: float       # the null's 95th percentile at this n — this is what "floor" means
    p_value: float         # fraction of null draws ≥ observed, with +1 correction
    n_draws: int


def permutation_floor(statistic: Callable[[np.ndarray], float], labels: np.ndarray,
                      n_draws: int, rng: np.random.Generator, percentile: float = 95.0) -> Floor:
    """Shuffle the labels (never the EEG) and recompute the statistic; report the 95th percentile.

    For sequential/behavioural statistics a label permutation is anti-conservative (RLNF C-P007);
    use a design-generative null there instead.  This function is for decoder-vs-label statistics.
    """
    obs = statistic(labels)
    null = np.array([statistic(rng.permutation(labels)) for _ in range(n_draws)])
    p = (np.sum(null >= obs) + 1) / (n_draws + 1)
    return Floor(obs, float(np.percentile(null, percentile)), float(p), n_draws)


def bootstrap_ci(values: np.ndarray, stat: Callable[[np.ndarray], float], n_draws: int,
                 rng: np.random.Generator, alpha: float = 0.05):
    values = np.asarray(values)
    draws = np.array([stat(values[rng.integers(0, len(values), len(values))]) for _ in range(n_draws)])
    return float(stat(values)), float(np.percentile(draws, 100 * alpha / 2)), float(np.percentile(draws, 100 * (1 - alpha / 2)))


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 0 else 0.0


def surrogate_table(rows: Dict[str, float]) -> str:
    """Markdown table of {decoder name: value}; the caller passes EEG-only, EOG-only, no-neural."""
    out = ["| decoder | value |", "|---|---|"]
    out += [f"| {k} | {v:.4f} |" for k, v in rows.items()]
    return "\n".join(out)
