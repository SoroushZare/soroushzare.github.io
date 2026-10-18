"""Closed-loop simulation: simulated user → calibration → decoder → Fitts blocks → endpoints + floors.

The simulated user "attends toward the goal": at every control step the latent intent is the unit
vector from the cursor to the hidden goal, so closed-loop feedback is modelled (the intent changes as
the cursor moves).  The decoder never sees the goal; the shared-autonomy posterior never sees it either.

Conditions (same simulated user, same targets, same seeds):
  oracle         command = true direction          (upper bound of the task dynamics)
  decoder        EEG velocity decoder only
  decoder+assist EEG decoder blended with the goal posterior (shared autonomy, no evidence channel)
  eog_only       surrogate decoder trained on FP1/FP2/F7/F8 only  (must be at floor unless gaze leaks)
  no_neural      assist alone on a zero command      (must be at floor: the posterior has no input)
  random         uniform random command              (the chance floor of the task)

Run:  python -m neural_cursor.simulate --users 6 --trials 24 --gain 8 --eye 0
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Dict, List

import numpy as np

from .acquisition import SimulatedEEG
from .config import DEFAULT, Config
from .decoder import VelocityDecoder
from .evaluation import bootstrap_ci, fitts_throughput, pearson_r, permutation_floor, success_rate, time_to_target
from .signal import CausalFilter, common_average
from .task import TrialRunner, fitts_targets


class SimUser:
    """Wraps the simulator so each control step produces the samples that 'arrived' since the last."""

    def __init__(self, cfg: Config, seed: int, gain: float, eye: float):
        self.cfg = cfg
        self.sim = SimulatedEEG(cfg.hardware, seed=seed, intent_gain_uv=gain, eye_gain_uv=eye)
        self.filt = CausalFilter(cfg.hardware, cfg.pre)
        self.fs = cfg.hardware.fs
        self.win = int(cfg.decoder.window_s * self.fs)
        self.hop = int(self.fs / cfg.task.control_hz)
        self.buf = np.zeros((self.win, cfg.hardware.n_channels))
        self.filt(self.sim.generate(self.fs * 3))                     # settle the filters

    def step(self, intent: np.ndarray) -> np.ndarray:
        """Advance one control period with the given latent intent; return the latest window (CAR)."""
        self.sim.set_intent(*intent)
        x = self.filt(self.sim.generate(self.hop))
        self.buf = np.concatenate([self.buf[self.hop:], x])
        return common_average(self.buf[-self.win:])


def calibration(user: SimUser, n_windows: int, rng: np.random.Generator):
    """Cued open-loop block: the cue (a random direction) is the label; it is independent of EEG."""
    ws, ys = [], []
    for _ in range(n_windows):
        ang = rng.uniform(0, 2 * np.pi)
        y = np.array([np.cos(ang), np.sin(ang)])
        for _ in range(int(user.win / user.hop)):                     # let the window fill with this cue
            w = user.step(y)
        ws.append(w)
        ys.append(y)
    return ws, np.array(ys)


def mean_axis_r(pred: np.ndarray, labels: np.ndarray) -> float:
    """The reported decoder statistic: Pearson r per command axis, averaged over the axes."""
    return float(np.mean([pearson_r(pred[:, k], labels[:, k]) for k in range(labels.shape[1])]))


def decoder_r_with_floor(pred: np.ndarray, labels: np.ndarray, n_draws: int, rng: np.random.Generator):
    """``mean_axis_r`` and its label-permutation floor, computed on the SAME statistic: whole label
    rows (both axes together) are permuted and the mean-over-axes r is recomputed, so the floor is
    the null of the number it is reported next to (a floor of the x-axis r alone is not)."""
    return permutation_floor(lambda lab: mean_axis_r(pred, lab), labels, n_draws, rng)


def run_condition(name: str, user: SimUser, dec, cfg: Config, blocks, rng: np.random.Generator):
    task_cfg = dataclasses.replace(cfg.task, assist_gain=0.5 if "assist" in name or name == "no_neural" else 0.0)
    runner = TrialRunner(task_cfg, rng)
    outs = []
    for targets, width, goal in blocks:
        gpos = targets[goal]

        def command(st, t):
            intent = gpos - st.pos
            intent = intent / (np.linalg.norm(intent) + 1e-9)
            w = user.step(intent)                                     # EEG always advances, every condition
            if name == "oracle":
                return intent
            if name == "random":
                return rng.uniform(-1, 1, 2)
            if name == "no_neural":
                return np.zeros(2)
            return dec.predict(w)

        outs.append(runner.run(targets, width, goal, command))
    return outs


def simulate(cfg: Config, users: int, trials: int, gain: float, eye: float, seed: int,
             feature_set: str = "bandpower", adapt: bool = False) -> Dict:
    rng = np.random.default_rng(seed)
    results: Dict[str, List] = {}
    decoder_r = []
    surrogate_r = []
    floors = []
    for u in range(users):
        user = SimUser(cfg, seed=seed + 100 * u, gain=gain, eye=eye)
        ws, ys = calibration(user, 300, rng)
        dec = VelocityDecoder(cfg.decoder, cfg.hardware.fs, feature_set=feature_set, adapt=adapt).fit(ws[:240], ys[:240])
        mask = np.array([c in cfg.hardware.eog_proxy for c in cfg.hardware.channels])
        eog = VelocityDecoder(cfg.decoder, cfg.hardware.fs, channels=mask, feature_set=feature_set).fit(ws[:240], ys[:240])
        pred = dec.predict_many(ws[240:])
        fl = decoder_r_with_floor(pred, ys[240:], 300, rng)
        decoder_r.append(fl.observed)
        floors.append(fl.floor_p95)
        pe = eog.predict_many(ws[240:])
        surrogate_r.append(mean_axis_r(pe, ys[240:]))
        # identical Fitts blocks for every condition of this user
        blocks = []
        for D, W in ((0.5, 0.08), (0.8, 0.08), (0.8, 0.05)):
            for _ in range(trials // 3):
                tg = fitts_targets(cfg.task.n_targets, D, W, rng)
                blocks.append((tg, W, int(rng.integers(0, cfg.task.n_targets))))
        for name, d in (("oracle", None), ("decoder", dec), ("decoder+assist", dec),
                        ("eog_only", eog), ("no_neural", None), ("random", None)):
            results.setdefault(name, []).extend(run_condition(name, user, d, cfg, blocks, np.random.default_rng(seed + u)))
    rng2 = np.random.default_rng(seed)
    table = {}
    for name, outs in results.items():
        sr, lo, hi = bootstrap_ci(np.array([o.success for o in outs], float), np.mean, 500, rng2)
        table[name] = dict(n=len(outs), success=sr, success_ci=(lo, hi), ttt_s=time_to_target(outs),
                           throughput_bps=fitts_throughput(outs))
    return dict(users=users, trials_per_user=trials, gain=gain, eye=eye, feature_set=feature_set,
                decoder_r_mean=float(np.mean(decoder_r)), decoder_r_floor_p95=float(np.mean(floors)),
                eog_surrogate_r_mean=float(np.mean(surrogate_r)),
                conditions=table)


def to_markdown(res: Dict) -> str:
    lines = [f"simulated users = {res['users']}, trials/user = {res['trials_per_user']}, "
             f"signal gain = {res['gain']}, eye gain = {res['eye']}",
             f"features = {res['feature_set']}",
             f"calibration decoder r = {res['decoder_r_mean']:.3f} (permutation floor p95 = {res['decoder_r_floor_p95']:.3f}); "
             f"EOG-only surrogate r = {res['eog_surrogate_r_mean']:.3f}",
             "", "| condition | n | success | 95% CI | time-to-target (s) | Fitts TP (bits/s) |", "|---|---|---|---|---|---|"]
    for k, v in res["conditions"].items():
        lo, hi = v["success_ci"]
        lines.append(f"| {k} | {v['n']} | {v['success']:.2f} | [{lo:.2f}, {hi:.2f}] | {v['ttt_s']:.1f} | {v['throughput_bps']:.2f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--gain", type=float, default=8.0)
    ap.add_argument("--eye", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=DEFAULT.eval.seed)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--adapt", action="store_true", help="online feature re-centring in the velocity decoder")
    ap.add_argument("--features", type=str, default="bandpower", choices=["bandpower", "bandpower+slow"])
    a = ap.parse_args()
    res = simulate(DEFAULT, a.users, a.trials, a.gain, a.eye, a.seed, a.features, a.adapt)
    print(to_markdown(res))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)
