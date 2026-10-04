"""Round-2 closed-loop simulation: the composite system.

Three control signals, all stimulus-free and imagery-free:
  (1) a continuous EEG velocity decoder (weak, r ≈ 0.3–0.5)             → biases the cursor
  (2) a shared-autonomy agent that steers in 1 s segments from a goal posterior it infers;
      it never reads the goal, and explores with probability ε so every label stays estimable
  (3) an evidence decoder on the 0.2–0.8 s window after each agent segment (the person's
      automatic "that was wrong" response) → Bayesian update of the posterior

Labels in calibration: velocity ← the CUED direction (independent of EEG); evidence ← whether the
agent's logged move reduced the distance to the cued goal (a function of the hidden goal and the
logged action, never a threshold on a displayed variable).

Conditions, identical blocks and seeds per simulated user:
  oracle                 true direction                                   (task upper bound)
  decoder+assist         round-1 reference: velocity decoder + velocity-only posterior
  agent+evidence         velocity decoder OFF; agent + EEG evidence channel only
  composite              velocity decoder + agent + EEG evidence
  composite_shuffled_ev  composite with the evidence stream replaced by shuffled outputs (surrogate)
  assist+evidence        decoder + continuous posterior blend; evidence about the last second's net
                         motion (no agent segments, no exploration cost)
  assist+shuffled_ev     the same with shuffled evidence (surrogate)
  agent_only             agent on a flat posterior, no EEG at all (non-neural floor of the agent)
  random                 uniform random command

Run:  python -m neural_cursor.simulate_composite --users 6 --trials 24 --gain 8 --errp 6 --drift 0
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

from .config import DEFAULT, Config
from .decoder import EvidenceDecoder, VelocityDecoder
from .evaluation import bootstrap_ci, fitts_throughput, pearson_r, permutation_floor, time_to_target
from .simulate import SimUser, calibration
from .task import TrialRunner, fitts_targets

EPS = 0.2            # agent exploration probability (positivity of every evidence label)
SEGMENT_S = 1.0      # agent segment length; a wrong segment costs one second, not five
COMMIT_P = 0.8       # the agent enters a target (allowing dwell-select) only above this posterior
HOVER = 0.14         # otherwise it hovers just outside the target radius (0.08) — no Midas touch
N_EVIDENCE_CAL = 400 # labelled segments for the evidence decoder (the real engine yields ~2,500/h)


class SimUser2(SimUser):
    """SimUser with a 1 s buffer so the post-segment evidence epoch (0.2–0.8 s) is available."""

    def __init__(self, cfg: Config, seed: int, gain: float, eye: float, errp: float, drift: float):
        super().__init__(cfg, seed, gain, eye)
        self.sim.errp_uv = errp
        self.sim.drift = drift
        self.buf = np.zeros((self.fs, cfg.hardware.n_channels))        # 1 s

    def window(self) -> np.ndarray:                                    # last 0.5 s for velocity
        from .signal import common_average
        return common_average(self.buf[-self.win:])

    def evidence_epoch(self) -> np.ndarray:                            # 0.2–0.8 s after segment onset
        from .signal import common_average
        a, b = int(0.2 * self.fs), int(0.8 * self.fs)
        return common_average(self.buf[a:b])


def evidence_calibration(user: SimUser2, n: int, rng: np.random.Generator):
    """Random agent moves against a cued goal; label = move reduced the distance to the cued goal."""
    ws, ys = [], []
    for _ in range(n):
        goal = rng.uniform(0, 2 * np.pi)
        g = np.array([np.cos(goal), np.sin(goal)])
        move = rng.uniform(0, 2 * np.pi)
        m = np.array([np.cos(move), np.sin(move)])
        wrong = bool(m @ g < 0)
        user.sim.evoke(wrong)
        for _ in range(int(SEGMENT_S * user.fs / user.hop)):
            user.step(g)
        ws.append(user.evidence_epoch())
        ys.append(0 if wrong else 1)
    return ws, np.array(ys)


def run_condition(name: str, user: SimUser2, vdec, edec, cfg: Config, blocks, rng):
    use_agent = name in ("agent+evidence", "composite", "composite_shuffled_ev", "agent_only")
    motion_ev = name in ("assist+evidence", "assist+shuffled_ev")
    gain = 0.5 if (name == "decoder+assist" or name.startswith("composite") or motion_ev) else (0.7 if use_agent else 0.0)
    runner = TrialRunner(dataclasses.replace(cfg.task, assist_gain=gain), rng)
    outs = []
    for targets, width, goal in blocks:
        gpos = targets[goal]

        def true_dir(st):
            d = gpos - st.pos
            return d / (np.linalg.norm(d) + 1e-9)

        def command(st, t):
            user.step(true_dir(st))                                   # EEG advances in every condition
            if name == "oracle":
                return true_dir(st)
            if name == "random":
                return rng.uniform(-1, 1, 2)
            if name in ("agent+evidence", "agent_only"):
                return np.zeros(2)
            return vdec.predict(user.window())

        def agent(st, post):
            if rng.random() < EPS or name == "agent_only":
                k = int(rng.integers(0, len(targets)))
            else:
                k = int(np.argmax(post.p))
            d = targets[k] - st.pos
            dist = np.linalg.norm(d) + 1e-9
            d = d / dist
            user.sim.evoke(wrong=bool(d @ true_dir(st) < 0))         # the person's brain reacts
            confident = post.p.max() >= COMMIT_P and k == int(np.argmax(post.p))
            stop = 0.0 if confident else HOVER                        # hover until the posterior commits
            # never push *away* from a target: when hovering the agent stops and the person's own
            # decoder finishes the approach (a sign bug here cost 0.6 of success in round 2)
            return d * float(np.clip((dist - stop) / 0.15, 0.0, 1.0))

        def evidence(st, seg_dir):
            if name in ("composite_shuffled_ev", "assist+shuffled_ev"):
                return float(rng.uniform(0.05, 0.95))                 # no information, same scale
            return edec.predict_proba(user.evidence_epoch())

        def watched_motion(st, disp):
            if np.linalg.norm(disp) > 1e-3:
                user.sim.evoke(wrong=bool(disp @ true_dir(st) < 0))   # the person saw the cursor go wrong

        kw = dict(agent_fn=agent, segment_s=SEGMENT_S) if use_agent else {}
        if motion_ev:
            kw = dict(on_motion_segment=watched_motion, segment_s=SEGMENT_S)
        if name in ("agent+evidence", "composite", "composite_shuffled_ev", "assist+evidence", "assist+shuffled_ev"):
            kw["evidence_fn"] = evidence
        outs.append(runner.run(targets, width, goal, command, **kw))
    return outs


def simulate(cfg: Config, users: int, trials: int, gain: float, eye: float, errp: float,
             drift: float, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    results: Dict[str, List] = {}
    vr, ev_auc, ev_floor = [], [], []
    for u in range(users):
        user = SimUser2(cfg, seed + 100 * u, gain, eye, errp, drift)
        ws, ys = calibration(user, 300, rng)
        vdec = VelocityDecoder(cfg.decoder, cfg.hardware.fs).fit(ws[:240], ys[:240])
        pred = vdec.predict_many(ws[240:])
        vr.append(np.mean([pearson_r(pred[:, k], ys[240:, k]) for k in range(2)]))
        ews, eys = evidence_calibration(user, N_EVIDENCE_CAL + 80, rng)
        edec = EvidenceDecoder(cfg.decoder).fit(ews[:N_EVIDENCE_CAL], eys[:N_EVIDENCE_CAL])
        pe = edec.predict_many(ews[N_EVIDENCE_CAL:])
        fl = permutation_floor(lambda lab: roc_auc_score(lab, pe), eys[N_EVIDENCE_CAL:], 300, rng)
        ev_auc.append(fl.observed)
        ev_floor.append(fl.floor_p95)
        blocks = []
        for D, W in ((0.5, 0.08), (0.8, 0.08), (0.8, 0.05)):
            for _ in range(trials // 3):
                tg = fitts_targets(cfg.task.n_targets, D, W, rng)
                blocks.append((tg, W, int(rng.integers(0, cfg.task.n_targets))))
        for name in ("oracle", "decoder+assist", "assist+evidence", "assist+shuffled_ev",
                     "agent+evidence", "composite", "composite_shuffled_ev", "agent_only", "random"):
            results.setdefault(name, []).extend(
                run_condition(name, user, vdec, edec, cfg, blocks, np.random.default_rng(seed + u)))
    rng2 = np.random.default_rng(seed)
    table = {}
    for name, outs in results.items():
        sr, lo, hi = bootstrap_ci(np.array([o.success for o in outs], float), np.mean, 500, rng2)
        table[name] = dict(n=len(outs), success=sr, success_ci=(lo, hi), ttt_s=time_to_target(outs),
                           throughput_bps=fitts_throughput(outs))
    return dict(users=users, trials_per_user=trials, gain=gain, eye=eye, errp=errp, drift=drift,
                velocity_r=float(np.mean(vr)), evidence_auc=float(np.mean(ev_auc)),
                evidence_floor_p95=float(np.mean(ev_floor)), conditions=table)


def to_markdown(res: Dict) -> str:
    lines = [f"simulated users = {res['users']}, trials/user = {res['trials_per_user']}, signal gain = {res['gain']}, "
             f"eye gain = {res['eye']}, ErrP = {res['errp']} µV, drift = {res['drift']}/min",
             f"velocity decoder r = {res['velocity_r']:.3f}; evidence decoder AUC = {res['evidence_auc']:.3f} "
             f"(permutation floor p95 = {res['evidence_floor_p95']:.3f})", "",
             "| condition | n | success | 95% CI | time-to-target (s) | Fitts TP (bits/s) |", "|---|---|---|---|---|---|"]
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
    ap.add_argument("--errp", type=float, default=9.0)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=DEFAULT.eval.seed)
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()
    res = simulate(DEFAULT, a.users, a.trials, a.gain, a.eye, a.errp, a.drift, a.seed)
    print(to_markdown(res))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)
