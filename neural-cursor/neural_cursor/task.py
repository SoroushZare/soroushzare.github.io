"""The cursor task, headless.

* ``CursorState``   position/velocity in normalised screen coordinates [−1, 1]²
* ``fitts_targets`` ISO 9241-9 style target sets (distance × width conditions) for throughput
* ``Cursor``        velocity-control dynamics: smoothed decoder output → velocity → position
* ``GoalPosterior`` shared-autonomy inference over on-screen candidate targets.  It never sees the
                    true goal (the RLNF generator did; that is the first thing fixed here).
* ``TrialRunner``   one trial = pick hidden goal → run until dwell-select or timeout → log outcome

A renderer (pygame/psychopy) subscribes to ``Cursor`` and ``TrialRunner`` and draws; nothing here
depends on it, so the same code runs in CI with the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import Task

SMOOTHING_STEP_S = 0.1
"""The step ``Task.velocity_smoothing`` is defined for (the original 10-Hz loop's period)."""


@dataclass
class CursorState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    t: float = 0.0


def fitts_targets(n: int, distance: float, width: float, rng: np.random.Generator) -> np.ndarray:
    """n targets on a ring of the given distance from the centre; width stored by the caller.

    Order is a random permutation so a sequence effect cannot masquerade as control.
    """
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 2 * np.pi)
    pts = np.stack([np.cos(ang), np.sin(ang)], axis=1) * distance
    return pts[rng.permutation(n)]


class Cursor:
    """Velocity control with exponential smoothing, speed cap and screen clamp."""

    def __init__(self, cfg: Task):
        self.cfg = cfg
        self.state = CursorState()
        self._v_smooth = np.zeros(2)

    def reset(self, pos=(0.0, 0.0)) -> None:
        self.state = CursorState(pos=np.asarray(pos, float).copy())
        self._v_smooth = np.zeros(2)

    def step(self, command: np.ndarray, dt: float) -> CursorState:
        """command: decoder output in [−1, 1]² (unit = full speed in that axis).  The smoothing constant
        is defined per ``SMOOTHING_STEP_S`` and rescaled to ``dt``, so the cursor's time constant is the
        same at the original 10-Hz loop and the champion's 16-Hz one."""
        a = self.cfg.velocity_smoothing ** (dt / SMOOTHING_STEP_S)
        self._v_smooth = a * self._v_smooth + (1 - a) * np.clip(command, -1, 1)
        v = self._v_smooth * self.cfg.max_speed
        p = np.clip(self.state.pos + v * dt, -1, 1)
        self.state = CursorState(pos=p, vel=v, t=self.state.t + dt)
        return self.state


class GoalPosterior:
    """Bayesian inference over candidate targets from two evidence streams.

    1. Continuous: decoded velocity direction.  Likelihood of target g given command u is a von Mises
       in the angle between u and (g − pos), with concentration κ = |u| × temperature so a weak
       command carries little evidence.
    2. Discrete: an evidence packet (accept/veto classifier output) about the agent's last action,
       p_accept ∈ [0, 1].  It upweights the targets the action moved toward.

    The posterior is the agent's belief; the hidden goal is never read.  Log every update.
    """

    def __init__(self, targets: np.ndarray, cfg: Task):
        self.targets = targets
        self.cfg = cfg
        self.logp = np.full(len(targets), -np.log(len(targets)))

    @property
    def p(self) -> np.ndarray:
        return np.exp(self.logp - self.logp.max()) / np.exp(self.logp - self.logp.max()).sum()

    def update_velocity(self, pos: np.ndarray, u: np.ndarray) -> None:
        mag = float(np.linalg.norm(u))
        if mag < 1e-6:
            return
        d = self.targets - pos
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        cos = d @ (u / mag)
        kappa = mag * self.cfg.posterior_temperature
        self.logp += kappa * cos
        self.logp -= self.logp.max()

    def update_evidence(self, pos: np.ndarray, action_dir: np.ndarray, p_accept: float) -> None:
        d = self.targets - pos
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        agree = (d @ action_dir / (np.linalg.norm(action_dir) + 1e-9) > 0).astype(float)
        p_accept = float(np.clip(p_accept, 1e-3, 1 - 1e-3))
        # likelihood of the evidence under "goal is g": accept if the action helps g
        lik = agree * p_accept + (1 - agree) * (1 - p_accept)
        self.logp += np.log(lik)
        self.logp -= self.logp.max()

    def assist_command(self, pos: np.ndarray) -> np.ndarray:
        """Expected direction to the goal under the posterior, unit length × confidence."""
        d = self.targets - pos
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        m = self.p @ d
        return m   # |m| ≤ 1; small when the posterior is diffuse


@dataclass
class TrialOutcome:
    goal_idx: int
    selected_idx: Optional[int]
    success: bool
    movement_time_s: float
    path_length: float
    endpoint_error: float          # distance from goal centre at selection/timeout
    distance: float
    width: float


class TrialRunner:
    """Runs one trial per call to ``run``; the decoder is any callable returning a command."""

    def __init__(self, cfg: Task, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.cursor = Cursor(cfg)

    def run(self, targets: np.ndarray, width: float, goal_idx: int, command_fn,
            on_step=None, evidence_fn=None, agent_fn=None, segment_s: float = 1.0,
            on_motion_segment=None) -> TrialOutcome:
        """command_fn(state, t) → decoder command in [−1, 1]²; evidence_fn(state, action) → p_accept
        about the agent's last action; agent_fn(state, posterior) → the agent's direction for the
        next segment (shared autonomy: the agent steers, the person's EEG judges and biases).

        Motion-evidence mode (agent_fn None, on_motion_segment given): the cursor is driven by the
        continuous posterior blend, and every segment_s the net displacement of the last segment is
        what the evidence is about — on_motion_segment(state, displacement) lets the environment
        react to the motion the person just watched; evidence_fn(state, displacement) is read one
        segment later and updates the posterior about that displacement.  No exploration cost."""
        cfg = self.cfg
        self.cursor.reset()
        post = GoalPosterior(targets, cfg)
        self.post = post
        seg_dir, seg_t = None, -1e9
        seg_start_pos, prev_disp, motion_t = self.cursor.state.pos.copy(), None, 0.0
        dt = 1.0 / cfg.control_hz
        t, dwell, path = 0.0, 0.0, 0.0
        selected = None
        last_action = None
        while t < cfg.trial_timeout_s:
            st = self.cursor.state
            u = np.asarray(command_fn(st, t), float)
            post.update_velocity(st.pos, u)
            if agent_fn is not None and t - seg_t >= segment_s:
                if evidence_fn is not None and seg_dir is not None:
                    post.update_evidence(st.pos, seg_dir, evidence_fn(st, seg_dir))
                seg_dir, seg_t = np.asarray(agent_fn(st, post), float), t
            if agent_fn is not None:
                u = (1 - cfg.assist_gain) * u + cfg.assist_gain * seg_dir
            elif cfg.assist_gain > 0:
                u = (1 - cfg.assist_gain) * u + cfg.assist_gain * post.assist_command(st.pos)
            if on_motion_segment is not None and t - motion_t >= segment_s:
                if evidence_fn is not None and prev_disp is not None and np.linalg.norm(prev_disp) > 1e-3:
                    post.update_evidence(st.pos, prev_disp, evidence_fn(st, prev_disp))
                disp = st.pos - seg_start_pos
                on_motion_segment(st, disp)
                prev_disp, seg_start_pos, motion_t = disp, st.pos.copy(), t
            last_action = u.copy()
            prev = st.pos.copy()
            st = self.cursor.step(u, dt)
            path += float(np.linalg.norm(st.pos - prev))
            t += dt
            dists = np.linalg.norm(targets - st.pos, axis=1)
            inside = int(np.argmin(dists)) if dists.min() < width else None
            dwell = dwell + dt if inside is not None else 0.0
            if on_step:
                on_step(st, u, post.p, inside, dwell)
            if inside is not None and dwell >= cfg.dwell_s:
                selected = inside
                break
            t = min(t, cfg.trial_timeout_s)
        goal = targets[goal_idx]
        return TrialOutcome(goal_idx=goal_idx, selected_idx=selected,
                            success=(selected == goal_idx),
                            movement_time_s=t, path_length=path,
                            endpoint_error=float(np.linalg.norm(self.cursor.state.pos - goal)),
                            distance=float(np.linalg.norm(goal)), width=width)
