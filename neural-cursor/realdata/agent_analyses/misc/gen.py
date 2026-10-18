import numpy as np, os, json
from neural_cursor.acquisition import SimulatedEEG
from neural_cursor.logger import RunLogger

def make_run(root, cfg, tag, seed, n_cued, t0, n_closed=0, n_free=0, trial_s=2.5, iti_s=0.5):
    fs = cfg.hardware.fs
    lg = RunLogger(root, cfg, tag=tag)
    sim = SimulatedEEG(cfg.hardware, seed=seed)
    rng = np.random.default_rng(seed)
    now = lambda: t0 + sim.n_sent / fs
    def emit(sec, intent):
        n = int(round(sec * fs)); ts = now() + np.arange(n) / fs
        sim.set_intent(*intent); lg.eeg(sim.generate(n), ts)
    emit(3.0, (0, 0))
    lg.event("marker", {"note": "rest"}, t=now())
    kinds = ["cued_blank"] * n_cued + ["closed_loop"] * n_closed + ["free"] * n_free
    for i, kind in enumerate(kinds):
        a = rng.uniform(0, 2 * np.pi); goal = 0.8 * np.array([np.cos(a), np.sin(a)])
        start = {"trial": i, "goal": None if kind == "free" else goal, "block": kind,
                 "feedback_visible": kind != "cued_blank"}
        lg.event("trial_start", start, t=now())
        if kind == "cued_blank":
            emit(trial_s, goal / np.linalg.norm(goal))
        else:
            pos = np.zeros(2); dt = 0.1
            for _ in range(int(round(trial_s / dt))):
                d = goal - pos; u = d / max(np.linalg.norm(d), 1e-9)
                lg.cursor(now(), pos, u, 0.5, -1, 0.0)
                emit(dt, u)
                pos = pos + 0.25 * u * dt
        lg.event("trial_end", {"trial": i, "goal": goal if kind == "free" else None, "outcome": "timeout"}, t=now())
        emit(iti_s, (0, 0))
    lg.event("trial_start", {"trial": 999, "goal": [0.5, 0.5], "block": "cued_blank", "feedback_visible": False}, t=now())
    emit(1.5, (0.7, 0.7))       # crash: never ended
    lg.close()
    rng.standard_normal((1000, 4)).astype(np.float32).tofile(os.path.join(lg.dir, "eyetracker.f32"))
    return lg.dir
