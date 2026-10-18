"""Append-only run logging on one clock.

Layout of a run directory::

    run_<utc>/config.json          the frozen Config used
    run_<utc>/eeg.f32              raw µV samples, row-major (n_samples × n_channels), append-only
    run_<utc>/eeg_ts.f64           one LSL/local timestamp per sample
    run_<utc>/events.csv           t, kind, payload(JSON)   — markers, trial outcomes, gate flags
    run_<utc>/cursor.csv           t, x, y, ux, uy, p_goal_max, inside, dwell   — every control step

Raw samples are written *before* filtering so any pipeline can be replayed bit-exactly, and every
writer is opened in append mode (a truncating write once lost a registry in the previous project).

Crash safety: every ``eeg`` call flushes the sample and timestamp files together, and every
``event`` / ``cursor`` row is flushed as it is written, so a crash (a liblsl segfault, an OOM kill)
loses at most the chunk being written.  Python's buffering would otherwise hold back kilobytes per
file, by different amounts for ``eeg.f32`` and ``eeg_ts.f64``, and the run would become unreadable.
``load_eeg`` truncates a crashed run's two files to their common length instead of rejecting it.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from .config import Config


class RunLogger:
    def __init__(self, root: str, cfg: Config, tag: str = "", clock=time.time):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = os.path.join(root, f"run_{stamp}{('_' + tag) if tag else ''}")
        os.makedirs(self.dir, exist_ok=False)
        self.clock = clock
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            f.write(cfg.to_json())
        self._eeg = open(os.path.join(self.dir, "eeg.f32"), "ab")
        self._ts = open(os.path.join(self.dir, "eeg_ts.f64"), "ab")
        self._ev = open(os.path.join(self.dir, "events.csv"), "a")
        self._cur = open(os.path.join(self.dir, "cursor.csv"), "a")
        if os.path.getsize(os.path.join(self.dir, "events.csv")) == 0:
            self._ev.write("t,kind,payload\n")
            self._cur.write("t,x,y,ux,uy,p_goal_max,inside,dwell\n")
        self.n_samples = 0
        self.n_channels = cfg.hardware.n_channels

    def eeg(self, x: np.ndarray, ts: np.ndarray) -> None:
        if len(x) == 0:
            return
        self._eeg.write(np.ascontiguousarray(x, dtype=np.float32).tobytes())
        self._ts.write(np.ascontiguousarray(ts, dtype=np.float64).tobytes())
        self._eeg.flush()
        self._ts.flush()
        self.n_samples += len(x)

    def event(self, kind: str, payload: Optional[Any] = None, t: Optional[float] = None) -> None:
        t = self.clock() if t is None else t
        self._ev.write(f"{t:.6f},{kind},{json.dumps(payload, default=_np)}\n")
        self._ev.flush()

    def cursor(self, t, pos, u, p_goal_max, inside, dwell) -> None:
        self._cur.write(f"{t:.4f},{pos[0]:.5f},{pos[1]:.5f},{u[0]:.5f},{u[1]:.5f},"
                        f"{p_goal_max:.4f},{-1 if inside is None else inside},{dwell:.3f}\n")
        self._cur.flush()

    def close(self) -> None:
        for f in (self._eeg, self._ts, self._ev, self._cur):
            f.flush()
            f.close()

    @staticmethod
    def load_eeg(run_dir: str, n_channels: int):
        """(samples (n, n_channels) float32, timestamps (n,)).  A run cut off by a crash can end in a
        partial sample row and with one file a chunk ahead of the other; both are truncated to the
        last complete sample they share (with a warning), so a crashed run stays usable."""
        raw = np.fromfile(os.path.join(run_dir, "eeg.f32"), dtype=np.float32)
        ts = np.fromfile(os.path.join(run_dir, "eeg_ts.f64"), dtype=np.float64)
        n_rows = len(raw) // n_channels
        n = min(n_rows, len(ts))
        if len(raw) != n_rows * n_channels or n_rows != len(ts):
            warnings.warn(f"{run_dir}: {n_rows} sample rows ({len(raw) - n_rows * n_channels} stray values) vs "
                          f"{len(ts)} timestamps — the run was cut off; keeping the first {n} samples",
                          RuntimeWarning, stacklevel=2)
        return raw[:n * n_channels].reshape(n, n_channels), ts[:n]


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
