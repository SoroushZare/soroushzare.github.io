"""Load ds004362 into per-subject epoch caches.

ds004362 = PhysioNet eegmmidb in BIDS: 109 subjects, 64-ch gel EEG, 160 Hz, BCI2000.
Runs 3/7/11 executed left/right fist, 4/8/12 imagined left/right fist, 5/9/13 executed both
fists/both feet, 6/10/14 imagined both fists/both feet.  Targets are shown on the screen
(left/right or top/bottom) for the whole 4.1 s trial — every label is a displayed position, so the
confound audit in experiments.py is not optional.

Cache layout (one file per subject, float32 µV):
    X_<task>: (n_trials, 64, n_samples)  1–40 Hz causal-free (zero-phase) band, -1.0 … 4.0 s
    y_<task>: (n_trials,)  0 = T1, 1 = T2
    o_<task>: (n_trials,)  run order index (0..), for sequential/online replay
    X_rest:   (n, 64, n_samples) T0 epochs from the task runs (idle class)
Technical exclusion (pre-declared, never performance-based): sampling rate != 160 Hz or fewer than
20 trials in a task.

Usage:  python -I -m realdata.load --data DIR --cache DIR [--jobs 4]
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

TASKS = {"exec_LR": (3, 7, 11), "imag_LR": (4, 8, 12), "exec_FF": (5, 9, 13), "imag_FF": (6, 10, 14)}
FS = 160
TMIN, TMAX = -1.0, 4.0
BAND = (1.0, 40.0)


def _read(path_set: str, path_ev: str):
    import mne
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_eeglab(path_set, preload=True, verbose="ERROR")
    sf = raw.info["sfreq"]
    raw.filter(*BAND, picks="all", verbose="ERROR")     # picks='all': FP channels filtered too (E-1)
    x = raw.get_data(picks="all") * 1e6                 # V → µV
    ev = []
    with open(path_ev) as f:
        next(f)
        for line in f:
            onset, _, sample, value = line.rstrip("\n").split("\t")
            ev.append((float(onset), value))
    return x.astype(np.float32), sf, raw.ch_names, ev


def subject(sub: int, data: str, cache: str):
    out = os.path.join(cache, f"sub-{sub:03d}.npz")
    if os.path.exists(out):
        return sub, "cached"
    res, ch_names, sfs = {}, None, set()
    rest = []
    n0, n1 = int(round(TMIN * FS)), int(round(TMAX * FS))
    for task, runs in TASKS.items():
        Xs, ys, os_ = [], [], []
        for k, r in enumerate(runs):
            base = os.path.join(data, f"sub-{sub:03d}_task-motion_run-{r}_")
            x, sf, ch_names, ev = _read(base + "eeg.set", base + "events.tsv")
            sfs.add(sf)
            if sf != FS:
                continue
            for onset, val in ev:
                s = int(round(onset * sf))
                if s + n0 < 0 or s + n1 > x.shape[1]:
                    continue
                seg = x[:, s + n0:s + n1]
                if val.endswith("T0"):
                    rest.append(seg)
                    continue
                Xs.append(seg)
                ys.append(0 if val.endswith("T1") else 1)
                os_.append(k)
        if Xs:
            res[f"X_{task}"] = np.stack(Xs)
            res[f"y_{task}"] = np.array(ys, np.int8)
            res[f"o_{task}"] = np.array(os_, np.int8)
    if rest:
        res["X_rest"] = np.stack(rest[::3])                # every third T0 epoch keeps the cache small
    res["ch_names"] = np.array(ch_names)
    res["sfreq"] = np.array(sorted(sfs), float)
    np.savez_compressed(out, **res)
    return sub, f"sf={sorted(sfs)} " + " ".join(f"{t}:{len(res.get('y_'+t, []))}" for t in TASKS)


def load_subject(cache: str, sub: int):
    return dict(np.load(os.path.join(cache, f"sub-{sub:03d}.npz"), allow_pickle=False))


def usable_subjects(cache: str, task: str, min_trials: int = 20):
    subs = []
    for f in sorted(os.listdir(cache)):
        if not f.startswith("sub-"):
            continue
        d = np.load(os.path.join(cache, f), allow_pickle=False)
        if list(d["sfreq"]) == [float(FS)] and f"y_{task}" in d and len(d[f"y_{task}"]) >= min_trials:
            subs.append(int(f[4:7]))
    return subs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--subjects", type=str, default="1-109")
    a = ap.parse_args()
    lo, hi = map(int, a.subjects.split("-"))
    os.makedirs(a.cache, exist_ok=True)
    from realdata.load import subject as _subject      # module-level reference so workers can pickle it
    with ProcessPoolExecutor(a.jobs) as ex:
        for sub, msg in ex.map(_subject, range(lo, hi + 1), [a.data] * (hi - lo + 1), [a.cache] * (hi - lo + 1)):
            print(f"sub-{sub:03d} {msg}", flush=True)
