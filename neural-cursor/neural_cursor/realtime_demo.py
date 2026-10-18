"""End-to-end pure-EEG real-time demo: a simulated SMR user imagines feet, hands, right and left, and a
screen cursor and a planar arm follow, live over LSL.

What it shows
-------------
The champion chain (INTERFACES_CHAMPION G, J and the realtime contract) running on a wall clock:

1. **Calibration (offline-paced, no wall clock).**  One ``sim_body.BodySubject`` records a setup rest
   (the session's re-centring reference and the idle gate's rest class, source "L0") and a block of L1
   fixation probes (central glyph, nothing spatial drawn), through ``RunLogger`` → ``SessionStore`` →
   ``Labeler`` exactly as ``simulate_champion.record_fixation`` does for every simulated session.
   ``data_engine.BodyTrainer`` fits the deployed ``BodyCoordinateDecoder`` on those L1 windows (it
   refuses any other source), ``earn_x`` decides the x axis on the same probes by the pre-declared rule
   — two consecutive passing grant tests, here on the first half of the probes and on all of them —
   and an earned x refits the y model with one-hand windows as neutral (so lateral imagery does not
   drive y), and ``fit_idle`` arms the idle gate only if it can tell rest from intent.  No label the
   display could identify reaches the decoder.
2. **Live run (wall clock).**  The SAME person, later in the same session (same electrode gains, so the
   setup rest reference applies, but a fresh block: new noise, blinks and rhythm phases), is published
   over a real LSL outlet by ``BodyLSLStreamer`` in 10-ms chunks.  ``acquisition.LSLEEG`` pulls it and
   ``realtime.RealTimeLoop`` runs causal filter → amplitude gate → CAR → decoder → identity mapping →
   effector at ``cfg.task.control_hz`` with the decoder's 1-s window (``loop_config``).  The effector is
   ``effectors.ScreenCursor`` with the log backend (headless: integrated and clamped, nothing moved),
   ``effectors.PlanarArm2Link``, or both at once (``TeeEffector``: one decoded command, two devices).
3. **The simulated person** follows a cue schedule of ``segment_s`` segments: FEET (body +y, cursor
   and arm up), HANDS (−y, down) and, only when x was earned, RIGHT (+x) and LEFT (−x), in turn.  Each
   segment starts a new imagery episode (``BodySubject.cue``: the ERD rises to its peak 1.5 s after the
   cue and decays toward half depth, ds004362's feet/fists course).

What it reports (``results/realtime_demo.md``)
----------------------------------------------
* latency p50 / p95 / max on the sample clock (newest LSL sample → ``effector.send`` returned: transport,
  tick wait, preprocessing, decoding and the device call), jitter, the achieved update rate, drops,
  overruns, invalid and stale ticks;
* per segment: the cued class, the mean decoded body velocity and each effector's displacement over the
  scored part of the segment, and whether the displacement's sign on the cued axis agrees with the
  imagined direction; and the CROSS-AXIS drift — the mean decoded y while the person imagines one hand
  (RIGHT/LEFT), and x while imagining feet or hands — since an arm uses sustained y for grasp and axis
  switches, and one-hand imagery reading as HANDS would issue y commands.  The scored part starts ``SCORE_LOCKOUT_S`` after the cue (the protocol's
  closed-loop control onset, ``ClosedLoopTiming.label_start_s``): before it the 1-s window still holds
  the previous segment's imagery, and the new ERD has not yet risen.  A segment cut short by the end of
  the run is scored only if its scored part has at least ``MIN_SCORED_S``.

Pure-EEG rule: the decoder receives only the loop's preprocessed EEG windows and its per-channel
validity mask.  The cue schedule (``intent_fn``) drives the simulated person and labels the report; its
return value is only logged by the loop.  Effector positions are read for the report only.

Run:  python3 -m neural_cursor.realtime_demo --seconds 30 --effector both
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .acquisition import LSLEEG
from .body_decoder import BodyCoordinateDecoder
from .config import DEFAULT, Config
from .data_engine import BodyTrainer, body_labels
from .effectors import Effector, PlanarArm2Link, ScreenCursor
from .mapping import BODY_VECTORS
from .protocol import Block, ClosedLoopTiming
from .realtime import RealTimeLoop, SimLSLStreamer
from .sim_body import BodySubject
from .simulate_champion import ChampionSimConfig, champion_config, record_fixation
from .simulate_engine import CLOCK_ORIGIN_S

EFFECTORS: Tuple[str, ...] = ("arm", "cursor", "both")
Y_CLASSES: Tuple[str, ...] = ("FEET", "HANDS")
X_CLASSES: Tuple[str, ...] = ("RIGHT", "LEFT")
SCORE_LOCKOUT_S = ClosedLoopTiming().label_start_s
"""Seconds after a segment's cue before it is scored (1.5 s: the protocol's control onset after a
cue; module docstring)."""
MIN_SCORED_S = 1.0
"""A segment truncated by the end of the run is scored only with at least this much scored time
(sixteen control updates at 16 Hz); a shorter tail is listed but not counted."""
MIN_CALIBRATION_MINUTES = 2.0
"""Shortest calibration: 16 probes (4 per class); its first half — the first grant test — then holds
20 x windows, twice ``n_splits``, and both y classes (a shorter one fails inside ``fit`` / ``earn_x``
after the whole recording)."""
DEFAULT_OUT = os.path.join("results", "realtime_demo.md")
ARM_LINKS_M: Tuple[float, float] = (0.3, 0.25)
"""Link lengths of the simulated arm (``PlanarArm2Link``'s defaults, named so the start pose's inverse
kinematics and the arm are built from the same numbers)."""


@dataclass(frozen=True)
class DemoConfig:
    """Hyperparameters of the demo (everything not in ``config.Config``).

    seconds            timed live run (after the loop's 1-s warm-up).
    effector           "arm", "cursor" or "both" (one loop driving both through ``TeeEffector``).
    seed               the simulated person (``BodySubject.seed``) and the probe schedule.  Person 2 is
                       a demo choice: with the depths below its x passes ``earn_x`` within the default
                       calibration, so the x path runs; it is not a population estimate.
    y_depth, x_depth   ERD depths of feet/hands and one-hand imagery: 0.4 / 0.5, the top of ``sim_body``'s
                       calibrated ranges (a good responder, disclosed as such in the report: the demo shows
                       the chain working, not a population estimate).  Measured for person 2 on 16 min
                       of L1 probes (5 contiguous folds, 1-s windows): window AUC 0.73 on y, 0.69 on x.
    x_responder        lateralised L/R ERD (only then can x be earned; ≈ 20 % of people).
    dry_penalty        dry-cap scaling of the motor ERD depth, in [0, 1) (``sim_body``; 0 = gel-equivalent;
                       not the d′ loss, which ``sim_body`` tabulates).
    calibration_minutes  L1 fixation-probe minutes (8 probes a minute, 2 per class), at least
                       ``MIN_CALIBRATION_MINUTES``.  x is granted on two consecutive passing grant tests:
                       on the first half of the probes and on all of them.  24 min (192 probes): person 2's
                       first half (12 min) passes at AUC 0.64 against a floor of 0.60, and all of it at
                       0.685 against 0.585; at 16 min the first half (AUC 0.56) fails and x stays off.
    rest_s             setup rest for the re-centring reference and the idle gate's rest class.
    label_hop_s        hop between labelled 1-s windows (``simulate_champion``'s 0.5 s).
    x_permutations     circular-shift null draws of ``earn_x`` (body_decoder default 200).
    segment_s          length of one imagery segment of the live cue schedule.  5 s holds the 1.5-s
                       lockout plus 3.5 s of scored control, about one L2b trial's useful span.
    chunk_s            LSL push granularity of the simulated amplifier (10 ms, as real drivers).
    screen_px          (width, height) of the log-backend screen.
    cursor_gain_px_s   ScreenCursor speed at full command.  80 px/s: a 5-s segment at a strong
                       |command| of 0.8 travels 320 px, inside the 540 px from the centre to an edge, so
                       the clamp does not mask the direction in a demo-length run.  (No targets are
                       drawn, so nothing re-centres the device; a run long enough to drift to a wall
                       would show zero displacement there.)
    arm_max_speed      PlanarArm2Link end-effector speed at full command (m/s).  0.04 for the same
                       reason: ≤ 0.16 m per segment against ≥ 0.19 m of free workspace around the start.
    arm_start_xy       start of the arm's end effector (m): mid-workspace, so up, down, left and right
                       all have room (the arm's default pose sits 0.16 m below its reach limit).
    """
    seconds: float = 30.0
    effector: str = "both"
    seed: int = 2
    y_depth: float = 0.4
    x_depth: float = 0.5
    x_responder: bool = True
    dry_penalty: float = 0.0
    calibration_minutes: float = 24.0
    rest_s: float = 30.0
    label_hop_s: float = 0.5
    x_permutations: int = 200
    segment_s: float = 5.0
    chunk_s: float = 0.010
    screen_px: Tuple[int, int] = (1920, 1080)
    cursor_gain_px_s: float = 80.0
    arm_max_speed: float = 0.04
    arm_start_xy: Tuple[float, float] = (0.0, 0.30)

    def __post_init__(self) -> None:
        if self.effector not in EFFECTORS:
            raise ValueError(f"effector must be one of {EFFECTORS}, got {self.effector!r}")
        if not self.seconds > 0 or not self.segment_s > SCORE_LOCKOUT_S + MIN_SCORED_S:
            raise ValueError(f"need seconds > 0 and segment_s > {SCORE_LOCKOUT_S + MIN_SCORED_S} s "
                             f"(lockout plus a scorable span), got {self.seconds}, {self.segment_s}")
        if not self.calibration_minutes >= MIN_CALIBRATION_MINUTES or not self.rest_s >= 1.0:
            raise ValueError(f"calibration_minutes must be >= {MIN_CALIBRATION_MINUTES:g} (both y classes and "
                             f"enough x windows for the first grant test) and rest_s >= 1 s, got "
                             f"{self.calibration_minutes}, {self.rest_s}")
        if not 0.0 <= self.dry_penalty < 1.0:
            raise ValueError(f"dry_penalty must be in [0, 1), got {self.dry_penalty}")


# ----------------------------------------------------------------------------------------------
# Streaming, effectors and the decoder tap
# ----------------------------------------------------------------------------------------------
class BodyLSLStreamer(SimLSLStreamer):
    """``SimLSLStreamer`` for a ``BodySubject``, adding ``cue``: a new imagery episode that restarts the
    ERD time course.  ``set_intent`` alone would continue the current episode, so a switch from feet to
    hands would inherit an already decayed ERD instead of a fresh one.  Takes the streamer's lock, like
    ``set_intent``, because the streaming thread is calling ``generate`` on the same subject."""

    def cue(self, vx: float, vy: float) -> None:
        with self._lock:
            self.sim.cue(vx, vy)


class TeeEffector(Effector):
    """Sends one command to several 2-D effectors in the same tick (``--effector both``).

    It forwards the clipped, unscaled command (its own ``max_speed`` is 1), so each device applies its
    own speed and safety limits; the loop's latency then includes every device call.  ``task_position``
    is the first child's that draws a cursor (the loop logs the cursor the person sees); ``close``
    closes every child, which brings each to rest."""

    def __init__(self, children: Dict[str, Effector]):
        super().__init__(dim=2, max_speed=1.0)
        if not children or any(c.dim != 2 for c in children.values()):
            raise ValueError("TeeEffector needs at least one 2-D effector")
        self.children = dict(children)

    def _apply(self, u: np.ndarray, dt: float) -> None:
        for child in self.children.values():
            child.send(u, dt)

    def task_position(self) -> Optional[np.ndarray]:
        for child in self.children.values():
            pos = child.task_position()
            if pos is not None:
                return pos
        return None

    def _stop(self) -> None:
        for child in self.children.values():
            child.close()

    def state(self) -> dict:
        return {**super().state(), "children": {k: c.state() for k, c in self.children.items()}}


class DecoderTap:
    """Passes the loop's window and mask to the decoder unchanged and records each output with the tick
    time the cue schedule set (``mark``), so the report can average decoded velocity per segment.  It
    adds nothing to the decoder's input: the pure-EEG rule holds through it."""

    def __init__(self, decoder: BodyCoordinateDecoder):
        self.decoder = decoder
        self.t = float("nan")
        self.rows: List[Tuple[float, float, float]] = []

    def mark(self, t: float) -> None:
        self.t = t

    def predict(self, window: np.ndarray, good: Optional[np.ndarray] = None) -> np.ndarray:
        out = self.decoder.predict(window, good=good)
        self.rows.append((self.t, float(out[0]), float(out[1])))
        return out


def arm_joint_angles(l1: float, l2: float, xy: Sequence[float]) -> Tuple[float, float]:
    """Elbow-up inverse kinematics (q2 ≥ 0, the branch ``PlanarArm2Link``'s default elbow limits keep)
    of a 2-link arm for end-effector point ``xy``."""
    x, y = float(xy[0]), float(xy[1])
    c2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if not -1.0 <= c2 <= 1.0:
        raise ValueError(f"point {xy} is out of reach of links {l1}, {l2}")
    q2 = float(np.arccos(c2))
    q1 = float(np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2)))
    return q1, q2


def effector_position(effector: Effector) -> np.ndarray:
    """Where a device is, for the report: the cursor in task units ([−1, 1]², +y up), the arm's end
    effector in metres (+y up, away from the base)."""
    pos = effector.task_position()
    return np.asarray(effector.state()["pos"] if pos is None else pos, float)


def make_effectors(demo: DemoConfig) -> Dict[str, Effector]:
    """The devices of one run, by name."""
    out: Dict[str, Effector] = {}
    if demo.effector in ("cursor", "both"):
        out["cursor"] = ScreenCursor(*demo.screen_px, gain_px_s=demo.cursor_gain_px_s, backend="log")
    if demo.effector in ("arm", "both"):
        out["arm"] = PlanarArm2Link(*ARM_LINKS_M, max_speed=demo.arm_max_speed,
                                    q0=arm_joint_angles(*ARM_LINKS_M, demo.arm_start_xy))
    return out


# ----------------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------------
def make_subject(demo: DemoConfig, cfg: Config) -> BodySubject:
    """The simulated person.  No screen or covert leak: nothing spatial is drawn in the calibration and
    the live demo has no targets, so only the motor pathway carries the intent."""
    return BodySubject(cfg.hardware, seed=demo.seed, y_depth=demo.y_depth, x_depth=demo.x_depth,
                       x_responder=demo.x_responder, dry_penalty=demo.dry_penalty)


def calibrate(demo: DemoConfig, cfg: Config, subject: BodySubject) -> Tuple[BodyCoordinateDecoder, dict]:
    """Setup rest + L1 fixation probes → the deployed decoder, re-centred on the session's rest, with
    x decided by ``earn_x`` and the idle gate fitted.  Returns the decoder and a calibration summary."""
    sim = ChampionSimConfig(days=1, users=1, pop=0, seed=demo.seed, dry_penalty=demo.dry_penalty,
                            screen_leak=0.0, rest_s=demo.rest_s, label_hop_s=demo.label_hop_s,
                            x_permutations=demo.x_permutations)
    block = Block("fixation_probe", demo.calibration_minutes * 60.0, seed=demo.seed)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nc_demo_") as root:
        probes, refs, rest = record_fixation(cfg, sim, subject, 1, block, 0, "demo_calibration", root,
                                             CLOCK_ORIGIN_S)
    trainer = BodyTrainer(cfg, decoder_options={"n_permutations": demo.x_permutations})
    decoder = trainer.fit(probes, session_references=refs)
    # earn_x's confirm rule: the grant test on the first half of the probes, then on all of them.
    t_mid = float(np.median([p["t"] for p in probes.provenance]))
    first = probes.mask(np.array([p["t"] <= t_mid for p in probes.provenance], bool))
    first_report = decoder.earn_x(list(first.windows), body_labels(first), first.provenance, refs)
    x_report = decoder.earn_x(list(probes.windows), body_labels(probes), probes.provenance, refs,
                              previous_grant_passed=first_report["passed"])
    if decoder.x_enabled:                           # an earner's y model treats one-hand imagery as neutral
        earned = trainer.fit(probes, session_references=refs, neutral_one_hand=True)
        earned.x_enabled, earned.x_report = True, x_report
        decoder = earned
    idle_report = decoder.fit_idle(list(rest.windows), rest.provenance, refs)
    (reference,) = refs.values()
    decoder.set_reference(reference=reference)
    labels = body_labels(probes)
    summary = {"n_windows": len(probes), "n_rest_windows": len(rest),
               "n_by_class": {name: int(np.sum(np.all(labels == np.array(vec, int), axis=1)))
                              for name, vec in BODY_VECTORS.items()},
               "sources": sorted({p.get("source") for p in probes.provenance}),
               "x_report": x_report, "x_first_report": first_report, "idle_report": idle_report,
               "x_enabled": bool(decoder.x_enabled), "neutral_one_hand": bool(decoder.neutral_one_hand),
               "seconds": time.perf_counter() - t0}
    return decoder, summary


# ----------------------------------------------------------------------------------------------
# Live run
# ----------------------------------------------------------------------------------------------
def schedule(x_enabled: bool) -> Tuple[str, ...]:
    """Cue classes in the order the person imagines them, cycled: feet then hands, and right then left
    once x is earned (before that the person imagines only y, as in ``simulate_champion``)."""
    return Y_CLASSES + (X_CLASSES if x_enabled else ())


def live_run(demo: DemoConfig, cfg: Config, subject: BodySubject, decoder: BodyCoordinateDecoder,
             block_index: int = 2) -> dict:
    """One wall-clock run over LSL: returns the loop's statistics and the per-segment scores.

    ``block_index`` 2 follows the calibration's rest (block 0) and probe block (1): a block's noise,
    blink and rhythm streams are keyed by its index, so reusing 1 would replay the calibration EEG."""
    loop_cfg = decoder.loop_config(cfg)
    classes = schedule(decoder.x_enabled)
    effectors = make_effectors(demo)
    device = TeeEffector(effectors) if len(effectors) > 1 else next(iter(effectors.values()))
    tap = DecoderTap(decoder)
    ticks: List[Tuple[float, int, Dict[str, np.ndarray]]] = []
    current = {"segment": -1}

    def positions() -> Dict[str, np.ndarray]:
        return {name: effector_position(e) for name, e in effectors.items()}

    def intent_fn(t: float):
        """The simulated person's cue schedule: a new imagery episode at every segment boundary."""
        seg = int(t // demo.segment_s)
        vec = BODY_VECTORS[classes[seg % len(classes)]]
        if seg != current["segment"]:
            streamer.cue(*vec)
            current["segment"] = seg
        tap.mark(t)
        ticks.append((t, seg, positions()))
        return vec

    subject.begin_block(1, block_index)             # same session (gains, reference), fresh block
    streamer = BodyLSLStreamer(subject, chunk_s=demo.chunk_s,
                               source_id=f"nc-demo-{uuid.uuid4().hex[:12]}")
    streamer.start()
    src = LSLEEG(cfg.hardware, timeout_s=5.0, source_id=streamer.source_id)
    try:
        src.start()
        loop = RealTimeLoop(src, tap, device, loop_cfg)
        stats = loop.run(demo.seconds, intent_fn=intent_fn)
        end = (stats["elapsed_s"], -1, positions())
    finally:
        src.close()                                 # consumer first, then the outlet
        streamer.stop()
        device.close()
    segments = score_segments(ticks + [end], tap.rows, classes, demo.segment_s, list(effectors))
    return {"stats": stats, "segments": segments, "effectors": list(effectors),
            "update_rate_hz": stats["n_updates"] / stats["elapsed_s"] if stats["elapsed_s"] > 0 else 0.0,
            "summary": summarise(segments, list(effectors))}


def score_segments(ticks: Sequence[Tuple[float, int, Dict[str, np.ndarray]]],
                   decoded: Sequence[Tuple[float, float, float]], classes: Sequence[str], segment_s: float,
                   names: Sequence[str]) -> List[dict]:
    """Per segment: the mean decoded body velocity and each effector's displacement over the scored
    span [cue + ``SCORE_LOCKOUT_S``, next cue), and the sign agreement on the cued axis.

    ``ticks`` are (t, segment, positions) at the top of each tick, before that tick's command, plus a
    final row after the run; a displacement is therefore the motion produced by the scored ticks'
    commands.  NaN decoded rows (refused windows: zero commands) are left out of the mean."""
    t = np.array([row[0] for row in ticks])
    dec = np.array(decoded, float).reshape(-1, 3)
    out = []
    n_seg = max(row[1] for row in ticks) + 1
    for seg in range(n_seg):
        cls = classes[seg % len(classes)]
        vec = np.array(BODY_VECTORS[cls], float)
        axis = int(np.flatnonzero(vec)[0])
        start, stop = seg * segment_s + SCORE_LOCKOUT_S, min((seg + 1) * segment_s, t[-1])
        i0 = int(np.searchsorted(t, start))
        i1 = int(np.searchsorted(t, stop))                # first tick of the next segment, or the end row
        row = {"segment": seg, "class": cls, "cue": vec.tolist(), "axis": "xy"[axis],
               "scored_s": max(0.0, stop - start), "scored": bool(stop - start >= MIN_SCORED_S and i1 > i0)}
        m = (dec[:, 0] >= start) & (dec[:, 0] < stop) & np.isfinite(dec[:, 1:]).all(axis=1)
        row["n_decoded"] = int(m.sum())
        row["decoded_mean"] = dec[m, 1:].mean(axis=0).tolist() if m.any() else [float("nan")] * 2
        row["decoded_agrees"] = bool(m.any() and np.sign(row["decoded_mean"][axis]) == vec[axis])
        row["cross_axis_mean"] = row["decoded_mean"][1 - axis]            # drift on the axis NOT imagined
        row["displacement"], row["agrees"] = {}, {}
        for name in names:
            d = ticks[i1][2][name] - ticks[i0][2][name] if row["scored"] else np.full(2, np.nan)
            row["displacement"][name] = d.tolist()
            row["agrees"][name] = bool(row["scored"] and np.sign(d[axis]) == vec[axis])
        out.append(row)
    return out


def summarise(segments: Sequence[dict], names: Sequence[str]) -> dict:
    """Sign agreement counts per axis and device over the scored segments ({axis: {device: [k, n]}});
    "decoded" counts the decoder's mean output itself.  ``cross_axis_drift`` {axis: mean decoded output
    on the OTHER axis over that axis's scored segments}: y while imagining one hand (x segments), x
    while imagining feet or hands."""
    out: Dict[str, Dict[str, List[int]]] = {}
    drift: Dict[str, float] = {}
    for axis in "yx":
        rows = [s for s in segments if s["scored"] and s["axis"] == axis]
        if not rows:
            continue
        out[axis] = {name: [sum(s["agrees"][name] for s in rows), len(rows)] for name in names}
        out[axis]["decoded"] = [sum(s["decoded_agrees"] for s in rows), len(rows)]
        vals = [s["cross_axis_mean"] for s in rows if np.isfinite(s["cross_axis_mean"])]
        drift[axis] = float(np.mean(vals)) if vals else float("nan")
    if drift:
        out["cross_axis_drift"] = drift
    return out


def run_demo(demo: DemoConfig, cfg: Config = DEFAULT) -> dict:
    """Calibrate the simulated person, then run the live loop; returns everything the report shows.
    ``cfg`` gets the body decoder's 1-s window (``champion_config``), so the labeler and the loop cut
    the windows the decoder is trained on."""
    cfg = champion_config(cfg)
    subject = make_subject(demo, cfg)
    decoder, calibration = calibrate(demo, cfg, subject)
    live = live_run(demo, cfg, subject, decoder)
    return {"demo": dataclasses.asdict(demo), "calibration": calibration, "live": live,
            "control_hz": cfg.task.control_hz, "window_s": decoder.window_s}


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def _f(v: float, nd: int = 3) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _vec(v: Sequence[float], nd: int = 3) -> str:
    return "(" + ", ".join(_f(float(a), nd) for a in v) + ")"


def to_markdown(res: dict) -> str:
    demo, cal, live = res["demo"], res["calibration"], res["live"]
    st, names = live["stats"], live["effectors"]
    units = {"cursor": "task units", "arm": "m"}
    xr, ir = cal["x_report"], cal["idle_report"]
    lines = [
        "# Real-time pure-EEG demo",
        "",
        "`python3 -m neural_cursor.realtime_demo --seconds {:g} --effector {}` (seed {}).  A simulated SMR "
        "user (`sim_body.BodySubject`, y depth {}, x depth {}, x responder {}, dry penalty {}: a top-of-range "
        "responder, so this shows the chain working, not a population estimate) is calibrated "
        "on L1 fixation probes only, then streamed over a real LSL outlet and decoded live by "
        "`realtime.RealTimeLoop` with the deployed `BodyCoordinateDecoder` ({:g}-s windows, {:g} Hz) "
        "driving {}.  Only EEG reaches the decoder; the cue schedule drives the simulated person and "
        "labels this report.".format(demo["seconds"], demo["effector"], demo["seed"], demo["y_depth"],
                                     demo["x_depth"], demo["x_responder"], demo["dry_penalty"],
                                     res["window_s"], res["control_hz"], " and ".join(names)),
        "",
        "## Calibration",
        "",
        f"- {cal['n_windows']} labelled 1-s L1 windows (sources {cal['sources']}; by class "
        + ", ".join(f"{k} {v}" for k, v in cal["n_by_class"].items())
        + f") from {demo['calibration_minutes']:g} min of probes, {cal['n_rest_windows']} rest (L0) windows; "
        f"{cal['seconds']:.1f} s offline.",
        f"- x axis: {'earned' if cal['x_enabled'] else 'not earned'} (first grant test on half the probes: AUC "
        f"{_f(cal['x_first_report']['auc'])}, {'passed' if cal['x_first_report']['passed'] else 'failed'}; on all: "
        f"AUC {_f(xr['auc'])}, circular-shift floor {_f(xr['floor'])}, purity retained "
        f"{_f(xr['purity_retained'], 2)}; {xr.get('reason', '')}).",
        f"- idle gate: {'armed' if ir.get('armed') else 'not armed'} (out-of-fold AUC {_f(ir.get('cv_auc'))}, "
        f"minimum {_f(ir.get('auc_min'), 2)}).",
        "",
        "## Live loop",
        "",
        "| metric | value |",
        "|---|---|",
        f"| latency p50 / p95 / max (ms, newest sample → effector.send returned) | "
        f"{_f(st['latency_ms']['p50'], 1)} / {_f(st['latency_ms']['p95'], 1)} / {_f(st['latency_ms']['max'], 1)} |",
        f"| tick jitter p50 / p95 (ms) | {_f(st['jitter_ms']['p50'], 2)} / {_f(st['jitter_ms']['p95'], 2)} |",
        f"| update rate (Hz) | {live['update_rate_hz']:.2f} ({st['n_updates']} decoded of {st['n_expected']} "
        f"ticks in {st['elapsed_s']:.2f} s) |",
        f"| overruns / stale / invalid / gated ticks | {st['n_overruns']} / {st['n_stale']} / {st['n_invalid']} / "
        f"{st['n_gated']} |",
        f"| samples / dropped / non-finite | {st['n_samples']} / {st['dropped_samples']} / {st['nonfinite_samples']} |",
        f"| aborted | {st['aborted']} |",
        "",
        "## Does the end effector move in the imagined body direction?",
        "",
        f"Scored from {SCORE_LOCKOUT_S:g} s after each cue to the next cue (segments of {demo['segment_s']:g} s). "
        "Body +y = feet = up for both devices (identity mapping), +x = right hand = right.",
        "",
        "| # | imagined | decoded mean (x, y) | "
        + " | ".join(f"{n} displacement ({units[n]})" for n in names) + " | sign agrees |",
        "|---|---|---|" + "---|" * len(names) + "---|",
    ]
    for s in live["segments"]:
        if not s["scored"]:
            lines.append(f"| {s['segment']} | {s['class']} | {_vec(s['decoded_mean'])} | "
                         + " | ".join("—" for _ in names) + f" | not scored ({s['scored_s']:.1f} s) |")
            continue
        agree = ", ".join(f"{n} {'yes' if s['agrees'][n] else 'NO'}" for n in names)
        lines.append(f"| {s['segment']} | {s['class']} ({s['axis']}) | {_vec(s['decoded_mean'])} | "
                     + " | ".join(_vec(s["displacement"][n]) for n in names) + f" | {agree} |")
    lines += ["", "Sign agreement over scored segments:", ""]
    drift = live["summary"].get("cross_axis_drift", {})
    for axis, counts in live["summary"].items():
        if axis != "cross_axis_drift":
            lines.append(f"- {axis}: " + "; ".join(f"{k} {a}/{n}" for k, (a, n) in counts.items()))
    lines += ["", "Cross-axis drift (mean decoded output on the axis NOT imagined, over the scored segments): "
              + "; ".join(f"{'y while imagining one hand' if a == 'x' else 'x while imagining feet/hands'} "
                          f"{_f(v)}" for a, v in drift.items())
              + (".  The earned x refits the y model with one-hand windows as neutral (y = 0) to shrink the y "
                 "commands lateral imagery issues (an arm uses sustained y for grasp and axis switches); the drift "
                 "above is what remains." if cal.get("neutral_one_hand") else ".")]
    if "x" not in live["summary"]:
        lines.append("- x: not tested (x not earned, so the person imagines only feet and hands)")
    else:
        lines += ["", f"x was earned at a 1-s window AUC of {_f(xr['auc'])} (y has no such gate).  A segment's "
                  "mean rests on only a few independent 1-s windows, so at that AUC a segment's x sign is "
                  "far less reliable than y's: lateralised SMR is the weak axis (P_smr_modern_FINAL §1, "
                  "\"x is earned\")."]
    return "\n".join(lines) + "\n"


def write_report(res: dict, path: str = DEFAULT_OUT) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(to_markdown(res))
    return path


def main(argv: Optional[Sequence[str]] = None) -> dict:
    d = DemoConfig()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=d.seconds, help="timed live run (after a 1-s warm-up)")
    ap.add_argument("--effector", choices=EFFECTORS, default=d.effector)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--segment-s", type=float, default=d.segment_s, help="seconds per imagined class")
    ap.add_argument("--dry-penalty", type=float, default=d.dry_penalty)
    ap.add_argument("--calibration-minutes", type=float, default=d.calibration_minutes)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    try:
        demo = DemoConfig(seconds=a.seconds, effector=a.effector, seed=a.seed, segment_s=a.segment_s,
                          dry_penalty=a.dry_penalty, calibration_minutes=a.calibration_minutes)
    except ValueError as exc:
        ap.error(str(exc))
    res = run_demo(demo)
    path = write_report(res, a.out)
    print(to_markdown(res))
    print(f"wrote {path}")
    return res


if __name__ == "__main__":
    main()
