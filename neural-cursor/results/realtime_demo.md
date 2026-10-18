# Real-time pure-EEG demo

`python3 -m neural_cursor.realtime_demo --seconds 30 --effector both` (seed 2).  A simulated SMR user (`sim_body.BodySubject`, y depth 0.4, x depth 0.5, x responder True, dry penalty 0.0: a top-of-range responder, so this shows the chain working, not a population estimate) is calibrated on L1 fixation probes only, then streamed over a real LSL outlet and decoded live by `realtime.RealTimeLoop` with the deployed `BodyCoordinateDecoder` (1-s windows, 16 Hz) driving cursor and arm.  Only EEG reaches the decoder; the cue schedule drives the simulated person and labels this report.

## Calibration

- 960 labelled 1-s L1 windows (sources ['L1']; by class FEET 240, HANDS 240, RIGHT 240, LEFT 240) from 24 min of probes, 1020 rest (L0) windows; 20.7 s offline.
- x axis: earned (first grant test on half the probes: AUC 0.641, passed; on all: AUC 0.685, circular-shift floor 0.585, purity retained 1.00; x earned (second consecutive grant pass)).
- idle gate: not armed (out-of-fold AUC 0.657, minimum 0.70).

## Live loop

| metric | value |
|---|---|
| latency p50 / p95 / max (ms, newest sample → effector.send returned) | 13.0 / 18.3 / 26.4 |
| tick jitter p50 / p95 (ms) | 0.16 / 0.24 |
| update rate (Hz) | 16.03 (480 decoded of 480 ticks in 29.95 s) |
| overruns / stale / invalid / gated ticks | 0 / 0 / 0 / 0 |
| samples / dropped / non-finite | 15470 / 0 / 0 |
| aborted | None |

## Does the end effector move in the imagined body direction?

Scored from 1.5 s after each cue to the next cue (segments of 5 s). Body +y = feet = up for both devices (identity mapping), +x = right hand = right.

| # | imagined | decoded mean (x, y) | cursor displacement (task units) | arm displacement (m) | sign agrees |
|---|---|---|---|---|---|
| 0 | FEET (y) | (-0.272, 0.326) | (-0.079, 0.169) | (-0.037, 0.043) | cursor yes, arm yes |
| 1 | HANDS (y) | (0.457, -0.217) | (0.133, -0.113) | (0.062, -0.028) | cursor yes, arm yes |
| 2 | RIGHT (x) | (-0.147, 0.109) | (-0.043, 0.057) | (-0.020, 0.014) | cursor NO, arm NO |
| 3 | LEFT (x) | (0.100, -0.297) | (0.029, -0.154) | (0.013, -0.039) | cursor NO, arm NO |
| 4 | FEET (y) | (-0.057, 0.217) | (-0.017, 0.113) | (-0.008, 0.028) | cursor yes, arm yes |
| 5 | HANDS (y) | (-0.174, -0.452) | (-0.051, -0.234) | (-0.024, -0.059) | cursor yes, arm yes |

Sign agreement over scored segments:

- y: cursor 4/4; arm 4/4; decoded 4/4
- x: cursor 0/2; arm 0/2; decoded 0/2

Cross-axis drift (mean decoded output on the axis NOT imagined, over the scored segments): x while imagining feet/hands -0.011; y while imagining one hand -0.094.  The earned x refits the y model with one-hand windows as neutral (y = 0) to shrink the y commands lateral imagery issues (an arm uses sustained y for grasp and axis switches); the drift above is what remains.

x was earned at a 1-s window AUC of 0.685 (y has no such gate).  A segment's mean rests on only a few independent 1-s windows, so at that AUC a segment's x sign is far less reliable than y's: lateralised SMR is the weak axis (P_smr_modern_FINAL §1, "x is earned").
