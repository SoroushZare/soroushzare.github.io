## Champion protocol (simulated): 3 days × 1 h, 3 deep participant(s), population prior N = 10

seed 20261004; screen leak 1 (targets drawn: S, H, R); covert leak 0 (blind L1/L2b blocks); replay attention 'cursor'; x-responder probability 0.2; y depth (0.1, 0.4), x depth (0.25, 0.5); user learning rate 0; assistance a = 0.5 (sighted, C2c); prior 3 min of L1 per member; label hop 0.5 s; slot minutes protocol; shadow null calibration on; runtime 2425 s.

x responders: 2 of 3 deep users, 5 of 10 prior members (7 of 13; the population rate is 0.2, so P(≥ 7 of 13) = 0.007): read x-axis and prior results as those of this draw, not of the stated population.

**Every CERTIFIED verdict below is conditional on the covert leak of its condition.**  At covert leak 0 the deployed decoder's training data (L1/L2b, and a prior of simulated L1) contain no screen-correlated non-motor signal by construction; covert attention toward the imagined direction in blind blocks — the leak the label rule cannot exclude (skeptic C3) — is what the covert-leak sweep (`--covert-leak 0 1 2`) tests.  A screen-leak sweep at covert leak 0 measures the SHADOW positive controls' sensitivity only (they learn the leak from their display-identifiable labels; the deployed decoder's training data carry none of it); whether a display leak reaches the DEPLOYED decoder is the 'leak transfer' column: a paired probe (identical noise streams at screen leak 0 and 1, rot±90 imagery on diagonal targets) of the frozen held-out decoder, its output slope on the screen direction ÷ the external margin.

Deployed decoder: trained on L1 + L2b only (+ prior), re-centred on each session's rest; its x model only on the person's own L1 probes (the windows x is earned on), and an x earner's y model treats one-hand windows as neutral.  L2b windows end ≤ 4 s after the cue and before the hidden cursor reaches the dwell radius.  Shadows: the same model on sighted ReFIT labels (positive controls, never deployed) — NOT matched-n twins: S windows only (about a third of the deployed decoder's own window count) and no prior.  L1 AUC: fixed fixation-probe test set (2 min per simulated day, never trained on), with its floor (p95 of a trial-level label permutation) in brackets; x AUC is the x model's even when x is not earned.  x earned: two consecutive passing grant tests on own L1 probes (AUC > 0.60 and > its circular-shift floor, and ≥ 70 % of the margin kept after regressing out posterior alpha lateralisation), then kept unless the floor or the purity rule rejects it.  Sighted: non-replay S trials (success, time-to-target, Fitts throughput); blind: L2b hidden-cursor success.

Certification: per test unit (held-out: one deployed decoder version, i.e. one session, with its own slopes and intercepts; replay: one person), output_a = β_body·body_a + β_screen·screen_a + β_cross·body_other + c_a, CR2 cluster-robust SEs with t(G − k) intervals (trials are the clusters); units of one person combined by inverse variance, people by a person-then-trial bootstrap (t with people − 1 df).  βs are relative to the deployed decoder's EXTERNAL margin: its slope on body intent in the identity trials (0.25 of each held-out block, interleaved, never certified), whose SE is propagated into every threshold test; 'margin L1' is the secondary slope on the fixed L1 pool.  Verdict family (Holm, per axis): β_body > 0, β_body > 50 % of the margin (and the point estimate ≥ 70 %), |β_screen| < 10 % (two one-sided tests), and each shadow's β_screen above its NULL — the same shadow, same units, at screen leak 0 (in brackets).  A β_screen significantly different from 0 in EITHER direction and not bounded inside ±10 % is SCREEN_READING, any unit's own SCREEN_READING vetoes the combined verdict, and a unit whose body and screen regressors correlate |r| > 0.5 in any block is refused (max |r| block).  Trials whose keyed report (C2b, post-replay) does not confirm the intent are dropped.  Held-out blocks are rot+90 and rot−90 in every session with balanced diagonal targets; replays do not consume a goal quadrant and replay diagonal trajectories inside held-out blocks.  Regressors use the cursor at the window start; C2b and C2c are also reported separately.  Intent is an oracle (exact cued body vector; instant remapping), so β_body is an upper bound.  'Just intend' blocks are not simulated.

Idle gate: rest windows (L0: setup rest and the rest between probes, which now carries the post-imagery beta rebound) against the decoder's training windows; armed only if its out-of-fold AUC ≥ 0.70, and then the output is exactly 0 when p(idle) > 0.7.  'rest gated / intent passed' are measured on the fixed pool's rest and probe windows.

### dry penalty 0, covert leak 0

| user | day | cum h | x earned (AUC / floor / purity kept) | L1 AUC y (floor) | L1 AUC x (floor) | L1 r y | margin y ± SE / x (I) | margin L1 y | leak transfer y / x | idle gate (AUC): rest gated / intent passed | sighted success | TTT (s) | TP (bits/s) | blind success | windows L1/L2b/S/H/R/I/L0 |
|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|---:|---:|---:|---:|---|
| 0 | 0 (prior only) | 0 | – | 0.555 (0.621) | 0.517 (0.617) | 0.082 | – | – | – | – | – | – | – | – | – |
| 0 | 1 | 1 | pending (0.703 / 0.596 / 0.91) | 0.559 (0.609) | 0.613 (0.624) | 0.075 | 0.301 ± 0.118 / 0.000 | 0.031 | 0.00 / – | off (0.61): 0.00 / 1.00 | 0.11 | 4.2 | 1.03 | 0.31 | 420/254/644/140/54/44/540 |
| 0 | 2 | 2 | yes (0.684 / 0.568 / 0.90) | 0.540 (0.603) | 0.560 (0.600) | 0.069 | 0.049 ± 0.024 / 0.075 | 0.016 | -0.01 / 0.27 | off (0.60): 0.00 / 1.00 | 0.32 | 5.1 | 0.69 | 0.32 | 420/264/562/140/72/44/540 |
| 0 | 3 | 3 | yes (0.668 / 0.557 / 0.95) | 0.561 (0.612) | 0.580 (0.597) | 0.102 | 0.042 ± 0.068 / -0.027 | 0.024 | 0.11 / – | off (0.61): 0.00 / 1.00 | 0.25 | 5.3 | 0.77 | 0.02 | 420/255/478/228/177/80/540 |
| 1 | 0 (prior only) | 0 | – | 0.632 (0.616) | 0.380 (0.637) | 0.228 | – | – | – | – | – | – | – | – | – |
| 1 | 1 | 1 | pending (0.633 / 0.588 / 1.01) | 0.717 (0.620) | 0.591 (0.577) | 0.368 | 0.120 ± 0.133 / 0.000 | 0.175 | -0.08 / – | off (0.63): 0.00 / 1.00 | 0.15 | 3.7 | 1.25 | 0.32 | 420/263/662/136/46/48/540 |
| 1 | 2 | 2 | no (0.498 / 0.564 / –) | 0.777 (0.629) | 0.542 (0.587) | 0.482 | 0.143 ± 0.093 / 0.000 | 0.218 | 0.02 / – | off (0.63): 0.00 / 1.00 | 0.12 | 5.8 | 0.72 | 0.39 | 420/278/694/136/67/48/540 |
| 1 | 3 | 3 | no (0.503 / 0.560 / 0.97) | 0.807 (0.630) | 0.557 (0.600) | 0.526 | – ± – / – | 0.239 | – / – | off (0.64): 0.00 / 1.00 | 0.11 | 6.9 | 0.91 | 0.37 | 420/276/665/380/80/114/540 |
| 2 | 0 (prior only) | 0 | – | 0.799 (0.629) | 0.513 (0.623) | 0.507 | – | – | – | – | – | – | – | – | – |
| 2 | 1 | 1 | no (0.434 / 0.622 / –) | 0.806 (0.624) | 0.534 (0.609) | 0.536 | 0.287 ± 0.126 / 0.000 | 0.289 | 0.03 / – | off (0.62): 0.00 / 1.00 | 0.21 | 4.2 | 0.98 | 0.27 | 420/246/600/136/94/48/540 |
| 2 | 2 | 2 | no (0.569 / 0.587 / 1.01) | 0.836 (0.637) | 0.528 (0.616) | 0.576 | 0.340 ± 0.151 / 0.000 | 0.322 | -0.01 / – | off (0.62): 0.00 / 1.00 | 0.16 | 3.7 | 0.99 | 0.36 | 420/240/614/136/49/48/540 |
| 2 | 3 | 3 | no (0.508 / 0.564 / 1.03) | 0.829 (0.632) | 0.541 (0.645) | 0.567 | – ± – / – | 0.312 | – / – | off (0.63): 0.00 / 1.00 | 0.25 | 3.6 | 0.89 | 0.24 | 420/222/579/361/67/133/540 |

| user | day | test | axes | units | trials | deployed β_body/margin | deployed β_screen/margin | deployed β_cross | shadow c11 β_screen/margin (null) | shadow all32 β_screen/margin (null) | max \|r\| block | leak transfer y / x | per axis | verdict |
|---:|---|---|---|---:|---:|---|---|---|---|---|---:|---|---|---|
| 0 | 1 | heldout | y | 1 | 35 | y: 0.57 [0.11, 3.10] | 0.08 [-0.55, 0.97] | 0.07 [-0.07, 0.21] | 0.09 [-0.53, 1.03] (null -0.06) | 1.60 [0.68, 9.89] (null -0.83) | 0.06 | 0.00 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | 1 | replay | – | – | 9 | – | – | – | – | – | – | 0.00 / – | – | skipped: u0: 9 trials (< 30 for cluster-robust inference) |
| 0 | 1 | heldout C2b | y | 1 | 35 | y: 0.57 [0.11, 3.10] | 0.08 [-0.55, 0.97] | 0.07 [-0.07, 0.21] | 0.09 [-0.53, 1.03] (null -0.06) | 1.60 [0.68, 9.89] (null -0.83) | 0.06 | 0.00 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | 2 | heldout | x+y | 1 | 35 | y: -0.41 [-324.08, 2.46] | -0.46 [-221.15, 2.16] | -0.06 [-0.13, 0.02] | -0.32 [–, –] (null -0.66) | 12.07 [–, –] (null -3.15) | 0.11 | -0.01 / 0.27 | x: NO_BODY_SIGNAL, y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | 2 | replay | – | – | 11 | – | – | – | – | – | – | -0.01 / 0.27 | – | skipped: u0: 11 trials (< 30 for cluster-robust inference) |
| 0 | 2 | heldout C2b | x+y | 1 | 35 | y: -0.41 [-324.08, 2.46] | -0.46 [-221.15, 2.16] | -0.06 [-0.13, 0.02] | -0.32 [–, –] (null -0.66) | 12.07 [–, –] (null -3.15) | 0.11 | -0.01 / 0.27 | x: NO_BODY_SIGNAL, y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | 3 | heldout | y | 1 | 26 | y: 1.25 [–, –] | -0.35 [–, –] | -0.04 [-0.11, 0.02] | 3.38 [–, –] (no null) | 15.98 [–, –] (no null) | 0.11 | 0.11 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | 3 | replay | – | – | 16 | – | – | – | – | – | – | 0.11 / – | – | skipped: u0: 16 trials (< 30 for cluster-robust inference) |
| 0 | 3 | heldout C2c | y | 1 | 26 | y: 1.25 [–, –] | -0.35 [–, –] | -0.04 [-0.11, 0.02] | 3.38 [–, –] (no null) | 15.98 [–, –] (no null) | 0.11 | 0.11 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | all | heldout | x+y | 3 | 96 | y: 0.44 [-0.12, 1.01] | 0.02 [-0.39, 0.42] | – [–, –] | 0.08 [-0.35, 0.51] (null -0.09) | 1.75 [0.36, 3.13] (null -0.95) | 0.11 | 0.03 / 0.27 | x: NO_BODY_SIGNAL, y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | all | replay | y | 1 | 36 | y: 0.63 [–, –] | -0.60 [–, –] | 0.03 [-0.01, 0.08] | 1.42 [–, –] (null -0.85) | 5.52 [–, –] (null -1.42) | – | 0.03 / 0.27 | y: SCREEN_READING | **SCREEN_READING** |
| 0 | all | heldout C2b | x+y | 2 | 70 | y: 0.43 [-0.14, 1.00] | 0.03 [-0.39, 0.46] | – [–, –] | 0.08 [-0.36, 0.51] (null -0.09) | 1.74 [0.35, 3.12] (null -0.95) | 0.11 | 0.03 / 0.27 | x: NO_BODY_SIGNAL, y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | all | heldout C2c | y | 1 | 26 | y: 1.25 [–, –] | -0.35 [–, –] | -0.04 [-0.11, 0.02] | 3.38 [–, –] (no null) | 15.98 [–, –] (no null) | 0.11 | 0.03 / 0.27 | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 1 | 1 | heldout | y | 1 | 34 | y: 2.26 [–, –] | 0.91 [–, –] | -0.03 [-0.17, 0.12] | 0.17 [–, –] (null -0.41) | 4.65 [–, –] (null 1.18) | 0.06 | -0.08 / – | y: TEST_INSENSITIVE | **TEST_INSENSITIVE** |
| 1 | 1 | replay | – | – | 8 | – | – | – | – | – | – | -0.08 / – | – | skipped: u1: 8 trials (< 30 for cluster-robust inference) |
| 1 | 1 | heldout C2b | y | 1 | 34 | y: 2.26 [–, –] | 0.91 [–, –] | -0.03 [-0.17, 0.12] | 0.17 [–, –] (null -0.41) | 4.65 [–, –] (null 1.18) | 0.06 | -0.08 / – | y: TEST_INSENSITIVE | **TEST_INSENSITIVE** |
| 1 | 2 | heldout | y | 1 | 34 | y: 1.60 [–, –] | 0.17 [–, –] | 0.09 [-0.05, 0.23] | 0.53 [–, –] (null -0.11) | 4.38 [–, –] (null 0.58) | 0.06 | 0.02 / – | y: INCONCLUSIVE | **INCONCLUSIVE** |
| 1 | 2 | replay | – | – | 9 | – | – | – | – | – | – | 0.02 / – | – | skipped: u1: 9 trials (< 30 for cluster-robust inference) |
| 1 | 2 | heldout C2b | y | 1 | 34 | y: 1.60 [–, –] | 0.17 [–, –] | 0.09 [-0.05, 0.23] | 0.53 [–, –] (null -0.11) | 4.38 [–, –] (null 0.58) | 0.06 | 0.02 / – | y: INCONCLUSIVE | **INCONCLUSIVE** |
| 1 | 3 | heldout | – | – | 20 | – | – | – | – | – | – | – / – | – | skipped: u1/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 1 | 3 | replay | – | – | 10 | – | – | – | – | – | – | – / – | – | skipped: u1: 10 trials (< 30 for cluster-robust inference) |
| 1 | 3 | heldout C2c | – | – | 20 | – | – | – | – | – | – | – / – | – | skipped: u1/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 1 | all | heldout | y | 2 | 88 | y: 1.71 [-0.34, 3.75] | 0.27 [-0.60, 1.15] | – [–, –] | 0.38 [-0.47, 1.23] (null -0.15) | 4.45 [-0.55, 9.45] (null 0.67) | 0.06 | -0.03 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 1 | all | replay | – | – | 27 | – | – | – | – | – | – | -0.03 / – | – | skipped: u1: 27 trials (< 30 for cluster-robust inference) |
| 1 | all | heldout C2b | y | 2 | 68 | y: 1.71 [-0.34, 3.75] | 0.27 [-0.60, 1.15] | – [–, –] | 0.38 [-0.47, 1.23] (null -0.15) | 4.45 [-0.55, 9.45] (null 0.67) | 0.06 | -0.03 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 1 | all | heldout C2c | – | – | 20 | – | – | – | – | – | – | -0.03 / – | – | skipped: u1/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 2 | 1 | heldout | y | 1 | 34 | y: 0.74 [0.19, 8.51] | -0.09 [-1.69, 0.80] | 0.04 [-0.11, 0.19] | 0.46 [0.04, 4.76] (null 0.02) | 2.50 [1.21, 80.36] (null 0.14) | 0.06 | 0.03 / – | y: INCONCLUSIVE | **INCONCLUSIVE** |
| 2 | 1 | replay | – | – | 9 | – | – | – | – | – | – | 0.03 / – | – | skipped: u2: 9 trials (< 30 for cluster-robust inference) |
| 2 | 1 | heldout C2b | y | 1 | 34 | y: 0.74 [0.19, 8.51] | -0.09 [-1.69, 0.80] | 0.04 [-0.11, 0.19] | 0.46 [0.04, 4.76] (null 0.02) | 2.50 [1.21, 80.36] (null 0.14) | 0.06 | 0.03 / – | y: INCONCLUSIVE | **INCONCLUSIVE** |
| 2 | 2 | heldout | y | 1 | 34 | y: 0.79 [0.19, 9.56] | -0.00 [-1.30, 1.26] | -0.05 [-0.25, 0.14] | 0.52 [0.14, 6.43] (null 0.08) | 1.63 [0.73, 39.07] (null -0.31) | 0.06 | -0.01 / – | y: INCONCLUSIVE | **INCONCLUSIVE** |
| 2 | 2 | replay | – | – | 10 | – | – | – | – | – | – | -0.01 / – | – | skipped: u2: 10 trials (< 30 for cluster-robust inference) |
| 2 | 2 | heldout C2b | y | 1 | 34 | y: 0.79 [0.19, 9.56] | -0.00 [-1.30, 1.26] | -0.05 [-0.25, 0.14] | 0.52 [0.14, 6.43] (null 0.08) | 1.63 [0.73, 39.07] (null -0.31) | 0.06 | -0.01 / – | y: INCONCLUSIVE | **INCONCLUSIVE** |
| 2 | 3 | heldout | – | – | 19 | – | – | – | – | – | – | – / – | – | skipped: u2/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 2 | 3 | replay | – | – | 10 | – | – | – | – | – | – | – / – | – | skipped: u2: 10 trials (< 30 for cluster-robust inference) |
| 2 | 3 | heldout C2c | – | – | 19 | – | – | – | – | – | – | – / – | – | skipped: u2/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 2 | all | heldout | y | 2 | 87 | y: 0.77 [0.17, 1.36] | -0.05 [-0.41, 0.32] | – [–, –] | 0.49 [0.09, 0.88] (null 0.06) | 1.91 [0.66, 3.15] (null -0.08) | 0.06 | 0.01 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | all | replay | – | – | 29 | – | – | – | – | – | – | 0.01 / – | – | skipped: u2: 29 trials (< 30 for cluster-robust inference) |
| 2 | all | heldout C2b | y | 2 | 68 | y: 0.77 [0.17, 1.36] | -0.05 [-0.41, 0.32] | – [–, –] | 0.49 [0.09, 0.88] (null 0.06) | 1.91 [0.66, 3.15] (null -0.08) | 0.06 | 0.01 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | all | heldout C2c | – | – | 19 | – | – | – | – | – | – | 0.01 / – | – | skipped: u2/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| all | all | heldout | x+y | 7 | 271 | y: 1.03 [-41.30, 43.36] | 0.09 [-14.63, 14.80] | – [–, –] | 0.56 [-24.96, 26.08] (null -0.19) | 5.30 [-123.00, 133.60] (null -0.40) | 0.11 | 0.01 / 0.27 | x: NO_BODY_SIGNAL, y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| all | all | replay | y | 1 | 92 | y: 0.63 [–, –] | -0.60 [–, –] | 0.03 [-0.01, 0.08] | 1.42 [–, –] (null -0.85) | 5.52 [–, –] (null -1.42) | – | 0.01 / 0.27 | y: SCREEN_READING | **SCREEN_READING** |
| all | all | heldout C2b | x+y | 6 | 206 | y: 0.93 [-40.98, 42.83] | 0.10 [-12.46, 12.66] | – [–, –] | 0.24 [-18.79, 19.28] (null -0.19) | 4.47 [-121.85, 130.79] (null -0.40) | 0.11 | 0.01 / 0.27 | x: NO_BODY_SIGNAL, y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| all | all | heldout C2c | y | 1 | 65 | y: 1.25 [–, –] | -0.35 [–, –] | -0.04 [-0.11, 0.02] | 3.38 [–, –] (no null) | 15.98 [–, –] (no null) | 0.11 | 0.01 / 0.27 | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |

Trials dropped because the keyed report did not confirm the intent: held-out 0, replay 4 (the simulated person always remaps).

### dry penalty 0.2, covert leak 0

| user | day | cum h | x earned (AUC / floor / purity kept) | L1 AUC y (floor) | L1 AUC x (floor) | L1 r y | margin y ± SE / x (I) | margin L1 y | leak transfer y / x | idle gate (AUC): rest gated / intent passed | sighted success | TTT (s) | TP (bits/s) | blind success | windows L1/L2b/S/H/R/I/L0 |
|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|---:|---:|---:|---:|---|
| 0 | 0 (prior only) | 0 | – | 0.525 (0.618) | 0.500 (0.617) | 0.021 | – | – | – | – | – | – | – | – | – |
| 0 | 1 | 1 | pending (0.686 / 0.593 / 0.91) | 0.502 (0.609) | 0.601 (0.628) | -0.004 | 0.277 ± 0.090 / 0.000 | -0.001 | 0.03 / – | off (0.58): 0.00 / 1.00 | 0.08 | 5.9 | 1.50 | 0.23 | 420/264/677/140/52/44/540 |
| 0 | 2 | 2 | yes (0.665 / 0.567 / 0.90) | 0.487 (0.608) | 0.544 (0.601) | -0.033 | 0.015 ± 0.024 / 0.060 | -0.006 | 0.08 / 0.32 | off (0.58): 0.00 / 1.00 | 0.31 | 5.2 | 1.03 | 0.22 | 420/270/529/140/88/44/540 |
| 0 | 3 | 3 | yes (0.649 / 0.556 / 0.95) | 0.529 (0.606) | 0.569 (0.602) | 0.029 | 0.004 ± 0.044 / -0.054 | 0.006 | 0.51 / – | off (0.59): 0.00 / 1.00 | 0.19 | 5.2 | 0.72 | 0.02 | 420/253/514/213/156/94/540 |
| 1 | 0 (prior only) | 0 | – | 0.589 (0.613) | 0.381 (0.637) | 0.133 | – | – | – | – | – | – | – | – | – |
| 1 | 1 | 1 | pending (0.633 / 0.588 / 1.01) | 0.633 (0.624) | 0.588 (0.575) | 0.234 | 0.097 ± 0.126 / 0.000 | 0.099 | -0.07 / – | off (0.61): 0.00 / 1.00 | 0.16 | 4.6 | 0.72 | 0.23 | 420/262/625/136/58/48/540 |
| 1 | 2 | 2 | no (0.498 / 0.565 / –) | 0.704 (0.614) | 0.540 (0.585) | 0.360 | 0.188 ± 0.069 / 0.000 | 0.154 | 0.02 / – | off (0.61): 0.00 / 1.00 | 0.11 | 5.4 | 0.46 | 0.25 | 420/268/668/136/113/48/540 |
| 1 | 3 | 3 | no (0.502 / 0.561 / 0.97) | 0.739 (0.618) | 0.555 (0.600) | 0.416 | – ± – / – | 0.178 | – / – | off (0.62): 0.00 / 1.00 | 0.10 | 4.9 | 1.24 | 0.24 | 420/260/598/380/98/114/540 |
| 2 | 0 (prior only) | 0 | – | 0.739 (0.617) | 0.492 (0.623) | 0.423 | – | – | – | – | – | – | – | – | – |
| 2 | 1 | 1 | no (0.420 / 0.621 / –) | 0.753 (0.616) | 0.519 (0.616) | 0.446 | 0.238 ± 0.118 / 0.000 | 0.210 | 0.02 / – | off (0.59): 0.00 / 1.00 | 0.18 | 4.2 | 1.08 | 0.33 | 420/266/665/136/56/48/540 |
| 2 | 2 | 2 | no (0.547 / 0.586 / 1.01) | 0.769 (0.623) | 0.513 (0.616) | 0.468 | 0.219 ± 0.138 / 0.000 | 0.212 | -0.02 / – | off (0.58): 0.00 / 1.00 | 0.15 | 4.1 | 0.99 | 0.25 | 420/251/606/136/75/48/540 |
| 2 | 3 | 3 | no (0.477 / 0.564 / –) | 0.797 (0.633) | 0.510 (0.644) | 0.510 | – ± – / – | 0.228 | – / – | off (0.60): 0.00 / 1.00 | 0.26 | 3.9 | 1.00 | 0.38 | 420/270/556/361/66/133/540 |

| user | day | test | axes | units | trials | deployed β_body/margin | deployed β_screen/margin | deployed β_cross | shadow c11 β_screen/margin (null) | shadow all32 β_screen/margin (null) | max \|r\| block | leak transfer y / x | per axis | verdict |
|---:|---|---|---|---:|---:|---|---|---|---|---|---:|---|---|---|
| 0 | 1 | heldout | y | 1 | 35 | y: 0.35 [-0.09, 1.32] | 0.07 [-0.47, 0.72] | 0.07 [-0.05, 0.20] | 0.46 [-0.13, 1.73] (null -0.12) | 1.86 [0.94, 6.27] (null -0.56) | 0.06 | 0.03 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | 1 | replay | – | – | 8 | – | – | – | – | – | – | 0.03 / – | – | skipped: u0: 8 trials (< 30 for cluster-robust inference) |
| 0 | 1 | heldout C2b | y | 1 | 35 | y: 0.35 [-0.09, 1.32] | 0.07 [-0.47, 0.72] | 0.07 [-0.05, 0.20] | 0.46 [-0.13, 1.73] (null -0.12) | 1.86 [0.94, 6.27] (null -0.56) | 0.06 | 0.03 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | 2 | heldout | – | – | 35 | – | – | – | – | – | – | 0.08 / 0.32 | – | skipped: u0/d2/v2: no external y margin from identity trials (0.015; needs ≥ 0.02 from ≥ 8 trials) |
| 0 | 2 | replay | – | – | 11 | – | – | – | – | – | – | 0.08 / 0.32 | – | skipped: u0: 11 trials (< 30 for cluster-robust inference) |
| 0 | 2 | heldout C2b | – | – | 35 | – | – | – | – | – | – | 0.08 / 0.32 | – | skipped: u0/d2/v2: no external y margin from identity trials (0.015; needs ≥ 0.02 from ≥ 8 trials) |
| 0 | 3 | heldout | – | – | 26 | – | – | – | – | – | – | 0.51 / – | – | skipped: u0/d3/v3: no external y margin from identity trials (0.004; needs ≥ 0.02 from ≥ 8 trials) |
| 0 | 3 | replay | – | – | 16 | – | – | – | – | – | – | 0.51 / – | – | skipped: u0: 16 trials (< 30 for cluster-robust inference) |
| 0 | 3 | heldout C2c | – | – | 26 | – | – | – | – | – | – | 0.51 / – | – | skipped: u0/d3/v3: no external y margin from identity trials (0.004; needs ≥ 0.02 from ≥ 8 trials) |
| 0 | all | heldout | y | 1 | 96 | y: 0.35 [-0.09, 1.32] | 0.07 [-0.47, 0.72] | 0.07 [-0.05, 0.20] | 0.46 [-0.13, 1.73] (null -0.14) | 1.86 [0.94, 6.27] (null -0.58) | 0.06 | 0.20 / 0.32 | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | all | replay | y | 1 | 35 | y: 0.60 [–, –] | -0.26 [–, –] | -0.02 [-0.07, 0.04] | 2.86 [–, –] (null 0.02) | 10.69 [–, –] (null -0.43) | – | 0.20 / 0.32 | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | all | heldout C2b | y | 1 | 70 | y: 0.35 [-0.09, 1.32] | 0.07 [-0.47, 0.72] | 0.07 [-0.05, 0.20] | 0.46 [-0.13, 1.73] (null -0.14) | 1.86 [0.94, 6.27] (null -0.58) | 0.06 | 0.20 / 0.32 | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | all | heldout C2c | – | – | 26 | – | – | – | – | – | – | 0.20 / 0.32 | – | skipped: u0/d3/v3: no external y margin from identity trials (0.004; needs ≥ 0.02 from ≥ 8 trials) |
| 1 | 1 | heldout | y | 1 | 34 | y: 1.82 [–, –] | 1.09 [–, –] | -0.04 [-0.16, 0.07] | -0.68 [–, –] (null 1.04) | 4.43 [–, –] (null 0.04) | 0.06 | -0.07 / – | y: TEST_INSENSITIVE | **TEST_INSENSITIVE** |
| 1 | 1 | replay | – | – | 8 | – | – | – | – | – | – | -0.07 / – | – | skipped: u1: 8 trials (< 30 for cluster-robust inference) |
| 1 | 1 | heldout C2b | y | 1 | 34 | y: 1.82 [–, –] | 1.09 [–, –] | -0.04 [-0.16, 0.07] | -0.68 [–, –] (null 1.04) | 4.43 [–, –] (null 0.04) | 0.06 | -0.07 / – | y: TEST_INSENSITIVE | **TEST_INSENSITIVE** |
| 1 | 2 | heldout | y | 1 | 34 | y: 0.97 [0.30, 4.14] | 0.36 [-0.30, 1.87] | 0.03 [-0.09, 0.15] | 0.72 [0.02, 3.16] (null -0.30) | 3.91 [2.01, 18.99] (null -0.49) | 0.06 | 0.02 / – | y: INCONCLUSIVE | **INCONCLUSIVE** |
| 1 | 2 | replay | – | – | 8 | – | – | – | – | – | – | 0.02 / – | – | skipped: u1: 8 trials (< 30 for cluster-robust inference) |
| 1 | 2 | heldout C2b | y | 1 | 34 | y: 0.97 [0.30, 4.14] | 0.36 [-0.30, 1.87] | 0.03 [-0.09, 0.15] | 0.72 [0.02, 3.16] (null -0.30) | 3.91 [2.01, 18.99] (null -0.49) | 0.06 | 0.02 / – | y: INCONCLUSIVE | **INCONCLUSIVE** |
| 1 | 3 | heldout | – | – | 20 | – | – | – | – | – | – | – / – | – | skipped: u1/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 1 | 3 | replay | – | – | 10 | – | – | – | – | – | – | – / – | – | skipped: u1: 10 trials (< 30 for cluster-robust inference) |
| 1 | 3 | heldout C2c | – | – | 20 | – | – | – | – | – | – | – / – | – | skipped: u1/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 1 | all | heldout | y | 2 | 88 | y: 1.00 [0.10, 1.90] | 0.39 [-0.25, 1.03] | – [–, –] | 0.56 [-0.24, 1.35] (null -0.28) | 3.94 [1.08, 6.81] (null -0.43) | 0.06 | -0.02 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 1 | all | replay | – | – | 26 | – | – | – | – | – | – | -0.02 / – | – | skipped: u1: 26 trials (< 30 for cluster-robust inference) |
| 1 | all | heldout C2b | y | 2 | 68 | y: 1.00 [0.10, 1.90] | 0.39 [-0.25, 1.03] | – [–, –] | 0.56 [-0.24, 1.35] (null -0.28) | 3.94 [1.08, 6.81] (null -0.43) | 0.06 | -0.02 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 1 | all | heldout C2c | – | – | 20 | – | – | – | – | – | – | -0.02 / – | – | skipped: u1/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 2 | 1 | heldout | y | 1 | 34 | y: 0.41 [–, –] | -0.16 [–, –] | 0.03 [-0.08, 0.15] | 0.19 [–, –] (null -0.10) | 2.06 [–, –] (null -1.04) | 0.06 | 0.02 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | 1 | replay | – | – | 8 | – | – | – | – | – | – | 0.02 / – | – | skipped: u2: 8 trials (< 30 for cluster-robust inference) |
| 2 | 1 | heldout C2b | y | 1 | 34 | y: 0.41 [–, –] | -0.16 [–, –] | 0.03 [-0.08, 0.15] | 0.19 [–, –] (null -0.10) | 2.06 [–, –] (null -1.04) | 0.06 | 0.02 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | 2 | heldout | y | 1 | 34 | y: 0.83 [–, –] | 0.11 [–, –] | -0.10 [-0.27, 0.08] | 1.39 [–, –] (null 0.48) | 2.27 [–, –] (null 0.41) | 0.06 | -0.02 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | 2 | replay | – | – | 10 | – | – | – | – | – | – | -0.02 / – | – | skipped: u2: 10 trials (< 30 for cluster-robust inference) |
| 2 | 2 | heldout C2b | y | 1 | 34 | y: 0.83 [–, –] | 0.11 [–, –] | -0.10 [-0.27, 0.08] | 1.39 [–, –] (null 0.48) | 2.27 [–, –] (null 0.41) | 0.06 | -0.02 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | 3 | heldout | – | – | 19 | – | – | – | – | – | – | – / – | – | skipped: u2/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 2 | 3 | replay | – | – | 11 | – | – | – | – | – | – | – / – | – | skipped: u2: 11 trials (< 30 for cluster-robust inference) |
| 2 | 3 | heldout C2c | – | – | 19 | – | – | – | – | – | – | – / – | – | skipped: u2/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 2 | all | heldout | y | 2 | 87 | y: 0.49 [-0.07, 1.05] | -0.08 [-0.49, 0.34] | – [–, –] | 0.34 [-0.29, 0.96] (null 0.20) | 2.13 [0.40, 3.87] (null -0.20) | 0.06 | -0.00 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | all | replay | – | – | 29 | – | – | – | – | – | – | -0.00 / – | – | skipped: u2: 29 trials (< 30 for cluster-robust inference) |
| 2 | all | heldout C2b | y | 2 | 68 | y: 0.49 [-0.07, 1.05] | -0.08 [-0.49, 0.34] | – [–, –] | 0.34 [-0.29, 0.96] (null 0.20) | 2.13 [0.40, 3.87] (null -0.20) | 0.06 | -0.00 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | all | heldout C2c | – | – | 19 | – | – | – | – | – | – | -0.00 / – | – | skipped: u2/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| all | all | heldout | y | 5 | 271 | y: 0.79 [-134.01, 135.59] | 0.26 [-113.03, 113.54] | – [–, –] | 0.42 [-151.68, 152.53] (null -0.06) | 2.73 [-539.45, 544.91] (null -0.59) | 0.06 | 0.08 / 0.32 | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| all | all | replay | y | 1 | 90 | y: 0.60 [–, –] | -0.26 [–, –] | -0.02 [-0.07, 0.04] | 2.86 [–, –] (null 0.02) | 10.69 [–, –] (null -0.43) | – | 0.08 / 0.32 | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| all | all | heldout C2b | y | 5 | 206 | y: 0.79 [-134.01, 135.59] | 0.26 [-113.03, 113.54] | – [–, –] | 0.42 [-151.68, 152.53] (null -0.06) | 2.73 [-539.45, 544.91] (null -0.59) | 0.06 | 0.08 / 0.32 | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| all | all | heldout C2c | – | – | 65 | – | – | – | – | – | – | 0.08 / 0.32 | – | skipped: u0/d3/v3: no external y margin from identity trials (0.004; needs ≥ 0.02 from ≥ 8 trials); u1/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials); u2/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |

Trials dropped because the keyed report did not confirm the intent: held-out 0, replay 5 (the simulated person always remaps).

### dry penalty 0.4, covert leak 0

| user | day | cum h | x earned (AUC / floor / purity kept) | L1 AUC y (floor) | L1 AUC x (floor) | L1 r y | margin y ± SE / x (I) | margin L1 y | leak transfer y / x | idle gate (AUC): rest gated / intent passed | sighted success | TTT (s) | TP (bits/s) | blind success | windows L1/L2b/S/H/R/I/L0 |
|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|---:|---:|---:|---:|---|
| 0 | 0 (prior only) | 0 | – | 0.493 (0.620) | 0.489 (0.616) | -0.040 | – | – | – | – | – | – | – | – | – |
| 0 | 1 | 1 | pending (0.667 / 0.596 / 0.93) | 0.440 (0.616) | 0.591 (0.632) | -0.109 | 0.229 ± 0.094 / 0.000 | -0.036 | 0.04 / – | off (0.55): 0.00 / 1.00 | 0.10 | 4.9 | 0.75 | 0.24 | 420/264/648/140/53/44/540 |
| 0 | 2 | 2 | yes (0.646 / 0.566 / 0.89) | 0.436 (0.610) | 0.529 (0.606) | -0.107 | 0.012 ± 0.025 / 0.047 | -0.019 | 0.45 / 0.40 | off (0.55): 0.00 / 1.00 | 0.13 | 5.3 | 0.68 | 0.19 | 420/267/561/140/107/44/540 |
| 0 | 3 | 3 | yes (0.630 / 0.557 / 0.94) | 0.474 (0.599) | 0.551 (0.608) | -0.033 | 0.021 ± 0.028 / 0.024 | -0.006 | 0.25 / 0.80 | off (0.56): 0.00 / 1.00 | 0.14 | 5.8 | 0.62 | 0.00 | 420/254/491/208/142/116/540 |
| 1 | 0 (prior only) | 0 | – | 0.533 (0.614) | 0.383 (0.633) | 0.029 | – | – | – | – | – | – | – | – | – |
| 1 | 1 | 1 | pending (0.634 / 0.587 / 1.00) | 0.644 (0.619) | 0.585 (0.575) | 0.260 | 0.034 ± 0.101 / 0.000 | 0.104 | -0.10 / – | off (0.58): 0.00 / 1.00 | 0.17 | 3.8 | 1.02 | 0.24 | 420/262/631/136/46/48/540 |
| 1 | 2 | 2 | no (0.498 / 0.564 / –) | 0.696 (0.617) | 0.539 (0.584) | 0.349 | 0.071 ± 0.051 / 0.000 | 0.122 | -0.16 / – | off (0.58): 0.00 / 1.00 | 0.07 | 4.8 | 0.52 | 0.19 | 420/271/652/136/80/48/540 |
| 1 | 3 | 3 | no (0.502 / 0.562 / 0.98) | 0.681 (0.615) | 0.553 (0.600) | 0.325 | – ± – / – | 0.105 | – / – | off (0.59): 0.00 / 1.00 | 0.08 | 5.5 | 0.71 | 0.29 | 420/275/623/380/81/114/540 |
| 2 | 0 (prior only) | 0 | – | 0.686 (0.605) | 0.474 (0.623) | 0.332 | – | – | – | – | – | – | – | – | – |
| 2 | 1 | 1 | no (0.408 / 0.620 / –) | 0.704 (0.607) | 0.507 (0.621) | 0.374 | 0.083 ± 0.104 / 0.000 | 0.163 | 0.10 / – | off (0.57): 0.00 / 1.00 | 0.14 | 4.8 | 0.94 | 0.21 | 420/259/579/136/124/48/540 |
| 2 | 2 | 2 | no (0.529 / 0.585 / 1.01) | 0.708 (0.610) | 0.489 (0.610) | 0.374 | 0.115 ± 0.098 / 0.000 | 0.153 | 0.05 / – | off (0.56): 0.00 / 1.00 | 0.14 | 5.5 | 0.50 | 0.31 | 420/275/646/136/85/48/540 |
| 2 | 3 | 3 | no (0.451 / 0.565 / –) | 0.736 (0.622) | 0.481 (0.631) | 0.415 | – ± – / – | 0.164 | – / – | off (0.57): 0.00 / 1.00 | 0.25 | 4.6 | 0.81 | 0.39 | 420/274/539/361/111/133/540 |

| user | day | test | axes | units | trials | deployed β_body/margin | deployed β_screen/margin | deployed β_cross | shadow c11 β_screen/margin (null) | shadow all32 β_screen/margin (null) | max \|r\| block | leak transfer y / x | per axis | verdict |
|---:|---|---|---|---:|---:|---|---|---|---|---|---:|---|---|---|
| 0 | 1 | heldout | y | 1 | 35 | y: 0.24 [-0.29, 1.87] | 0.24 [-0.31, 1.85] | 0.04 [-0.07, 0.16] | -0.31 [-2.27, 0.23] (null -0.51) | 1.57 [0.58, 12.32] (null -0.41) | 0.06 | 0.04 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | 1 | replay | – | – | 9 | – | – | – | – | – | – | 0.04 / – | – | skipped: u0: 9 trials (< 30 for cluster-robust inference) |
| 0 | 1 | heldout C2b | y | 1 | 35 | y: 0.24 [-0.29, 1.87] | 0.24 [-0.31, 1.85] | 0.04 [-0.07, 0.16] | -0.31 [-2.27, 0.23] (null -0.51) | 1.57 [0.58, 12.32] (null -0.41) | 0.06 | 0.04 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | 2 | heldout | – | – | 35 | – | – | – | – | – | – | 0.45 / 0.40 | – | skipped: u0/d2/v2: no external y margin from identity trials (0.012; needs ≥ 0.02 from ≥ 8 trials) |
| 0 | 2 | replay | – | – | 10 | – | – | – | – | – | – | 0.45 / 0.40 | – | skipped: u0: 10 trials (< 30 for cluster-robust inference) |
| 0 | 2 | heldout C2b | – | – | 35 | – | – | – | – | – | – | 0.45 / 0.40 | – | skipped: u0/d2/v2: no external y margin from identity trials (0.012; needs ≥ 0.02 from ≥ 8 trials) |
| 0 | 3 | heldout | x+y | 1 | 24 | y: 1.88 [–, –] | -0.27 [–, –] | -0.03 [-0.07, 0.02] | 1.20 [–, –] (no null) | 30.68 [–, –] (no null) | 0.17 | 0.25 / 0.80 | x: NO_BODY_SIGNAL, y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | 3 | replay | – | – | 17 | – | – | – | – | – | – | 0.25 / 0.80 | – | skipped: u0: 17 trials (< 30 for cluster-robust inference) |
| 0 | 3 | heldout C2c | x+y | 1 | 24 | y: 1.88 [–, –] | -0.27 [–, –] | -0.03 [-0.07, 0.02] | 1.20 [–, –] (no null) | 30.68 [–, –] (no null) | 0.17 | 0.25 / 0.80 | x: NO_BODY_SIGNAL, y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | all | heldout | x+y | 2 | 94 | y: 0.26 [-0.24, 0.76] | 0.21 [-0.28, 0.71] | – [–, –] | -0.30 [-0.86, 0.25] (null -0.49) | 1.58 [0.09, 3.07] (null -0.42) | 0.17 | 0.25 / 0.60 | x: NO_BODY_SIGNAL, y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | all | replay | y | 1 | 36 | y: 0.88 [-0.06, 6.03] | -0.14 [-2.03, 1.00] | -0.04 [-0.08, 0.00] | 1.52 [-0.31, 10.70] (null 0.54) | 9.12 [4.42, 65.71] (null 1.68) | – | 0.25 / 0.60 | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | all | heldout C2b | y | 1 | 70 | y: 0.24 [-0.29, 1.87] | 0.24 [-0.31, 1.85] | 0.04 [-0.07, 0.16] | -0.31 [-2.27, 0.23] (null -0.49) | 1.57 [0.58, 12.32] (null -0.42) | 0.06 | 0.25 / 0.60 | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | all | heldout C2c | x+y | 1 | 24 | y: 1.88 [–, –] | -0.27 [–, –] | -0.03 [-0.07, 0.02] | 1.20 [–, –] (no null) | 30.68 [–, –] (no null) | 0.17 | 0.25 / 0.60 | x: NO_BODY_SIGNAL, y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 1 | 1 | heldout | y | 1 | 34 | y: 4.66 [–, –] | 2.92 [–, –] | -0.03 [-0.14, 0.08] | -0.37 [–, –] (null -2.79) | 17.46 [–, –] (null 4.65) | 0.06 | -0.10 / – | y: TEST_INSENSITIVE | **TEST_INSENSITIVE** |
| 1 | 1 | replay | – | – | 8 | – | – | – | – | – | – | -0.10 / – | – | skipped: u1: 8 trials (< 30 for cluster-robust inference) |
| 1 | 1 | heldout C2b | y | 1 | 34 | y: 4.66 [–, –] | 2.92 [–, –] | -0.03 [-0.14, 0.08] | -0.37 [–, –] (null -2.79) | 17.46 [–, –] (null 4.65) | 0.06 | -0.10 / – | y: TEST_INSENSITIVE | **TEST_INSENSITIVE** |
| 1 | 2 | heldout | y | 1 | 34 | y: 1.46 [–, –] | 0.57 [–, –] | 0.06 [-0.04, 0.17] | 0.53 [–, –] (null -0.25) | 9.85 [–, –] (null 1.01) | 0.06 | -0.16 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 1 | 2 | replay | – | – | 10 | – | – | – | – | – | – | -0.16 / – | – | skipped: u1: 10 trials (< 30 for cluster-robust inference) |
| 1 | 2 | heldout C2b | y | 1 | 34 | y: 1.46 [–, –] | 0.57 [–, –] | 0.06 [-0.04, 0.17] | 0.53 [–, –] (null -0.25) | 9.85 [–, –] (null 1.01) | 0.06 | -0.16 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 1 | 3 | heldout | – | – | 20 | – | – | – | – | – | – | – / – | – | skipped: u1/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 1 | 3 | replay | – | – | 10 | – | – | – | – | – | – | – / – | – | skipped: u1: 10 trials (< 30 for cluster-robust inference) |
| 1 | 3 | heldout C2c | – | – | 20 | – | – | – | – | – | – | – / – | – | skipped: u1/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 1 | all | heldout | y | 2 | 88 | y: 1.49 [-0.99, 3.97] | 0.59 [-1.01, 2.19] | – [–, –] | 0.36 [-1.48, 2.20] (null -0.26) | 9.99 [-3.93, 23.92] (null 1.05) | 0.06 | -0.13 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 1 | all | replay | – | – | 28 | – | – | – | – | – | – | -0.13 / – | – | skipped: u1: 28 trials (< 30 for cluster-robust inference) |
| 1 | all | heldout C2b | y | 2 | 68 | y: 1.49 [-0.99, 3.97] | 0.59 [-1.01, 2.19] | – [–, –] | 0.36 [-1.48, 2.20] (null -0.26) | 9.99 [-3.93, 23.92] (null 1.05) | 0.06 | -0.13 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 1 | all | heldout C2c | – | – | 20 | – | – | – | – | – | – | -0.13 / – | – | skipped: u1/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 2 | 1 | heldout | y | 1 | 34 | y: 0.46 [–, –] | -0.16 [–, –] | 0.01 [-0.10, 0.13] | 0.99 [–, –] (null -0.38) | 7.11 [–, –] (null -1.24) | 0.06 | 0.10 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | 1 | replay | – | – | 10 | – | – | – | – | – | – | 0.10 / – | – | skipped: u2: 10 trials (< 30 for cluster-robust inference) |
| 2 | 1 | heldout C2b | y | 1 | 34 | y: 0.46 [–, –] | -0.16 [–, –] | 0.01 [-0.10, 0.13] | 0.99 [–, –] (null -0.38) | 7.11 [–, –] (null -1.24) | 0.06 | 0.10 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | 2 | heldout | y | 1 | 34 | y: 0.94 [–, –] | 0.76 [–, –] | -0.06 [-0.21, 0.09] | 1.69 [–, –] (null 0.13) | 6.14 [–, –] (null 1.49) | 0.06 | 0.05 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | 2 | replay | – | – | 10 | – | – | – | – | – | – | 0.05 / – | – | skipped: u2: 10 trials (< 30 for cluster-robust inference) |
| 2 | 2 | heldout C2b | y | 1 | 34 | y: 0.94 [–, –] | 0.76 [–, –] | -0.06 [-0.21, 0.09] | 1.69 [–, –] (null 0.13) | 6.14 [–, –] (null 1.49) | 0.06 | 0.05 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | 3 | heldout | – | – | 19 | – | – | – | – | – | – | – / – | – | skipped: u2/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 2 | 3 | replay | – | – | 10 | – | – | – | – | – | – | – / – | – | skipped: u2: 10 trials (< 30 for cluster-robust inference) |
| 2 | 3 | heldout C2c | – | – | 19 | – | – | – | – | – | – | – / – | – | skipped: u2/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 2 | all | heldout | y | 2 | 87 | y: 0.67 [-0.66, 2.00] | 0.19 [-0.91, 1.29] | – [–, –] | 1.34 [-0.77, 3.44] (null 0.04) | 6.39 [-2.56, 15.35] (null 0.71) | 0.06 | 0.08 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | all | replay | y | 1 | 30 | y: 1.48 [–, –] | -0.21 [–, –] | 0.04 [-0.06, 0.13] | 1.95 [–, –] (null 1.16) | 6.83 [–, –] (null -0.13) | – | 0.08 / – | y: TEST_INSENSITIVE | **TEST_INSENSITIVE** |
| 2 | all | heldout C2b | y | 2 | 68 | y: 0.67 [-0.66, 2.00] | 0.19 [-0.91, 1.29] | – [–, –] | 1.34 [-0.77, 3.44] (null 0.04) | 6.39 [-2.56, 15.35] (null 0.71) | 0.06 | 0.08 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | all | heldout C2c | – | – | 19 | – | – | – | – | – | – | 0.08 / – | – | skipped: u2/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| all | all | heldout | x+y | 6 | 269 | y: 1.56 [-38.21, 41.32] | 0.69 [-29.98, 31.36] | – [–, –] | 0.58 [-29.44, 30.59] (null -0.57) | 11.23 [-203.09, 225.56] (null 0.65) | 0.17 | 0.09 / 0.60 | x: NO_BODY_SIGNAL, y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| all | all | replay | y | 2 | 94 | y: 0.88 [-0.06, 6.03] | -0.14 [-2.03, 1.00] | – [–, –] | 1.52 [-0.31, 10.70] (null 0.85) | 9.12 [4.42, 65.71] (null 0.77) | – | 0.09 / 0.60 | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| all | all | heldout C2b | y | 5 | 206 | y: 1.34 [-39.61, 42.28] | 0.76 [-26.25, 27.77] | – [–, –] | 0.37 [-15.46, 16.20] (null -0.57) | 7.29 [-169.79, 184.36] (null 0.65) | 0.06 | 0.09 / 0.60 | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| all | all | heldout C2c | x+y | 1 | 63 | y: 1.88 [–, –] | -0.27 [–, –] | -0.03 [-0.07, 0.02] | 1.20 [–, –] (no null) | 30.68 [–, –] (no null) | 0.17 | 0.09 / 0.60 | x: NO_BODY_SIGNAL, y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |

Trials dropped because the keyed report did not confirm the intent: held-out 0, replay 3 (the simulated person always remaps).

