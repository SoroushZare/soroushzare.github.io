## Champion protocol (simulated): 3 days × 1 h, 3 deep participant(s), population prior N = 10

seed 7; screen leak 1 (targets drawn: S, H, R); covert leak 0 (blind L1/L2b blocks); replay attention 'cursor'; x-responder probability 0.2; y depth (0.1, 0.4), x depth (0.25, 0.5); user learning rate 0; assistance a = 0.5 (sighted, C2c); prior 3 min of L1 per member; label hop 0.5 s; slot minutes protocol; shadow null calibration on; runtime 800 s.

x responders: 1 of 3 deep users, 1 of 10 prior members (2 of 13; the population rate is 0.2, so P(≥ 2 of 13) = 0.766): read x-axis and prior results as those of this draw, not of the stated population.

**Every CERTIFIED verdict below is conditional on the covert leak of its condition.**  At covert leak 0 the deployed decoder's training data (L1/L2b, and a prior of simulated L1) contain no screen-correlated non-motor signal by construction; covert attention toward the imagined direction in blind blocks — the leak the label rule cannot exclude (skeptic C3) — is what the covert-leak sweep (`--covert-leak 0 1 2`) tests.  A screen-leak sweep at covert leak 0 measures the SHADOW positive controls' sensitivity only (they learn the leak from their display-identifiable labels; the deployed decoder's training data carry none of it); whether a display leak reaches the DEPLOYED decoder is the 'leak transfer' column: a paired probe (identical noise streams at screen leak 0 and 1, rot±90 imagery on diagonal targets) of the frozen held-out decoder, its output slope on the screen direction ÷ the external margin.

Deployed decoder: trained on L1 + L2b only (+ prior), re-centred on each session's rest; its x model only on the person's own L1 probes (the windows x is earned on), and an x earner's y model treats one-hand windows as neutral.  L2b windows end ≤ 4 s after the cue and before the hidden cursor reaches the dwell radius.  Shadows: the same model on sighted ReFIT labels (positive controls, never deployed) — NOT matched-n twins: S windows only (about a third of the deployed decoder's own window count) and no prior.  L1 AUC: fixed fixation-probe test set (2 min per simulated day, never trained on), with its floor (p95 of a trial-level label permutation) in brackets; x AUC is the x model's even when x is not earned.  x earned: two consecutive passing grant tests on own L1 probes (AUC > 0.60 and > its circular-shift floor, and ≥ 70 % of the margin kept after regressing out posterior alpha lateralisation), then kept unless the floor or the purity rule rejects it.  Sighted: non-replay S trials (success, time-to-target, Fitts throughput); blind: L2b hidden-cursor success.

Certification: per test unit (held-out: one deployed decoder version, i.e. one session, with its own slopes and intercepts; replay: one person), output_a = β_body·body_a + β_screen·screen_a + β_cross·body_other + c_a, CR2 cluster-robust SEs with t(G − k) intervals (trials are the clusters); units of one person combined by inverse variance, people by a person-then-trial bootstrap (t with people − 1 df).  βs are relative to the deployed decoder's EXTERNAL margin: its slope on body intent in the identity trials (0.25 of each held-out block, interleaved, never certified), whose SE is propagated into every threshold test; 'margin L1' is the secondary slope on the fixed L1 pool.  Verdict family (Holm, per axis): β_body > 0, β_body > 50 % of the margin (and the point estimate ≥ 70 %), |β_screen| < 10 % (two one-sided tests), and each shadow's β_screen above its NULL — the same shadow, same units, at screen leak 0 (in brackets).  A β_screen significantly different from 0 in EITHER direction and not bounded inside ±10 % is SCREEN_READING, any unit's own SCREEN_READING vetoes the combined verdict, and a unit whose body and screen regressors correlate |r| > 0.5 in any block is refused (max |r| block).  Trials whose keyed report (C2b, post-replay) does not confirm the intent are dropped.  Held-out blocks are rot+90 and rot−90 in every session with balanced diagonal targets; replays do not consume a goal quadrant and replay diagonal trajectories inside held-out blocks.  Regressors use the cursor at the window start; C2b and C2c are also reported separately.  Intent is an oracle (exact cued body vector; instant remapping), so β_body is an upper bound.  'Just intend' blocks are not simulated.

Idle gate: rest windows (L0: setup rest and the rest between probes, which now carries the post-imagery beta rebound) against the decoder's training windows; armed only if its out-of-fold AUC ≥ 0.70, and then the output is exactly 0 when p(idle) > 0.7.  'rest gated / intent passed' are measured on the fixed pool's rest and probe windows.

### dry penalty 0, covert leak 0

| user | day | cum h | x earned (AUC / floor / purity kept) | L1 AUC y (floor) | L1 AUC x (floor) | L1 r y | margin y ± SE / x (I) | margin L1 y | leak transfer y / x | idle gate (AUC): rest gated / intent passed | sighted success | TTT (s) | TP (bits/s) | blind success | windows L1/L2b/S/H/R/I/L0 |
|---:|---:|---:|---|---:|---:|---:|---|---:|---|---|---:|---:|---:|---:|---|
| 0 | 0 (prior only) | 0 | – | 0.660 (0.623) | 0.479 (0.632) | 0.290 | – | – | – | – | – | – | – | – | – |
| 0 | 1 | 1 | pending (0.715 / 0.603 / 0.99) | 0.717 (0.625) | 0.484 (0.602) | 0.397 | 0.410 ± 0.104 / 0.000 | 0.173 | 0.04 / – | off (0.64): 0.00 / 1.00 | 0.17 | 5.1 | 0.84 | 0.30 | 420/266/615/144/42/40/540 |
| 0 | 2 | 2 | no (0.599 / 0.572 / 0.98) | 0.777 (0.635) | 0.542 (0.602) | 0.490 | 0.149 ± 0.092 / 0.000 | 0.199 | 0.12 / – | off (0.65): 0.00 / 1.00 | 0.12 | 5.1 | 1.09 | 0.31 | 420/268/641/136/96/48/540 |
| 0 | 3 | 3 | pending (0.616 / 0.562 / 1.01) | 0.791 (0.634) | 0.556 (0.615) | 0.510 | – ± – / – | 0.206 | – / – | off (0.64): 0.00 / 1.00 | 0.20 | 3.5 | 1.05 | 0.27 | 420/273/589/380/84/114/540 |
| 1 | 0 (prior only) | 0 | – | 0.658 (0.594) | 0.570 (0.608) | 0.266 | – | – | – | – | – | – | – | – | – |
| 1 | 1 | 1 | no (0.412 / 0.600 / –) | 0.692 (0.598) | 0.461 (0.634) | 0.332 | 0.145 ± 0.089 / 0.000 | 0.162 | 0.23 / – | off (0.61): 0.00 / 1.00 | 0.19 | 6.2 | 0.60 | 0.25 | 420/243/590/136/120/48/540 |
| 1 | 2 | 2 | no (0.498 / 0.580 / –) | 0.711 (0.598) | 0.435 (0.617) | 0.371 | 0.369 ± 0.095 / 0.000 | 0.181 | 0.04 / – | off (0.62): 0.00 / 1.00 | 0.22 | 3.6 | 1.09 | 0.32 | 420/227/627/140/96/44/540 |
| 1 | 3 | 3 | no (0.532 / 0.567 / 0.99) | 0.751 (0.609) | 0.443 (0.628) | 0.437 | – ± – / – | 0.211 | – / – | off (0.62): 0.00 / 1.00 | 0.27 | 4.2 | 0.83 | 0.27 | 420/248/600/380/79/114/540 |
| 2 | 0 (prior only) | 0 | – | 0.591 (0.639) | 0.405 (0.603) | 0.153 | – | – | – | – | – | – | – | – | – |
| 2 | 1 | 1 | no (0.481 / 0.596 / –) | 0.568 (0.632) | 0.367 (0.633) | 0.121 | 0.043 ± 0.115 / 0.000 | 0.050 | 0.37 / – | off (0.61): 0.00 / 1.00 | 0.10 | 5.1 | 0.73 | 0.23 | 420/265/626/136/88/48/540 |
| 2 | 2 | 2 | no (0.550 / 0.576 / 1.00) | 0.584 (0.619) | 0.359 (0.603) | 0.169 | 0.168 ± 0.089 / 0.000 | 0.060 | 0.11 / – | off (0.60): 0.00 / 1.00 | 0.11 | 5.4 | 0.59 | 0.27 | 420/277/645/140/86/44/540 |
| 2 | 3 | 3 | no (0.574 / 0.567 / 1.00) | 0.600 (0.626) | 0.325 (0.612) | 0.189 | – ± – / – | 0.061 | – / – | off (0.62): 0.00 / 1.00 | 0.11 | 6.5 | 0.93 | 0.27 | 420/278/660/380/70/114/540 |

| user | day | test | axes | units | trials | deployed β_body/margin | deployed β_screen/margin | deployed β_cross | shadow c11 β_screen/margin (null) | shadow all32 β_screen/margin (null) | max \|r\| block | leak transfer y / x | per axis | verdict |
|---:|---|---|---|---:|---:|---|---|---|---|---|---:|---|---|---|
| 0 | 1 | heldout | y | 1 | 36 | y: 0.21 [-0.16, 0.72] | 0.02 [-0.39, 0.45] | -0.09 [-0.23, 0.06] | -0.03 [-0.49, 0.40] (null -0.03) | 1.59 [0.93, 3.65] (null 0.05) | 0.11 | 0.04 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | 1 | replay | – | – | 6 | – | – | – | – | – | – | 0.04 / – | – | skipped: u0: 6 trials (< 30 for cluster-robust inference) |
| 0 | 1 | heldout C2b | y | 1 | 36 | y: 0.21 [-0.16, 0.72] | 0.02 [-0.39, 0.45] | -0.09 [-0.23, 0.06] | -0.03 [-0.49, 0.40] (null -0.03) | 1.59 [0.93, 3.65] (null 0.05) | 0.11 | 0.04 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | 2 | heldout | y | 1 | 34 | y: 1.15 [–, –] | -0.34 [–, –] | -0.02 [-0.15, 0.11] | 0.36 [–, –] (null -0.28) | 4.51 [–, –] (null 0.02) | 0.06 | 0.12 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | 2 | replay | – | – | 9 | – | – | – | – | – | – | 0.12 / – | – | skipped: u0: 9 trials (< 30 for cluster-robust inference) |
| 0 | 2 | heldout C2b | y | 1 | 34 | y: 1.15 [–, –] | -0.34 [–, –] | -0.02 [-0.15, 0.11] | 0.36 [–, –] (null -0.28) | 4.51 [–, –] (null 0.02) | 0.06 | 0.12 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | 3 | heldout | – | – | 20 | – | – | – | – | – | – | – / – | – | skipped: u0/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 0 | 3 | replay | – | – | 11 | – | – | – | – | – | – | – / – | – | skipped: u0: 11 trials (< 30 for cluster-robust inference) |
| 0 | 3 | heldout C2c | – | – | 20 | – | – | – | – | – | – | – / – | – | skipped: u0/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 0 | all | heldout | y | 2 | 90 | y: 0.25 [-0.10, 0.60] | -0.02 [-0.34, 0.31] | – [–, –] | 0.02 [-0.32, 0.36] (null -0.05) | 1.66 [0.80, 2.52] (null 0.04) | 0.11 | 0.08 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | all | replay | – | – | 26 | – | – | – | – | – | – | 0.08 / – | – | skipped: u0: 26 trials (< 30 for cluster-robust inference) |
| 0 | all | heldout C2b | y | 2 | 70 | y: 0.25 [-0.10, 0.60] | -0.02 [-0.34, 0.31] | – [–, –] | 0.02 [-0.32, 0.36] (null -0.05) | 1.66 [0.80, 2.52] (null 0.04) | 0.11 | 0.08 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 0 | all | heldout C2c | – | – | 20 | – | – | – | – | – | – | 0.08 / – | – | skipped: u0/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 1 | 1 | heldout | y | 1 | 34 | y: 2.32 [–, –] | -0.41 [–, –] | -0.00 [-0.12, 0.12] | 0.26 [–, –] (null -0.48) | 2.58 [–, –] (null 0.43) | 0.06 | 0.23 / – | y: TEST_INSENSITIVE | **TEST_INSENSITIVE** |
| 1 | 1 | replay | – | – | 9 | – | – | – | – | – | – | 0.23 / – | – | skipped: u1: 9 trials (< 30 for cluster-robust inference) |
| 1 | 1 | heldout C2b | y | 1 | 34 | y: 2.32 [–, –] | -0.41 [–, –] | -0.00 [-0.12, 0.12] | 0.26 [–, –] (null -0.48) | 2.58 [–, –] (null 0.43) | 0.06 | 0.23 / – | y: TEST_INSENSITIVE | **TEST_INSENSITIVE** |
| 1 | 2 | heldout | y | 1 | 35 | y: 0.83 [0.41, 1.90] | -0.02 [-0.44, 0.39] | 0.08 [-0.05, 0.21] | 0.17 [-0.17, 0.65] (null -0.06) | 1.63 [0.93, 3.75] (null -0.07) | 0.11 | 0.04 / – | y: INCONCLUSIVE | **INCONCLUSIVE** |
| 1 | 2 | replay | – | – | 8 | – | – | – | – | – | – | 0.04 / – | – | skipped: u1: 8 trials (< 30 for cluster-robust inference) |
| 1 | 2 | heldout C2b | y | 1 | 35 | y: 0.83 [0.41, 1.90] | -0.02 [-0.44, 0.39] | 0.08 [-0.05, 0.21] | 0.17 [-0.17, 0.65] (null -0.06) | 1.63 [0.93, 3.75] (null -0.07) | 0.11 | 0.04 / – | y: INCONCLUSIVE | **INCONCLUSIVE** |
| 1 | 3 | heldout | – | – | 20 | – | – | – | – | – | – | – / – | – | skipped: u1/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 1 | 3 | replay | – | – | 11 | – | – | – | – | – | – | – / – | – | skipped: u1: 11 trials (< 30 for cluster-robust inference) |
| 1 | 3 | heldout C2c | – | – | 20 | – | – | – | – | – | – | – / – | – | skipped: u1/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 1 | all | heldout | y | 2 | 89 | y: 0.88 [0.35, 1.41] | -0.06 [-0.38, 0.26] | – [–, –] | 0.18 [-0.13, 0.50] (null -0.07) | 1.70 [0.80, 2.60] (null -0.01) | 0.11 | 0.13 / – | y: INCONCLUSIVE | **INCONCLUSIVE** |
| 1 | all | replay | – | – | 28 | – | – | – | – | – | – | 0.13 / – | – | skipped: u1: 28 trials (< 30 for cluster-robust inference) |
| 1 | all | heldout C2b | y | 2 | 69 | y: 0.88 [0.35, 1.41] | -0.06 [-0.38, 0.26] | – [–, –] | 0.18 [-0.13, 0.50] (null -0.07) | 1.70 [0.80, 2.60] (null -0.01) | 0.11 | 0.13 / – | y: INCONCLUSIVE | **INCONCLUSIVE** |
| 1 | all | heldout C2c | – | – | 20 | – | – | – | – | – | – | 0.13 / – | – | skipped: u1/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 2 | 1 | heldout | y | 1 | 34 | y: 1.13 [–, –] | -1.80 [–, –] | 0.01 [-0.13, 0.16] | 2.73 [–, –] (no null) | 15.30 [–, –] (no null) | 0.06 | 0.37 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | 1 | replay | – | – | 10 | – | – | – | – | – | – | 0.37 / – | – | skipped: u2: 10 trials (< 30 for cluster-robust inference) |
| 2 | 1 | heldout C2b | y | 1 | 34 | y: 1.13 [–, –] | -1.80 [–, –] | 0.01 [-0.13, 0.16] | 2.73 [–, –] (no null) | 15.30 [–, –] (no null) | 0.06 | 0.37 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | 2 | heldout | y | 1 | 35 | y: 0.07 [–, –] | -0.26 [–, –] | -0.02 [-0.11, 0.07] | 0.54 [–, –] (null 0.33) | 3.42 [–, –] (null -0.01) | 0.06 | 0.11 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | 2 | replay | – | – | 9 | – | – | – | – | – | – | 0.11 / – | – | skipped: u2: 9 trials (< 30 for cluster-robust inference) |
| 2 | 2 | heldout C2b | y | 1 | 35 | y: 0.07 [–, –] | -0.26 [–, –] | -0.02 [-0.11, 0.07] | 0.54 [–, –] (null 0.33) | 3.42 [–, –] (null -0.01) | 0.06 | 0.11 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | 3 | heldout | – | – | 20 | – | – | – | – | – | – | – / – | – | skipped: u2/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 2 | 3 | replay | – | – | 8 | – | – | – | – | – | – | – / – | – | skipped: u2: 8 trials (< 30 for cluster-robust inference) |
| 2 | 3 | heldout C2c | – | – | 20 | – | – | – | – | – | – | – / – | – | skipped: u2/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| 2 | all | heldout | y | 2 | 89 | y: 0.07 [-0.46, 0.61] | -0.27 [-0.87, 0.33] | – [–, –] | 0.55 [-0.42, 1.51] (null 0.33) | 3.45 [-0.38, 7.27] (null -0.01) | 0.06 | 0.24 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | all | replay | – | – | 27 | – | – | – | – | – | – | 0.24 / – | – | skipped: u2: 27 trials (< 30 for cluster-robust inference) |
| 2 | all | heldout C2b | y | 2 | 69 | y: 0.07 [-0.46, 0.61] | -0.27 [-0.87, 0.33] | – [–, –] | 0.55 [-0.42, 1.51] (null 0.33) | 3.45 [-0.38, 7.27] (null -0.01) | 0.06 | 0.24 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| 2 | all | heldout C2c | – | – | 20 | – | – | – | – | – | – | 0.24 / – | – | skipped: u2/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |
| all | all | heldout | y | 6 | 268 | y: 0.94 [-265.05, 266.92] | -0.46 [-168.52, 167.60] | – [–, –] | 0.66 [-154.70, 156.03] (null -0.03) | 4.79 [-712.97, 722.56] (null 0.07) | 0.11 | 0.15 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| all | all | replay | – | – | 81 | – | – | – | – | – | – | 0.15 / – | – | skipped: u0: 26 trials (< 30 for cluster-robust inference); u1: 28 trials (< 30 for cluster-robust inference); u2: 27 trials (< 30 for cluster-robust inference) |
| all | all | heldout C2b | y | 6 | 208 | y: 0.94 [-265.05, 266.92] | -0.46 [-168.52, 167.60] | – [–, –] | 0.66 [-154.70, 156.03] (null -0.03) | 4.79 [-712.97, 722.56] (null 0.07) | 0.11 | 0.15 / – | y: NO_BODY_SIGNAL | **NO_BODY_SIGNAL** |
| all | all | heldout C2c | – | – | 60 | – | – | – | – | – | – | 0.15 / – | – | skipped: u0/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials); u1/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials); u2/d3/v3: no external y margin from identity trials (nan; needs ≥ 0.02 from ≥ 8 trials) |

Trials dropped because the keyed report did not confirm the intent: held-out 0, replay 4 (the simulated person always remaps).

