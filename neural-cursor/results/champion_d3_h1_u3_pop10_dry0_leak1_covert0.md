## Champion protocol (simulated): 3 days × 1 h, 3 deep participant(s), population prior N = 10

seed 20261004; screen leak 1 (targets drawn: S, H, R); covert leak 0 (blind L1/L2b blocks); replay attention 'cursor'; x-responder probability 0.2; user learning rate 0; assistance a = 0.5 (sighted, C2c); prior 3 min of L1 per member; label hop 0.5 s; runtime 252 s.
Deployed decoder: trained on L1 + L2b only (+ prior), re-centred on each session's rest.  Shadow: same model on sighted ReFIT labels (positive control, never deployed).  L1 AUC: fixed fixation-probe test set (2 min per simulated day, never trained on); x AUC is the x model's even when x is not earned.  Sighted: non-replay S trials (success, time-to-target, Fitts throughput); blind: L2b hidden-cursor success.  β: OLS of decoder output on body intent and on the screen direction [95 % trial-cluster bootstrap CI], under held-out mappings (H) and pixel replay (R).  Intent is an oracle (exact cued body vector; instant remapping), so β_body is an upper bound.  'Just intend' blocks are not simulated.

### dry penalty 0

| user | day | cum h | x earned (AUC / floor) | L1 AUC y | L1 AUC x | L1 r y | sighted success | TTT (s) | TP (bits/s) | blind success | windows L1/L2b/S/H/R |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 (prior only) | 0 | – | 0.571 | 0.608 | 0.119 | – | – | – | – | – |
| 0 | 1 | 1 | yes (0.804 / 0.588) | 0.589 | 0.639 | 0.160 | 0.29 | 4.7 | 0.67 | 0.28 | 420/1213/579/230/93 |
| 0 | 2 | 2 | no (0.592 / 0.545) | 0.584 | 0.666 | 0.155 | 0.20 | 5.0 | 0.88 | 0.00 | 420/1292/676/230/63 |
| 0 | 3 | 3 | yes (0.620 / 0.540) | 0.585 | 0.690 | 0.150 | 0.34 | 4.8 | 0.85 | 0.20 | 420/1215/517/257/138 |
| 1 | 0 (prior only) | 0 | – | 0.632 | 0.439 | 0.228 | – | – | – | – | – |
| 1 | 1 | 1 | no (0.593 / 0.597) | 0.718 | 0.476 | 0.375 | 0.15 | 3.8 | 0.92 | 0.22 | 420/1215/719/230/53 |
| 1 | 2 | 2 | no (0.460 / 0.569) | 0.766 | 0.485 | 0.450 | 0.19 | 4.4 | 1.13 | 0.32 | 420/1193/635/230/87 |
| 1 | 3 | 3 | no (0.495 / 0.557) | 0.782 | 0.526 | 0.474 | 0.14 | 5.2 | 0.68 | 0.28 | 420/1200/704/294/103 |
| 2 | 0 (prior only) | 0 | – | 0.805 | 0.636 | 0.527 | – | – | – | – | – |
| 2 | 1 | 1 | no (0.439 / 0.637) | 0.879 | 0.625 | 0.672 | 0.25 | 4.5 | 0.78 | 0.22 | 420/1209/581/230/111 |
| 2 | 2 | 2 | no (0.585 / 0.585) | 0.892 | 0.650 | 0.685 | 0.15 | 4.0 | 0.80 | 0.21 | 420/1215/680/230/91 |
| 2 | 3 | 3 | no (0.555 / 0.562) | 0.900 | 0.689 | 0.693 | 0.25 | 5.4 | 0.77 | 0.22 | 420/1249/654/297/108 |

| user | day | test | axes | trials | deployed β_body | deployed β_screen | shadow β_body | shadow β_screen | verdict |
|---:|---|---|---|---:|---|---|---|---|---|
| 0 | 1 | heldout | x+y | 46 | 0.16 [0.09, 0.24] | -0.04 [-0.11, 0.03] | 0.02 [-0.07, 0.12] | 0.04 [-0.05, 0.12] | **TEST_INSENSITIVE** |
| 0 | 1 | replay | x+y | 11 | 0.21 [0.08, 0.34] | 0.01 [-0.07, 0.14] | 0.20 [0.06, 0.35] | -0.04 [-0.16, 0.09] | **TEST_INSENSITIVE** |
| 0 | 2 | heldout | y | 46 | 0.13 [0.06, 0.22] | 0.09 [-0.02, 0.19] | 0.07 [-0.00, 0.14] | 0.03 [-0.05, 0.11] | **TEST_INSENSITIVE** |
| 0 | 2 | replay | y | 10 | 0.16 [-0.08, 0.32] | 0.09 [-0.12, 0.25] | -0.08 [-0.18, 0.06] | -0.06 [-0.19, 0.05] | **NO_BODY_SIGNAL** |
| 0 | 3 | heldout | x+y | 41 | 0.05 [-0.02, 0.12] | -0.00 [-0.07, 0.06] | 0.04 [-0.02, 0.09] | 0.07 [0.01, 0.13] | **NO_BODY_SIGNAL** |
| 0 | 3 | replay | x+y | 17 | 0.12 [0.02, 0.24] | 0.08 [-0.01, 0.16] | 0.04 [-0.04, 0.13] | 0.04 [-0.01, 0.08] | **TEST_INSENSITIVE** |
| 0 | all | heldout | y | 133 | 0.10 [0.06, 0.15] | 0.07 [0.00, 0.12] | 0.04 [-0.01, 0.09] | 0.04 [-0.02, 0.09] | **SCREEN_READING** |
| 0 | all | replay | y | 38 | 0.13 [0.04, 0.21] | 0.06 [-0.02, 0.13] | 0.07 [-0.00, 0.15] | 0.01 [-0.04, 0.06] | **TEST_INSENSITIVE** |
| 1 | 1 | heldout | y | 46 | 0.23 [0.11, 0.34] | -0.02 [-0.12, 0.07] | 0.12 [0.02, 0.22] | -0.05 [-0.15, 0.05] | **TEST_INSENSITIVE** |
| 1 | 1 | replay | y | 8 | 0.06 [-0.09, 0.28] | 0.09 [-0.02, 0.17] | -0.03 [-0.29, 0.10] | 0.09 [-0.03, 0.23] | **NO_BODY_SIGNAL** |
| 1 | 2 | heldout | y | 46 | 0.32 [0.18, 0.45] | 0.01 [-0.13, 0.13] | 0.03 [-0.09, 0.15] | 0.01 [-0.11, 0.10] | **TEST_INSENSITIVE** |
| 1 | 2 | replay | y | 11 | -0.05 [-0.29, 0.16] | 0.05 [-0.10, 0.21] | -0.02 [-0.16, 0.05] | 0.00 [-0.09, 0.10] | **NO_BODY_SIGNAL** |
| 1 | 3 | heldout | y | 40 | 0.16 [0.04, 0.29] | -0.01 [-0.13, 0.11] | 0.02 [-0.07, 0.11] | 0.06 [-0.01, 0.14] | **TEST_INSENSITIVE** |
| 1 | 3 | replay | y | 13 | 0.22 [0.03, 0.35] | 0.11 [-0.03, 0.25] | 0.12 [-0.02, 0.31] | 0.08 [-0.06, 0.21] | **TEST_INSENSITIVE** |
| 1 | all | heldout | y | 132 | 0.23 [0.17, 0.30] | -0.03 [-0.09, 0.04] | 0.05 [-0.00, 0.11] | 0.01 [-0.05, 0.06] | **TEST_INSENSITIVE** |
| 1 | all | replay | y | 32 | 0.10 [-0.01, 0.22] | 0.09 [0.00, 0.17] | 0.07 [-0.03, 0.15] | 0.04 [-0.02, 0.10] | **SCREEN_READING** |
| 2 | 1 | heldout | y | 46 | 0.42 [0.32, 0.51] | 0.06 [-0.05, 0.19] | 0.05 [-0.04, 0.13] | 0.10 [0.03, 0.18] | **INCONCLUSIVE** |
| 2 | 1 | replay | y | 13 | 0.25 [0.06, 0.47] | 0.05 [-0.05, 0.16] | 0.05 [-0.09, 0.17] | -0.14 [-0.22, -0.01] | **TEST_INSENSITIVE** |
| 2 | 2 | heldout | y | 46 | 0.38 [0.18, 0.58] | 0.05 [-0.11, 0.21] | 0.10 [-0.01, 0.21] | 0.01 [-0.08, 0.11] | **TEST_INSENSITIVE** |
| 2 | 2 | replay | y | 10 | 0.33 [0.12, 0.49] | -0.07 [-0.15, 0.05] | 0.03 [-0.09, 0.13] | -0.09 [-0.17, -0.01] | **TEST_INSENSITIVE** |
| 2 | 3 | heldout | y | 40 | 0.30 [0.22, 0.40] | 0.02 [-0.07, 0.13] | 0.07 [-0.01, 0.15] | 0.02 [-0.04, 0.09] | **TEST_INSENSITIVE** |
| 2 | 3 | replay | y | 14 | 0.40 [0.26, 0.53] | 0.00 [-0.12, 0.12] | 0.10 [0.00, 0.16] | 0.00 [-0.05, 0.06] | **TEST_INSENSITIVE** |
| 2 | all | heldout | y | 132 | 0.37 [0.32, 0.43] | 0.03 [-0.03, 0.10] | 0.07 [0.02, 0.12] | 0.03 [-0.02, 0.07] | **TEST_INSENSITIVE** |
| 2 | all | replay | y | 37 | 0.37 [0.28, 0.46] | 0.01 [-0.05, 0.07] | 0.06 [0.01, 0.11] | -0.08 [-0.13, -0.02] | **TEST_INSENSITIVE** |
| all | all | heldout | y | 397 | 0.23 [0.19, 0.26] | 0.01 [-0.03, 0.05] | 0.05 [0.01, 0.09] | 0.02 [-0.02, 0.06] | **TEST_INSENSITIVE** |
| all | all | replay | y | 107 | 0.24 [0.18, 0.29] | 0.05 [0.01, 0.09] | 0.01 [-0.05, 0.08] | 0.01 [-0.04, 0.06] | **SCREEN_READING** |

