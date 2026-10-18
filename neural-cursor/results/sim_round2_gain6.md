simulated users = 6, trials/user = 24, signal gain = 6.0, eye gain = 0.0, ErrP = 9.0 µV, drift = 0.0/min
velocity decoder r = 0.366; evidence decoder AUC = 0.702 (permutation floor p95 = 0.600)

| condition | n | success | 95% CI | time-to-target (s) | Fitts TP (bits/s) |
|---|---|---|---|---|---|
| oracle | 144 | 1.00 | [1.00, 1.00] | 2.1 | 3.49 |
| decoder+assist | 144 | 0.66 | [0.59, 0.74] | 5.4 | 0.71 |
| assist+evidence | 144 | 0.66 | [0.58, 0.74] | 5.2 | 0.75 |
| assist+shuffled_ev | 144 | 0.60 | [0.52, 0.67] | 5.4 | 0.68 |
| agent+evidence | 144 | 0.07 | [0.03, 0.12] | 4.7 | 0.87 |
| composite | 144 | 0.40 | [0.33, 0.48] | 9.4 | 0.35 |
| composite_shuffled_ev | 144 | 0.38 | [0.31, 0.46] | 8.8 | 0.40 |
| agent_only | 144 | 0.01 | [0.00, 0.02] | 4.9 | nan |
| random | 144 | 0.01 | [0.00, 0.04] | 11.7 | 0.46 |
