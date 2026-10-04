simulated users = 6, trials/user = 24, signal gain = 6.0, eye gain = 0.0, ErrP = 9.0 µV, drift = 0.0/min
velocity decoder r = 0.324; evidence decoder AUC = 0.681 (permutation floor p95 = 0.610)

| condition | n | success | 95% CI | time-to-target (s) | Fitts TP (bits/s) |
|---|---|---|---|---|---|
| oracle | 144 | 1.00 | [1.00, 1.00] | 2.1 | 3.49 |
| decoder+assist | 144 | 0.72 | [0.64, 0.79] | 5.3 | 0.74 |
| assist+evidence | 144 | 0.75 | [0.68, 0.83] | 5.1 | 0.72 |
| assist+shuffled_ev | 144 | 0.69 | [0.62, 0.78] | 5.5 | 0.72 |
| agent+evidence | 144 | 0.12 | [0.07, 0.17] | 6.7 | 0.61 |
| composite | 144 | 0.38 | [0.29, 0.46] | 8.6 | 0.38 |
| composite_shuffled_ev | 144 | 0.33 | [0.26, 0.39] | 8.9 | 0.38 |
| agent_only | 144 | 0.01 | [0.00, 0.02] | 4.9 | nan |
| random | 144 | 0.01 | [0.00, 0.04] | 11.7 | 0.46 |
