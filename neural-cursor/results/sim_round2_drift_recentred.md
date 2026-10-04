simulated users = 6, trials/user = 24, signal gain = 8.0, eye gain = 0.0, ErrP = 9.0 µV, drift = 0.1/min
velocity decoder r = 0.396; evidence decoder AUC = 0.681 (permutation floor p95 = 0.609)

| condition | n | success | 95% CI | time-to-target (s) | Fitts TP (bits/s) |
|---|---|---|---|---|---|
| oracle | 144 | 1.00 | [1.00, 1.00] | 2.1 | 3.49 |
| decoder+assist | 144 | 0.38 | [0.30, 0.45] | 4.3 | 0.91 |
| assist+evidence | 144 | 0.63 | [0.56, 0.72] | 4.5 | 0.85 |
| assist+shuffled_ev | 144 | 0.73 | [0.66, 0.81] | 4.8 | 0.81 |
| agent+evidence | 144 | 0.11 | [0.07, 0.17] | 8.6 | 0.39 |
| composite | 144 | 0.31 | [0.23, 0.38] | 8.6 | 0.40 |
| composite_shuffled_ev | 144 | 0.39 | [0.31, 0.46] | 8.9 | 0.39 |
| agent_only | 144 | 0.01 | [0.00, 0.02] | 4.9 | nan |
| random | 144 | 0.01 | [0.00, 0.04] | 11.7 | 0.46 |
