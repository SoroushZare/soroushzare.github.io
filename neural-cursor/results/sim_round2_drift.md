simulated users = 6, trials/user = 24, signal gain = 8.0, eye gain = 0.0, ErrP = 9.0 µV, drift = 0.1/min
velocity decoder r = 0.396; evidence decoder AUC = 0.681 (permutation floor p95 = 0.609)

| condition | n | success | 95% CI | time-to-target (s) | Fitts TP (bits/s) |
|---|---|---|---|---|---|
| oracle | 144 | 1.00 | [1.00, 1.00] | 2.1 | 3.49 |
| decoder+assist | 144 | 0.26 | [0.19, 0.34] | 4.8 | 0.77 |
| assist+evidence | 144 | 0.26 | [0.19, 0.34] | 4.6 | 0.73 |
| assist+shuffled_ev | 144 | 0.23 | [0.17, 0.31] | 4.6 | 0.77 |
| agent+evidence | 144 | 0.08 | [0.03, 0.12] | 7.4 | 0.74 |
| composite | 144 | 0.03 | [0.01, 0.06] | 7.1 | 1.02 |
| composite_shuffled_ev | 144 | 0.01 | [0.00, 0.04] | 10.2 | 0.30 |
| agent_only | 144 | 0.01 | [0.00, 0.02] | 4.9 | nan |
| random | 144 | 0.01 | [0.00, 0.04] | 11.7 | 0.46 |
