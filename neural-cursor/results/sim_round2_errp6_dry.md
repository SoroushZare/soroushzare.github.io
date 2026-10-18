simulated users = 6, trials/user = 24, signal gain = 8.0, eye gain = 0.0, ErrP = 6.0 µV, drift = 0.0/min
velocity decoder r = 0.458; evidence decoder AUC = 0.611 (permutation floor p95 = 0.599)

| condition | n | success | 95% CI | time-to-target (s) | Fitts TP (bits/s) |
|---|---|---|---|---|---|
| oracle | 144 | 1.00 | [1.00, 1.00] | 2.1 | 3.49 |
| decoder+assist | 144 | 0.79 | [0.73, 0.85] | 4.9 | 0.78 |
| assist+evidence | 144 | 0.89 | [0.83, 0.93] | 4.9 | 0.77 |
| assist+shuffled_ev | 144 | 0.83 | [0.77, 0.90] | 5.1 | 0.75 |
| agent+evidence | 144 | 0.08 | [0.04, 0.12] | 5.2 | 1.03 |
| composite | 144 | 0.40 | [0.31, 0.47] | 9.8 | 0.36 |
| composite_shuffled_ev | 144 | 0.41 | [0.33, 0.49] | 7.7 | 0.43 |
| agent_only | 144 | 0.01 | [0.00, 0.02] | 4.9 | nan |
| random | 144 | 0.01 | [0.00, 0.04] | 11.7 | 0.46 |
