simulated users = 6, trials/user = 24, signal gain = 8.0, eye gain = 0.0, ErrP = 9.0 µV, drift = 0.1/min
velocity decoder r = 0.452; evidence decoder AUC = 0.704 (permutation floor p95 = 0.602)

| condition | n | success | 95% CI | time-to-target (s) | Fitts TP (bits/s) |
|---|---|---|---|---|---|
| oracle | 144 | 1.00 | [1.00, 1.00] | 2.1 | 3.49 |
| decoder+assist | 144 | 0.39 | [0.31, 0.47] | 4.8 | 0.76 |
| assist+evidence | 144 | 0.63 | [0.56, 0.72] | 4.8 | 0.80 |
| assist+shuffled_ev | 144 | 0.79 | [0.72, 0.85] | 5.0 | 0.79 |
| agent+evidence | 144 | 0.10 | [0.06, 0.15] | 8.2 | 0.45 |
| composite | 144 | 0.35 | [0.28, 0.43] | 8.7 | 0.41 |
| composite_shuffled_ev | 144 | 0.42 | [0.33, 0.51] | 9.1 | 0.38 |
| agent_only | 144 | 0.01 | [0.00, 0.02] | 4.9 | nan |
| random | 144 | 0.01 | [0.00, 0.04] | 11.7 | 0.46 |
