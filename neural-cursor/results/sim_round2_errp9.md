simulated users = 6, trials/user = 24, signal gain = 8.0, eye gain = 0.0, ErrP = 9.0 µV, drift = 0.0/min
velocity decoder r = 0.406; evidence decoder AUC = 0.682 (permutation floor p95 = 0.609)

| condition | n | success | 95% CI | time-to-target (s) | Fitts TP (bits/s) |
|---|---|---|---|---|---|
| oracle | 144 | 1.00 | [1.00, 1.00] | 2.1 | 3.49 |
| decoder+assist | 144 | 0.83 | [0.78, 0.89] | 4.7 | 0.82 |
| assist+evidence | 144 | 0.83 | [0.77, 0.89] | 4.8 | 0.80 |
| assist+shuffled_ev | 144 | 0.87 | [0.81, 0.92] | 4.6 | 0.87 |
| agent+evidence | 144 | 0.12 | [0.07, 0.17] | 7.0 | 0.47 |
| composite | 144 | 0.34 | [0.27, 0.42] | 9.5 | 0.37 |
| composite_shuffled_ev | 144 | 0.45 | [0.38, 0.53] | 9.1 | 0.35 |
| agent_only | 144 | 0.01 | [0.00, 0.02] | 4.9 | nan |
| random | 144 | 0.01 | [0.00, 0.04] | 11.7 | 0.46 |
