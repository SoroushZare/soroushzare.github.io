simulated users = 6, trials/user = 24, signal gain = 8.0, eye gain = 0.0, ErrP = 9.0 µV, drift = 0.1/min
velocity decoder r = 0.452; evidence decoder AUC = 0.704 (permutation floor p95 = 0.602)

| condition | n | success | 95% CI | time-to-target (s) | Fitts TP (bits/s) |
|---|---|---|---|---|---|
| oracle | 144 | 1.00 | [1.00, 1.00] | 2.1 | 3.49 |
| decoder+assist | 144 | 0.19 | [0.12, 0.26] | 4.4 | 1.00 |
| assist+evidence | 144 | 0.19 | [0.13, 0.26] | 4.0 | 0.96 |
| assist+shuffled_ev | 144 | 0.19 | [0.13, 0.26] | 4.0 | 0.91 |
| agent+evidence | 144 | 0.10 | [0.06, 0.16] | 7.9 | 0.50 |
| composite | 144 | 0.10 | [0.06, 0.15] | 10.6 | 0.39 |
| composite_shuffled_ev | 144 | 0.07 | [0.03, 0.11] | 9.3 | 0.37 |
| agent_only | 144 | 0.01 | [0.00, 0.02] | 4.9 | nan |
| random | 144 | 0.01 | [0.00, 0.04] | 11.7 | 0.46 |
