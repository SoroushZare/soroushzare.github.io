simulated users = 6, trials/user = 24, signal gain = 8.0, eye gain = 1.0
features = bandpower
calibration decoder r = 0.419 (permutation floor p95 = 0.142); EOG-only surrogate r = 0.001

| condition | n | success | 95% CI | time-to-target (s) | Fitts TP (bits/s) |
|---|---|---|---|---|---|
| oracle | 144 | 1.00 | [1.00, 1.00] | 2.1 | 3.49 |
| decoder | 144 | 0.57 | [0.50, 0.65] | 9.2 | 0.37 |
| decoder+assist | 144 | 0.80 | [0.73, 0.86] | 4.9 | 0.82 |
| eog_only | 144 | 0.03 | [0.01, 0.06] | 9.5 | 0.28 |
| no_neural | 144 | 0.00 | [0.00, 0.00] | nan | nan |
| random | 144 | 0.03 | [0.01, 0.07] | 11.5 | 0.30 |
