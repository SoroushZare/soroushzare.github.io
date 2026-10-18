simulated users = 6, trials/user = 24, signal gain = 8.0, eye gain = 0.0
features = bandpower
calibration decoder r = 0.421 (permutation floor p95 = 0.144); EOG-only surrogate r = -0.026

| condition | n | success | 95% CI | time-to-target (s) | Fitts TP (bits/s) |
|---|---|---|---|---|---|
| oracle | 144 | 1.00 | [1.00, 1.00] | 2.1 | 3.49 |
| decoder | 144 | 0.41 | [0.33, 0.49] | 9.1 | 0.35 |
| decoder+assist | 144 | 0.56 | [0.48, 0.65] | 5.4 | 0.70 |
| eog_only | 144 | 0.01 | [0.00, 0.02] | 14.5 | nan |
| no_neural | 144 | 0.00 | [0.00, 0.00] | nan | nan |
| random | 144 | 0.03 | [0.01, 0.07] | 11.5 | 0.30 |
