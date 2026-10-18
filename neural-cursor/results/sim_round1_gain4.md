simulated users = 6, trials/user = 24, signal gain = 4.0, eye gain = 0.0
features = bandpower
calibration decoder r = 0.192 (permutation floor p95 = 0.148); EOG-only surrogate r = -0.028

| condition | n | success | 95% CI | time-to-target (s) | Fitts TP (bits/s) |
|---|---|---|---|---|---|
| oracle | 144 | 1.00 | [1.00, 1.00] | 2.1 | 3.49 |
| decoder | 144 | 0.15 | [0.10, 0.21] | 10.4 | 0.30 |
| decoder+assist | 144 | 0.43 | [0.35, 0.51] | 5.9 | 0.64 |
| eog_only | 144 | 0.01 | [0.00, 0.03] | 11.0 | nan |
| no_neural | 144 | 0.00 | [0.00, 0.00] | nan | nan |
| random | 144 | 0.03 | [0.01, 0.07] | 11.5 | 0.30 |
