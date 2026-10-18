## Data-engine learning curve (simulated): 5 days × 1 h, 3 target user(s), population prior N = 20

seed 20261004; assistance a = 0.5; ocular-leak fraction 0.0; user learning rate 0.0 (0 = users do not learn); topography heterogeneity h ~ U(0.3, 0.7); prior pseudo-count 300 windows; intent mismatch 30° sd, lapse p 0.15; runtime 1927 s.
decoder cosine [95 % trial CI] / r: the decoder deployed that day, on a FIXED evaluation pool (2 min of auditory cued_blank per simulated day, never trained on), so every day meets the same windows.  ceiling: the best mean cosine a constant output reaches on that pool (cosine is not chance-corrected; day-to-day gain drift also adds an offset that lowers cosine but not r).  cued_blank audit cosine: out-of-fold cosine of a decoder refit on the day's blank-cue windows only (nothing spatial on screen, the person fixating), with its permutation floor (p95, circular shifts within session); its verdict is the purity statement of record.  Cursor-visible windows are display-identifiable by design and train the decoder but never certify it.  Success / TTT / TP: closed-loop Fitts trials.
Caveats: '0 (def.)' = nothing deployed, so 0 by definition, not a measurement.  The population prior's benefit is set by the assumed heterogeneity range; quote it as a sweep.

| condition | day | cum. h | deployed | decoder cosine [95 % CI] | ceiling | decoder r | n eval | CL success | TTT (s) | TP (bits/s) | hidden-cursor success | cued_blank audit cosine (floor) | promoted | purity verdict (cued_blank) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| own | 1 | 1 | none | 0 (def.) | 0.019 | 0 (def.) | 680 | 0.00 | – | – | 0.00 | 0.283 (0.049) | 3/3 | PASS×3 |
| own | 2 | 2 | v1 | 0.356 | 0.019 | 0.394 | 680 | 0.40 | 5.0 | 0.78 | 0.27 | 0.254 (0.055) | 0/3 | PASS×3 |
| own | 3 | 3 | v1 | 0.356 | 0.019 | 0.394 | 680 | 0.33 | 4.7 | 0.89 | 0.22 | 0.297 (0.060) | 2/3 | PASS×3 |
| own | 4 | 4 | v1 / v3 | 0.381 | 0.019 | 0.408 | 680 | 0.41 | 4.9 | 0.83 | 0.22 | 0.244 (0.063) | 0/3 | PASS×3 |
| own | 5 | 5 | v1 / v3 | 0.381 | 0.019 | 0.408 | 680 | 0.38 | 5.0 | 0.80 | 0.24 | 0.202 (0.065) | 1/3 | PASS×3 |
| population | 1 | 1 | population prior | 0.176 | 0.019 | 0.183 | 680 | 0.28 | 6.2 | 0.66 | 0.23 | 0.294 (0.044) | 3/3 | PASS×3 |
| population | 2 | 2 | v1 | 0.359 | 0.019 | 0.390 | 680 | 0.51 | 5.8 | 0.66 | 0.35 | 0.273 (0.058) | 0/3 | GAZE_TRACKS_LABEL×2, PASS |
| population | 3 | 3 | v1 | 0.359 | 0.019 | 0.390 | 680 | 0.37 | 4.8 | 0.82 | 0.27 | 0.245 (0.066) | 1/3 | PASS×3 |
| population | 4 | 4 | v1 / v3 | 0.377 | 0.019 | 0.404 | 680 | 0.40 | 5.0 | 0.86 | 0.21 | 0.246 (0.056) | 0/3 | PASS×3 |
| population | 5 | 5 | v1 / v3 | 0.377 | 0.019 | 0.404 | 680 | 0.38 | 5.1 | 0.83 | 0.27 | 0.227 (0.068) | 0/3 | PASS×3 |
