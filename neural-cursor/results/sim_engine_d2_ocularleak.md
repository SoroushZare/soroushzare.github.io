## Data-engine learning curve (simulated): 2 days × 0.5 h, 2 target user(s), population prior N = 0

seed 20261004; assistance a = 0.5; ocular-leak fraction 1.0; user learning rate 0.0 (0 = users do not learn); topography heterogeneity h ~ U(0.3, 0.7); prior pseudo-count 300 windows; intent mismatch 30° sd, lapse p 0.15; runtime 88 s.
decoder cosine [95 % trial CI] / r: the decoder deployed that day, on a FIXED evaluation pool (2 min of auditory cued_blank per simulated day, never trained on), so every day meets the same windows.  ceiling: the best mean cosine a constant output reaches on that pool (cosine is not chance-corrected; day-to-day gain drift also adds an offset that lowers cosine but not r).  cued_blank audit cosine: out-of-fold cosine of a decoder refit on the day's blank-cue windows only (nothing spatial on screen, the person fixating), with its permutation floor (p95, circular shifts within session); its verdict is the purity statement of record.  Cursor-visible windows are display-identifiable by design and train the decoder but never certify it.  Success / TTT / TP: closed-loop Fitts trials.
Caveats: '0 (def.)' = nothing deployed, so 0 by definition, not a measurement.  The population prior's benefit is set by the assumed heterogeneity range; quote it as a sweep.

| condition | day | cum. h | deployed | decoder cosine [95 % CI] | ceiling | decoder r | n eval | CL success | TTT (s) | TP (bits/s) | hidden-cursor success | cued_blank audit cosine (floor) | promoted | purity verdict (cued_blank) |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| own | 1 | 0.5 | none | 0 (def.) | 0.027 | 0 (def.) | 272 | 0.00 | – | – | 0.00 | 0.311 (0.050) | 2/2 | PASS×2 |
| own | 2 | 1 | v1 | 0.364 | 0.027 | 0.378 | 272 | 0.49 | 5.3 | 0.76 | 0.31 | 0.270 (0.059) | 0/2 | PASS×2 |
