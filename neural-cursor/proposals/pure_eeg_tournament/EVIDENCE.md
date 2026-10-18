# Evidence produced during the tournament (real data + simulation). Judges: weigh proposals against this.

## Real data: OpenNeuro ds004362 (PhysioNet EEG Motor Movement/Imagery), 106 usable subjects
64-ch gel, 160 Hz, BCI2000; analysed on the 32 channels that match the g.tec dry cap. 45 trials per
subject per task (three 2-min runs). Targets are DRAWN on the screen (left/right or top/bottom) for the
whole trial. 3 subjects recorded at 128 Hz excluded (technical, pre-declared). Code:
neural-cursor/realdata/experiments.py. Model unless stated: 8–30 Hz, 0.5–3.5 s, OAS covariance →
Riemannian tangent space → logistic regression.

E1 Within-subject 5-fold CV (3 seeds), matched-n permutation floor p95 = 0.644 (n≈45):
| task | mean acc | 95% CI | median | fraction above floor |
|---|---|---|---|---|
| imagined left/right fist | 0.643 | [0.613, 0.672] | 0.615 | 0.38 |
| executed left/right fist | 0.668 | [0.639, 0.696] | 0.659 | 0.54 |
| imagined both fists/feet | 0.710 | [0.682, 0.741] | 0.696 | 0.65 |
| executed both fists/feet | 0.832 | [0.804, 0.860] | 0.874 | 0.90 |

E2 Cross-subject zero-shot (no labels from the new person; covariances re-centred on each person's own
REST epochs), 25 fixed held-out subjects, imagined left/right:
| # training people | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 | 81 |
|---|---|---|---|---|---|---|---|---|---|
| rest-aligned | 0.583 | 0.573 | 0.632 | 0.627 | 0.639 | 0.656 | 0.662 | 0.658 | 0.660 |
| unaligned | 0.540 | 0.535 | 0.588 | 0.585 | 0.589 | 0.605 | 0.626 | 0.626 | 0.637 |
Executed left/right, rest-aligned: 0.542 → 0.625 (8) → 0.664 (32) → 0.675 (48) → 0.673 (81).
Linear Riemannian model saturates at ~32–48 people. With ≥32 people, zero-shot ≥ own 45-trial calibration.
Deep model (EEGNet-8,2, rest Euclidean alignment, 60 epochs, zero-shot, same 25 test subjects):
4 people 0.48; 8 people 0.50–0.53; 16 people 0.53–0.61; 32 people 0.636–0.642; 64 people 0.647–0.663
(81 people still running). The deep model starts worse than the linear one (it needs more people) and
reaches or passes the linear plateau (~0.66) by 64 people — the classic crossover the bitter lesson predicts.

E2b Population prior + k own labelled trials (fixed test = each person's last 15 trials):
pooled zero-shot 0.736; pooled + 30 own 0.741; own-only with 30 trials 0.677; own-only with 8 trials 0.616.
(Every model scores higher on the last run than on the session overall; compare within one test set only.)

E6 Collect → retrain → adapt, replayed trial by trial (predict, then learn), 25 held-out subjects,
accuracy in blocks of 9 trials:
| | 1–9 | 10–18 | 19–27 | 28–36 | 37–45 |
|---|---|---|---|---|---|
| population model, frozen | 0.644 | 0.618 | 0.622 | 0.716 | 0.702 |
| own data from scratch | 0.484 | 0.480 | 0.609 | 0.613 | 0.698 |
| population then adapted | 0.644 | 0.631 | 0.640 | 0.720 | 0.698 |
Population prior makes a new person usable from trial 1; own data catches up by ~40 trials; the
dataset cannot show what happens beyond 45 trials per person — exactly the regime a long-recording
(live-in) study would measure.

E4/E5: executed→imagined transfer 0.656; imagined CV improves 0.643 → 0.709 when the person's 45 executed
trials are added to training (more related own data helps); idle-vs-imagine 0.844 (display-confounded:
rest epochs have no target on screen).

E3 Confound audit (within-subject CV, mean acc; fraction of subjects above floor in brackets):
| decoder | imagined L/R | executed L/R | imagined fists/feet |
|---|---|---|---|
| sensorimotor 11 ch, 8–30 Hz power | 0.629 (0.42) | 0.666 (0.49) | 0.706 (0.63) |
| frontal/ocular 7 ch, 8–30 Hz power | 0.507 (0.10) | 0.519 (0.14) | 0.515 (0.12) |
| frontal/ocular 7 ch, 1–4 Hz slow potentials | 0.566 (0.18) | 0.545 (0.10) | 0.540 (0.11) |
| occipital 8 ch, 8–30 Hz power | 0.604 (0.31) | 0.600 (0.39) | 0.661 (0.52) |
| all 64 ch, 8–30 Hz power | 0.617 | 0.643 | 0.693 |
Reading: the 8–30 Hz motor signal is not ocular; a small slow-potential eye leak exists (18% of subjects
above floor where 5% is expected); occipital channels decode the DISPLAYED target nearly as well as
sensorimotor ones (visual attention to the on-screen target) — the public benchmark carries the
display-identifiable-label trap. All-64 < 32-ch with 36 training trials: the problem is data-limited.

## Simulation (neural-cursor), pure-EEG runtime built and reviewed this session
- Real-time loop over LSL, effectors (screen cursor, UDP, ROS2, simulated 2-link arm), collect-label-
  retrain-promote data engine with calibrated champion/challenger on frozen held-out trials, purity audit
  (time-block permutation floors; frontal, slow, occipital, display, intercept and auditor surrogates),
  protocol with non-spatial cues and hidden-cursor probes, multi-subject simulator. 97 tests pass.
- Earlier: weak velocity decoder r≈0.4 → 0.49 success alone, ≈0.8 with a goal posterior; ErrP channel at
  AUC 0.68 added nothing over the dense posterior; frozen decoder collapses under dry-electrode drift.

## Multi-day engine simulation (5 simulated days × 1 h, 3 users, intent mismatch 30° sd, lapse 0.15)
Decoder r on a FIXED evaluation pool: own data 0 (day 1, nothing deployed) → 0.39 (day 2) → 0.41 (day 4–5);
population prior (20 simulated others): 0.18 on day 1 zero-shot → 0.39 → 0.40. Closed-loop success 0.33–0.51,
throughput ~0.66–0.89 bits/s (simulated; the simulator's SNR is fixed, so saturation by day 2 is a property of a
linear decoder at fixed SNR, not evidence about real brains). Purity verdict on blank-cue windows: PASS on
nearly all days; with gaze leak in every simulated person but fixation held on blank-cue windows, the verdict
stays PASS (the certifying windows are clean by design).
