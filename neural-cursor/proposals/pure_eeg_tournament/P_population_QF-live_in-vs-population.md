# P_population (amended after QF vs live_in): a generic sensorimotor-only EEG decoder trained on many people, used by a new person with no labelled calibration, then co-adapted

## 1. One-line idea and signal basis

**Idea.** We carry the Kaifosh et al. 2025 (Nature) sEMG recipe over to EEG and audit it. 130 training participants each give one standardized 90-min session. One generic decoder is trained across them. A new person starts with zero labelled calibration. The decoder is then co-adapted using only display-unidentifiable trials. The headline result is a **cross-participant scaling curve** on a **sensorimotor-only montage**, with eye, muscle and display surrogate curves measured on the same N-ladder, plus the number of own-data hours the generic prior is worth.

**Signal.** Two contrasts that the 106-subject public data decodes zero-shot from sensorimotor channels alone (§10): **horizontal**, lateralized mu/beta (8–30 Hz) ERD for left versus right hand; **vertical**, hand versus foot somatotopy (Pfurtscheller et al. 2006 NeuroImage), both hands up and feet down.

**Instruction (amended; costs some naturalness).** "Make the hidden cursor go to the ring you chose", plus, if asked, "many people imagine their hands for sideways, both hands for up, feet for down." A generic decoder needs a code shared across people, which fully free strategies conflict with; we accept the cost, log strategies and stratify by them.

**Prior art, stated honestly.** Subject-independent SMR classifiers already exist (Fazli et al. 2009 Neural Networks). So does zero training from a user's own past sessions (Krauledat et al. 2008 PLoS ONE), and co-adaptive calibration that starts from a subject-independent classifier (Vidaurre et al. 2011 Neural Comput). What is new here is a **prospective, purity-audited, closed-loop scaling law over participants** that uses labels the display cannot identify.

## 2. Label source: never display-identifiable

**Pre-committed private goals (main training source; grafted from live_in).** Four identical unmarked rings stay on screen all session, so no target has a visual onset. The participant chooses one and taps it on a side tablet whose layout is rotated by a random multiple of 90° each trial, so finger and gaze direction carry no goal information. A 1-s blank precedes a 3-s intent window (fixation dot, hidden cursor). The label is the committed goal; the screen is pixel-identical across classes.

**Cued dark-intent trials.** A 250-ms luminance-matched central glyph; window 1.25–4.25 s after cue. **Amended:** the glyph→direction mapping is fixed within each person and counterbalanced across people (Latin square over 24 permutations), removing the working-memory remapping load. Glyph-specific visual responses map to different labels in different people, so a cross-person decoder cannot exploit them.

**Free-choice, report-after trials (gold test).** The participant picks a direction, intends it, and reports it by keypress only after the window closes.

**Closed loop.** Visible-cursor trials never provide labels. Co-adaptation (§7) uses only hidden-cursor probe trials with pre-committed goals.

## 3. Data engine, scaling mechanism, expected curve

**Mechanism (a hypothesis, not a result).** Pooling people lets a model learn the parts of the signal shared across people, while unsupervised re-centring removes each person's offset.

**Conceded.** The most person-invariant signal is the corneoretinal dipole. Microsaccades are biased toward covertly attended locations (Hafed & Clark 2002; Engbert & Kliegl 2003, both Vision Research), so they persist even under fixation. A pooled decoder will therefore select for eye signals unless the architecture prevents it. That is why the decoder montage now **excludes every frontal and temporal channel** (§5, §7). Surrogates check the result; they are not the only defence.

**Evidence (re-run for this round, §10, sensorimotor 21-ch, frontal channels excluded):**

| Axis | EEG N=8 → 76 | EEG gain, 95% CI | Eye-proxy gain | EEG − eye slope |
|---|---|---|---|---|
| L/R, broadband | .669 → .695 | +.026 [+.008, +.045] | +.073 | **−.047 [−.079, −.016]** |
| Hands/feet, filter bank | .662 → .721 | **+.059 [+.042, +.078]** | +.030 | **+.029 [+.002, +.057]** |

**Reading.** There is a participant axis. **On L/R it scales more slowly than the eyes**, so the original L/R-centred claim fails our own slope test. **On hands/feet, cortex out-scales the eye proxy** (eye accuracy only .553, output correlation 0.06), so this somatotopic axis is now **primary** and L/R secondary. L/R broadband is flat from N=32 to 76 (.692 → .695); hands/feet filter bank still rises (.692 → .721).

**Withdrawn.** "The limit is model capacity" (now measured by a 0.2M/2M/8M capacity ladder at every N); "8× more trials per person justify a higher plateau" (it contradicted our flat per-person axis); "zero-shot equals per-person calibration" (measured with about 36 own trials only; the depth arm now measures hours-scale personalization).

**Data volume.** 130 training participants × 376 labelled windows ≈ 49k windows (about 45 h labelled, 195 h total); 30 held-out test participants; depth arm 4 × 30 h. Self-supervised pretraining on all unlabelled cohort recordings (same dry cap), TUEG secondary on the same 14 positions (+0–3 points). Display-cued public data (ds004362, Stieger 2021, Lee 2019, Cho 2017) serves only pretraining and the month-1 gate, never purity evidence.

**Expected curve.** Fit acc(N) = a − b·N^(−c) per axis on the fixed test cohort. Gel numbers are anchored in ds004362 (cued, displayed targets); dry numbers are projected, assuming a dry penalty of 4–8 points (unmeasured).

| N train | Hands/feet (gel, anchored) | Hands/feet (dry, projected) | L/R (dry, projected) |
|---|---|---|---|
| 10 | ~.66 | .58–.62 | .58–.62 |
| 40 | ~.69 | .60–.64 | .60–.63 |
| 130 | ~.72–.74 (extrapolated) | .62–.67 | .61–.64 |

Saturation is declared when doubling N adds less than 0.5 points. Four-direction velocity correlation has **no empirical anchor** and will be simulated from the two measured binary axes before pre-registration. We claim no number for it.

**Month-1–2 gate (kills or redirects the project before recording).** Pool ds004362, Stieger, Lee and Cho on the sensorimotor montage (N≈250). The population design proceeds only if both hold on the hands/feet axis:
1. N≈250 beats N=16 by at least 3 points, with the CI excluding 0.
2. The EEG slope beats the eye-proxy slope, with the CI excluding 0.

Otherwise resources shift to the depth arm.

## 4. Expected performance (honest)

- **Zero-shot:** 2-class 0.62–0.67 per 2-s decision gives 0.02–0.04 bits/s, the bottom of the established 0.03–0.3 range. Throughput is conceded to be low; the contribution is the scaling law and zero calibration.
- **Assistance:** success is re-simulated from the measured axes; raw and assisted are reported separately, and the headline uses **raw hidden-probe directional accuracy**.
- **Co-adaptation:** Vidaurre 2011 reached control within minutes from a subject-independent start; Benaroch 2021 reported about +30% over 20 sessions. We expect +3–8 points; the open quantity is how many own hours beat the prior.

## 5. Purity proof

**Architectural exclusion (new).** The decoder sees only 14 positions (FC5, FC1, FC2, FC6, C3, Cz, C4, CP5, CP1, CP2, CP6, P3, Pz, P4), referenced to their own average. Fp1, Fp2, AF3, AF4, F7, F3, Fz, F4, F8, T7 and T8 never enter the decoder path; they feed only surrogate S2. No EOG regression in the decoder path (it can inject inverted eye signal); regression is a sensitivity analysis only.

**Auditors (recorded, never input and never a loss).** Remote eye tracker at ≥500 Hz (microsaccades by the Engbert–Kliegl algorithm); 4 EMG (bilateral forearm, bilateral masseter); head IMU; 2-min artifact calibration.

**Surrogates on the same N-ladder.** S1 gaze plus microsaccade direction; S2 edge EEG (Fp/AF/F7/F8/T7/T8); S3 display, glyph and tablet tap; S4 label permutation (1,000 at matched n; 5 permutations already gave up to 0.64 on hands/feet broadband, so a full null is mandatory); S5 circular shift.

**Claims that must hold (pre-registered, per axis).**
1. EEG is above the 95th percentile of S1–S5 at every N.
2. **Slope dissociation:** a one-sided paired bootstrap on test subjects of (EEG gain per doubling − S1 gain) has a lower 95% bound above 0. This replaces the TOST.
3. **Microsaccade-incongruent trials:** where the dominant microsaccade direction in the window disagrees with the label, EEG accuracy is above the floor and at least 80% of congruent-trial accuracy.
4. **Saccade-locked output:** the decoder output averaged around (micro)saccade onsets shows no direction-specific deflection beyond the S5 floor. Spike potentials are a known microsaccade artifact (Yuval-Greenberg et al. 2008 Neuron).
5. The partial correlation with the label, controlling for S1 output, stays significant.
6. Free-choice report-after accuracy is above the permutation 95th percentile.

**Closed-loop circularity controls (amended).**
- Targets are pre-committed private goals. There is no flash and no visual onset.
- **Hidden-cursor probes** on 20% of trials score **direction, not speed**: the cosine of decoded velocity with the committed goal must be at least 70% of the visible-trial cosine and above S5.
- **Sham-feedback trials** (5%, grafted): the cursor replays an unrelated trajectory, and the output must follow the goal.
- **Replay-without-intent** viewing.
- **Pre-registered technical exclusions only.** No performance-based exclusion.

## 6. Protocol

**Participants.** 160 in total: **130 training and 30 held-out test**. The test participants are recruited interleaved through the cohort period and are never trained on.

**90-min session.**

| Min | Activity |
|---|---|
| 20 | Dry cap, eye-tracker calibration, 4 EMG, IMU |
| 3 + 2 | Rest (for re-centring only); artifact calibration |
| 24 | 240 cued dark-intent trials (5 blocks) |
| 12 | 96 pre-committed private-goal trials |
| 5 | 40 free-choice report-after trials |
| 3 | Replay without intent |
| 15 | Closed loop: 5 min frozen snapshot, zero-shot, then 10 min co-adaptive |
| 3 | Strategy questionnaire |

**Snapshots and the curve of record.**
- Snapshots are frozen at N = 10, 20, 40, 80 and 130.
- **Curve of record:** all 5 snapshots evaluated offline on the 30 test participants' sessions. At the end, each test participant returns for a 75-min visit and runs all 5 snapshots in blinded, counterbalanced order, giving the closed-loop curve.
- **Secondary prospective curve:** each training arrival is tested with the latest snapshot. Snapshots are retrained every 10 arrivals, so each arrival sees a model trained on everyone up to 10 before them.

**Depth arm (amended, grafted from live_in).**
- 4 participants × 30 h of **co-adaptive closed loop starting from the N=80 prior**.
- Each session includes a frozen-prior probe and fixed hidden-probe test sets.
- Offline, an own-data-only decoder ladder (1, 2, 4, …, 30 h) and the prior-plus-own ladder are fitted on the same test sets. This yields the **own-hours equivalent of the prior** and the human-versus-machine split.

**Dry-cap heterogeneity.** Technical channel criteria (line noise, flat or saturated) are computed before any decoding; failed channels are dropped, and training uses channel dropout to tolerate them.

**Feasibility (amended).** 2 rigs, about 14 sessions/week: 160 sessions in about 12 weeks plus slack (fallback 80 training + 20 test). Gel arm dropped; robot demo is 2 sessions per depth participant. Team: 1 postdoc, 2 RAs.

**Timeline.** M1–2 build and gate; M3 10-person pilot and pre-registration; M3–8 cohort; M6–13 depth arm; M9–11 test revisits; M12–13 robot; M13–18 analysis and paper.

## 7. Real-time architecture

**Input.** LSL at 500 Hz, causal 4–40 Hz filter, decimation to 250 Hz, the 14 channels with local average reference. 2-s window, 100-ms hop.

**Unsupervised alignment.** Riemannian re-centring to the person's 3-min rest plus a 5-min-half-life running mean for gain drift. "Zero calibration" means zero *labelled* calibration; the rest is disclosed.

**Models.**
- Primary: a 4-band filter-bank tangent-space linear model, with interpretable spatial patterns.
- Contender: a compact spatio-temporal transformer, masked-reconstruction pretrained, at 0.2M, 2M and 8M parameters (the capacity ladder).
- Head 1 outputs two axis logits (L/R and hands/feet) mapped to a 2-D direction. Head 2 outputs intent present versus absent.

**Adaptation.**
1. Zero-shot block: frozen snapshot plus re-centring. This is the headline.
2. Co-adaptive block: every 60 s, a SmoothBatch-style last-layer update (W ← 0.9W + 0.1W_batch), only on hidden-probe trials with pre-committed goals (Vidaurre 2011 precedent).
3. Depth arm: nightly fine-tune.

**Latency and effector.** Under 30 ms compute, about 120 ms end-to-end. The decoder emits (v, confidence) at 10 Hz; a goal posterior that never reads the true goal blends in assistance. The same frozen decoder drives the cursor or a planar robot end-effector (≤5 cm/s) and never sees the effector type.

## 8. The Nature claim

**Headline figure.**
- **(a)** Zero-shot accuracy on 30 held-out people versus N, per axis, with the S1/S2/S3/permutation curves and the capacity ladder.
- **(b)** Closed-loop hidden-probe directional accuracy versus snapshot N, raw and assisted.
- **(c)** Own-hours equivalent of the generic prior (depth arm).
- **(d)** Cursor and robot driven by one frozen decoder.

**One sentence.** *A sensorimotor-only scalp-EEG decoder trained on 130 people decodes a new person's intended direction with no labelled calibration, its accuracy grows with the number of training people along a measured power law that eye, muscle and display surrogates do not match, and the prior is worth a measured number of hours of that person's own data.*

**Why it is new.** Subject-independent EEG decoding is not. New here: the participant-scaling law measured prospectively in closed loop; display-unidentifiable labels; a slope dissociation from eye surrogates, which already separates the axes in public data (eyes scale faster on L/R, cortex on hands/feet); and the prior's own-hours value.

If the curve is flat, the result is a measured ceiling, which we judge to be Nature Communications level, not Nature.

## 9. Falsifiers and risks

**Falsifiers (pre-registered).**
1. Hands/feet zero-shot at N=130 minus N=10 on the test cohort is under 3 points, or its CI includes 0.
2. The EEG − S1 slope lower bound is ≤ 0 on hands/feet. On L/R alone, this restricts the claim to the vertical axis.
3. Pre-committed or free-choice accuracy is at or below the permutation 95th percentile.
4. Hidden-probe cosine is below 70% of visible, or at the S5 floor.
5. Microsaccade-incongruent accuracy is at the floor.

**Risks.** A dry-cap plateau below 0.60; L/R scaling dominated by eyes (already seen in public data); lower naturalness from the strategy hint; low raw throughput; a failed month-1 gate, which pivots the project to the depth arm.

## 10. ds004362: results (original and this round)

**Original run** (32 ch incl. frontal, recorded reference, experiments.py): L/R zero-shot .599 at N=76; eye proxy (7 frontal channels, 1–4 Hz) .730; a combined model weighted eyes 1.17 versus central 0.36.

**This round (run in memory, no files written).** Setup: 106 subjects, 30 fixed test and 76 pool; per-subject rest re-centring; tangent space at identity; logistic regression with C=0.01; N=8 is the mean of 6 draws.

| Model | Imagined L/R, N=8 / 32 / 76 | Imagined hands/feet, N=8 / 32 / 76 |
|---|---|---|
| 32-ch, CAR over 64 (frontal included) | .660 / .683 / .690 | .685 / .698 / .710 |
| **Sensorimotor 21-ch, no frontal, posterior/central reference** | .669 / .692 / .695 | .682 / .696 / .708 |
| Same, after unsupervised frontal regression | .663 / .696 / .700 | .683 / .693 / .708 |
| Same, regression + 4-band filter bank | .640 / .667 / .673 | **.662 / .692 / .721** |
| Eye proxy | .646 / .700 / .719 | .524 / .544 / .553 |

**Further results.** Paired gains and slope differences are in §3. Individually significant test subjects at N=76: 57–60% (L/R), **77% (hands/feet, filter bank)**. Eye-proxy independence: output correlation 0.21 (L/R) and 0.06 (hands/feet); on the third of trials where the eye proxy is least confident, EEG reaches .662 and **.740**.

**Unreconciled:** a ~9-point gap between the original pipeline (.599) and this one (.690 on 32 ch). Sensorimotor channels on the recorded reference give .691, so referencing does not explain it. Both are reported; reconciling them is the first month-1 task. The design relies on paired slope comparisons within one pipeline. **Caveat:** ds004362 targets are displayed and there is no feedback, so none of this is closed-loop purity evidence.
