# P_population (SF1): a generic sensorimotor-only EEG decoder trained on many people, zero labelled calibration, then co-adapted

## 1. One-line idea and signal basis

**Idea.** We apply the Kaifosh et al. 2025 (Nature) sEMG recipe to EEG and audit it. 100 training participants each give one standardized 90-min session, and one generic decoder is trained across all of them. A new person starts with zero labelled calibration. The decoder is then co-adapted using only display-unidentifiable trials.

**Measured quantity.** The PI's bitter-lesson hypothesis on two axes: **gain per doubling of training people** (sensorimotor-only montage, eye/muscle/attention/display surrogates on the same ladder) and **gain per doubling of a person's own labelled windows**, i.e. how many own windows the prior is worth.

**Signal.** Mu/beta (8–30 Hz) ERD/ERS over sensorimotor cortex, with a **4-class somatotopic code**: left hand, right hand, both hands, feet. These map to left, right, up and down. Lateralization separates left from right. Hand versus foot somatotopy separates up from down (Pfurtscheller et al. 2006 NeuroImage).
- **Amended:** we no longer use two independent axis heads. The measurement in §10 showed they are collinear: the hands/feet head calls 63% of single-hand trials "hands".

**Instruction (costs some naturalness, conceded).** "Make the hidden cursor go to the ring you chose." A strategy hint is given only after a **hint-free block**, and we measure the cost of the hint (§6).

**Prior art.** Not new: subject-independent SMR (Fazli et al. 2009), co-adaptation from a generic start (Vidaurre et al. 2011), Riemannian re-centring (Zanini et al. 2018 IEEE TBME). New is a **prospective, purity-audited measurement of both scaling axes, using labels the display cannot identify**.

## 2. Label source: never display-identifiable

**Pre-committed private goals (main source).** Four identical unmarked rings stay on screen. The participant picks one and taps it on a side tablet whose layout is rotated by a random multiple of 90° on every trial. Then follow a 1-s blank and a 3-s intent window (fixation dot, hidden cursor). The label is the committed goal, and the screen is pixel-identical across classes.

**Cued dark-intent trials (cohort decoder only).** A 250-ms luminance-matched glyph; window 1.25–4.25 s; glyph→direction mapping fixed within a person, Latin-square counterbalanced across people. **Amended:** glyph trials train only the cross-person decoder; every within-person model (co-adaptation, depth ladders) uses glyph-free labels only.

**Free-choice, report-after trials (gold test).** The participant reports the chosen direction by keypress after the window closes.

**Closed loop.** Labelled windows come only from a 2-s hidden-cursor phase at the start of every closed-loop trial, with the goal pre-committed (§7). Visible-cursor segments never provide labels.

## 3. Data engine, scaling mechanism, expected curve

**Mechanism (a hypothesis).** Pooling lets the model learn the person-shared part of the somatotopic code. Unsupervised re-centring removes each person's offset.

**Statistic (amended).** The power-law claim is withdrawn: five snapshots over a few points do not identify three parameters. Pre-registered: **one parameter, gain per doubling of N** (slope of per-test-subject accuracy on log₂N, bootstrap over test subjects); a − b·N^(−c) is descriptive only.

**New evidence (this round, ds004362, files in `pop_sf1/`).** Setup:
- the **14 dry-cap positions of record**, with local average reference and no frontal or temporal channels;
- 30 fixed test subjects and 76 pool subjects;
- N ∈ {8, 16, 32, 64, 76}, with 10/8/6/4/1 draws per rung;
- L2 penalty **chosen per draw by grouped inner CV across training subjects**. This answers the bias-variance objection.

| Imagined, 14-ch | Model | N=8 → 76 | Gain 8→76 [95% CI] | Gain per doubling [95% CI] |
|---|---|---|---|---|
| Hands/feet | broadband, tuned | .669 → .704 | +.034 [.016, .053] | +.010 [.005, .015] |
| Hands/feet | filter bank, tuned | .646 → .694 | +.048 [.028, .069] | +.013 [.007, .019] |
| L/R | broadband, tuned | .661 → .685 | +.024 [.013, .034] | +.007 [.003, .011] |
| Hands/feet | eye proxy E2 (1–8 Hz + derived VEOG/HEOG) | .554 → .607 | +.052 [.027, .078] | +.017 [.010, .024] |
| L/R | eye proxy E2 | .720 → .766 | +.046 | +.014 |

**Reading.**
1. A participant axis exists on the hardware montage, also for a broadband model tuned at every N, so it is not only filter-bank catch-up.
2. **The gain is small: about 1 point per doubling.** From N=10 to 100 (3.3 doublings), we expect +3 to +4 points.
3. **Withdrawn:** last round's FB .721 (21-ch). On the 14 positions of record it is .694.
4. **Withdrawn: "cortex out-scales the eyes on hands/feet".** A stronger eye proxy (E2) scales at least as fast: EEG − E2 slope −.004 [−.013, +.005]. The earlier proxy was weak, as the opponent said.

**Why the small slope is still measurable.** The dry penalty mainly shifts the **level**; the slope is paired within one pipeline on the same test subjects. With 45 trials per test subject its CI half-width is ±0.5 points per doubling; our test participants give 376 windows each.

**Risk.** Dry noise may flatten the slope; the dry penalty is now measured (§6), not assumed.

**Data volume.** Cohort 100 × 376 labelled windows ≈ 38k; 24 test participants; depth arm 4 × 20 h. Self-supervised pretraining uses unlabelled cohort data; public display-cued data is used **only for pretraining**.

**Month-1–2 gate (amended; no cross-dataset pooling).** The gate is run separately within ds004362 and within Stieger 2021 (62 subjects, its own test split), so dataset shift cannot create the gain. Proceed only if both hold in both datasets:
1. 4-class gain per doubling > 0 (lower 95% bound).
2. EEG beats the strongest eye proxy **in level** on hands-versus-feet, by at least 5 points.

Otherwise, resources shift to the depth arm.

**Expected curve (dry, projected).**

| N train | 2-class (per axis) | 4-class |
|---|---|---|
| 8 | .58–.63 | .33–.37 |
| 100 | .61–.67 | .36–.41 |
| Gel anchor (ds004362, N=76) | .69–.70 | .44 |

## 4. Expected performance (honest)

**4-class.** Zero-shot 4-class accuracy is .442 on ds004362 (gel, cued; chance .25; pre-cue run-confound floor .269). That is 0.12 bits per 2-s decision, about 0.06 bits/s. Dry projected .36–.41 gives 0.03–0.05 bits/s, at the bottom of the established 0.03–0.3 range. **Throughput is low, conceded.** The contribution is the scaling measurement, not speed.

**Lag.** A 2-s window adds ~1 s effective lag, so "continuous" is nominal (conceded); a 1-s-window variant is reported.

**SMR non-responders** (Blankertz et al. 2010 NeuroImage). The resting-mu predictor is computed from rest before decoding; scaling is reported overall and by predictor tercile, with no exclusion.

**Co-adaptation (lowered).** +0–4 points per block (~60 labelled windows is few).

## 5. Purity proof

**Architecture.** The decoder sees only FC5, FC1, FC2, FC6, C3, Cz, C4, CP5, CP1, CP2, CP6, P3, Pz, P4. Frontal and temporal channels feed only the surrogates.

**Auditors (never input or loss).** **Amended:** head-stabilized (chin rest) tower eye tracker at 1,000 Hz for microsaccades (Engbert & Kliegl 2003); **12 EMG**: bilateral forearm, masseter, **tibialis anterior, gastrocnemius, upper trapezius, sternocleidomastoid** (foot and posture tension audited); head and seat IMU.

**Surrogates on the same ladder.** S1 gaze/microsaccade library plus E1–E3 frontal proxies (strongest reported); S2 edge EEG; S3 display, glyph, tablet; S4 label permutation (1,000, matched n); S5 circular shift; **S6 (new)** EMG union; **S7 (new)** posterior alpha lateralization (P3/P4/PO7/PO8, 8–12 Hz), the covert-attention reader.

**Surrogate positive control (new).** Each session includes a 3-min block where the participant overtly looks at the chosen ring. S1 must decode it at ≥.90. This shows the eye reader is not weak by construction.

**Claims that must hold (pre-registered).** The slope-dissociation test is **withdrawn as a purity criterion** (conceded: a weak surrogate passes it trivially). The tests are now based on level and on increment:
1. On pre-committed and free-choice trials, EEG is above the 1,000-permutation 95th percentile **and above the strongest surrogate in S1, S6 and S7, and their union**.
2. **Incremental information:** stacked model (EEG + all surrogates) versus surrogates alone, nested by subject, with a lower 95% bound above 0. The **scaling claim is made on this increment**: its gain per doubling must exceed 0.
3. **Microsaccade-incongruent trials:** accuracy is at least 80% of congruent-trial accuracy.
4. **Saccade-locked decoder output:** no direction-specific deflection beyond S5. Spike potentials are a known artifact (Yuval-Greenberg et al. 2008).
5. **Attention dissociation (new):** in half the blocks the rings sit on the diagonals (45° rotation) while the motor code stays left/right/up/down. Transfer from axis-aligned blocks must lose ≤5 points with no rotated confusion pattern; an attention reader would rotate with the rings.
6. **Glyph control (new):** a within-person decoder on the 0–1-s glyph epoch sets the glyph-readable floor; cohort accuracy on cued trials must not correlate with it across people.

**Circularity controls.** Rings always present (no onsets); hidden-cursor probes scored by direction cosine to the committed goal; 5% sham-feedback trials; replay-without-intent block; technical pre-registered exclusions only.

## 6. Protocol

**Participants.** 124: **100 training and 24 held-out test**, with test participants interleaved through recruitment. Fallback: 64 + 20, which still spans 3 doublings from N=8.

**90-min session.**

| Min | Activity |
|---|---|
| 20 | Dry cap, eye tracker, 12 EMG, IMU |
| 3 + 2 | Rest (re-centring, non-responder predictor); artifact calibration |
| 8 | **Hint-free** pre-committed block (32 trials): measures natural zero-shot |
| 22 | 220 cued dark-intent trials |
| 12 | 96 pre-committed trials (half with the 45°-rotated layout) |
| 4 | 28 free-choice, report-after trials |
| 3 | Overt-gaze surrogate positive control |
| 13 | Closed loop: 4 min frozen zero-shot, then 9 min co-adaptive |
| 3 | Strategy questionnaire |

**Snapshots** frozen at N = 8, 16, 32, 64, 100; curve of record = all snapshots offline on the 24 test participants, who also return for a 75-min visit running the 5 snapshots blinded and counterbalanced (closed-loop curve).

**Naturalness gap (new):** hint-free versus hinted accuracy, per person, is a reported result.

**Dry-penalty measurement (new, month 3).** 10 pilots each do one session with the hybrid electrodes dry and one gelled. This gives a measured dry penalty in level and slope before pre-registration.

**Depth arm (amended).** 4 participants × 20 h from the N=64 prior. Every 2-h session begins with the standard labelled protocol (glyph-free trials only), so labels per hour match the cohort; own-only and prior-plus-own ladders are indexed by **labelled windows**, not hours, on fixed test sets.

**Feasibility (amended).** 2 rigs at ~10 sessions/week: 124 sessions in ~13 weeks plus 24 revisits. We budget 15% of sessions losing ≥1 vertex channel (channel-dropout training; results also reported without Cz, FC1, FC2, CP1, CP2). Robot: 1 session per depth participant. Team: 1 postdoc, 2 RAs.

**Timeline.** M1–2 gate; M3 pilot, dry penalty, pre-registration; M4–8 cohort; M6–12 depth arm; M9–10 revisits; M12 robot; M13–18 analysis and paper.

## 7. Real-time architecture

**Input.** LSL at 500 Hz, causal 4–40 Hz filter, decimation to 250 Hz, the 14 channels with local average reference. 2-s window (1-s variant also reported), 100-ms hop.

**Alignment.** Riemannian re-centring to the person's rest, plus a running mean with a 5-min half-life. "Zero calibration" means zero *labelled* calibration.

**Models.** Primary: 4-band filter-bank tangent-space **4-class multinomial**, penalty by grouped CV; contender: compact transformer at 0.2M/2M/8M parameters. Velocity v = Σ_k p_k·u_k over the four unit directions (replaces independent axis heads); a second head gives intent present/absent.

**Adaptation.** Zero-shot (headline): frozen snapshot plus re-centring. Co-adaptive (amended): every closed-loop trial opens with a 2-s hidden-cursor phase after a pre-committed goal, giving 1 labelled window per trial (~60 per block); SmoothBatch last-layer update (W ← 0.9W + 0.1W_batch) every 20 windows. Depth arm: nightly fine-tune.

**Effector.** Under 30 ms compute. (v, confidence) is emitted at 10 Hz. A goal posterior blends in assistance and never reads the true goal. The same frozen decoder drives the cursor or a planar robot (≤5 cm/s).

## 8. The Nature claim

**Headline figure.**
(a) Zero-shot 4-class accuracy on 24 held-out people vs log₂N: gain per doubling, EEG-beyond-surrogates increment, S1/S6/S7/permutation curves, capacity ladder. (b) Own-labelled-window ladder vs the prior (depth arm). (c) Closed-loop hidden-probe cosine vs snapshot N, raw and assisted. (d) Cursor and robot driven by one frozen decoder.

**One sentence.** *For scalp EEG intent decoding, measured with labels the display cannot identify and with eyes, muscles and spatial attention audited, adding people buys about one point per doubling and adding a person's own data buys a measured amount more. This gives the PI's bitter-lesson hypothesis its first quantitative test for non-invasive brain control.*

**Novelty, stated honestly.** It is incremental in method. It is new as a measurement: the first prospective, purity-audited estimate of both scaling axes. The Kaifosh analogy is weaker than first claimed, because EMG has far higher SNR and they had thousands of participants. The realistic venue is **Nature Communications**. Nature becomes possible only if the increment curve does not saturate by N=100.

## 9. Falsifiers and risks

**Falsifiers.**
1. The gain per doubling of the EEG-beyond-surrogates increment has a lower bound ≤ 0.
2. Pre-committed or free-choice accuracy is at or below the permutation 95th percentile, or at or below the strongest surrogate.
3. The 45° rotation produces a rotated confusion pattern, meaning the decoder reads attention.
4. Hidden-probe cosine is at the S5 floor.

**Risks.** Dry 4-class level below .33; slope flattens on dry electrodes; hint-free block at chance (low naturalness); failed gate pivots the project to the depth arm.

## 10. ds004362 results (this round; scripts and JSON in `scratchpad/pop_sf1/`)

**Scaling and eye proxies.** See §3. Fixed-penalty results are similar: hands/feet broadband +.042 [.019, .065] from 8 to 76.

**Four-direction (new; 14 ch, N=76, 30 test subjects).** The hands/feet head calls 63% of single-hand trials "hands" (both hands 72%, feet 33%); the L/R head is neutral on hands/feet trials (44–49% "right"). The two-head readout reaches .436 (chance .25) with 34–37% of L/R trials sent "up": **collinearity confirmed**. A joint 4-class multinomial reaches **.442**. Run-confound checks: 4-class on the pre-cue window .269; run type from the task window .553.

**Permutation null (new; 1,000 within-subject label permutations through the full N=76 pipeline, 30 test subjects × 45 trials).**

| Axis, model | Real | Null mean | Null p95 (cohort mean) | Null max | Per-subject null p95 | Subjects p<.05 (BH-FDR) |
|---|---|---|---|---|---|---|
| L/R broadband | .693 | .507 | .581 | .634 | .667 | 18/30 (15) |
| L/R filter bank | .672 | .504 | .542 | .585 | .644 | 14/30 (14) |
| Hands/feet broadband | .696 | .506 | .589 | .643 | .711 | 16/30 (15) |
| Hands/feet filter bank | .694 | .503 | .542 | .576 | .644 | 15/30 (15) |

All 1,000 nulls fall below the real value in every row (p ≤ .001). The null is centred on .50, but it is wide. One random weight vector applied to test subjects who share a class structure moves all of them together, above or below chance. Values like last round's "0.64" are the width of this matched-n null, not label-independent leakage. **Withdrawn:** "77% individually significant". With the proper null, it is 47–60% (14–15 of 30 after FDR).

**Order surrogate.** Run index plus within-run position decodes the label at .486 (L/R) and .517 (hands/feet). On the pre-cue window, the sensorimotor decoder scores .518 and .506. Trial order and pre-cue drift do not carry the label.

**Pipeline gap (conceded, still open).** The old .599 came from a different penalty (C=1 on 528 32-ch tangent features) and test split. My re-run to isolate the penalty did not finish this round. Every number here comes from one new file-backed pipeline on the 14 positions of record, and nothing relies on the old figure. The gap must be reconciled before the M2 gate.

**Caveat.** ds004362 targets are displayed and there is no feedback, so none of this is closed-loop purity evidence. That is why the decisive tests are prospective.
