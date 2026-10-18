# P_smr_modern (FINAL): SMR control whose decoder never trains on a display-identifiable label, certified in closed loop against a leak positive control, with hours split into decoder volume, robustness and user learning

## 1. Idea and signal basis

**Idea.** Decode mu/beta (8–30 Hz) ERD over sensorimotor cortex into continuous velocity in *body coordinates*:
- **y (primary):** feet vs both hands.
- **x (earned per person):** right vs left hand.

**Change from SF2.** The decoder is trained **only** on fixation-verified probes and on *blind* closed loop with body-part cues. Sighted cursor use trains the *user*. The decoder sees sighted data only through unsupervised re-centring.

Endpoints are certified where claimed (closed-loop held-out mappings, pixel replay, blind "just intend"), beside a deliberately leaky *shadow* decoder that shows the tests can detect a leak.

**Signal.** Imagined hand or foot movement suppresses mu/beta at C3/C4 (contralateral hand) or at the vertex (feet). Gel precedents: Wolpaw & McFarland 2004 PNAS; McFarland 2010; Edelman 2019 Sci Robot; Meng 2016 Sci Rep.

**Time course on ds004362** (new reading, §10). On the central 11 channels with 1-s windows (cross-subject means, gel, untrained, no feedback):

| Task | 0–1 s | 0.5–1.5 s | 1–2 s | 2–3 s | 3–4 s |
|---|---|---|---|---|---|
| L/R | 0.605 | 0.616 | 0.588 | 0.574 | 0.537 |
| Feet/fists | 0.554 | 0.646 | 0.667 | 0.629 | 0.610 |

- Both tasks decline after their peak (untrained, feedback-free imagery fades); for L/R the sustained part is weak (conceded).
- In the top third of subjects ranked on the *independent* feet/fists task, L/R stays at 0.729/0.691/0.656 (0.5–1.5/1–2/2–3 s): sustained lateralised SMR exists in responders, not in the population mean.
- Therefore **y is primary, x is earned** (§4), and training windows start ≥ 1.0 s after any cue.

**Naturalness (honest).** We decode *body-part intent*, not goal-coordinate intent. Rotate-180 training is removed (§2). Instruction is withdrawn for use and for the "just intend" endpoint, and both are certified blind.

## 2. Label source

**Rule.** No decoder label is ever computed from anything visible.

**L1: fixation probes (~43% of labelled time).**
- A central glyph cues one of four classes. The glyph-to-class map is re-drawn every session.
- A gaze-contingent 1.0° fixation window is enforced (from generative); the trial aborts and repeats if gaze leaves it.
- Windows run 1.0–4.0 s after the cue.

**L2b: blind closed loop (~57%).**
- The cue is a central glyph (half of trials) or a duration- and loudness-matched synthesised body-part word ("feet", "left hand", or a two-part cue for diagonals). **It is never a location.**
- Cursor hidden, fixation enforced; the decoder drives the hidden cursor toward a hidden goal.
- Feedback is a **label-symmetric** tone: its pitch encodes progress (decrease in distance), identically for every direction. A 0.5-s end-point flash follows.
- Label = cued body direction. Windows start ≥ 1.0 s after the cue and end before the flash.
- Closed-loop statistics, with nothing direction-specific on screen or in the tone.

**S: sighted closed loop, identity mapping.** User practice and the performance endpoint. **No labels.**

**Shadow decoder (positive control, never deployed).** It is trained on S with ReFIT labels u = (target − cursor)/‖·‖, i.e. exactly the display-identifiable label the brief warns about. It is the leak positive control in §5.

**Held-out mappings (test only):** rotate ±90° (primary; it swaps the axes), plus mirror-x and mirror-y.

**Label audit.** Each session, L1/L2b labels are predicted from the pixel log, audio envelope, gaze, IMU and trial timing with gradient boosting. The result must sit at the matched-n permutation p95.

**Conceded.** Dropping sighted labels gives up ReFIT-style sighted CLDA; the cost is measured as the blind-to-sighted gap. A blind-trained decoder that works sighted is itself evidence against display reading.

## 3. Data engine and scaling mechanism

**(a) Population prior, used from day 1.**
- Stieger 2021 (62 participants, ~600 h) and ds004362, central montage, re-centred on own rest (Zanini 2018). Used from day 1 (conceded: E6, 0.644 vs 0.484 in trials 1–9).
- **Every deployed decoder must pass §5.** An own-data-only decoder runs in parallel for the prior-vs-own contest.

**(b) The PI's hypothesis, split into three separately falsifiable predictions.** The evidence at fixed SNR says a decoder for one person saturates fast:
- E2b: the prior with 0 own trials scores 0.736; adding 30 own trials gives 0.741.
- E6: own data from scratch catches the prior by ~40 trials.
- The repo engine simulation saturates by day 2.

We therefore pre-register that the bitter lesson is **half right for one person**:

| | Mechanism | Design | Prediction |
|---|---|---|---|
| **H1: decoder volume → peak** | More labels at fixed person and SNR | Random-k vs all vs last-k training sessions, tested on **fixed** held-out sessions (last 4 per deep participant): user state at test is constant; random-k separates volume from recency | **≤ +0.03 r** from 5 h to 60 h (saturation) |
| **H2: sessions → robustness** | Dry-cap non-stationarity (re-fit, impedance, gain drift); more cap fits span more of it (repo: frozen decoder collapses under drift) | First-block-of-day r on unseen days, re-centring only, vs number of training sessions | **≥ +0.05 r** from 3 to 20 sessions |
| **H3: practice → user** | Operant SMR learning (Wolpaw 2004; Perdikis 2018) | L1 separability at fixed n = 120 per session; decoder frozen at session 2; frozen-epoch shadow refits (from manifold) | **≥ +0.05 acc** on y by 30 h |

**Statistics.**
- Primary statistic: gain per doubling (one parameter, bootstrap over sessions and participants; from population).
- Descriptive: a − b·n^(−c), with a bootstrap CI on the asymptote a (from vocabulary), fitted on per-session rungs (~40 per deep participant).

**Confounds fixed.** L1 and L2b keep one cued protocol in every session, so the ladders' label stream never changes; there is no rotate-180 schedule; instruction is withdrawn only in S and "just intend".

**(c) People axis: not our contribution (conceded).**
- E2 locates it on gel: linear saturation at 32–48 people; deep crossover at ~64.
- Our 20 training dry participants give a matched-N dry slope for N = 2–20 against E2. They cannot locate saturation.
- Pooled dry ladder: 20 × 7.5 h = 150 h, with rungs 20/40/80/150 h, tested on 8 held-out people (corrected from "180 h").

## 4. Expected performance

**Conversion.** r = d′/√(d′²+4), d′ = 2Φ⁻¹(acc): the balanced point-biserial correlation, a *heuristic* for velocity r.

**Dry penalty.** 20–40% of d′, a planning prior with **no source (conceded)**, measured in month 2 (10 dry-vs-gel pilots; from population) and the monthly gel arm.

**Per 1-s window** (the real-time unit):
- Gel population means: y 0.40 (1–2 s), x 0.28 → 0.18 (0.5–1.5 → 2–3 s).
- Dry: y 0.25–0.33; x 0.11–0.23. Responder-third x: 0.23–0.44.

**Integration.** The cursor integrates successive windows. Over 3 s the information is bounded by the 3-s-window figures: gel L/R 0.631 (r 0.32), feet/fists 0.700 (r 0.46).

| Stage (dry, everyone reported) | Metric | Prediction |
|---|---|---|
| Session 1 L1 | feet vs hands; 4-class | 0.60–0.68; 0.33–0.42 |
| Untrained, per 1 s | r on y; r on x | 0.25–0.33; 0.11–0.23 |
| 30–60 h (H3) | r on y, per 1 s | 0.30–0.45 |
| 30–60 h | 1-D y, 2 targets, hit rate at 10 s | 70–85% |
| x earned (pre-registered, §5) | fraction of participants | 30–45% (E1: 38% above floor imagined L/R; 33% at 2–3 s) |
| Earners, 8 targets at 15 s | unassisted (assisted) | 30–55% (60–80%; repo simulation: r≈0.4 → 0.49 alone, ≈0.8 with goal posterior) |
| Non-earners | 1-D y + dwell + goal posterior | 0.03–0.15 bits/s |
| Earners | bits/s; time-to-target | 0.05–0.3; 4–8 s |

**x-axis rescue (from vocabulary).** For non-earners, alternative pairs on the central montage (e.g. subtraction vs rest, a self-proposed act) are screened on L1. They are ranked with empirical-Bayes shrinkage on nested sessions. A pair is admitted only through the P1 gate (§5).

**Conceded.** The rate does not exceed the project's established 0.03–0.3 bits/s. We expect 15–30% non-responders, none excluded.

## 5. Purity proof

**Auditors (no socket to the decoder).** Eye tracker 1,000 Hz; H/V EOG; bilateral EMG (forearm, tibialis, gastrocnemius, trapezius, masseter, SCM); head/seat IMU.

**Floors.** Matched-n permutation and circular shift, p95; no performance-based exclusion; the claim is "auditor-bounded".

**Primary tests (four).**

**P1. Incremental information (from vocabulary; replaces the 25% surrogate rule).**
- Auditor-only vs auditor+EEG, same learner (gradient boosting), nested.
- The auditor set is gaze/microsaccades, EOG, EMG, IMU, the display/audio log, and **S7 posterior alpha lateralisation** (P3/P4/PO7/PO8, 8–12 Hz; from population).
- Pass: lower bound of the increment > 0, and increment ≥ 70% of the EEG margin.
- **Auditor-matched propensity resampling:** EEG keeps ≥ 70% of its margin.
- Run on L1, L2b, S and the held-out-mapping blocks.

**P2. Held-out mappings, closed and open loop.**
- The decoder is frozen and has never seen these mappings. Rotate ±90° makes feet/hands move the cursor *horizontally*.
- **C2c (new, closed loop):** a visible moving cursor. Decoder output is regressed on the body label M⁻¹(target − cursor) and on the screen label (target − cursor).
- **C2b (open loop):** 3 s of imagery with the cursor frozen, then a keyed body-part report; scored decoder vs report, so a remap failure is still a labelled trial.
- Pass: β_body ≥ 70% of the trained margin, and the upper CI of β_screen < 10%.
- **Declared inconclusive zone:** β_body between 50% and 70%, or a β_screen CI straddling 10%.
- Pooled mixed model over all 32 participants (deep, plus broad sessions 4–5). **y is primary**; x only for earners.
- **Positive control:** the sighted-label shadow decoder must show β_screen > 0 in the same blocks; otherwise the test is declared insensitive.

**P3. Pixel replay (from generative).**
- On 15% of S and C2c trials, the cursor silently replays a past trajectory aimed at a different target.
- Decoder output is regressed on intended direction and on replayed cursor velocity. Pass: upper CI of β_replay < 10% of the margin.
- Noticed replays (post-trial query) are analysed separately.
- Passive replay without intent must be at floor.

**P4. Artefact levers.**
- **Instructed-artefact localiser every 10 sessions (from manifold):** jaw, brow, squint, eye closure, blinks, saccades, neck, swallow; 3 × 10 s each. Components responding > 2 SD are projected out before training, and decoder output on localiser epochs must be < 25% of the margin.
- **EMG- and gaze-silent subset:** EEG-only 30–90 Hz rim power (from manifold), plus max-over-channels EMG RMS < rest p97.5, plus gaze < 1°. Pass: ≥ 80% of the margin retained; the subset fraction is reported.

**Secondary.** Central-given-posterior increment (central+posterior vs posterior alone; the old C6 tested the wrong direction); Haufe patterns at C3/C4/Cz; 8–20 Hz-only decoder; saccade-locked output vs circular-shift floor; gaze-replay mimic in blind blocks (from vocabulary); axis independence |ρ| < 0.3.

**Montage (conceded).** The central 11 reduce risk but do not prove anything: CP channels share posterior sources through volume conduction. Purity rests on the label rule, P1–P4 and the shadow positive control.

## 6. Protocol

**Months 1–2.** Dry-pin tolerability pilot (3 people × 3 days × 2 h); dry-vs-gel penalty in 10 pilots; pre-registration.

**Deep cohort.** 4 × ~60 h (1.5 h/day, 4 days/week, ~10 weeks), from month 3. Robot from week 7.

**Broad cohort.** 28 × 5 × 1.5 h; 8 held out as the fixed test set.

**Total.** ≈ 450 h dry, plus one 64-ch gel session per deep participant per month (€20–40k).

**Staff.** 2 rigs, 1 postdoc, 2 RAs. M3–8 broad; M3–12 deep; M13–18 analysis.

| min | Block |
|---|---|
| 0–15 | Setup, impedance, 3-min rest (re-centring); artefact localiser every 10th session |
| 15–30 | L1, 120 fixation probes |
| 30–50 | L2b blind closed loop |
| 50–65 | S sighted identity, 15% replay |
| 65–73 | P2 held-out mapping (from session 3; C2b and C2c alternate) |
| 73–85 | "Just intend": no instruction, half blind, half sighted, identity |
| 85–90 | Strategy questions, NASA-TLX, auditory-probe reaction time |

Session 1 opens with a hint-free block (from population). Imagery instruction is withdrawn in S and "just intend" after session 10 (deep) or 3 (broad); L1 and L2b stay cued.

## 7. Real-time architecture

**Input.** Central 11 (FC5/FC1/FC2/FC6, C3/Cz/C4, CP5/CP1/CP2/CP6); LSL 500 Hz; causal 8–30 Hz filter bank; 1-s windows every 62.5 ms; latency ~300–500 ms.

**Model A.** OAS covariances → running re-centring (τ = 5 min, on all blocks; unlabelled) → tangent space → ridge to body velocity, plus a logistic idle detector.

**Model B.** EEGNet on the same input, pre-trained on the central prior and fine-tuned nightly. It is kept only if it passes P1–P4 alone.

**Ensemble.** Weights are fitted on L1/L2b only.

**Adaptation.** Kalman filter with idle gating; supervised update every 3 min from an L2b-only 24-min buffer; nightly retrain on L1 + L2b; test blocks frozen.

**Effector.** v_screen = g·M·v_body. The robot arm moves planar end-effector velocity ≤ 10 cm/s; grasp and z-switching are dwell actions.

**Idle safety (from vocabulary).** Disarm after 20 s idle; re-arm after a 2-s hold of the most idle-distinct state. The false-command rate in distractor-idle blocks is pre-registered at ≤ 1/min.

## 8. The Nature claim

**Headline figure.**
- **(a) Closed-loop provenance.** Under rotate ±90° and pixel replay, the deployed decoder follows the body (β_screen ≈ 0); the sighted-label shadow (same person, data, model) follows the screen. *The RLNF trap, quantified and closed.*
- **(b) Naturalness.** Blind vs sighted "just intend" across hours.
- **(c) Hours decomposed.** Decoder-volume peak (predicted flat), robustness (rising) and user learning (rising), each with an asymptote.
- **(d)** Robot pursuit and Haufe topographies.

**Sentence.** "A dry-cap EEG decoder trained only on labels the screen cannot identify gives continuous closed-loop control that follows body intent, not the display, under mappings and replays it never saw, where a decoder trained on on-screen labels follows the screen; for one person, more hours buy robustness and user skill, not decoder peak accuracy."

**Novelty, honestly stated.** Neither the paradigm nor the rate is new. What is new:
1. The first closed-loop EEG control certified against a leak positive control.
2. A pre-registered, signed decomposition of the bitter-lesson hypothesis.

Venue: Nature Communications.

## 9. Falsifiers and biggest risk

- **F1 (not cortex).** Any of:
  - pooled P2 lower CI of β_body < 50%;
  - upper CI of β_screen > 10%;
  - P1 increment lower bound ≤ 0;
  - shadow positive control at floor (the test is then uninformative).

  The inconclusive zone is reported as such.
- **F2 (circularity).** Upper CI of β_replay ≥ 25% of the margin, or passive replay above floor.
- **F3a.** Peak gain from random-k volume > +0.03 r. This falsifies *our* prediction, in the PI's favour.
- **F3b.** CI of the robustness gain per doubling includes 0.
- **F3c.** Pooled frozen-decoder L1 slope CI includes 0.
- **F4 (not natural).** After 30 h, blind "just intend" is < 70% of cued performance.
- **F5 (no 2-D).** Fewer than 25% of participants earn x. The headline is then 1-D y + goal posterior, stated as such.

**Biggest risks.**
- Dry L/R at floor in most users (likely).
- Slower user learning from blind closed loop (auditory-feedback SMR works but is initially slower: Nijboer et al. 2008 J Neurosci Methods); sighted S practice mitigates. A large blind-to-sighted gap.

## 10. ds004362: done and next

**Done** (106 subjects; cross-subject means; scratchpad `sf2_smr/res.json`).

| 0.5–3.5 s | L/R | feet/fists |
|---|---|---|
| Central 8–30 Hz | 0.631 | 0.700 |
| Central 8–12 / 16–30 Hz | 0.600 / 0.591 | 0.641 / 0.673 |
| Posterior 8–30 Hz | 0.604 | 0.662 |
| Posterior, central regressed out | 0.529 | 0.541 |
| One-shuffle permutation | 0.491 | 0.495 |

- E3: occipital channels decode the displayed target at 0.604 (L/R) and 0.661 (feet/fists). **The feet/fists anchor is as display-confounded as L/R** (conceded).
- Alpha and beta contribute almost equally, so band cannot separate mu from attention alpha; the P1/P2 dissociations must.

**Next, with kill rules.**
1. **Stieger 2021** (multi-session gel; download to be verified): first-block accuracy without recalibration vs number of training sessions. If its gain-per-doubling CI includes 0, H2 is demoted before collection.
2. **ds004362**, decoder trained on 1–4 s, evaluated per sliding window: if < 25% of subjects are above floor on sustained L/R, x is dropped from the pre-registration.
3. **Repo simulator with gaze leak:** blind-trained decoder vs sighted-label shadow; unless P2/P3 separate them at the planned n, the tests are redesigned before month 3.
