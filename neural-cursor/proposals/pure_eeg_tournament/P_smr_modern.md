# P_smr_modern: Mirror-proof sensorimotor-rhythm control on a dry cap, scaled by hours

## 1. One-line idea and signal basis

**Idea.** Use the one stimulus-free scalp signal with 20 years of closed-loop replication behind it: mu/beta (8–30 Hz) event-related desynchronisation (ERD) over the sensorimotor cortex during imagined or attempted movement. Decode it into continuous 2-D velocity in *body coordinates* with a Riemannian plus small-deep ensemble that co-adapts with the user. Prove purity with a **mirror-mapping crossover** that no eye, muscle or display reader can pass. Then measure how control scales with hours in three deeply trained people and thirty briefly trained ones.

**Signal basis.** Imagining or attempting a hand or foot movement suppresses 8–13 Hz mu and 13–30 Hz beta over the contralateral hand area (C3/C4) or the vertex (feet). This is covert motor intention, not a reflex to the screen.

- **Proven with gel EEG.** Wolpaw & McFarland 2004 (PNAS 101:17849): four users reached peripheral targets within 10 s on 89/70/78/92 % of trials, mean movement times 1.9/3.9/3.3/1.9 s. McFarland 2010 extended this to 3-D. Edelman 2019 (Sci Robot; 68 subjects per press materials) did continuous pursuit and robotic-arm tracking; Meng 2016 (Sci Rep, 13 subjects) robotic reach-and-grasp; Ding 2025 (Nat Commun, 21 users) online finger-level robotic hand (80.56 % two-finger, 60.61 % three-finger).
- **Fading instruction.** Wadsworth users report using imagery less as training proceeds and control becoming automatic (Wolpaw, McFarland & Vaughan 2000 review). We use imagery as a short bootstrap, then withdraw it and measure the effect.

## 2. Label source (and why it is not display-identifiable)

In every classic SMR dataset (ds004362, Stieger 2021, Forenzo 2024) the target side equals the imagined limb, so gaze, VEPs and cursor motion all decode the label. We break that equivalence.

**L1 (open-loop bootstrap and recurring probes, ~10 % of data).** One of four equal-luminance glyphs at fixation means *left hand / right hand / both feet / relax*; glyph-to-class mapping randomised per participant and re-drawn each session. No target, cursor or spatial position is shown. Label = cued body part; retinal location is identical across classes. Fixation within 1.5° is verified; decoding windows start 0.5 s after cue onset.

**L2 (closed-loop co-adaptation, mirror-balanced).** The decoder outputs velocity in a *body frame* (x = right- minus left-hand intent, y = feet vs relax). A mapping M ∈ {identity, mirror-x, mirror-y, rotate-180} carries it to the screen, changing every 4-min block, announced at block start and invisible during trials. Training label = ReFIT-style intended velocity (Gilja 2012) pulled back to the body frame, **M⁻¹·(target − cursor)**. M is balanced in every training set, so each screen-direction component has exactly zero correlation with each label component: gaze, hemifield VEPs and visible cursor motion carry zero linear information about the label. The only carrier of M within a trial is the user's motor plan.

**L3 (free use)** feeds only unsupervised re-centring, never supervised labels.

Cost: half the training blocks are mirrored; we measure this effort cost (§5 C3).

## 3. Data engine and scaling mechanism

Three axes, each with evidence and a saturation point.

**(a) Population prior (cross-subject).** Pre-train on 32-ch, 8–30 Hz versions of Stieger 2021 (62 participants, >600 h, 598 sessions, >250,000 trials), Forenzo 2024 (28 participants, ~168 h online pursuit) and ds004362 (109), each subject re-centred on its own unlabelled rest covariance (Zanini 2018). Their labels are display-confounded, so they are a prior only; every claim is evaluated on L1/L2 data. This repo already shows where the axis saturates (`results/realdata/E2_scaling_imag_LR.json`; zero-shot imagined L/R, 32-ch montage, tangent space + logistic regression, 25 fixed held-out subjects): rest-aligned 0.583/0.632/0.639/0.656/0.662/0.658/0.660 at N = 1/4/16/32/48/64/81 subjects; unaligned 0.540→0.637; executed 0.542→0.673. **The prior saturates near 0.66 by ~32–48 subjects**: it removes calibration but does not deliver control. Kaifosh 2025 (Nature) scaled generic sEMG decoders across thousands of people; EEG SMR hits its ceiling far earlier.

**(b) Within-person data.** Adding the same person's executed trials raises imagined L/R CV from 0.643 to 0.709 in ds004362 (`E4E5_transfer_idle.json`). Forenzo 2024 reports deep decoders improving across sessions as online data accumulated, beating the traditional decoder by the last session. Expected: success roughly linear in log(own hours) from 2 to ~30 h, then flattening at the person's SNR ceiling.

**(c) User learning.** An estimated 20–25 % of people fail standard SMR; co-adaptive calibration (Vidaurre 2011) gave novices and previous non-performers significant control within a session; Perdikis 2018 (PLOS Biol) won Cybathlon with two tetraplegic pilots via longitudinal mutual learning; Stieger's dataset documents learning across up to 11 sessions.

**Separating machine from human.** Every session is decoded by the *frozen session-1 decoder* (gain = user learning) and the *nightly retrained decoder* (extra gain = machine), on the same fixed test blocks.

**Predicted saturation.** Machine ~30–50 own hours; human ~20–30 sessions; plateau past ~100 h, pre-registered so a plateau is a finding. RLNF's "only volume helped" is tested directly: architecture fixed, only hours vary.

## 4. Expected performance on this hardware

**Anchors.** My Wolpaw-ITR calculation from Wolpaw 2004 (8 targets, timeouts as errors, inter-trial interval ignored) gives ≈0.8–0.9 bits/s for the best two trained gel users and ≈0.2–0.35 for the other two. On ds004362 (gel, untrained, no feedback, our 32-ch subset) imagined L/R reaches 0.643 [0.613, 0.672], with only 38 % of 106 subjects above the per-subject floor 0.644 (n ≈ 45); feet vs fists 0.710. Dry-vs-wet MI evidence is thin and on other hardware (one 10-subject lower-limb study: 58.44 vs 58.66 %); we still budget a 20–40 % SNR penalty because the g.tec layout lacks FC3/FC4/C1/C2/CP3/CP4 for a tight Laplacian.

| Stage (responders, dry 32-ch) | Metric | Prediction |
|---|---|---|
| Session 1, L1 cued, open loop | 2-class / 4-class accuracy | 0.60–0.68 / 0.35–0.45 |
| After 10–20 h co-adaptive | 1-D 2-target hit rate (10 s) | 75–85 % |
| After 10–20 h | 2-D velocity r per axis | 0.30–0.45 |
| After 30–100 h | 8-target centre-out success at 15 s, unassisted | 45–65 % (repo sim: r 0.4 → 0.49) |
| Same, with the repo's goal-posterior assist | success | 70–85 % (repo sim: 0.8) |
| Throughput, unassisted | Fitts/Wolpaw bits/s | 0.10–0.35 |
| Time-to-target | median | 3–7 s |
| Robot arm, planar pursuit | normalised tracking error vs chance | clearly below the shuffled floor; no absolute claim |

Non-responders (an expected 15–30 %) are reported, never excluded. This is the honest ceiling: about a third of Wolpaw's best gel users, with true cortical provenance.

## 5. Purity proof

**Auditors (recorded, never fed to the decoder):** eye tracker ≥250 Hz; bipolar H/V EOG; EMG on both forearm flexors/extensors, both tibialis anterior, bilateral upper trapezius and masseter; 3-axis head accelerometer.

**Controls** (matched n; label-permutation and block-wise circular-shift 95th-percentile floors; no performance-based exclusion):
- **C1. Matched non-neural surrogates.** Same pipeline and same L1/L2 labels on (i) eye-tracker+EOG, (ii) EMG RMS, (iii) display variables (cursor/target positions, cursor velocity history, time in trial), (iv) their union. Each must sit at floor; EEG must beat the union.
- **C2. Mirror crossover (headline).** On frozen-decoder test blocks, regress decoder output on body-frame intent and screen-frame direction. Cortex predicts β_body ≫ 0, β_screen ≈ 0; any gaze or display reader predicts the reverse.
- **C3. Fixation blocks.** Gaze held within 1.5° of a central dot while the cursor moves peripherally; only tracker-verified epochs scored; retained performance and mirror-block effort cost reported.
- **C4. Replay (feedback circularity).** The user passively watches replays of their own sessions ("do not try"); decoder-output vs replayed-velocity correlation must be ≤ floor and < 25 % of the closed-loop value.
- **C5. EMG gate.** Epochs with EMG RMS > rest + 3 SD are excluded from test scoring (fraction reported); attempted movement is allowed only under that gate.
- **C6. Physiological signature.** The 11 central channels (FC5, FC1, FC2, FC6, C3, Cz, C4, CP5, CP1, CP2, CP6) alone must retain ≥80 % of performance; deleting FP1/FP2/AF3/AF4/F7/F8 must cost < 10 %; Haufe patterns must peak contralaterally at C3/C4 (Cz for feet) in mu/beta. 8–30 Hz features exclude slow ocular potentials and most EMG power by construction.

## 6. Protocol

**Deep cohort (the PI's "person sits and collects").** Three participants × about 100 h each: 2 h/day, 4 days/week for about 13 weeks, roughly 50 sessions. Robotic arm in weeks 9–13.

**Broad cohort.** Thirty participants × 5 sessions × 1.5 h, about 225 h. Four further participants are held out entirely, and their last two sessions form the fixed test set on every resource ladder.

**Total.** About 525 h of own dry-cap data, plus about 800 h of public pre-training data.

**Session template (90 min).**

| min | Block |
|---|---|
| 0–10 | Dry-cap setup with impedance and line-noise QC |
| 10–15 | Rest, eyes open and closed (re-centring reference) |
| 15–25 | L1 cued probes: 4 classes × 20 trials × 4 s |
| 25–65 | Co-adaptive L2: centre-out to 8 targets and continuous pursuit (Edelman-style), mirror-balanced 4-min blocks |
| 65–75 | Frozen-decoder test: C2 and C3 |
| 75–82 | Replay, C4 |
| 82–90 | "Just intend": no imagery instruction ("move it however feels natural"), frozen decoder, then a strategy questionnaire |

**Imagery withdrawal schedule.** Explicit imagery cues are dropped from L2 instructions after session 10 for the deep cohort and after session 3 for the broad cohort. L1 probes keep the cued labels, which are needed for the clean-label stream.

**Gel arm (science, optional).** One gel 64-channel session per month per deep participant (12 sessions) to quantify the dry penalty and validate the Laplacian. Equipment would be rented or borrowed; budget placeholder €20–40k, quote needed.

## 7. Real-time architecture

- **Input:** LSL 500 Hz; causal 4th-order IIR bank 8–12/12–16/16–20/20–26/26–30 Hz. **Windows:** 1.0 s, updated every 62.5 ms; latency ≈300–500 ms (window-bound), compute < 20 ms.
- **Model A (Riemannian):** per band 32×32 OAS covariance → re-centred on a running Riemannian mean (τ = 5 min, includes rest) → tangent space (5 × 528 = 2,640 features) → ridge to 2-D body-frame velocity + logistic idle detector; α by nested CV.
- **Model B (deep):** EEGNet-class CNN (Lawhern 2018) on 2 s windows, pre-trained on the pooled corpus, fine-tuned nightly. Ensemble weights fitted on the previous day's held-out blocks; B's weight is reported, not assumed.
- **Smoothing:** Kalman velocity filter (gain 0.9); idle probability gates velocity to zero.
- **Adaptation:** (1) continuous unsupervised re-centring (the repo sim showed it recovers part of the dry-drift collapse); (2) CLDA update (Orsborn 2014) every 3 min on mirror-balanced L2 labels, learning rate halved per session after convergence; (3) nightly retrain. Test sessions use a frozen decoder.
- **Effector:** decoder emits v_body ∈ ℝ² + p_idle; cursor v_screen = g·M·v_body; 6-DoF arm gets planar end-effector velocity ≤ 10 cm/s, z/grasp toggled by sustained "both feet". The repo's goal-posterior shared autonomy is optional; the unassisted arm is always reported.

## 8. The Nature claim

**Headline figure.**
- (a) Mirror crossover: β_body versus β_screen per participant, with surrogate floors.
- (b) Control versus own hours on log axes for the three deep participants on fixed test blocks, split into frozen-decoder (human) and retrained (machine) components.
- (c) The population-prior ladder saturating at about 40 subjects, contrasted with the continued within-person climb.
- (d) Robot-arm pursuit trajectories.
- (e) Decoder spatial patterns at C3/C4/Cz, with EOG/EMG/display surrogates at floor.

**The sentence.** "Hours of co-adaptive use on a wearable dry EEG cap yield continuous 2-D cursor and robot control that provably follows the user's imagined body part rather than the screen, eyes or muscles, and the gains come from the person and the person's own data, not from more people."

**Why new.** SMR control itself is not new; we say so. What is new:
1. A crossover proof of provenance. No continuous EEG cursor study we know of has decoupled label from display.
2. A dry-cap, fixed-test-set scaling law that separates human learning from machine learning across about 100 h per person.
3. A measured imagery-withdrawal endpoint.

Honest venue estimate: Nature Communications is likely; Nature needs (b) to keep climbing past 50 h.

## 9. Falsifier and biggest risk

**Pre-registered falsifiers (frozen test sessions, fixation-verified).**
- **F1 (no cortex).** β_body ≤ circular-shift floor + 0.05, or β_screen ≥ β_body, in ≥2 of 3 deep participants.
- **F2 (circularity).** Replay correlation is ≥50 % of the closed-loop value.
- **F3 (no scaling).** The slope of success versus log(hours) between 10 and 100 h has a 95 % CI that includes 0 in all three deep participants. This would falsify the bitter-lesson hypothesis for SMR on this cap.
- **F4 (not natural).** Performance in the "just intend" block falls by more than 50 % relative to the cued blocks after 30 h.

**Biggest risk.** Dry-pin SNR under hair at C3/C4, combined with a sparse central montage, could leave 2-D velocity r below 0.25. Usable control would then rest on shared autonomy, and 15–30 % of users would remain non-responders. **Mitigations:** gel-arm calibration of the dry penalty, the executed→imagined bootstrap (+6.6 points in ds004362) and pre-registered reporting of every participant.

## 10. What can be tested now on ds004362

**Already in this repo**, 32-channel g.tec subset, tangent-space SMR, 106 subjects:
- E1 within-subject accuracy: imagined L/R 0.643, executed L/R 0.668, imagined fists/feet 0.710, executed fists/feet 0.832.
- E2: population ladder saturating at about 0.66 (§3).
- E4: executed→imagined transfer 0.656, and imagined plus executed data 0.709.
- E5: idle versus imagine reaches 0.844, but **we discount it**, because rest trials have no target on screen, so the label is display-identifiable. It is a worked example of the trap.

**To run next (each with a kill rule).**
1. **E3 provenance.** The same pipeline on the ocular-frontal, sensorimotor and occipital subsets, in the 8–30 Hz band and in 1–4 Hz slow features. *Kill SMR on this montage* if sensorimotor ≤ ocular-frontal in 8–30 Hz. The trap is *demonstrated* if slow frontal features decode the L/R target above floor.
2. **Time and topography.** Time-resolved decoding (0–4 s). Predicted: cortical ERD builds over 0.5–2 s, while gaze and VEP signals peak in the first 0.1–0.4 s. Contralateral sign flip of C3 versus C4 ERD between hands. *Kill* if accuracy peaks before 0.5 s.
3. **Channel ablation.** Removing FP1/FP2/AF3/AF4/F7/F8 must cost less than 0.02.
4. **E2b few-shot.** Population prior plus k own trials (k = 0…40), to estimate the hours-to-plateau parameter used in §3.
5. **E6 replay of the collect-retrain-adapt loop.** Sequential retraining over a subject's runs. This predicts the within-person slope, with a bounded caveat: there are only 3 imagined runs per class.

**What ds004362 cannot test.** It has no mapping swap, so the mirror crossover (C2) requires our own data. The first two weeks of the deep cohort are devoted to it.
