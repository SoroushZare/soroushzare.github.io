# P_manifold — Intrinsic-manifold operant control: the decoder never sees a label, the person is the learner

## 1. One-line idea and signal basis

**Learn each person's intrinsic EEG manifold from many hours of unlabelled recording, freeze a 2-D velocity decoder that points along its dominant, stable directions, and let the person learn by trial and error ("brain babbling") to steer the cursor or robot. No intent label is ever fitted.**

*Signal.* Each 1-s window gives three 32-channel covariances (4–8, 8–13, 13–30 Hz) → Riemannian tangent vectors (3 × 528 features) → latent z ∈ R^10. On a scalp cap, the high-variance, stable modes in 4–30 Hz are sensorimotor mu/beta desynchronisation, posterior alpha and frontal-midline theta. These are the rhythms people have learned to regulate operantly for decades: Wolpaw & McFarland 2004 PNAS (4 users, 2-D SMR cursor), McFarland 2010 (3-D), Stieger et al. 2021 Sci Data (62 participants, 598 sessions, >600 h). Content above 30 Hz is excluded, which removes most of the EMG and the saccadic spike-potential band.

*Why this counts as "thinking".* The only instruction is "move it to the target, any way that works." Motor imagery is offered as a first hint, not required, and strategy reports are logged every session. Nothing is calibrated on what the person says they intend.

*Why the manifold.* Sadtler et al. 2014 Nature showed that monkeys quickly relearned BCI mappings inside the intrinsic manifold (estimated by 10-D factor analysis) and largely failed outside it. Busch, Fincke, Lajoie, Krishnaswamy & Turk-Browne 2026 Nat Neurosci found the same in humans with fMRI (2-s volumes, avatar navigation): within-manifold perturbations were relearned and outside-manifold ones were not. Oby et al. 2019 PNAS: multi-day practice creates *new* patterns. Golub et al. 2018 Nat Neurosci: short-term learning re-uses existing patterns. **To our knowledge, nobody has tested this with scalp EEG, at sub-second continuous control, driving a robot, or over long practice.**

## 2. Label source and why it is not display-identifiable

| component | what it is fitted on | label |
|---|---|---|
| manifold encoder f | unlabelled EEG: spontaneous, idle wear, free babbling, and all task windows pooled across target directions | none (self-supervised) |
| decoder D (2 × 10) | chosen once from f by EEG-only criteria: variance rank, split-half stability (principal angle < 30°), no frontal-polar dominance | none |
| online adaptation | Riemannian re-centring to a running mean (Zanini et al. 2018 IEEE TBME); nightly Procrustes re-alignment of new latents to old (Degenhart et al. 2020 Nat Biomed Eng; intent-free) | none |
| evaluation only | target direction (experimenter-known), used to score | never enters any fit |

Every decoder parameter depends only on EEG statistics pooled over all directions, so it cannot learn to read the target, gaze or the cursor's visible motion. This closes the RLNF failure mode by construction.

**The trap moves; it does not vanish.** Operant learning exploits *any* route that moves the cursor. If a manifold dimension carries ocular or visual-attention variance, the person can learn to steer with their eyes and we would call it thought. So:

1. Bands start at 4 Hz.
2. A pre-registered EEG-only exclusion rule removes any latent whose squared loading on FP1/FP2/AF3/AF4/F7/F8 entries exceeds 1.5× the chance share (0.335).
3. Purity is then *proven* with auditors (§5), never assumed.

The eye tracker is never used to pick dimensions, so it stays an independent auditor.

## 3. Data engine and scaling mechanism

The bitter lesson applies to the right object: unlabelled volume buys a better **representation**; the human supplies the **skill**. Every minute of cap wear counts.

**Loop A (machine): unlabelled hours → manifold estimate.**
- Stage 1 (tested, not assumed): population pretraining on ds004362, Stieger 2021 (>600 h) and our cohort, rest re-centred per subject.
- Stage 2: per-person fine-tuning with relative-positioning and temporal-contrastive objectives (Banville et al. 2021 J Neural Eng). Linear tangent-space PCA/FA is the always-reported baseline.
- Metrics on a fixed held-out test session: split-half principal angles, and the decodable direction information kept in the top-k latents (target used only as an evaluation key).
- Expected curve: angles fall roughly as 1/√hours and saturate at about 5–10 h of *own, task-state* data per person. Pilot (§10): own unlabelled task-run data helped (+0.034); others' rest data did not, so the population-prior benefit is a hypothesis to test, not an assumption.
- The live test of scaling is three manifold rungs (0.5 h, 5 h, 40 h). Each gets its own frozen D, and blocks are randomised within session.

**Loop B (human): practice hours → operant skill.** Precedents: SMR skill over 7–11 sessions (Stieger 2021), pursuit learning gains (Edelman et al. 2019 Sci Robotics), months-long mutual learning (Perdikis et al. 2018). Expected: a power law in practice hours, saturating at 20–40 h.

**Loop C (coupling).** Practice creates new covariance structure (Oby 2019). Nightly re-estimation absorbs it, and Procrustes keeps D's meaning fixed, so the manifold grows with the user and the decoder never has to be relearned.

**Where it saturates.** At the number of independently controllable dimensions that a dry 32-channel cap resolves. We expect 2–3, which is enough for 2-D velocity. Loop A cannot exceed what the person can modulate.

## 4. Expected performance on this hardware

The score is r_int, the correlation between produced velocity and the target unit vector, plus success and time-to-target. Anchors:

- **Repo simulation:** r ≈ 0.4 gives 0.49 success on 8 targets alone and about 0.8 with the Bayesian goal posterior.
- **Pilot on ds004362** (gel, 32-ch subset, open loop): a label-free top-8 manifold keeps hands-vs-feet imagery at 0.636 accuracy, against 0.694 for all 528 features (§10).
- **Wolpaw & McFarland 2004:** 2-D SMR control after extensive multi-session training, gel.

Predicted results on the dry cap, stated honestly:

| | session 1 | after 20 h practice (≥ half of long-arm users) |
|---|---|---|
| r_int | 0.05–0.15 | 0.25–0.40 |
| 4-target center-out success (15 s timeout) | ~35% (chance ≈ 25%) | 60–80% |
| median time-to-target | n/a | 6–10 s |
| throughput, EEG alone | n/a | 0.05–0.2 bits/s |
| throughput, with shared-autonomy posterior (secondary arm) | n/a | 0.2–0.4 bits/s |

Within- versus outside-manifold: Δr_int ≥ 0.10 after 2 h of practice. The approach does not beat the 0.03–0.3 bits/s envelope; its contribution is how control is learned and how it scales.

## 5. Purity proof

**Auditors** (separate amplifier, never fed to the decoder): 250 Hz screen-based eye tracker; bipolar HEOG/VEOG; EMG on masseter, temporalis, splenius and bilateral forearm flexors/extensors; 3-axis accelerometer on the cap.

**Controls (all pre-registered):**

1. **Matched non-neural surrogates.** The best ridge decoder from auditors alone predicts (a) target direction and (b) the realised cursor velocity, under the same CV and the same trials. Requirements:
   - EEG r_int must exceed the auditor-only r_int;
   - after regressing all auditor channels out of z offline, r_int must keep at least 80% of its value.
2. **Blind-target blocks, the core display control.** A spoken cue ("up-left") starts the trial; the screen shows only a fixation cross; cursor and target are hidden; the score is given by a tone at the end. Nothing on screen relates to the label. Requirement: blind r_int ≥ 0.7 × visible r_int and above its permutation floor.
3. **Fixation-enforced blocks.** Gaze is held within 1.5° of centre. Breaks are logged and counted, never excluded, and break rates are compared across conditions.
4. **Mapping-rotation probes.** Once a week, D is silently rotated 90° in latent space. Volitional control shows a drop, relearning and an after-effect on undoing; a retinotopic/attention confound would keep pushing toward the target in screen coordinates, and we quantify that velocity component.
5. **Within-manifold vs outside-manifold vs random-in-manifold mappings**, with per-dimension gains equalised so cursor-speed statistics match. A muscle or eye artefact has no reason to favour within-manifold mappings.
6. **Feedback-circularity "replay-watch" blocks.** The person passively watches a replayed trajectory, and we measure the display → latent open-loop gain. Requirement: below 0.1 of the closed-loop gain. The decoder is frozen, so it cannot learn its own visual consequence.
7. **Statistics.** Permutation floors at matched n, circular-shift floors, fixed test sessions on every ladder, no performance-based exclusion, and forward-model spatial patterns (Haufe 2014) for every latent.

## 6. Protocol

The study runs 15 months, pre-registered on OSF.

- **Months 0–3: build and offline work.** Build the rig and auditors; run the offline ladders on ds004362 and Stieger.
- **Long arm (L): 4 participants × 40 sessions × 1.5 h practice, plus 20 h of passive wear each (≈ 80 h per person, 320 h in total).** Twice-weekly sessions with a passive cap, during weeks 3–20.
- **Cohort arm (C): 24 participants × 3 sessions × 2 h (144 h).** Within-subject counterbalanced mappings: within-manifold, outside-manifold, and a conventional supervised MI decoder (reference only).
- **Gel arm (optional):** 6 cohort participants do 2 extra sessions with 64-ch gel, comparing manifold dimensionality and SNR (order €30–50k for amplifier and consumables).

Session structure (120 min):

| minutes | block |
|---|---|
| 10 | setup and impedance |
| 10 | spontaneous recording: fixation 3, eyes closed 2, free viewing 5 |
| 5 | free babbling: cursor visible, no target |
| 4 × 12 | center-out blocks (8 targets, 40 trials, 15-s timeout), rotating visible / fixation / blind-target / robot |
| 5 | replay-watch |
| 5 | strategy questionnaire |
| weekly | rotation probe |

## 7. Real-time architecture

**Pipeline:** LSL 500 Hz → causal 4th-order Butterworth bands → 1.0-s window with 50-ms hop (20 Hz) → OAS covariance → re-centring (exponential forgetting, half-life 5 min) → tangent space → f → z ∈ R^10, standardised → v = g·D·z → exponential smoothing (τ = 300 ms) with a dead zone.

**Encoder f:** a linear FA/PCA baseline, or a 1–5M-parameter temporal-conv/transformer retrained nightly on a GPU.

**Latency:** ≈0.5 s window-centre lag + <10 ms compute + smoothing ≈ 0.7 s effective. **Gain g** is set once so 95th-percentile excursions cross the workspace in ≈2 s.

**Adaptation** is label-free only: continuous re-centring, plus a nightly manifold refit with Procrustes alignment. D is never refitted to targets.

**Effector interface:** one LSL stream `intent_velocity` (2-D, later 3-D, 20 Hz). The cursor integrates it; the robot takes it as Cartesian end-effector velocity (ROS 2 bridge, workspace limits). Selection is by 1-s dwell. The same D drives every effector.

## 8. The Nature claim

**Headline figure:**
- (a) the intrinsic manifold learned from unlabelled scalp EEG, with sensorimotor topographies;
- (b) learning curves by session for within-manifold, outside-manifold and random mappings;
- (c) blind-target (no screen) performance against visible performance;
- (d) rotation after-effects;
- (e) a scaling surface of performance against unlabelled hours × practice hours, on a fixed test session;
- (f) auditor-only surrogates at chance.

**The one sentence:** *Humans learn to steer a cursor and a robot through directions of their own scalp-EEG manifold, found without a single label, while directions outside it stay unlearnable; control improves with unlabelled data and practice, and is shown not to come from eyes, muscles or the display.*

**Why it is new:** Sadtler 2014 was intracortical, in monkeys; Busch 2026 used fMRI at 0.5 Hz in higher-order cortex. Nobody has shown the manifold constraint, label-free decoder construction and its data-scaling law in non-invasive, sub-second, effector-general control with a display-free purity proof.

Honest reading: Nature Communications is likely; Nature needs the scaling surface plus a clean outside-manifold dissociation.

## 9. Falsifier and biggest risk

**Pre-registered falsifiers:**
1. After 6 h of practice, within-manifold r_int is TOST-equivalent (±0.05) to outside-manifold r_int. Then the manifold hypothesis fails for EEG.
2. Blind-target r_int is below 0.5 × visible r_int, or at its permutation floor. Then control is display-driven.
3. The auditor-only surrogate is at least as good as EEG.
4. Δr_int < 0.03 between the 0.5-h and 40-h manifold rungs. Then the data-scaling claim is dropped.

**Biggest risk: variance is not controllability.** On a dry cap the top latents may be arousal, alpha state or drowsiness: large, not finely steerable. The pilot warns of this: L/R imagery is barely better in the top-8 manifold than in random 8-D (0.539 vs 0.506).

Mitigation is a pre-registered fallback. Dimensions are selected by *controllability* in eyes-closed, audio-cued "push up / push down" probes. These cues are not display-identifiable, but this adds a supervised selection step, which we would report as such.

**Second risk: slow learning.** SMR skill takes weeks, so the long arm carries the claim.

## 10. What ds004362 can test now

In-memory pilot run this session on the repo's cached epochs (no files written; reproducible from the stated pipeline).
- Data: 106 usable subjects (3 excluded at 128 Hz), the 32 g.tec channels, 8–30 Hz.
- Features: OAS tangent space re-centred on each subject's inter-trial T0 epochs (no labels).
- Manifold: PCA on the 56 T0 epochs only (≈3 min, label-free).
- Decoding: 5-fold × 3 logistic regression on k = 8 projections.

| imagined task | top-8 manifold | random 8-D | bottom-8 | all 528 |
|---|---|---|---|---|
| hands vs feet | **0.636** | 0.562 | 0.548 | 0.694 |
| left vs right fist | 0.539 | 0.506 | 0.528 | 0.633 |

- **Hands vs feet:** top-8 minus random is +0.074 (75% of subjects positive, Wilcoxon p = 5.5e-9). Top-8 minus bottom-8 is +0.088 (p = 1.8e-7).
- **Left vs right:** top-8 minus random is +0.033 (p = 0.008), but top-8 is not better than bottom-8 (p = 0.55).
- **Own task-state unlabelled data helps; more rest does not.** Half the T0 epochs → T0 + 135 unlabelled trials from other runs raised hands/feet 0.629 → 0.663 (+0.034, p = 1.7e-5; L/R +0.004). Doubling rest alone (28 → 56 epochs) changed nothing (−0.004, n.s.).
- **Not ocular-dominated in 8–30 Hz.** The frontal-polar loading of the top-8 latents is 0.314, against a chance share of 0.335.

**Population ladder (negative result).** On a fixed set of 30 test subjects (2 × 5-fold), linear manifolds from 5, 20 or 76 *other* subjects' rest data (up to about 4 h) did not beat the subject's own 3 minutes. Hands/feet: own 0.623; others 0.645 / 0.632 / 0.639; all differences n.s. (p ≥ 0.37). L/R was also n.s. and non-monotonic.

With linear PCA, scaling comes from the **person's own task-state** unlabelled data, not others' rest; SSL encoders must change this to justify a population prior.

**Tests still to run, with kill criteria:**
- (i) The ladders above with SSL encoders and 2–60 min of own data. Kill if the top-k manifold never beats random-k by ≥ 0.05 for any imagery contrast.
- (ii) Does the manifold estimated from executed trials (unlabelled) contain the imagined-task modulation? If not, the person cannot babble into the dimensions that matter.
- (iii) A display-trap audit: decode target side from the 0–0.5 s visual-onset window within the manifold latents. If that early visual decodability exceeds the 0.5–3.5 s imagery decodability, the manifold is visually driven and must be rebuilt from ≥ 4 Hz non-occipital terms.

**Caveat:** ds004362 labels are on-screen targets, the data are gel and open loop, and there are about 3 minutes of rest per subject. These data can kill the idea; only the closed loop can confirm it.
