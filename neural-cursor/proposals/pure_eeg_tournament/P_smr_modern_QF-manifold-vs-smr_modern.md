# P_smr_modern (amended after QF vs manifold): mirror-proof SMR control on a dry cap, with held-out-mapping transfer, scaled by hours

## 1. One-line idea and signal basis

**Idea.** Decode mu/beta (8–30 Hz) event-related desynchronisation (ERD) over sensorimotor cortex into continuous 2-D velocity in *body coordinates*. The decoder is a Riemannian plus small-deep ensemble that co-adapts with the user. Two tests establish provenance:
- **Mirror crossover:** the body label is decorrelated from the screen.
- **Held-out-mapping transfer:** at test, the body-to-screen mapping is one never used in training. A decoder that reads body-part intent keeps working. A gaze-by-context reader cannot.

Scaling with hours is then measured in four deep participants and 24 brief ones.

**Signal.** Imagining a hand or foot movement suppresses mu/beta at the contralateral hand area (C3/C4) or at the vertex (feet). Gel precedents: Wolpaw & McFarland 2004 PNAS (four trained, selected users, two with spinal cord injury); McFarland 2010 (3-D); Edelman 2019 Sci Robot (pursuit, robotic arm); Meng 2016 Sci Rep (reach and grasp).

**Honest naturalness position.** The design certifies *instructed body-part intent*, not goal-coordinate intent ("move it there"), which on a visible screen is display-identifiable by construction (the RLNF trap). Body coordinates are the price of a provable claim; we measure how much instruction can be withdrawn (§6).

## 2. Label source (why it is not display-identifiable)

**Classes.**
- x = right vs left hand.
- y = feet (+) vs both hands (−).
- *Relax* is a separate idle state, so "down" no longer collides with "stop".
- The hands/feet contrast is our strongest: 0.710 on the g.tec 32-channel subset of ds004362, 0.719 on the g.tec central 11 channels alone (§10).

**L1: open-loop probes (about 10 % of data).**
- One of four equal-luminance glyphs appears at fixation. The glyph-to-class map is randomised per participant and re-drawn each session.
- No target, cursor or spatial position is shown. Fixation within 1.5° is verified, and decoding windows start 0.5 s after the cue.

**L2: closed-loop co-adaptation.**
- A mapping M carries body-frame velocity to the screen.
- **Training uses only {identity, rotate-180}, 50 % each in every session**, in 12-min epochs announced at epoch start. P(id) = P(rot180) is exactly the condition for zero linear correlation between every screen-direction component and every body-label component.
- Two mappings instead of four every 4 min reduce load; rotate-180 is one consistent remap, practisable like mirror drawing.
- Label = ReFIT-style intended velocity (Gilja 2012) pulled back to the body frame: **M⁻¹·(target − cursor)**.

**Held-out mappings, used at test only:** mirror-x, mirror-y and **axis-swap** (body x → screen y). The participant is told the mapping and has never trained on it.

**The nonlinear leak (conceded, and closed).** M is announced, so it may be visible in the user's cognitive set. A nonlinear decoder could then compute label = f(posterior-alpha side, M-state) and still pass a linear β_screen test. Four defences:
1. **Axis-swap transfer.** No trained mapping ever made screen-vertical attention predict body-x, so a gaze×M reader fails here.
2. **M-decodability.** A classifier predicts M from inter-trial fixation EEG. Its accuracy is reported.
3. **Leak-emulating surrogate.** A nonlinear model (gradient-boosting and the same CNN) is given eye, EOG and display features plus the EEG-decoded M-state. It can therefore learn the exact interaction, and it must stay far below the EEG decoder.
4. **Channel-group double dissociation.** Posterior channels should track screen side and central channels body side, never the reverse.

**L3: free use** feeds only unsupervised re-centring.

## 3. Data engine and scaling mechanism

**(a) Population prior.**
- Sources: Stieger 2021 (62 participants, about 600 h), Forenzo 2024 and ds004362.
- Each subject is re-centred on its own unlabelled rest covariance (Zanini 2018).
- These labels are display-confounded, so they serve as a prior only.
- Repo E2: zero-shot imagined L/R saturates near 0.66 by 32–48 subjects.

**I withdraw the earlier claim that gains come from own data and "not from more people".**
- Repo E2b (25 fixed test subjects, last 15 trials): prior with 0 own trials 0.736 [0.68, 0.79]; prior plus 30 own trials 0.741; 30 own trials alone 0.677.
- Population versus own hours is now a **pre-registered contest**. Fixed test blocks are decoded by (i) the re-centred prior, (ii) the prior fine-tuned on k own hours, and (iii) own data alone. Whichever wins is reported.

**(b) Within-person data.**
- My earlier 0.643 → 0.709 figure came from adding *executed* trials. It is cross-condition augmentation, not more hours of the same imagery.
- Forenzo 2024 reports deep decoders improving as online sessions accumulated.
- The log-hours shape is an unanchored prior, to be estimated rather than assumed.

**(c) User learning.** Vidaurre 2011 (co-adaptive calibration); Perdikis 2018 (mutual learning); Stieger 2021 (cross-session learning).

**Identifying human vs machine (revised).** A frozen decoder confounds learning with dry-cap drift, and nightly retraining mixes data volume with recency. So:
- **Human: L1 separability at fixed n.** Each session's 80 L1 probes are decoded within that session only, after rest re-centring, alongside a Riemannian class-distinctiveness index. This is independent of accumulated machine data.
- **Machine: volume vs recency ladder.** On the same fixed test sessions, train on the last k sessions, on k sessions sampled at random across history, and on all sessions.
  - Volume gain = all minus last-k.
  - Recency gain = last-k minus random-k.
- **Drift floor.** A frozen session-1 decoder with re-centring only is reported as a drift floor, not as "human learning".

**Predicted saturation** (pre-registered): machine ~30–50 own hours; human ~20–30 sessions.

## 4. Expected performance on this hardware

**Anchors.**
- **Wolpaw 2004** (8 targets, timeouts counted as errors): about 0.8–0.9 bits/s for the two best gel users and 0.2–0.35 for the other two.
- **ds004362** (gel, untrained, g.tec 32-channel subset):
  - imagined L/R: 0.643 [0.613, 0.672], with only about 38 % of subjects above the 0.644 floor;
  - feet vs fists: 0.710.
- **g.tec central 11 alone** (run for this rebuttal): L/R 0.635. This equals the denser gel sensorimotor set with FC3/FC4/C1/C2/CP3/CP4 (0.629). The sparse montage is not the bottleneck; L/R imagery itself is weak.

**Derived velocity r (not assumed).**
- Formula: d′ = 2Φ⁻¹(acc), r = d′/√(d′² + 4).
- 0.64 gives r ≈ 0.34; 0.71 gives r ≈ 0.48.
- A 20–40 % dry-cap d′ penalty gives r ≈ 0.21–0.39 untrained.
- No dry-cap anchor exists for trained 2-D control.

| Stage (dry 32-ch, everyone reported) | Metric | Prediction |
|---|---|---|
| Session 1, L1 | hands/feet / 4-class accuracy | 0.60–0.68 / 0.33–0.42 |
| Untrained (derived) | r on y / x | 0.25–0.40 / 0.15–0.30 |
| 10–20 h | 1-D (y) hit rate at 10 s | 70–85 % |
| 30–80 h (no dry anchor) | 2-D r per axis | 0.25–0.40 |
| 30–80 h | 8-target success at 15 s, unassisted | 35–60 % (repo simulation: r = 0.4 → 0.49) |
| Same, with goal-posterior assist | success | 60–80 % |
| Unassisted | bits/s; median time-to-target | 0.05–0.3; 4–8 s |

We expect 15–30 % non-responders, none excluded. If x stays at floor, y gives 1-D control and 2-D runs under shared autonomy, reported as an outcome.

## 5. Purity proof

**Auditors** (recorded, never fed to the decoder): eye tracker ≥250 Hz, H/V EOG, bilateral EMG (forearm flexors/extensors, tibialis anterior, trapezius, masseter), head accelerometer.

**Floors.** Matched-n label permutation and block-wise circular shift, 95th percentile. No performance-based exclusion. **Every retention threshold is stated on the above-floor margin, not on raw accuracy.**

- **C1. Surrogates.**
  - Inputs: eye/EOG; EMG; display variables; their union; and the **leak-emulating surrogate** from §2.
  - Each must reach less than 25 % of the EEG margin.
  - After regressing auditor channels out of the EEG features, the decoder must keep ≥80 % of its margin (an idea taken from the manifold proposal).
- **C2. Mirror crossover.** Frozen decoder output is regressed on body-frame and screen-frame direction, within each M and pooled. Prediction: β_body ≫ 0, β_screen ≈ 0.
- **C2b. Held-out-mapping transfer (headline).** On first exposure to axis-swap, mirror-x and mirror-y, body-frame decoding must keep ≥70 % of the margin seen under trained mappings.
- **C3. Fixation and blind blocks.**
  - Fixation blocks: gaze within 1.5°, verified by the tracker.
  - **Blind blocks** (taken from manifold): a central symbolic glyph gives the target, cursor and target are hidden, and a tone gives the outcome. Run under both identity and rotate-180.
- **C4. Replay.**
  - Passive replay, plus **active replay**, in which the user predicts the replayed cursor's next move. This matches attention without motor intent.
  - In both, the decoder must sit at floor and below 25 % of its closed-loop value.
- **C5. EMG.**
  - Imagery only; no attempted movement by able-bodied participants.
  - Epochs with EMG above rest + 2 SD are excluded from **training and test**, and the fraction is reported.
  - EEG–EMG coherence is reported.
  - An 8–20 Hz-only decoder must keep ≥80 % of the margin.
- **C6. Topography.**
  - The central 11 channels alone must keep ≥80 % of the margin.
  - Deleting FP1/FP2/AF3/AF4/F7/F8 must cost less than 10 % of the margin.
  - **Posterior ablation:** deleting P7/P3/Pz/P4/P8/PO7/PO3/PO4/PO8/Oz must cost less than 10 % on L1 and C2b, and posterior-only must not beat central-only on the body label.
  - Haufe patterns must peak at C3/C4 (Cz for feet).

## 6. Protocol

**Deep cohort.**
- Four participants × about 80 h (2 h/day, 4 days/week, about 10 weeks); robotic arm from week 7.
- Four enrolled, so one dropout leaves three; falsifiers use mixed models over completers (≥20 sessions).

**Broad cohort.**
- 24 participants × 5 sessions × 1.5 h.
- Four more held out; their last two sessions are the fixed test set on every ladder.

**Total.** About 530 h of own dry-cap data.

| min | Block |
|---|---|
| 0–10 | Setup, impedance, line-noise QC |
| 10–15 | Rest, eyes open and closed (re-centring) |
| 15–25 | L1: 4 classes × 20 trials × 4 s |
| 25–61 | L2: 3 × 12-min epochs, identity/rotate-180 balanced, centre-out plus pursuit |
| 61–73 | Frozen test: C2, C2b, then C3 fixation or blind (rotating) |
| 73–80 | C4 replay, passive or active (alternating) |
| 80–90 | "Just intend": **always identity**, frozen decoder; then strategy questionnaire and NASA-TLX |

**Imagery withdrawal.** Imagery instructions are dropped from L2 after session 10 (deep cohort) or session 3 (broad cohort). In rotated epochs the participant is told only "the display is flipped".

**Optional gel arm.** One 64-channel gel session per month per deep participant, to measure the dry-cap penalty (€20–40k).

## 7. Real-time architecture

- **Input:** LSL 500 Hz; causal IIR bank of five 8–30 Hz bands; 1.0-s windows every 62.5 ms; latency about 300–500 ms.
- **Model A:** OAS covariances per band, running Riemannian re-centring (τ = 5 min), tangent space (2,640 features), ridge regression to body-frame velocity, and a logistic idle detector.
- **Model B:** EEGNet-class CNN, pre-trained on the pooled corpus, fine-tuned nightly; its ensemble weight is fitted on held-out blocks and reported.
- **Output filter:** Kalman smoothing; the idle probability gates velocity to zero.
- **Adaptation:**
  1. Unsupervised re-centring.
  2. CLDA (Orsborn 2014) every 3 min on a **rolling 24-min buffer, reweighted so identity and rotate-180 carry equal weight**. The first epoch of a session borrows the previous session's tail.
  3. Nightly retrain.
- Test blocks are always frozen.
- **Effector.**
  - Cursor: v_screen = g·M·v_body.
  - Arm: planar end-effector velocity ≤ 10 cm/s.
  - Grasp, release and z-plane switching are **dwell actions**: 1.5 s idle within 3 cm of an object or mode zone. No class carries two meanings.
  - Generality: 2 continuous DoF plus dwell events.

## 8. The Nature claim

**Headline figure.**
- **(a)** Held-out-mapping transfer on first exposure, against the leak-emulating surrogate and floors.
- **(b)** β_body vs β_screen, with the posterior↔screen / central↔body dissociation.
- **(c)** Control vs own hours on fixed tests: L1 separability (human), the volume/recency ladder (machine), and the population-vs-own contest.
- **(d)** Robot pursuit and topographies.

**The sentence.** "On a wearable dry EEG cap, a co-adaptive decoder reads imagined body-part intent so specifically that it controls a cursor or robot through a body-to-screen mapping it has never seen, which excludes eyes, muscles, display and context. We quantify what more hours, more people and the person's own learning each contribute."

**Why new.** SMR control is not new, and performance will stay below the best 2004 gel users. What is new:
1. Zero-shot held-out-mapping transfer as a provenance proof that closes linear and nonlinear display leaks. No EEG cursor study we know of has done this.
2. A dry-cap scaling decomposition of volume, recency, population and user learning on fixed tests.
3. A measured imagery-withdrawal endpoint.

Honest venue: Nature Communications likely; Nature needs (a) to hold and (c) to keep climbing.

## 9. Falsifiers and risks

All falsifiers are evaluated on frozen tests verified by fixation or blind presentation.
- **F1 (not cortex).** In ≥2 completers: held-out margin < 50 % of the trained margin, or β_screen ≥ β_body, or the leak-emulating surrogate reaches ≥ 50 % of the EEG margin.
- **F2 (circularity).** Active-replay correlation ≥ 50 % of the closed-loop value.
- **F3 (no scaling).** In a mixed model with random participant slopes, the fixed slope of success on log(hours) over 10–80 h has a 95 % CI that includes 0, **and** fewer than 2 completers have individually positive slopes.
- **F3b (population wins).** If the re-centred population prior alone matches the own-data decoder at 80 h, the own-data bitter lesson is falsified for SMR on this cap.
- **F4 (not natural).** After 30 h, "just intend" (identity mapping) reaches less than 50 % of cued performance.

**Biggest risk.** Left/right imagery was below floor in about 60 % of untrained gel users, so the dry-cap x axis may stay near floor. Mitigations: the hands/feet y axis gives 1-D control; the gel arm measures the dry penalty; shared autonomy is reported separately; every participant is reported.

**Second risk.** Rotated epochs cost effort and may slow automaticity; measured (identity vs rotate-180, NASA-TLX).

## 10. ds004362: done, and next

**Done** (repo `results/realdata/`): E1, E2 and E2b are summarised in §3–4. E5 (idle detection, 0.844) is discounted as display-confounded. **E3** (imagined L/R, 8–30 Hz): sensorimotor 0.629, **occipital 0.604**, ocular-frontal 0.507; slow 1–4 Hz ocular-frontal features reach 0.566, so the display trap is real in public data.

**Run for this rebuttal** (8–30 Hz, 0.5–3.5 s, tangent space + logistic regression, 5-fold × 3 seeds, 106 subjects):

| g.tec subset | imagined L/R | imagined fists/feet |
|---|---|---|
| all 32 | 0.643 [0.613, 0.672] | 0.710 [0.680, 0.739] |
| central 11 | 0.635 [0.603, 0.668] | 0.719 [0.689, 0.750] |
| no posterior (22) | 0.619 [0.590, 0.650] | 0.690 [0.657, 0.719] |
| posterior only (10) | 0.617 [0.593, 0.640] | 0.675 [0.649, 0.704] |
| no ocular-frontal (26) | 0.653 [0.624, 0.682] | 0.724 [0.695, 0.754] |

**Reading.** Central channels carry the signal and removing frontal ones *helps*, but posterior channels nearly match central ones when targets are shown left/right. ds004362 cannot separate cortex from display; the mirror, held-out-mapping, posterior-ablation and blind-block controls can.

**Next, each with a kill rule.**
1. Time-resolved decoding: kill if accuracy peaks before 0.5 s.
2. C3/C4 ERD sign flip between hands, against posterior alpha following target side. A double dissociation is predicted.
3. E6 sequential retraining with a volume-vs-recency ladder.
4. Week 1–2 deep-cohort pilot of M-decodability and axis-swap transfer. Redesign if the transfer margin is below 30 % in all participants.
