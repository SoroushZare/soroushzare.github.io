# P_smr_modern (SF2): auditor-bounded SMR control on a dry cap, certified by held-out-mapping transfer and blind "just intend", scaled by hours

## 1. Idea and signal basis

**Idea.** Decode mu/beta (8–30 Hz) ERD over sensorimotor cortex into continuous 2-D velocity in *body coordinates*. The decoder is a central-montage Riemannian plus small-CNN ensemble that co-adapts with the user. Each endpoint is certified *in its own condition* by three tests:
1. **Held-out-mapping transfer.** The decoder is tested under rotate ±90°, which it never trained on, and scored against the body part the user *reports*.
2. **Leak signature.** Under those mappings, a gaze×mapping reader predicts a specific *wrong* output. We measure that output directly.
3. **Blind "just intend".** The naturalness endpoint is also run with no target or cursor on screen.

Hours-scaling is measured in 4 deep and 28 brief participants.

**Signal.** Imagined hand or foot movement suppresses mu/beta at C3/C4 (contralateral hand) or at the vertex (feet). Gel precedents: Wolpaw & McFarland 2004 PNAS; McFarland 2010; Edelman 2019 Sci Robot; Meng 2016 Sci Rep.

**Naturalness.** We certify *body-part intent*, not goal-coordinate intent. On a visible screen, goal-coordinate intent is display-identifiable by construction. Instruction is then withdrawn, and the withdrawn state is certified blind (C3b).

## 2. Label source

**Axes.** As in Wolpaw & McFarland 2004 (weighted sum and difference of left and right sensorimotor amplitudes):
- x ≈ C4-minus-C3 contrast (right vs left hand);
- y ≈ vertex-vs-bilateral contrast (feet + / both hands −).

Both-hands imagery is symmetric, so it projects to x ≈ 0. **Conceded:** diagonals such as up-left are not a natural mental act, so independence is *measured* (C7). *Relax* is a separate idle state.

**L1: open-loop probes (~10 % of data).** A glyph at fixation cues one of four classes; the glyph-to-class map is re-drawn each session. No spatial information is shown. Gaze is verified within 1.5°, and windows start 0.5 s after the cue.

**L2: sighted closed loop.**
- Mapping M carries body velocity to the screen. Training uses only {identity, rotate-180}, in announced 12-min epochs.
- Label = ReFIT intended direction (Gilja 2012) pulled back to the body frame: **u = M⁻¹·(target − cursor)/‖·‖**.
- **Moment matching (fix).** Equal time does not equalise moments, because rotate-180 performs worse. The CLDA buffer is importance-weighted over direction (8 bins) × distance (3) × time-in-trial (3) × M, so that E[ssᵀ|id] = E[ssᵀ|rot180].
- **Label audit.** On every buffer, the linear and gradient-boosted predictability of u from screen variables, with M withheld, must sit at the permutation p95.
- **Conceded:** with *oracle* M, any sighted closed-loop label is fully gaze-predictable. Purity therefore cannot rest on the label. It rests on three things:
  - (i) the decoder cannot see where the leak lives: central montage only, 8–30 Hz (§7);
  - (ii) fixation-verified L1 and blind L2b supply labels too;
  - (iii) every endpoint is tested where gaze×M predicts the *wrong* answer.

**L2b: blind closed loop (new, ~15 % of training).** The target is a spoken word or central glyph. The cursor is hidden, and feedback is a tone plus a 0.5-s end-point flash. The label is the instructed body direction.

**Held-out mappings** never enter training, CLDA or nightly fits. Rotate ±90° swaps both axes and is primary. Mirror-x and mirror-y are secondary, because each leaves one axis unchanged (conceded).

**L3: free use** feeds only unsupervised re-centring.

## 3. Data engine and scaling

**(a) Population prior.**
- Sources: Stieger 2021 (62 participants, ~600 h), Forenzo 2024 and ds004362, each re-centred on its own rest data (Zanini 2018).
- The prior is **restricted to the central montage**, and it must pass C1–C6 *on its own* before use, because its labels are display-confounded.
- Repo E2: zero-shot L/R saturates near 0.66 by 32–48 subjects.

**(b) Own hours: a contest, not an assumption.**
- Repo E2b: prior with 0 own trials 0.736, with 30 own trials 0.741, own data only 0.677.
- Repo E6: own-from-scratch catches the prior after ~40 trials (0.698 vs 0.702).
- Both use minutes of gel data and say nothing about tens of hours. The earlier 0.643 → 0.709 gain came from executed-trial augmentation and is withdrawn.
- On fixed tests we compare (i) the prior, (ii) the prior fine-tuned on k own hours, and (iii) own data only. The winner is reported.

**(c) "More robust", as the PI's hypothesis states.**
- **Robustness endpoints:** first-block-of-day performance without recalibration; across-session SD; loss after a cap re-fit.
- **Machine:** volume (all sessions) vs recency (last k) vs random-k on fixed test sessions, stratified before and after imagery withdrawal.
- **Pooled-hours ladder:** 25/50/100/180 broad-cohort dry hours, tested on 8 held-out people. Its power comes across people, not from 3–4 individual slopes.
- **Human:** L1 separability at fixed n, from 120 probes per session (4-class SE ≈ 0.045), with impedance and cap-fit covariates.
- **Drift floor:** a frozen session-1 decoder with re-centring only.

**Pre-registered saturation:** peak accuracy at ~30–50 own hours, with robustness still improving to ~80 h. The deep arm is meant to *locate* the plateau.

## 4. Expected performance

**Anchors (corrected).**
- The full-cap ds004362 numbers contain display information, so they are upper bounds.
- The new anchors use the **decoder montage** (central 11, §10), bracketed by its beta-only version:
  - fists/feet 0.700 (beta-only 0.673) → r ≈ 0.46 (0.41);
  - L/R 0.631 (0.591) → r ≈ 0.32 (0.22).
- These are still gel, untrained, upper-bound figures.
- Formula: d′ = 2Φ⁻¹(acc), r = d′/√(d′²+4).
- The dry penalty has **no source (conceded)**. A 20–40 % loss in d′ is a planning prior only; the gel arm measures it.
- Wolpaw 2004 (8 targets): ~0.8–0.9 bits/s for the best two gel users and 0.2–0.35 for the other two.

| Stage (dry, everyone reported) | Metric | Prediction |
|---|---|---|
| Session 1 L1 | hands/feet; 4-class | 0.60–0.68; 0.33–0.42 |
| Untrained | r on y; r on x | 0.26–0.39; 0.14–0.26 |
| 10–20 h | 1-D hit rate at 10 s | 70–85 % |
| 30–80 h | r per axis | 0.25–0.40 |
| 30–80 h | 8-target success at 15 s, unassisted (assisted) | 30–60 % (60–80 %) |
| Unassisted | bits/s; time-to-target | 0.05–0.3; 4–8 s |

**Conceded:** this does not exceed the project's established 0.03–0.3 bits/s. The contribution is the first *auditor-bounded*, scaling-decomposed version of that rate. We expect 15–30 % non-responders, none excluded. If x stays at floor, the result is 1-D control plus shared autonomy, reported as such.

## 5. Purity proof

**Auditors** (no socket to the decoder): eye tracker ≥250 Hz, H/V EOG, bilateral EMG (forearm, tibialis anterior, trapezius, masseter), accelerometer.

**Floors.** Matched-n permutation and circular shift, 95th percentile. No performance-based exclusion. All thresholds are stated on the above-floor margin.

**Wording (adopted from vocabulary).** The claim is "auditor-bounded": we report the upper 95 % CI of the fraction of the margin attributable to recorded eye, muscle and display sources.

- **C1. Surrogates.**
  - Inputs: eye/EOG, EMG, display, their union, and an **oracle-M leak surrogate** (gradient boosting and the same CNN, given auditors, display and the *true* M).
  - All are scored on the **test conditions** (held-out mappings, blind and fixation blocks), where gaze×M predicts the wrong label. Each must stay below 25 % of the EEG margin. On L2 the oracle surrogate is near-perfect by construction; this is reported.
  - **Nonlinear conditional test (adopted):** gradient boosting predicts decoder output from auditor features. It must explain < 10 % of the variance, and the residual must keep ≥ 80 % of the margin.
- **C2. Mirror crossover.** Prediction: β_body ≫ 0, β_screen ≈ 0.
- **C2b. Held-out-mapping transfer (headline; redesigned).**
  - Schedule: rotate ±90° for 8 min per session from session 3 (~1,000 trials per deep participant), plus broad sessions 4–5. First exposure is reported descriptively only.
  - **The decoder never trains on these blocks.** Its novelty, not the human's, is what tests provenance.
  - **Remap failure is separated from impurity.** Each trial is 3 s of open-loop imagery (cursor frozen), followed by a report of the body part imagined on a randomised key. The score is **decoder vs report**, so a remap failure is still a labelled trial.
  - **Leak signature.** A {id, rot180}-trained gaze×M reader maps screen-vertical targets to body-y under ±90°. Decoder output is regressed on the body label and on the leak-predicted label. Pass: β_body ≥ 70 % of the trained margin **and** the upper CI of β_leak < 10 %. This replaces the bare 70 % rule.
  - **Power.** The strong y axis is tested here: hands/feet imagery now moves the cursor *horizontally*.
- **C3. Fixation and blind blocks** under identity and rotate-180.
- **C3b. Blind "just intend" (new).** The no-instruction block alternates sighted and blind (spoken target, hidden cursor), always under identity. **Naturalness (F4) is scored on the blind half**, with C1 and C6 run on the same trials.
- **C4. Circularity.**
  - Passive and active replay: decoder output at floor and below 25 % of its closed-loop value.
  - **Freeze probes (adopted):** the cursor freezes for 2 s in 10 % of windows. Decoder output must not change beyond its CI.
- **C5. EMG (fixed).**
  - Imagery only. All epochs are kept in training and test.
  - **EMG- and gaze-silent subset (adopted):** max-over-channels EMG RMS below its rest 97.5th percentile. Being family-wise, this rejects ~2.5 % of clean epochs, not ~22 %. Gaze must also stay within 1.5°. Pass: ≥ 80 % of the margin retained. The subset fraction is reported.
  - An 8–20 Hz-only decoder must keep ≥ 80 % of the margin.
- **C6. Topography.** A decoder built only from the channels *excluded* from the montage must stay below 25 % of the margin on C2b and C3b. Haufe patterns must peak at C3/C4 (Cz for feet).
- **C7. Axis independence (new).** Pass: decoded x and y errors have |ρ| < 0.3, and diagonal-target success is ≥ 70 % of cardinal. On failure we report "2 sequential DoF" and time-multiplex the axes.

## 6. Protocol

**Deep cohort.** 4 participants × ~80 h (2 h/day, 4 days/week); arm from week 7.

**Broad cohort.** 28 participants × 5 × 1.5 h; 8 are held out as the fixed test set. Total ≈ 530 h of dry-cap data.

| min | Block |
|---|---|
| 0–15 | Setup, impedance, cap-fit, rest |
| 15–28 | L1, 120 probes |
| 28–58 | L2 (weighted id/rot180) + L2b blind |
| 58–66 | C2b rotate ±90°, frozen, with reports |
| 66–74 | C3 fixation/blind or C4 replay |
| 74–86 | "Just intend": identity, sighted/blind, frozen |
| 86–90 | Strategy questions, NASA-TLX |

**Imagery withdrawal** after session 10 (deep) or session 3 (broad).

**Rotate-180 load (conceded).** Opposing mappings can interfere (Krakauer et al. 1999, 2005; contested by Caithness et al. 2004). Our remap is explicit re-aiming, which can become cached with practice (McDougle & Taylor 2019 Nat Commun). After session 10 its time drops from 50 % to 30 %; importance weights keep the moments balanced at the cost of effective n. Automaticity is tracked with NASA-TLX and a 1-in-10 auditory-probe reaction time.

**Gel arm.** One 64-channel session per month per deep participant (€20–40k).

## 7. Real-time architecture

- **Montage: the g.tec central 11** (FC5/FC1/FC2/FC6, C3/Cz/C4, CP5/CP1/CP2/CP6). No posterior, FP/AF, F or temporal channels, so no component can see posterior alpha side, frontopolar EOG or temporalis EMG.
- LSL 500 Hz; causal five-band 8–30 Hz IIR; 1-s windows every 62.5 ms; latency ~300–500 ms.
- **Model A:** OAS covariances, running re-centring (τ = 5 min), tangent space, ridge to body velocity, logistic idle detector.
- **Model B:** EEGNet on the same montage and bands, pre-trained on the central prior, fine-tuned nightly. It is kept only if it passes C1/C2b/C6 alone.
- **Ensemble weight** is fitted on fixation, blind and L1 blocks only, never on sighted identity blocks (fix).
- **Output and adaptation:** Kalman filter with idle gating; re-centring; CLDA every 3 min on a moment-matched, label-audited 24-min buffer (L2 + L2b); nightly retrain. Test blocks are frozen.
- **Effector:** v_screen = g·M·v_body. The arm moves planar end-effector velocity ≤ 10 cm/s. Grasp, release and z-switching are dwell actions (1.5 s idle within 3 cm).

## 8. The Nature claim

**Figure.**
- (a) C2b decoder-vs-report transfer, with β_leak ≈ 0 beside the oracle-M surrogate.
- (b) Blind vs sighted "just intend" across hours.
- (c) Peak and robustness vs hours: volume/recency, pooled-hours ladder, population-vs-own contest, human L1.
- (d) Robot pursuit and topographies.

**Sentence.** "On a wearable dry EEG cap, a co-adaptive decoder reads imagined body-part intent through body-to-screen mappings it never trained on, and with no visible target, with an auditor-bounded limit on eye, muscle and display contributions. We measure what hours, people and user learning each add."

**Novelty, honestly stated.** Neither the paradigm nor the rate is new. What is new:
1. A held-out-mapping, leak-signature provenance test for closed-loop EEG control.
2. Naturalness certified blind.
3. A dry-cap decomposition of scaling and robustness.

The likely venue is Nature Communications.

## 9. Falsifiers and risks

- **F1 (not cortex).** In ≥ 2 completers, on C2b (vs report) or C3b blind, any of:
  - β_body < 50 % of the trained margin;
  - the upper CI of β_leak > 10 %;
  - any C1 surrogate ≥ 50 % of the EEG margin.
  Remap failure cannot trigger F1, and its rate is published.
- **F2 (circularity).** Active replay or freeze probes ≥ 50 % of the closed-loop value.
- **F3 (no scaling).** Both conditions hold:
  - the CI of the pooled-hours ladder slope (bootstrap over 8 held-out people) includes 0;
  - not every deep completer has a session-permutation-significant positive slope in peak or robustness.
- **F3b.** The central prior alone matches own data at 80 h.
- **F4 (not natural).** After 30 h, *blind* "just intend" is < 50 % of cued performance.

**Risks.**
- **Dry-cap L/R at floor.** About 60 % of untrained gel users in ds004362 were. We would then have 1-D y control, and C2b keeps its power through the y-to-screen-x swap.
- **Rotate-180 slows automaticity.** This is measured, and its share is reduced (§6).

## 10. ds004362: done, new, next

**Done** (repo `results/realdata/`). Method: 8–30 Hz, 0.5–3.5 s, tangent space + logistic regression, 106 subjects, 3 seeds.

| g.tec subset | L/R | fists/feet |
|---|---|---|
| all 32 | 0.643 | 0.710 |
| central 11 | 0.635 | 0.719 |
| posterior 10 | 0.617 | 0.675 |
| no ocular-frontal | 0.653 | 0.724 |

E3: slow ocular-frontal features reach 0.566, so the display trap is real.

**New for this rebuttal** (scratchpad `sf2_smr/an.py`). Method: 1 seed, 5-fold, 106 subjects. Posterior = P7/P3/Pz/P4/P8/PO7/PO3/PO4/PO8/Oz.

| | L/R | fists/feet |
|---|---|---|
| central 8–30 Hz | 0.631 [0.600, 0.666] | 0.700 [0.666, 0.732] |
| central 16–30 Hz | 0.591 | 0.673 |
| posterior 8–30 Hz | 0.604 | 0.662 |
| **posterior, central regressed out (train-fold ridge)** | **0.529 [0.510, 0.548]** | **0.541 [0.523, 0.560]** |
| permutation floor | 0.491 | 0.495 |
| 0–1 s, central / posterior | 0.605 / 0.579 | 0.554 / 0.564 |
| 1–2 s, central / posterior | 0.588 / 0.569 | 0.667 / 0.620 |

**Reading (partly concedes the attack).**
- About 70 % of the posterior margin is shared with central channels (28–34 % survives residualisation).
- The surviving part is above floor and plausibly display or attention, and the central montage excludes it.
- Fists/feet: posterior leads in the first second, then central dominates, as ERD build-up predicts.
- L/R: central information is near its peak already at 0–1 s, so part of it may be cue-locked or display-driven. This is one reason C2b scores against reports.

**Next, with kill rules.**
1. Week 1–2 C2b pilot: redesign if β_body < 30 % of the margin in all participants.
2. Moment-matching simulation in the repo simulator, with a rotate-180 deficit: kill it if residual |corr| > 0.05.
3. E6 volume-vs-recency ladder.
