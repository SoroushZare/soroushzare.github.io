# P_generative: generative inverse decoding with a per-person world model of EEG

## 1. One-line idea and signal basis

**Idea.** Train one conditional generative model of a person's dry-cap EEG, p(EEG_t | EEG_past, effector state, candidate goal g). Decode intent by Bayesian inversion: every 100 ms, score each candidate goal by how well it explains the incoming EEG, then move the cursor or robot toward the posterior. While the goal is still uncertain, the effector also makes **information-seeking probe moves** chosen so that the brain's reaction to them is diagnostic.

**Signals.** No single paradigm is chosen. Goal-dependent cortical processes all enter one likelihood:

| Relational feature | Phenomenon | Locus | Evidence |
|---|---|---|---|
| Last move toward or away from *my* goal | Expectation-violation / error negativity and positivity, 200–600 ms | FCz/Cz | Zander et al. 2016 PNAS; Chavarriaga 2014; Iturrate 2015 |
| Hemifield of my goal | Posterior alpha lateralization | PO/O | Worden 2000; Thut et al. 2006 J Neurosci |
| Distance to goal | Slow negativity / CNV | Cz | Walter 1964 |
| Desired direction | Low-frequency directional tuning | C3/Cz/C4 | Kobler et al. 2018 Sci Rep; Pulferer et al. 2022 JNE |

It is "thinking" because the user only *wants* the effector somewhere and watches it. There is no imagery instruction and no stimulation.

**Closed-loop precedent.** Row 1 has one. In Zander et al. 2016, a random cursor on a 4×4 grid needed about 27 moves to reach a target and a perfect cursor needed 10. Their neuroadaptive cursor needed about 13, closing 82% of that gap with no explicit commands. The group's earlier offline classification averaged 71% (16 subjects, binary).

**What I add:**
- (a) labels that provably carry no display information;
- (b) one likelihood that also absorbs the weaker rows 2–4;
- (c) a model that improves with every hour of recording, labeled or not.

## 2. Label source: never display-identifiable

**The problem with prior labels.** In Zander 2016 and Iturrate 2015, as I read them, the "toward/away" label was a function of the displayed or instructed target. That is the RLNF trap.

**My labels:**
1. **Private goal.** The screen shows K identical, unlabeled placeholders: 8 on a ring at 8° eccentricity, later a 64-point grid. The participant silently picks one. A choice counter encourages balanced choices.
2. **Goal-blind display in label blocks.** The cursor follows a goal-blind probe policy: random directions, or trajectories replayed from other trials. The model conditions on the exact cursor state, and I(display; g) = 0 by construction.
3. **Late report.** After the trial ends and a 1.5 s blank, the participant clicks the chosen placeholder. Report-related EEG, EMG and gaze fall outside every decoding window.
4. **Where goal information can come from.** Only the *interaction* between the visible cursor and the private goal carries it, for example an error response when the cursor moves away. The model has a background term f(EEG_past, cursor) that absorbs all display-driven responses. A separate goal-relational term supplies the evidence that discriminates between goals.
5. **Closed-loop blocks.** Here the cursor depends on decoded intent, so these blocks train only the goal-marginalized background term. The goal term is fit only on independent-display blocks, which blocks the "decoder reads its own visual consequence" loop.

**Remaining label noise.**
- Free choice is sequentially biased. A history-only prior p(g | history) is modeled and used as a surrogate floor.
- Changes of mind add label noise. Catch trials, where the participant reports twice, estimate how much.

## 3. Data engine, scaling mechanism, expected curve

**Why generative should scale where discriminative stalled.** RLNF's 24 decoders and 6 deep nets were all equivalent, and only data *volume* helped. A discriminative decoder learns from labeled trials only, and at this SNR most of its error is unexplained background variance.

A generative model also learns from every unlabeled second through the goal-marginalized likelihood: rest, closed-loop use, free use and other people's data. That likelihood models:
- gain drift;
- alpha and arousal fluctuations;
- blink and EMG bursts as they appear in EEG (the auditors are never inputs);
- session offsets.

The goal term is a small residual whose log-likelihood ratio sharpens as the background predictive variance shrinks. The label-hungry part is small and the data-hungry part is self-supervised. That is the PI's bitter lesson, applied to an objective that can actually absorb hours.

**Figure of merit.** I_goal = E_t[log p(x_t | x_<t, c_t, g*) − log Σ_g p(g) p(x_t | x_<t, c_t, g)] (nats/s). This is the conditional mutual information between EEG and the private goal, with no threshold or classifier choice.
- **Fixed test set:** every 10th session per participant, interleaved across the whole study, never trained on.
- **Ladder:** 1, 2, 5, 10, 20, 50 and 100 h of own data, crossed with 0, 30 and 120 h of others' unlabeled data. Each rung has a permutation floor at matched n.

**Expected curve.** This is a prediction, not a measurement. I expect roughly log-linear growth from 2 to 30–50 h, then saturation at the neural ceiling of the evaluation and attention signals.

**Anchors.**
- Kaifosh et al. 2025 Nature: sEMG decoders improved with more training participants.
- NDT2 (Ye et al. 2023 NeurIPS) and POYO (Azabou et al. 2023): self-supervised pretraining on spikes helps, with diminishing returns.
- RLNF: volume was the only thing that helped.

**Counter-evidence.** EEG foundation models (BENDR, LaBraM, EEGPT, CBraMod) give only modest downstream gains. Section 10, test T2, checks whether background pretraining transfers to goal information *before* we commit.

## 4. Expected performance (dry 32-ch)

**Arithmetic.**
- **Evaluation:** an AUC of 0.68–0.72 per probe move (matching Zander 2014's 71% and our simulation's 0.68) gives 1 − H(0.71) ≈ 0.13 bits per probe. At one probe per 0.75 s, that is at most about 0.17 bits/s, and only if the probes split the posterior optimally.
- **Attention:** assuming 58% hemifield accuracy per second, this adds about 0.02 bits/s.
- **Anticipation and preparation:** small under fixation.

| Metric (deep participants, 100 h, fixation) | Expected | Floor |
|---|---|---|
| I_goal | 0.08–0.20 bits/s | permutation and auditor surrogates |
| 4 goals, time to 90% posterior | 12–25 s | random probing alone does not converge |
| 8 goals, success within 60 s | 65–85% | 12.5% plus prior surrogate |
| Generative vs discriminative (xDAWN+Riemann classifier in the same filter) | +20–60% bits/s at ≥20 h; ≈ equal at ≤5 h | — |
| Broad cohort (cross-subject model + 2 h own) | 0.03–0.10 bits/s | — |

This sits inside the project's established 0.03–0.3 bits/s band. **The headline is purity plus a measured scaling law, not speed.**

## 5. Purity proof

- **Auditors (recorded, never decoder inputs).**
  - Eye tracker at ≥500 Hz.
  - Bipolar EOG on a separate amplifier.
  - EMG: temporalis, masseter, SCM/trapezius, forearm.
  - Head accelerometer.
  - Screen photodiode.
- **Fixation blocks.** A 2° online window. Any gaze exit pauses the trial regardless of direction, so the pause itself carries no goal information.
  - Primary analysis: all trials.
  - Secondary: saccade-free trials.
  - No performance-based exclusion.
- **Matched surrogates.** The *identical* inversion pipeline, on the same trials, is trained on:
  - (i) auditors only;
  - (ii) the screen stream (rendered frames plus cursor log);
  - (iii) choice history only.
  
  (ii) must sit at the floor by construction, and (i) at the floor under fixation. EEG must beat each surrogate and their union.
- **Regress-out.** Linearly removing auditor regressors (lags −200 to +200 ms) from the EEG must leave at least 70% of I_goal. So must dropping FP1, FP2, F7 and F8.
- **Pixel-identical replay (the key control).** A closed-loop run aimed at goal g_A is replayed frame-for-frame to the same participant, who now privately picks g_B ≠ g_A.
  - A display-reading decoder recovers g_A. An intent decoder recovers g_B.
  - Statistic: mean log P(g_B)/P(g_A) > 0, pre-registered, over at least 200 replay trials per deep participant.
- **Statistics.** 1,000-permutation floors at matched n, all analyses pre-registered.
- **Physiology.** Topographies of each channel's likelihood gradient should put evaluation evidence near FCz and attention evidence in parieto-occipital alpha. The optional 64-ch gel sessions run only if the lab already owns an amplifier; I request no extra cost.

## 6. Protocol

**Participants.**
- **Deep cohort:** 4 participants × 100 h, in 50 two-hour sessions each at 2–3 sessions per week. That gives about 8,000 labeled trials per person.
- **Broad cohort:** 30 participants × 2 sessions × 2 h.
- **Total:** about 520 h of data.

**Session (120 min).**

| Segment | Duration | Notes |
|---|---|---|
| Setup and quality check | 15 min | — |
| Rest, eyes open/closed | 5 min | Unlabeled |
| A. Label blocks, fixation | 2 × 10 min | About 30 trials of 20 s each |
| B. Label block, free viewing | 10 min | Never used for purity claims |
| C. Closed-loop | 2 × 10 min | — |
| D. Pixel replay | 10 min | From session 10 onward |
| E. Robot | 10 min | Deep cohort, months 8–14 |
| Free use | 10 min | Unlabeled |
| Breaks | 10 min | — |

**Retraining.** Nightly on all data. The fixed test set is evaluated weekly.

**Timeline.**
- Months 1–3: build, ds004362 analyses, 2 pilots.
- Months 3–10: deep cohort.
- Months 6–12: broad cohort.
- Months 8–14: robot and replay.
- Months 14–18: writing.

**Staff:** 2 RAs and 1 ML engineer.

## 7. Real-time architecture

- **Acquisition and preprocessing.** 500 Hz over LSL; causal 0.1–40 Hz filter; robust common average reference.
- **Features, per channel.** 0.3–8 Hz amplitude at 50 Hz, plus log theta, alpha and beta power at 10 Hz.
- **Model.**
  - Causal transformer: 6 layers, d = 256, about 6 M parameters, 4 s context.
  - Conditioning: effector position and velocity, time since each probe onset, a 16-dim session embedding, and relational goal features (angle between move and goal, distance, hemifield, time-to-arrival).
  - Output: the next 100 ms of features as a Student-t distribution (robust to dry-electrode spikes) with low-rank-plus-diagonal covariance.
- **Inversion.** All K ≤ 64 goals are scored in one batched pass every 100 ms, at under 15 ms on a laptop-class GPU. A goal-switch hazard of 0.01/s lets the posterior follow changes of mind.
- **Probe policy.** Each step maximizes the expected information gain about g, blended toward the posterior mean with weight α = max posterior.
- **Adaptation.**
  - Every 10 s: update the session embedding and per-channel gain/offset by gradient ascent on the *goal-marginalized* likelihood. No labels are used, so it cannot learn display-to-goal shortcuts.
  - Nightly: full retraining, with the goal term fit only on independent-display blocks.
- **Effector interface.** The decoder outputs a goal posterior in effector-relative coordinates.
  - The cursor uses it directly.
  - For a 6-DoF arm (xArm or Kinova), candidate goals are object poses from a fiducial-tag scene camera. This is environment perception, not user input. Probe moves are 3–5 cm, and the planner executes toward the posterior.
- **Generality.** The relational features are the same for every effector, so one EEG model drives both. I will test zero-shot cursor→robot transfer.

## 8. The Nature claim

**One sentence.** *A generative model of one person's scalp EEG, inverted over candidate goals, infers where they privately want a cursor or robot to go, without stimulation, instruction or eye input. Its information grows with hours of recording, and it follows the private goal even when the screen is pixel-identical to a run aimed elsewhere.*

**Headline figure.**
- (a) I_goal against log hours for the 4 deep participants, with permutation, auditor, screen and prior floors, with and without pretraining.
- (b) The pixel-replay dissociation.
- (c) The split of information across evaluation, attention, anticipation and preparation, with topographies.
- (d) One posterior steering both the cursor and a robot arm to the privately chosen object.

**Why it is new.**
- Prior EEG goal inference (Zander 2016, Iturrate 2015) used targets that were displayed or instructed, and single-signal discriminative classifiers.
- Model-based decoders were intracortical or fMRI, and open-loop or stimulus-driven: Srinivasan et al. 2006 Neural Comput, Yu et al. 2007 J Neurophysiol, and encoding-model identification in Kay et al. 2008 Nature.
- To my knowledge, nobody has shown non-invasive intent decoding with display-independent labels, a pixel-replay dissociation, or a within-person scaling law for intent information driven by self-supervised background modeling.

## 9. Falsifiers and biggest risk

**Falsifiers (pre-registered).**
- **F1.** In pixel replay, log P(g_B)/P(g_A) ≤ 0 → the decoder reads the display, and the idea is dead.
- **F2.** In fixation blocks at 100 h, EEG I_goal does not exceed both the permutation floor and the best auditor surrogate (95% CI).
- **F3.** The 95% bootstrap CI on the I_goal slope from 10 h to 100 h includes 0 → no bitter-lesson effect.
- **F4.** Generative inversion never beats the discriminative classifier in the same filter → drop the generative framing. The purity and replay results still stand.

**Biggest risk.** Nearly all the information comes from the evaluation response, and two things limit it:
- It may **habituate** over 100 h as probes become expected. Mitigations: jittered probe timing, tracking amplitude drift in the background term, and reporting habituation per session as a result.
- Its rate is capped at about 0.15 bits/s by the probe rate.

**Secondary risks.**
- Dry-electrode noise at 1–8 Hz.
- Likelihood capacity spent on background structure that carries no goal information. T2 below tests this.

## 10. Tests on ds004362 now

**Dataset.** 109 subjects, 64-ch, 160 Hz, about 26 min each (two 1-min baselines plus 12 two-minute runs), roughly 47 h in total.
- Runs 3/4/7/8/11/12: a target appears on the left or right, and the subject executes or imagines the matching fist.
- Runs 5/6/9/10/13/14: top = both fists, bottom = both feet.
- **The targets are displayed, so this dataset is itself the trap.** I use it to calibrate the purity pipeline.

**Setup.**
- Use the 32 channels closest to the g.tec layout.
- Fix a 19-subject test set up front.
- Exclude recordings only from an a-priori list of known timing irregularities, never by performance.

**Tests.**
- **T1 (does inversion work?).** Fit the class-conditional predictive model, invert it, and compare against FBCSP+LDA, Riemannian tangent space and EEGNet, at 10, 20 and 45 trials per class and leave-subjects-out.
  - *Kill:* generative is more than 5 points worse than the best discriminative model at every size.
- **T2 (does background pretraining raise goal information?).** Pretrain the goal-marginalized model, without labels, on 0, 10, 30, 60 and 90 training subjects. Fit the class term on a fixed small labeled set and measure I_class on the test subjects.
  - *Kill:* the slope CI includes 0.
- **T3 (how big is the trap?).** Compare class information at 0–400 ms vs 1–4 s, and a frontal-only "EOG proxy" (Fp1/Fp2/AF7/AF8/F7/F8) vs a centro-parietal model with frontal channels regressed out, in imagined runs.
  - *Prediction:* early information is largely ocular or visual. The signal must survive in the 1–4 s centro-parietal band.
- **T4 (shared intent?).** Train on executed runs and test on imagined runs.

**Limit.** ds004362 cannot test the private-goal evaluation signal: it has no feedback cursor. The first 2 pilots test that.

**References:**
- Zander et al. 2016 PNAS 113:14898
- Iturrate et al. 2015 Sci Rep 5:13893
- Thut et al. 2006 J Neurosci 26:9494
- Kobler et al. 2018 Sci Rep 8:17713
- Pulferer et al. 2022 JNE 19:036005
- Kay et al. 2008 Nature 452:352
- Srinivasan et al. 2006
- Yu et al. 2007
- Kaifosh et al. 2025
- Ye et al. 2023
- Stieger et al. 2021
