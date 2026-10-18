# P_brain_reward: the brain as the reward function. EEG evaluations train a machine policy, then the EEG is unplugged

## 1. One-line idea and signal basis

**Idea.** An agent moves a cursor or robot by itself. The person only decides privately what they want and watches. Each action evokes an evaluative response in the person's brain, a "yes/no, that was right/wrong" reaction. A reward model decodes that response, and the agent's policy trains on it by reinforcement learning, the way RLHF trains language models. The headline test: after training, the **EEG is disconnected** and the frozen policy has to do what this person wants, at machine speed.

**Signal basis.** When the person watches a device act, its actions evoke evaluative responses over medial frontal cortex:
- The error-related negativity/positivity complex and reward positivity (Holroyd & Coles 2002 Psychol Rev; Proudfit 2015 Psychophysiology).
- Mid-frontal theta at 4–8 Hz (Cavanagh & Frank 2014 TICS), peaking 200–500 ms after the action at Fz/FC1/FC2/Cz.

These are "thinking" in the strictest sense: they are the brain's judgement of whether the world moved toward the person's goal. They need no instruction, no imagery and no added stimulus.

There is one honest grey zone. The machine's own action is the event that locks the epoch, so this is an evoked response. It is evoked by behaviour the person cares about, not by a flash we add for control. Iturrate et al. 2015 (Sci Rep) argue these signals reflect task-independent monitoring.

**Precedents (verified):**
- **Iturrate 2015:** 12 subjects taught a 1-D cursor, a simulated robot and a real arm to reach targets, after about 350 training actions (20% wrong).
- **Zander et al. 2016 (PNAS):** implicit cursor control on a 6×6 grid in 23 moves, against 90 for random and 14 for perfect.
- **Kim et al. 2017 (Sci Rep):** a real robot learned a gesture→action mapping from ErrPs, with about 90% balanced accuracy.
- **Batzianoulis et al. 2021 (Commun Biol):** 13 subjects; ErrP-driven inverse RL.
- **Lopes-Dias et al. 2019 (Sci Rep):** 15 participants; asynchronous online ErrP detection, true-positive rate 70%, true-negative rate 86.8%.
- **Grizou et al. 2014 (AAAI):** calibration-free ErrP goal learning.
- **Xu et al. 2021 (Neurocomputing):** ErrPs as RL reward.

To our knowledge, none of these combined display-unidentifiable private goals, eye and face auditors, surrogate-reward policies, scaling over hundreds of hours, and an EEG-unplugged test.

## 2. Label source and why it is NOT display-identifiable

The reward model's label is the **goal-relative value of an action**, r = cos(θ_action − θ_goal), taking values {1, 0.71, 0, −0.71, −1} for 8 directions. It is computed from a goal the person committed to privately **before** any action. It is never a threshold on a displayed variable.

**Display.** Eight identical, unmarked rings sit around a fixation point. The cursor stays at fixation and the ring world translates, so the eyes need not chase anything.

**Commitment.** The person picks a goal (free-goal blocks) or a private icon→goal mapping (preference blocks). They log it on a side tablet whose layout is rotated by a random multiple of 45° on every commit, so finger and gaze movement carry no screen-direction information. Commitment ends ≥1.5 s before the first action. The agent never sees the goal.

**Why the display carries zero information about r on probe actions.** On ε-probe actions (a uniformly random direction, 20% of all actions, in every session forever), the action is independent of the goal. The goal is uniform and private, so P(r | everything on screen) = P(r). That includes the turn angle Δθ, cursor history and motion-onset visual evoked potentials. **All headline reward-model AUCs are computed only on probe actions.** This answers the previous verdict's Δθ objection by construction.

**The closed-loop trap.** On policy actions the trajectory does leak the goal ("keep going" is usually right). A display-only surrogate is fitted on the same epochs, labels and folds, using action direction, Δθ, the last 5 actions, cursor position, time in trial and the agent's action probability. We report the EEG AUC beside it, plus the partial correlation of EEG output with r controlling for it.

**Leave-mapping-out.** Within one mapping, correctness is a function of (icon, action), so reward models are always tested on unseen mappings, where the display surrogate is at chance.

**Goal-switch catch trials** (10%). The person silently switches goal mid-trial, as pre-logged on the tablet. Trajectory momentum now predicts the wrong label. Only a decoder that reads the person can follow the switch.

## 3. Data engine, scaling mechanism, expected curve

**The bitter lesson applies to the reward model, and its labels are free.** Every action is a labelled epoch: one per 1.2 s, about 2,400 per hour after breaks. This fits the PI's "sit, collect, retrain" loop exactly, and the labels never touch the display. RLNF supports the mechanism: across 11.5k trials, only data volume helped. That was the right lesson attached to the wrong label. Kaifosh et al. 2025 (Nature) show volume-driven generalisation for a non-invasive sEMG interface.

**Three scaling axes:**
1. **Within-person:** 3 deep participants × 100 h, about 240k epochs each.
2. **Across people:** 24 × 4 h, about 230k epochs, for a zero-calibration population model tested on 6 held-out people.
3. **Pretraining:** a CBraMod/LaBraM-style encoder on unlabelled stream data. Expect modest gains.

**Expected within-person curve** on a fixed probe test set (the last 5 sessions, frozen from day 1), cortical channels only:

| epochs | AUC (dry, hypothesis) |
|---|---|
| 300 (~8 min) | 0.62–0.66 |
| 2,400 (1 h) | 0.67–0.72 |
| 24k (10 h) | 0.71–0.75 |
| 240k (100 h) | plateau 0.73–0.78 |

**Saturation** is set by single-trial SNR, with the noise ceiling estimated from split-half ERP reliability. Gradual errors are decoded worse (Omedes et al. 2015), so we report AUC per angle bin.

**Where scaling actually pays.** The agent integrates evidence, so speed scales with information per evaluation. For a balanced binary label and a Gaussian decoder (computed):

| AUC | bits per evaluation | evaluations for 3 bits (ideal) |
|---|---|---|
| 0.65 | 0.052 | 58 |
| 0.70 | 0.093 | 32 |
| 0.75 | 0.148 | 20 |
| 0.80 | 0.218 | 14 |

Moving from 0.70 to 0.75 makes teaching 1.6× faster. **The second scaling mechanism is amortisation.** A policy learned once is reused at zero EEG cost. The number of bits a person must ever send grows with the number of distinct preferences, not with the number of uses.

## 4. Expected performance on this hardware (honest)

**Real-time, as a steering channel**, this is weak:
- Anchor values: RLNF's dry-cap AUC of 0.66–0.69 and Lopes-Dias' gel detection rates.
- About 0.05–0.12 bits/s at one evaluation per 1.2 s.
- Selecting one of 8 free goals takes about 40–60 practical evaluations (≈50–75 s) at AUC 0.70, and 25–35 (≈30–45 s) at 0.75.

This matches the previous competition's finding that ErrP adds nothing on top of a dense velocity decoder. **We do not claim to beat that.**

**As a teaching channel:**
- An 8-context × 8-goal private mapping is 24 bits: about 10–20 min of passive watching at AUC 0.70–0.75, assuming 2× inefficiency.
- The frozen policy then executes each context in about 2 s with EEG unplugged.
- Predicted frozen-policy accuracy is ≥85% of contexts correct, against 12.5% chance.
- Re-adaptation after an unannounced change of preference: ≤40 evaluations to detect it.

We report amortised task throughput (about 1.5 bits/s of executed choices) **separately** and labelled as amortised. It is not a brain transfer rate.

## 5. Purity proof

**Auditors**, recorded and never fed to any model: an eye tracker (≥250 Hz, with pupil), bipolar EOG, temporalis EMG, a head accelerometer, and facial EMG over corrugator and zygomaticus, because frowning at an error is the obvious confound for an evaluative signal.

**Decoder hygiene:** a 1–10 Hz band, and a cortical-only model of record that excludes FP1, FP2, F7, F8, T7 and T8.

**Surrogate-trained policies, the core control.** Identical agents are trained on the same action stream with different rewards: (a) EEG cortical-only, (b) eye and pupil, (c) facial EMG, (d) display-only, (e) shuffled EEG, (f) EEG shifted by ≥10 actions. All use the same epochs, with permutation floors at matched n, and each frozen policy is tested on held-out sessions. The claim needs (a) > (d), (e) and (f) at floor, and (a) not explained by (b) and (c): the partial correlation of EEG reward with r controlling for the eye and EMG surrogates must be > 0.

Pupil may itself carry an error signal. If so, we report it as a finding; the EEG claim stands on the partial correlation and the cortical-only montage.

**Fixation-clean subset.** We report all metrics on epochs with no saccade >0.5° and no blink from −200 to 1000 ms. Exclusion is pre-registered by eye-tracker criteria, never by performance.

**Topography.** The model's pattern must peak frontocentrally at 200–500 ms; a frontopolar or temporal peak counts as a fail.

**Rejection cost, the RLNF failure mode.** No epoch is ever discarded online and nothing is vetoed. The action always executes, and the reward only shapes future behaviour, so a noisy epoch costs nothing. The reward model outputs a calibrated probability (Platt-scaled on recent probes). Updates are weighted by confidence |2p−1|, so artefact epochs self-down-weight.

## 6. Protocol

- **Phase 0, pilot (months 0–2):** 6 participants × 2 × 1 h; 3 of them add one 64-channel gel session on an existing lab amplifier to measure the dry penalty. Gate in §9.
- **Phase 1, data engine (months 2–10):** 3 deep participants at 2 h/day, 4 days/week, about 100 h each; a cohort of 24 × 4 × 1 h.
- **Phase 2, transfer (months 10–14):** 30 more hours per deep participant. The cursor-trained reward model drives a robot arm with no recalibration; exploratory Tier 2 (§7).
- **Phase 3, frozen evaluation (months 14–18):** 5 frozen sessions per deep participant, and 6 new participants on the population model with no calibration.

**Session structure.** 6 × 8-min blocks: 3 free-goal blocks (≤15 actions per trial) and 3 preference blocks (a new private mapping per block, icon shown 1 s before each trial). 20% probes, 10% goal switches, and a 2-min fixation-only rest per block. The only instruction: "decide what you want; watch it."

## 7. Real-time architecture

- **Acquisition and epochs:** LSL at 500 Hz with photodiode-timestamped action markers. Epochs run 0–1,000 ms after each action (1.2 s apart, no overlap); causal 1–10 Hz filter, decimated to 64 Hz.
- **Reward model:**
  - Day 1: xDAWN plus Riemannian tangent-space ERP covariances with shrinkage LDA. At ≥50k epochs: an EEGNet-scale CNN initialised from the population/pretrained encoder.
  - Nightly retraining on all labels, and within-session covariance re-centring against dry gain drift (the repo simulation showed re-centring partly rescues a frozen decoder).
  - Output arrives 1.0 s after the action (about 15 ms compute).
- **Agent, Tier 1:**
  - A Bayesian posterior over 8 goals per context, with Gaussian class-conditional likelihoods of the reward-model score given r.
  - Thompson sampling, ε = 0.2 probes.
  - For richer robot tasks: soft actor-critic on a learned reward, off-policy from the replay of all logged epochs.
- **Agent, Tier 2** (exploratory, the answer to "the user never steers"):
  - A policy π(a | EEG from −1,000 to 0 ms before the action, context), trained by REINFORCE with a baseline. The reward is the EEG reward model alone, so the steering decoder learns intention with **no display label at all**.
  - Cortical channels only, fixation enforced.
  - Its eye-tracker-input twin, trained identically, is the surrogate.
- **Effector interface:** effector-agnostic Cartesian displacements (8 directions in 2-D, 26 in 3-D): a 1.5° cursor step, or a 3-cm step through a robot's Cartesian impedance controller with workspace limits. The same reward model serves both.

## 8. The Nature claim

**Headline figure (4 panels):**
- **(a)** Frozen-policy success with EEG unplugged against minutes of passive watching. Curves for the EEG-cortical reward and five surrogate rewards (eye/pupil, facial EMG, display, shuffled, shifted); only EEG rises.
- **(b)** Probe-AUC against hours (log axis) on a fixed test set, for 3 deep participants and the population model, with the noise ceiling.
- **(c)** Cursor→robot transfer of the same reward model.
- **(d)** Frontocentral topography and 200–500 ms timing on fixation-clean epochs.

**One sentence.** "From unprompted brain evaluations alone, with goals the screen never reveals and eyes and face audited, a machine learns a person's private intentions and then carries them out with the EEG disconnected; the brain-derived reward improves with hours of data like a scaling law."

**Why it is new.** Prior ErrP-RL work showed feasibility. To our knowledge, none of it used display-unidentifiable labels, surrogate-reward policies, tens of hours of data per person, or a test with the brain removed. This is RLHF with a brain-derived reward, under the controls RLNF lacked. Honestly, Nature Communications is the realistic venue unless panel (b) shows a clean log-linear slope over 100 h.

## 9. Falsifier and biggest risk

**Pre-registered kills:**
- **Pilot gate.** Probe AUC on cortical channels only, at n = 2,000 probe epochs, must exceed the permutation p95 floor + 0.03 in ≥4 of 6 pilot participants. Otherwise the project is stopped before Phase 1.
- **Purity.** On frozen sessions, frozen-policy success for the EEG reward minus the best of the display, shuffled and shifted surrogates must exceed 10 percentage points, with a CI excluding 0. The partial correlation controlling for eye and EMG surrogates must be > 0. Failing either kills the claim.
- **Scaling.** If probe AUC on the fixed test set improves by < 0.02 from 10 h to 100 h (CI including 0), the bitter-lesson claim is falsified for this signal. This is still a publishable negative.

**Biggest risk.** On a dry 32-channel cap, single-trial evaluative AUC for graded (45°–90°) errors stays at RLNF levels (0.62–0.68), so teaching needs ≥60–90 evaluations per 3 bits. On top of that, ErrP amplitude may habituate over 100 h, since it depends on expectation violation and 20% probes make errors familiar.

Mitigations: ramp probe rates, larger-angle actions, weighting by error magnitude, and the gel arm to separate hardware limits from physiology. Secondary risk: boredom over 100 h, countered with the person's own real choices and a 2-h daily cap.

## 10. What can be tested now on ds004362

**Plainly: ds004362 contains no evaluative events, so it can neither prove nor kill the core claim.** It can test the scaffolding:

1. **Display-leak harness.** Decode target side (T1 vs T2, left/right runs, 0–800 ms after the cue) from FP1/FP2/F7/F8 only, from a 32-channel-equivalent cortical subset without them, and from occipital only, with permutation floors at matched n. Prediction: frontal/occipital AUC ≥ cortical, which is exactly the display trap our probe design removes. If the harness cannot detect this leak, it cannot certify purity.
2. **Population ERP scaling ladder.** Detect the single-trial frontocentral (Fz/FCz/Cz, 1–10 Hz) cue-evoked response against the pre-cue baseline. Train on n ∈ {5, 10, 20, 40, 80} pooled subjects (5 draws each) and test on a fixed 29-subject set, using a 32-channel-equivalent montage. Kill for the zero-calibration leg: AUC gains < 0.01 from 10 to 80 subjects (CI including 0).
3. **Noise-calibrated policy simulation.** Feed the per-subject AUCs from test 2, minus the dry penalty, into the Tier 1 agent in `neural_cursor/` to predict minutes-to-24-bits. The earlier "agent+evidence" simulation reached only 0.08 success at AUC ≈ 0.58. We expect 50% frozen-policy success only above AUC ≈ 0.68, which fixes the pilot gate numerically before anyone sits down.
