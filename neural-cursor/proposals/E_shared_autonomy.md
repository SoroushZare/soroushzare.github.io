# Proposal E — Shared Autonomy with Implicit Neural Feedback (SA-INF)

*An agent drives the cursor; the brain only says "no", implicitly, about once per second; Bayesian goal inference turns sparse "no"s into target acquisition. The RLNF successor with every registered flaw fixed by construction.*

## 1. Paradigm and signal basis

**Loop.** The screen shows K candidate targets (K = 8 in the lab task; a UI's actionable elements in deployment). The participant holds a goal in mind. An agent moves the cursor in 1-s constant-velocity segments (8 % screen/s, 100 ms cosine-blended direction changes). The agent sees the targets and its own history but **never the goal**. It keeps a posterior over goals; each segment's direction change is a small, salient event the participant's brain evaluates. The decoder reads 0.15–0.8 s post-segment EEG and emits P(error); the posterior is updated for every candidate g by whether the segment was "wrong relative to g" (angle > 60°); the next segment is chosen by hindsight optimisation (Javdani, Srinivasa & Bagnell 2015 RSS; Javdani et al. 2018 IJRR): expected progress under the posterior plus λ × expected information gain, with an ε = 0.2 random-probe rate. When the cursor sits on a target with posterior ≥ 0.85 it dwell-selects (1 s). The participant never issues a command.

**Signal.** Error-related potentials to observed agent actions: fronto-central negativity ~250 ms followed by a centro-parietal positivity ~300–500 ms, maximal at FCz/Cz, 1–10 Hz — the "interaction ErrP" of Ferrez & Millán (2008 IEEE TBME) and Chavarriaga & Millán (2010 IEEE TNSRE), elicited by watching a cursor move the wrong way, requiring no instruction beyond "keep your goal in mind". Optional priors: Zander et al. (2016 PNAS) implicit-evaluation signals and, with an eye tracker, gaze.

**Why it is natural.** No flicker, no oddball, no imagery. The person watches, intends, and silently disapproves. The ErrP is automatic; Chavarriaga, Sobolewski & Millán (2014 Front Neurosci, "Errare machinale est") review that it needs no training and is stable for months (Chavarriaga & Millán 2010 re-tested subjects ~600 days later). **Caveat:** the cursor is agent-driven; the person is an evaluator, not the driver. Free-form drawing is not supported (a 0.1 bit/s channel cannot carry it); goal-directed, continuous, natural cursor motion on structured screens is.

## 2. Expected performance on this hardware

**Single-trial ErrP accuracy, gel.** Ferrez & Millán 2008: recognition ≈0.80 (error) / ≈0.83 (correct). Chavarriaga & Millán 2010 (6 subjects): ≈0.75–0.80 per class, stable across sessions months apart. Iturrate et al. 2015 Sci Rep (ErrP-taught reaching BMI, 12 subjects): per-subject single-trial accuracies ≈0.7–0.8, enough to learn target maps from a few hundred trials. Salazar-Gomez et al. 2017 ICRA: online primary-ErrP accuracy ≈0.70 on a binary robot choice. Zander et al. 2016: single-trial accuracies ≈0.6–0.8. Continuous settings are harder — Lopes-Dias, Sburlea & Müller-Putz 2019 Sci Rep (continuous robot control, 15 subjects) report asynchronous online true-positive rates ≈0.7 at low false-alarm rates — and Spüler & Niethammer 2015 and Omedes et al. 2015 show ErrPs survive continuous feedback but gradual errors evoke smaller, later potentials — which is why SA-INF uses discrete 1-s direction changes as eliciting events. Summary (Chavarriaga 2014): **0.65–0.85 single-trial, mode ≈0.75, gel**.

**Dry penalty.** Guger et al. 2012 (Front Neurosci) found g.SAHARA dry electrodes within a few percentage points of gel on P300 spelling; Di Flumeri et al. 2019 report comparable classification with lower SNR and more artefacts. ErrP is a 5–10 µV slow wave at a frontal-central site where dry contacts are noisiest; RLNF's dry cap reached only 0.66–0.69 AUC on a flawed label. I apply **−0.05 accuracy** and widen the lower tail: **planning value p = 0.70, plausible 0.62–0.78**, with an expected new-participant prediction interval of [0.57, 0.80].

**Bits per decision.** Binary-symmetric capacity C = 1 − H(p): p = 0.65 → 0.066; **0.70 → 0.119**; 0.75 → 0.189; 0.80 → 0.278 bits. At 1 s/segment plus ~0.8 s overhead per selection, **≈40 decisions/min** ⇒ **2.6 / 4.8 / 7.6 / 11.1 bits/min** (0.04–0.19 bits/s). Reference: mouse 3.7–4.9 bits/s (Soukoreff & MacKenzie 2004); P300 spellers ≈0.3–0.5 bits/s — SA-INF is **an order below P300, ~30× below a mouse**.

**Where autonomy multiplies.** Targets/min ≤ (decisions/min × C(p)) / H(goal | context). Physiology fixes the numerator; learning shrinks the denominator. Simulation (script `sim_shared_autonomy.py` beside this file; 3,000 trials/cell; K = 8 on a ring, D = 0.40, W = 0.08, Fitts ID = 2.58 bits; MAP policy, ε = 0.2, commit at 0.85):

| p | prior top-1 q | mean time-to-correct-target (s) | success | bits/min |
|---|---|---|---|---|
| 0.65 | 1/8 | 45 | 0.63 | 3.4 |
| **0.70** | **1/8** | **32** | **0.90** | **4.9** |
| 0.70 | 0.4 | 28 | 0.92 | 5.6 |
| 0.70 | 0.6 | 23 | 0.93 | 6.8 |
| 0.75 | 1/8 | 22 | 0.98 | 6.9 |
| 0.75 | 0.6 | 17 | 0.99 | 9.1 |
| 0.80 | 0.6 | 13 | 1.00 | 11.8 |

The naive MAP policy already runs at ≈100 % of channel capacity. Agent-only (no EEG, select the prior's guess): 6 s but success = q. **Realistic claim for this cap: 20–35 s per target among 8 at ≥ 90 % correctness, 5–7 bits/min, falling to 13–17 s once the prior reaches q ≈ 0.6.** With an eye tracker (gaze as prior, q ≈ 0.9) it becomes a 3–5 s confirm/veto of the gazed target — mostly a gaze interface, reported as such.

## 3. Scaling behaviour (bitter lesson)

**(a) ErrP decoder — volume scales it, provenance does not (RLNF's finding, kept).** Every 1-s segment is a labelled epoch (label from the revealed goal, Section 4), so one 40-min session yields ≈2,000 labels versus ≈400/participant in RLNF: a 5× label rate from cheap rejection alone. Pretraining: public ErrP corpora (BNCI Horizon 2020 "013-2015 Monitoring error-related potentials"; the BCI Challenge @ NER 2015 speller-ErrP set, 26 subjects) plus all SA-INF participants, with Euclidean alignment (He & Wu 2020 IEEE TBME) or Riemannian re-centring (Zanini et al. 2018) per session. Expected curve (dry, held-out-session AUC): zero-calibration cross-subject ≈0.60–0.65 (ErrP topography is stereotyped at FCz/Cz; fine-tuning on ≈500 own labels, 12 min, should close most of the gap) → one own session (2k labels) ≈0.70 → five sessions (10k) ≈0.73–0.75 → **plateau ≈0.75–0.78**. The plateau is physics: single-trial ErrP SNR (5–10 µV on ~20 µV background), dry-contact noise, and a label-noise floor from genuine lapses (≈5–10 % of segments). EEGNet-8,2 (Lawhern 2018) runs on a fixed resource ladder; I predict parity with Riemannian + LR below 10k labels and ≤ +0.03 AUC above 50k pooled labels.

**(b) Goal prior — where data buys most.** H(goal | context) shrinks with a prior over targets conditioned on history, layout and app context, trained on every logged (context, revealed goal) pair. In the lab task with uniformly random cued goals this prior is provably stuck at 1/K — I say so, and test prior-scaling only in free-choice and Zipf-menu blocks, where habit makes next-target prediction strong (expected top-1 ≈0.4 after one session, ≈0.6 after five, saturating at the entropy of the person's own habits). Dirichlet counts first; a small context-conditioned sequence model from session 3.

**(c) Policy.** The hindsight-optimisation policy improves as the user simulator (p, latency, ErrP timing) is fitted to logs; λ and ε are re-tuned per participant offline. No online RL with a human in the loop (cf. Reddy, Dragan & Levine 2018).

**Co-adaptation.** Decoder: unsupervised re-centring within session; supervised retrain between sessions on all accumulated labels. Prior/policy: between sessions. The participant is asked to adapt nothing (RLNF found no such effect; I do not rely on one).

## 4. Data-collection protocol

**Participants.** 24 adults, 3 sessions each ≥ 2 days apart; a longitudinal arm of 6 with 10 sessions for the scaling curve. No performance-based exclusion (pre-registered).

**Session (75 min).** Cap + impedance log (10 min) → 2 min resting EEG (re-centring covariance) → 6 × 6-min blocks: 2 **cued-goal** (one target flashes 500 ms, 1 s blank, cursor appears; the flash is a label cue the decoder never sees), 2 **free-choice** ("choose any target, press space when chosen; click it with the mouse when the trial ends"), 2 **Zipf-menu** (goal frequencies 0.35/0.2/0.15/…; participant told the "popular" items). Latin-square order. One **surrogate block** per session: decoder outputs replaced by a time-shuffled copy of the previous block's outputs (matched prevalence, no neural coupling), single-blind.

**Trial.** Cursor at centre; 1-s segments; max 60 s; ends at dwell-select; ITI 1.5 s. ≈10–15 trials/block, ≈350–500 segments/block, ≈2,000–2,500 labelled segments/session, ≈150k study-wide (RLNF: 11.5k).

**Labels.** y_i = 1[angle(segment_i, goal − x_i) > 60°], computed **after** the goal is revealed (cue, or the participant's click). The label is a function of hidden goal and logged action, never a threshold on a displayed variable. The 45–75° band is kept and flagged; a sensitivity analysis drops it. A "changed my mind" key voids the trial. Delayed report: in one block per session, after each trial the participant rates "how often was it wrong?" (1–5) to estimate label noise.

**Randomisation / positivity.** ε = 0.2 segments are uniformly random directions; the rest are softmax over the posterior (T = 0.5). Pre-registered check: in every posterior-decile bin, P(y = 1) ∈ [0.1, 0.9].

**LSL streams.** EEG (32 ch, 500 Hz); impedance per block; eye tracker (gaze x,y, pupil, 60–120 Hz — required addition, ≈€200–2,000); keyboard/mouse; markers: block/trial start, cue id, segment onset + direction + cursor x,y, decoder P(error) + timestamp, posterior vector, commit, revealed goal, dwell outcome. LSL time-sync audited day 1 (target < 10 ms).

**Gating.** Channel RMS > 5× session median → masked for that segment (FP1/FP2 exempt, logged); segments with gaze velocity > 100°/s flagged saccadic; impedance per block. No exclusions.

**Pre-registration list.** (1) decoder AUC on held-out sessions with permutation floors at matched n; (2) co-primary: time-to-correct-target and bits/min, EEG vs surrogate blocks; (3) falsifiers: display-only model AUC > floor + 0.02 (goal leaks); ocular-only model within 0.02 of the full decoder (eye, not cortex); EEG not better than surrogate on throughput (no utility); (4) resource ladder with fixed test set; (5) scaling-arm fit (log-linear + plateau).

**Power.** Decoder: AUC 0.70 vs floor 0.60, between-subject SD 0.07 (RLNF) → d ≈ 1.4; n = 24 gives > 0.99 power. Utility: simulated 4.9 vs ≈1 bits/min, within-subject SD ≈ 2 → d ≈ 1.5; n = 24 gives > 0.99 at α = 0.05 and 0.8 power for d = 0.6. Pre-declared: if p < 0.62 the utility test is not expected to pass (45 s/target, 63 %).

## 5. Decoder and system

**Preprocessing.** Common-average reference (gated channels excluded); 1–10 Hz 4th-order Butterworth (zero-phase offline, causal online with ~40 ms group delay); downsample to 64 Hz; epoch 0.15–0.80 s after segment onset (42 samples × 32 ch). FP1/FP2 and eye-tracker regressors are inputs to the surrogate models, not regressed out of the primary pipeline (RLNF: regression bought nothing).

**Primary decoder.** xDAWN (Rivet et al. 2009), 4 filters/class → 8 components; ERP-prototype-augmented covariance (Barachant & Congedo 2014) → tangent space at the session-re-centred mean → logistic regression (L2, C = 1.0, balanced). Platt-calibrated on the first own session; the p used in the Bayes update is a running Beta estimate of accuracy from revealed goals (prevents over-confident posteriors).

**Scaling decoder.** EEGNet-8,2 (F1 = 8, D = 2, F2 = 16, dropout 0.5, temporal kernel 32 at 64 Hz), Adam 1e-3, 100 epochs, early stop on a held-out *session*; Euclidean alignment per session. Same test set at every ladder rung (250, 500, 1k, 2k, 5k, 10k, pooled).

**Splits.** Session-wise: train sessions 1–2, test session 3 (4–10 in the scaling arm); leave-one-subject-out for the zero-calibration number; trials are the CV unit.

**Latency budget.** Onset t0 → EEG to t0 + 0.80 s → LSL jitter ≤ 10 ms → filter + xDAWN + tangent + LR ≤ 15 ms → posterior + planning ≤ 5 ms → next action ready ≈ t0 + 0.85 s, 150 ms margin; measured end-to-end on day 1.

**Cursor dynamics.** Agent velocity control, 8 % screen/s, fixed direction within a segment, 100 ms cosine blend between segments (continuous motion, sharp evaluable event). Assistance = the policy itself; "autonomy level" = softmax temperature and ε. Commit = dwell 1 s at posterior ≥ 0.85 within W/2. A wrong commit evokes a strong discrete ErrP that down-weights that target; the cursor leaves.

**Co-adaptation rule.** Within session: Riemannian re-centring every 2 min on unlabelled covariances, no supervised updates. Between sessions: full retrain on own + pooled labels; prior counts updated; λ, ε re-tuned on the fitted simulator. Within-session adaptation is predicted null and tested as a secondary endpoint, not assumed.

## 6. Evaluation protocol

**Co-primary endpoints.** (E1) Decoder AUC on held-out sessions against a per-participant permutation floor at matched n (1,000 within-block label permutations preserving trial structure); mean, 95 % CI, and a mixed-model prediction interval for a new participant. (E2) Controller utility: time-to-correct-target and bits/min, EEG vs surrogate blocks, paired within participant; Fitts throughput including error-corrected time. Secondary: selection accuracy, posterior calibration (Brier), label-noise ratings, within-session AUC drift, scaling-arm curve.

**Non-neural surrogates (pre-registered, identical pipeline).** S1 time-shuffled decoder outputs online (utility floor). S2 ocular-only decoder (FP1, FP2, FP1−FP2, (FP1+FP2)/2 − Fz, gaze/pupil features). S3 display-only model (segment direction, cursor x,y, time-in-trial, agent posterior) — must sit at floor, else the goal leaks and ε is raised. S4 noise-EEG surrogate (resting EEG cut into pseudo-epochs with real labels). S5 fixation-constrained analysis: trials where gaze stayed within 2° of a central dot — the cortical claim must survive here.

**Tests.** Mixed-effects (participant intercept, session slope) for AUC-above-floor and bits/min; TOST (±0.02 AUC) for equivalence claims; Holm over secondaries; effect sizes with CIs; the test session never informs any choice.

**Held-out structure.** Session 3 (and 4–10) per participant; 4 participants fully excluded from cross-subject pretraining for the zero-calibration estimate.

**Reproducibility.** Fixed seeds, pinned pyriemann/MNE/PyTorch in a container, raw XDF + markers and the simulation script released under the pre-registration DOI.

## 7. Failure modes and debugging

1. **ErrP too weak in continuous motion** (Omedes 2015). Detect: no FCz negativity in the segment-locked ERP image by block 2. Fix: 1.5-s segments, unblended direction changes; fall back to discrete 20° jumps.
2. **Ocular dominates.** S2 within 0.02 of the full decoder. Fix: report the cortical estimate from S5 only; add adhesive EOG electrodes on two re-routed channels from session 2.
3. **Goal leaks through the agent.** S3 above floor. Fix: ε = 0.3, higher temperature, re-run positivity audit.
4. **Lapses / goal switching.** Noise rating > 3 in > 20 % of trials. Fix: 4-min blocks, drop the ambiguous band.
5. **Over-confident posterior.** Wrong commits > 25 %. Fix: Beta-tracked p, threshold 0.9, longer dwell.
6. **Dry FCz noise.** Gates fire > 10 % at FCz/Cz. Fix: re-seat, use xDAWN components, report with/without gated segments.
7. **Decoder plateau < 0.62.** Utility test cannot pass (pre-declared); the study still returns a clean negative and 150k labelled epochs.
8. **Agent too good** (q → 0.9 in Zipf blocks). Incremental EEG value shrinks; report it — the system degrades gracefully to "agent with a veto".

## 8. Honest self-score

- **Naturalness: 8/10.** No stimulation, imagery or commands; the person watches and intends — but as evaluator of an agent's motion, not driver, which some users will not feel as "control".
- **Achievable performance: 4/10.** 0.04–0.19 bits/s, 20–35 s per target among 8 at p ≈ 0.70 dry; an order below P300/SSVEP, mitigated only by the learned prior and an optional eye tracker.
- **Scaling with data: 8/10.** 5× label rate, a prior whose entropy term shrinks with usage logs, cross-subject pretraining, a pre-registered ladder with fixed test set; the decoder itself plateaus near 0.75–0.78.
- **Validity / identifiability: 9/10.** Goal hidden from the generator, label from the revealed goal, ε-probes guarantee positivity, five surrogates including display-only and ocular-only, fixation-constrained blocks, eye tracker required.
- **Feasibility & burden: 7/10.** Sub-second latency, 2 min resting + optional 12 min fine-tune, standard LSL/pyriemann stack; needs an eye tracker and a well-engineered agent.
- **Rigour & reproducibility: 8/10.** Pre-registered endpoints, floors at matched n, held-out sessions and subjects, deterministic pipeline, explicit hyperparameters, and a released simulation that predicts the numbers in advance.

## 9. Pitch

RLNF failed not because the brain was silent but because the label was the display, a rejection cost five seconds, and the eye was never separated from the cortex; SA-INF fixes all three by construction. An agent that cannot see your goal moves the cursor, your brain's automatic "that's wrong" updates a posterior over targets every second, and a wrong move costs one second, not five. At the dry-cap accuracy of ~0.70 that is only 0.12 bits per decision, yet because the agent carries the dynamics and a learned prior carries the entropy, it reaches targets among eight in 20–35 s with ≥ 90 % correctness, and every hour of use yields ~2,500 clean labels plus a sharper prior. It is the only paradigm here whose labels come from a hidden intention rather than from what is on the screen, so a positive result will mean what it says. It is slow, and I say so; but it is the one design that gets monotonically better the more people simply use it.
