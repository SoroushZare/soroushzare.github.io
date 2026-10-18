# P_vocabulary — A purity-gated personal mental vocabulary driving a goal-level autonomous effector

## 1. One-line idea and signal basis

**Each person gets the 3–5 comfortable mental tasks that their cortex separates best *after* eye and muscle information is removed, chosen from an 11-class library on held-out data. The tasks map to commands, and an autonomous controller turns the few bits/s into cursor or robot motion.**

The signal is ongoing oscillatory reorganisation with no stimulus: sensorimotor mu/beta ERD (motor imagery), frontal-midline theta and alpha–beta ERD (subtraction, word generation), parietal alpha ERD (navigation, rotation), temporal/occipital alpha (auditory, face imagery), and the rest alpha baseline.

**Precedents:** Friedrich et al. 2012 (Int J Psychophysiol) compared seven strategies. Friedrich et al. 2013 (PLoS ONE, "Whatever works") gave 8 users personal 4-class sets, with mean accuracies of 44–84% (chance 25%) and sets that differed between users. In Scherer et al. 2015 (PLoS ONE, 9 end-users), a brain-teaser paired with dynamic imagery beat hand-vs-feet. None recorded muscle or gaze auditors at this level, so **whether "whatever works" survives EMG and gaze control is open. This proposal answers it either way.** This is "thinking" as a self-generated mental act, but it is **not** natural intention (§9).

## 2. Label source and why it is NOT display-identifiable

Every label is **the mental task performed**. It is never a direction, a target or a threshold on a displayed variable.

1. **Cue-remapped calibration trials.** An abstract glyph is shown for 1 s, with the glyph→task key re-drawn each calibration block and given by a spoken key. Then come 1.5 s of blank fixation and 6 s of task on a blank screen, and windows start ≥1.5 s after the cue.
2. **Free-choice trials.** On a blank screen, the person picks a task and reports it after the window closes, using randomised key positions.
3. **Closed-loop adaptation labels: instructed goals with all outcomes kept.**
   - In adaptation blocks the goal is spoken ("object three"), and the task→command mapping is permuted per block.
   - The label of the **first decision window** (before any effector motion) is the task the instructed goal requires, **whether or not the decoder succeeded**.
   - Confirmed choices are used for evaluation only, never as labels. This removes the self-confirmation loop.

The decoder outputs task identity. Screen geometry exists only in the autonomy layer.

## 3. Data engine and scaling mechanism

**L1 — Library breadth, purity-gated.**
- *Gate first:* a task enters ranking only if its auditor surrogate is within the permutation p95, its mimic block is at chance, and it keeps ≥80% of bits on EMG-silent trials (§5).
- *Then rank* with the **purified decoder** (4–20 Hz, 26 channels without FP1/FP2/F7/F8/T7/T8).

The optimiser now works *toward* purity, not against it.

Correction to the earlier draft: σ√(2 ln M) is the winner's curse, not a real gain. The out-of-sample gain depends on how much *true* separability varies between tasks compared with noise; with 60 trials per class per pair, the SE is about 0.04. Selection therefore uses **empirical-Bayes shrinkage** toward the cohort atlas, and it is **nested** (sessions 1–2 select, session 3 tests).

In the repo's ds004362 results (106 subjects), fists-vs-feet scored 0.710 and left-vs-right 0.643. Idle-vs-imagine scored 0.844, but it uses inter-trial rest and is display- and arousal-confounded, so **no design claim rests on it**.

**L2 — Hours per person.**
The test set is fixed (the last 10 sessions are never trained on). We run a ladder from 30 to 960 trials per class, fit acc(n) = a − b·n^(−c), and report the asymptote *a* with a bootstrap CI. Pre-registered expectation: a plateau near 300 trials per class, limited by non-stationarity (as in RLNF).

**L2b — Vocabulary evolution with use.**
- Each week a challenger task (n≥60 per class) replaces the weakest task only if it passes the gate and wins on that week's held-out day by ≥5 pp after shrinkage.
- Over 100 h that is about 25 challengers per person. This is search over the task space paid for by hours, and the main EEG scaling mechanism beyond L2.
- It is reported as purified bits/decision per week on fixed test days.

**L3 — Population prior over tasks, not decoders.** Zero-shot cross-person EEG is weak (repo E2: L/R rose only from 0.58 to 0.66 between 1 and 32 pooled subjects). A 5-min rest recording plus a 15-min mini-screen predicts each newcomer's top tasks. The precedent is Blankertz 2010, where a rest predictor reached r=0.53 in 80 users. L3 is dropped if r<0.2.

**L4 — Goal habit prior (engineering, kept out of the EEG claim).**
A goal prior saves at most log₂(goals) − H(goal | context) bits, which is **zero for uniform goals**, and the earlier "3→1.5 bits" figure is withdrawn. **Primary evaluations use uniformly random instructed goals.** L4 appears only in a secondary free-use arm, against a matched **prior-only, no-EEG controller** (§5, test 8). The EEG scaling claim rests on L1, L2 and L2b.

## 4. Expected performance on this hardware

Wolpaw bits per decision at 4 classes: 50% gives 0.21, 55% gives 0.29, 65% gives 0.51.

The earlier 10–15 pp dry-electrode penalty was an **unsourced assumption** and is now measured (§6). The figures below assume it and use the purified decoder.

| Quantity | Median user | Top quartile | Anchor / status |
|---|---|---|---|
| 4-class purified held-out accuracy | 50–60% | 65–75% | Friedrich 2013 (44–84%, gel, no auditors), minus purification and dry losses |
| EEG rate (4–6 s per decision) | **0.04–0.1 bits/s** | 0.12–0.25 | project range 0.03–0.3 |
| 1 of 8 goals, uniform prior, with confirm/undo | 30–75 s, 80–90% correct | 12–25 s | 3 bits ÷ rate |
| Steer-mode cursor, 8 targets | **not claimed until re-simulated** | | the repo's r≈0.4 continuous simulation does not apply to a categorical signal; it is re-run in month 1 with the measured confusion matrix |
| Robot pick-and-place, 6 objects (goal mode) | ≥80% completion, 45–120 s per object | | Millán 2004 (3 mental states + behaviours); Leeb 2015 (end-user shared control) |

Royer et al. 2011 (J Neural Eng) found that goal selection beat process control. **This design does not deliver dense continuous control.** It delivers discrete goals plus a coarse 2-D steer mode, and a 6-DoF arm uses goal mode only.

## 5. Purity proof

**Auditors** (recorded on a process with no socket to the decoder): a 250-Hz eye tracker, 4-electrode EOG, EMG (temporalis, masseter, frontalis, submental/laryngeal, splenius, forearm flexors), a head accelerometer and a throat microphone.

**Threats:**
- Whitham 2007: scalp power above 20 Hz during cognitive tasks is mostly EMG.
- Goncharova 2003: EMG reaches below 20 Hz at frontal and temporal sites.
- Loetscher 2010: eye position tracks number generation.
- Brandt & Stark 1997: imagery evokes scanpaths.

The **primary decoder is therefore 4–20 Hz on the 26 non-rim channels**. The full-band decoder is reported only as a *contamination index*.

**Pre-registered tests.** They use matched n and fixed held-out sessions, and they are applied **per task, before selection** (the gate) and again to the final vocabulary. Thresholds are in bits.

1. **Auditor-surrogate decoder.** Same architecture and search, auditor channels only, same trials. Pass: the surrogate is ≤ the permutation p95, and the surrogate's bits are ≤10% of EEG bits.
2. **EMG- and gaze-silent subset** (adopted from the opponent). Trials where every EMG RMS is inside the 95% rest band and gaze stays within 1.5° with no blink. Pass: ≥80% of bits retained, with the surrogate at chance on the subset. The test is nonparametric and needs no model of the leakage.
3. **Non-linear conditional test.** Gradient boosting predicts per-trial EEG log-odds from auditor features. Pass: <10% of the variance explained, and ≥80% of bits retained on the residual. This replaces linear regression, which **cannot remove non-linear, smeared leakage**.
4. **Mimic and gradient blocks.**
   - Mimic: the person produces each task's tells without doing the task. Pass: the output is at chance.
   - Gradient (adopted): word association and subtraction are also done aloud and mouthed. If EEG decodability follows EMG decodability across overt→mouthed→covert, the task fails.
5. **Fixation and eyes-closed blocks.** Pass: ≥80% of bits retained. At a 55% baseline this allows about 3 pp, not 10.
6. **Glyph-transfer.** Train on key π₁ and test on π₂. A display reader would collapse.
7. **Feedback decoupling** (adopted). In 15% of closed-loop windows the effector is frozen for 3 s or replays an old trajectory. Pass: task probabilities match non-frozen windows, and closed-loop accuracy does not exceed open-loop blank-screen accuracy beyond its CI. (This replaces the mapping-reversal test, whose predictions were not separable.)
8. **Prior-only, no-EEG controller.** Every closed-loop trial is re-run offline with the same goals, prior and autonomy, driven by phase-randomised EEG, by the auditor-surrogate output, and by a label-permuted stream. Brain-attributable success = success − max(surrogates). Ten percent of live blocks are blinded shams driven by another day's output.
9. **Replay.** The person watches their own earlier session while resting. Pass: the output is classified as **Rest** at the same rate as plain rest (the earlier "at chance" criterion was wrong).
10. **Bookkeeping.** Permutation and circular-shift floors; a time/order surrogate at chance; no performance-based exclusion.

**Claim wording.** "Provably cortical" is replaced by **"auditor-bounded"**: we report the upper 95% CI of the fraction of bits attributable to the measured eye and muscle sources. No test suite can rule out sources it did not record.

## 6. Protocol

**Library (11 classes):** serial-7 subtraction, word association, mental rotation, navigation, song imagery, face imagery, left hand, right hand, feet, one self-proposed task, and rest. A task rated below 4/7 for comfort is ineligible.

**Week 1:** the public-data tests in §10.

**Phase A, cohort (months 1–5).** 24 participants × 3 sessions × 2 h.
- 11 classes × 30 trials per session, plus auditor calibration, mimic and gradient blocks. Sessions 1–2 select and session 3 is held out.
- 6 participants add a same-day 1-h gel session (existing lab amplifier), which measures the dry penalty directly.
- Output: a purity-gated vocabulary atlas.

**Phase B, deep arm (months 3–14).** 4 participants × ~100 h, assigned by availability before screening. A typical 2-h day:

| Block | Min |
|---|---|
| Rest + auditor calibration | 8 |
| Cue-remapped trials (vocabulary + 1 challenger) | 20 |
| Closed loop: goal selection, arm pick-and-place, steer mode (stable, user-chosen mapping) | 35 |
| Adaptation/validation blocks (permuted mapping, instructed goals) | 10 |
| Distractor idle (video, conversation, planning, mind-wandering) | 10 |
| Free choice + mimic/replay | 10 |

- Cognitive-task load is ≤30 min per day, with breaks.
- Fridays are frozen-decoder tests.

**Phase C (months 13–16).** 8 new participants: an L3-guided mini-screen, then 4 × 2 h sessions, the last a fixed test.

## 7. Real-time architecture

- **Acquisition:** LSL at 500 Hz, causal 4–20 Hz FIR, CAR over the 26 non-rim channels, with no artifact rejection in the decoder path.
- **Features:** 2-s windows every 250 ms. Filter-bank covariances (4–8, 8–12, 12–16, 16–20 Hz) are mapped to tangent space and re-centred on 3 min of rest per session (Zanini 2018), then continuously (λ=0.995).
- **Model:** per-person L2 multinomial logistic regression, with an optional EEGNet ensemble that is kept only if it wins on held-out days.
- **Decision:** leaky log-odds accumulation (τ=3 s), with Rest as the idle class. The threshold is set for **≤1 false command per 5 min in distractor-idle blocks**, and a 2-s sustained "wake" task re-arms the system after long idle.
- **Adaptation:** re-centring always runs. Nightly refits use cue-remapped, free-choice and instructed-goal first-window trials (all outcomes kept), never confirmed choices or decoder output.
- **Effector interface:** in *goal mode*, hierarchical menus with a Bayesian posterior; above 0.9 the cursor or robot (MoveIt) executes a minimum-jerk trajectory, and undo is available. In *steer mode*, 2-D velocity = Σₖ pₖuₖ with assistance a ∈ {0, 0.3, 0.6}; a=0 is always measured. The decoder is device-blind. Any reliable 4-symbol switch could drive this layer, so generality is an interface property, not a novelty.

## 8. The Nature claim

**Headline figure:**
- (a) A 24-person × 10-task atlas of separability before and after the purity gate: how much of "whatever works" survives auditing, and whether it is idiosyncratic.
- (b) 100-h curves on fixed test days: purified bits per decision against hours (L2) and against weeks of vocabulary evolution (L2b), with the surrogate, mimic and no-EEG curves at the floor.
- (c) Invariance to glyph transfer and feedback decoupling.
- (d) One output driving a cursor, an arm and a mobile base, with success shown above the prior-only controller.

**Sentence:** *"Scalp EEG alone, with no stimulus, movement or gaze, gives each person a self-discovered vocabulary of mental acts whose information survives muscle and eye auditing, improves with weeks of use, and drives any device through goal-level autonomy."*

**Honest novelty.** The components have precedents: Millán 2004, Friedrich 2012/2013, Scherer 2015, Leeb 2015 and Royer 2011. What is new:
1. The first auditor-bounded test of mental-task BCIs, whose brain-teaser tasks are the most contamination-prone. A negative outcome ("the advantage was EMG and eyes") is equally publishable.
2. Purity-gated selection at cohort scale.
3. A 100-h test of whether vocabulary search keeps paying.

The realistic venue is Nature Communications.

## 9. Falsifier and biggest risk

**Pre-registered falsifiers:**
- **F1:** the purity-gated personal vocabulary does not beat the fixed canonical set (both-hands/feet/rest) by ≥5 pp in median held-out purified accuracy. Personalisation is then noise or artefact, and the design reduces to SMR.
- **F2:** ≥50% of the cognitive tasks' bits fail tests 1–5.
- **F3:** glyph transfer or feedback decoupling fails.
- **F4:** neither the L2 asymptote nor the L2b weekly curve rises on fixed test days (the bootstrap CI of the slope includes 0).

**Biggest risk: F2.** If the gate returns mostly motor imagery and rest, the project delivers an auditor-controlled refutation plus a standard SMR goal BCI, not the vocabulary headline.

**Second risk: naturalness.** The tasks are instructed and have no semantic link to commands. User-chosen stable pairings in routine use (permutation only in validation blocks), comfort gating and capped load soften this but do not remove it; a fair score is **about 5/10**. Spontaneous thought uses the same operations, so the distractor-idle false-command rate is a pre-registered outcome.

## 10. What can be tested NOW

**On ds004362** (5 states, 32-ch subset):
1. **Personalisation.** Select on runs 4/6/8/10 and test on 12/14, with shrinkage. Support: a ≥3 pp gain in ≥55% of subjects. Kill: median gain ≤0.
2. **Cross-run floor** (rest decoded between run types).
3. **Display audit:** frontopolar-only, occipital-only and 4–20 Hz non-rim decoders against the full decoder.
4. **L3 feasibility** and the **L2 power-law** curve.

**On BNCI Horizon 2015-004** (the Scherer 2015 data: 9 end-users, 30 channels, 2 days; word association, subtraction, navigation, hand and feet imagery; also in MOABB as `BNCI2015_004`). Run F1 under nested selection (day 1 selects, day 2 tests), plus the rim-free 4–20 Hz and frontal/temporal-only confound screens. The official server returned 403 through this machine's proxy, so access through MOABB is still to be confirmed. With no auditors, this data screens but **cannot prove purity**.
