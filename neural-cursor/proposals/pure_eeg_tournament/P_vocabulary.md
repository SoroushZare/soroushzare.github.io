# P_vocabulary — A personal mental vocabulary, discovered from data, driving a goal-level autonomous effector

## 1. One-line idea and signal basis

**Each person gets the 3–5 mental tasks their own cortex separates best (chosen from an 11-class library on held-out data and kept only if comfortable). These map to commands, and an autonomous controller turns the few bits/s into smooth cursor or robot motion.**

The signal is ongoing oscillatory reorganisation with no stimulus:
- Motor imagery: mu/beta ERD over the sensorimotor strip.
- Mental subtraction and word generation: frontal-midline theta and parietal/frontal alpha–beta ERD.
- Navigation and rotation: parietal alpha ERD.
- Auditory and face imagery: temporal/occipital alpha changes.
- Rest: the alpha baseline.

Friedrich, Scherer & Neuper (Int J Psychophysiol 2012) compared seven such strategies. Friedrich, Neuper & Scherer (PLoS ONE 2013, "Whatever works") screened them in 8 naive users over 10 sessions (~5 weeks) and chose a 4-class set per person. Mean per-user accuracies were 44–84% (peak sessions 58–93%; chance 25%), and the chosen sets differed between users. Scherer et al. (PLoS ONE 2015, 9 end-users with stroke, SCI or CP) found that pairing a "brain-teaser" with a "dynamic imagery" task beat hand-vs-feet, which was up to 15% worse on average in most users.

This is "thinking" in the literal sense: a self-generated mental act, with no stimulus and no movement. It is **not** natural intention (§9). I argue it is the most honest "by thinking" design the physiology supports, provided the person chose the tasks, the tasks are comfortable, and a robot does the low-level work.

## 2. Label source and why it is NOT display-identifiable

Every label is **the mental task performed**. It is never a direction, a target or a threshold on a displayed variable.

1. **Cue-remapped trials.**
   - Cue: an abstract glyph shown for 1 s, then removed. The glyph→task assignment is **re-drawn at random every block** and taught by a spoken key at block start.
   - Then 1.5 s of blank screen with fixation, then 6 s of task on a blank screen. Windows start ≥1.5 s after cue offset.
   - Over ≥24 blocks, the cue's visual features are independent of the label. This is tested directly (§5, glyph-transfer).
2. **Free-choice trials.**
   - The screen stays blank. The person picks a task, performs it for about 6 s, and reports it by keypress after the window closes.
   - The display carries nothing related to the label.
3. **Closed-loop decisions.**
   - Used for evaluation only. From month 9 they may also be used for adaptation, but only from blocks with a randomly permuted **task→command mapping**.
   - The label is the task mapped to the command the user confirmed. Because the mapping changes across blocks, "goal is on the left" carries no information about task identity once blocks are pooled. This removes the trap of "decode intended direction".

The decoder outputs task identity. Screen geometry exists only in the autonomy layer, which the EEG model never sees.

## 3. Data engine and scaling mechanism

The bitter lesson, aimed where the evidence says it pays.

**L1 — Library breadth (per person).** K tasks give K(K−1)/2 pairs. The best held-out pair improves like the maximum of correlated noisy draws (~σ√(2 ln M)): fast up to about 6–8 tasks, flat beyond about 10. Evidence:
- Scherer 2015: person-specific pairs beat the canonical pair.
- Friedrich 2013: the selected sets were idiosyncratic.
- This repo's ds004362 analysis (106 subjects, 32-ch g.tec subset, within-subject CV):

| Contrast | Accuracy | Subjects above permutation floor (0.644, n≈45) |
|---|---|---|
| Imagined left vs right fist | 0.643 | 38% |
| Imagined both fists vs feet | 0.710 | 65% |
| Idle vs imagine | 0.844 | n/a |

**The textbook left/right pair is the worst choice for most people, and the null command is the most reliable.** These numbers carry that dataset's display confound (§10).

Saturation: screening costs about 6 min per task per session, so the library is capped at 11.

**L2 — Hours per person.** A tangent-space classifier and a small CNN improve with trials per class. We pre-register a plateau at about 300 trials per class (≈30–40 h of use), driven by session non-stationarity rather than model capacity. That matches RLNF, where volume helped and architecture did not.

**L3 — Population prior.** Zero-shot cross-person EEG is weak: in this repo's E2 ladder, imagined L/R rose only from 0.58 to about 0.66 as the training pool grew from 1 to 32 subjects. We therefore pool **which tasks work for whom**, not decoders. A 5-min rest recording plus a 15-min mini-screen predicts each person's top tasks. The precedent is Blankertz et al. 2010 (NeuroImage): a 2-min rest predictor reached r=0.53 with SMR performance in 80 naive users. This cuts screening from 3 sessions to 1.

**L4 — Bits *needed* (the real bitter-lesson lever).** EEG bits/s will plateau at about 0.05–0.3 bits/s. The autonomy layer, however, learns a habit/context prior over goals from every logged session, without limit. If the prior's entropy over 8 goals falls from 3 to 1.5 bits, selection time halves at a constant EEG rate. This curve keeps improving after L2 saturates, and because it uses no EEG it cannot create a purity problem.

## 4. Expected performance on this hardware

Wolpaw bits per decision:
- 4 classes at 65%: **0.51 bits**
- 4 classes at 70%: **0.64 bits**
- 2 classes at 75%: **0.19 bits**

| Quantity | Median user, dry 32-ch | Top quartile | Anchor |
|---|---|---|---|
| 4-class held-out accuracy | 55–65% | 70–80% | Friedrich 2013, 44–84% on gel; 10–15 pp dry penalty assumed |
| EEG rate (asynchronous accumulation, 4–6 s per decision) | **0.05–0.12 bits/s** | 0.15–0.3 | project range 0.03–0.3 |
| Select 1 of 8 goals, flat prior | 25–60 s, 80–90% correct with confirm/undo | 12–25 s | 3 bits ÷ rate |
| Same, with habit prior (~1.5 bits) | 12–30 s | 6–12 s | L4 |
| Assisted continuous cursor, 8 targets | 0.5–0.75 success at 15 s | ~0.8 | repo sim: r≈0.40 decoder 0.49→0.78 with goal posterior; to be re-run at the measured confusion matrix |
| Robot pick-and-place, 6 objects | ≥80% completion, 30–90 s per object | | Millán 2004 (IEEE TBME; 3 mental states + robot behaviours, mental/manual ratio 0.74); Leeb 2015 (Proc IEEE; 9 end-users, shared-control telepresence, similar to 10 healthy controls) |

Royer, Rose & He (J Neural Eng 2011; 20 subjects, 8 sessions) found goal selection needed less training than process control to reach better speed, accuracy and information transfer. That is why autonomy sits at the high level.

## 5. Purity proof

**Auditors (recorded, never decoder inputs):**
- 250-Hz eye tracker and 4-electrode EOG.
- EMG on temporalis, masseter, frontalis, submental/laryngeal (subvocalisation), splenius and both forearm flexors.
- Head accelerometer and throat microphone.

The threats are concrete. Whitham et al. 2007 (Clin Neurophysiol, neuromuscular blockade) showed that scalp power above about 20 Hz during cognitive tasks is largely EMG. Loetscher et al. 2010 (Curr Biol) showed that eye position predicts the size of the next generated number.

**Pre-registered tests (matched n, fixed held-out sessions):**
1. **Auditor-surrogate decoder.** Same architecture and search, auditor channels only, same trials. Pass: surrogate ≤ permutation p95 + 5 pp, and EEG bits ≥ 2× surrogate bits.
2. **Conditional decodability.** Ridge-regress gaze, pupil, blink and 6-band EMG RMS out of the EEG features, then re-decode. Pass: ≥70% of bits retained.
3. **Band and site restriction.** The primary decoder is 1–30 Hz. A 1–20 Hz decoder without FP1/FP2/F7/F8/T7/T8 must keep ≥75% of bits. Any task that fails is struck from the vocabulary.
4. **Mimic blocks.** The person produces each task's tells (subvocal murmur, jaw tension, scanning saccades, paced breathing) without doing the task. Pass: decoder at chance.
5. **Eyes-closed and fixation-verified blocks.** Fixation is tracker-verified at <1.5°. Pass: drop ≤10 pp.
6. **Glyph-transfer and mapping-shuffle.** Train on assignment π₁, test on π₂. Pass: no collapse (a display reader would collapse).
7. **Feedback circularity.**
   - *Mapping reversal*: the mapping flips unannounced mid-block. A decoder reading its own visual consequence would follow the flip; a task decoder misdirects until the user switches tasks.
   - *Replay*: the person watches their own earlier session while resting. Pass: output at chance.
8. **Bookkeeping.** Permutation and circular-shift floors; no performance-based exclusion (deep-arm members are assigned by availability before screening); nested selection, so the winner's curse never reaches test data.

## 6. Protocol

**Library (11 classes):**
- Cognitive: serial-7 subtraction, word association, mental rotation, home navigation, song imagery, familiar-face imagery.
- Motor imagery: left hand, right hand, feet.
- One self-proposed task per person.
- Rest.

Comfort and effort are rated 1–7 after each run, and a task rated below 4 for comfort is ineligible.

**Phase A, cohort (months 1–5).** 24 participants × 3 sessions × 2 h.
- Each session: 11 classes × 30 trials × ~10 s (≈55 min of task), plus auditor calibration and mimic blocks.
- Sessions 1–2 are for selection; session 3 is held out.
- Output: a 24-person vocabulary atlas, which also trains L3.

**Phase B, deep arm (months 3–14).** 4 participants × ~100 h (2 h/day, 3–4 days/week). Daily blocks:

| Block | Minutes |
|---|---|
| Rest and auditor calibration | 8 |
| Cue-remapped trials (vocabulary + 2 challenger tasks) | 25 |
| Closed loop: cursor goal selection, 6-DoF arm pick-and-place, assisted cursor | 45 |
| Free choice | 10 |
| Mimic or replay probes | 5 |

- **Weekly vocabulary tournament:** a challenger replaces the weakest task only if it wins on that week's held-out day by ≥5 pp and is no worse on comfort.
- Fridays are frozen-decoder tests.

**Phase C (months 13–16).** 8 new participants with the frozen pipeline: an L3-guided 15-min mini-screen, then 4 × 2-h sessions, the last one a fixed test.

**Hardware.** The g.tec 32-ch dry cap. Optionally, one deep participant also records 10 sessions with a 64-ch gel cap for source localisation, using an existing lab amplifier; the project does not depend on it.

## 7. Real-time architecture

- **Acquisition:** LSL at 500 Hz, causal 1–30 Hz FIR, common average reference. No artifact rejection in the decoder path; rejection is analysed offline only.
- **Features:** 2-s windows every 250 ms. Filter-bank covariances (4–8, 8–12, 12–16, 16–20, 20–30 Hz) go to Riemannian tangent space. Each session is re-centred on 3 min of rest (Zanini et al. 2018, IEEE TBME), then continuously with λ=0.995.
- **Model:** per-person L2 multinomial logistic regression (C by nested CV), ensembled with EEGNet-8,2 (~3k parameters). The ensemble is kept only if it wins on held-out days. Output: task probabilities at 4 Hz.
- **Decision:** leaky log-odds accumulation (τ=3 s) with a threshold set for 85% precision on calibration data. Rest is the idle class, so the system is asynchronous. Latency is about 3–6 s per command.
- **Adaptation:** unsupervised re-centring always runs. Supervised updates happen nightly, only from cue-remapped and free-choice trials (plus mapping-shuffled confirmed choices after month 9). There are no online gradients on closed-loop output.
- **Effector interface (generality):** one probability vector serves two modes.
  - *Goal mode:* a hierarchical goal menu. A Bayesian posterior combines EEG likelihood with the habit prior. When it exceeds 0.9, the cursor or robot (MoveIt) executes a minimum-jerk trajectory. An undo task is available.
  - *Steer mode:* velocity = Σₖ pₖuₖ, blended with posterior assistance a ∈ {0, 0.3, 0.6}; a=0 is always measured.
  - The decoder never knows which device is attached.

## 8. The Nature claim

**Headline figure:**
- (a) A 24-person × 10-task held-out separability atlas: the best vocabulary is idiosyncratic and rarely the canonical motor imagery pair.
- (b) 100-h curves for 4 people: EEG bits/s plateau, bits needed keep falling, and the auditor-surrogate and mimic curves stay at the floor.
- (c) Invariance to mapping shuffle and reversal.
- (d) One decoder output driving a cursor, a robot arm and a mobile base.

**Sentence:** *"Scalp EEG alone, with no stimulus, movement or gaze, gives each person a self-discovered vocabulary of mental acts that is provably cortical and stable over 100 hours, and goal-level autonomy turns its ~0.1 bit/s into general device control that keeps getting faster with use."*

**Honest novelty.** Millán 2004 already drove a robot with 3 mental states, and Friedrich and Scherer already selected tasks per person. What is new:
1. Vocabulary discovery as a scaled, held-out tournament engine.
2. The first mental-task BCI with a full auditor suite and mimic controls, which matters because brain-teaser tasks are the most contamination-prone.
3. Scaling decomposed into bits delivered (plateaus) and bits needed (keeps falling).

Realistically this is a Nature Communications paper. Nature would need top-quartile users at ≥0.25 bits/s in real-world use.

## 9. Falsifier and biggest risk

**Pre-registered falsifiers:**
- **F1:** the per-person vocabulary does not beat the fixed canonical set (both-hands/feet/rest) by ≥5 pp in median held-out accuracy under nested selection. Personalisation is then noise, and the design reduces to standard SMR.
- **F2:** after purity tests 1–4, the cognitive tasks lose ≥50% of their bits. They were EMG or eyes.
- **F3:** glyph-transfer or mapping reversal fails. The decoder is reading the display.
- **F4:** frozen-test bits/s at hour 100 are no better than at hour 20, *and* the habit prior saves <0.5 bits. Neither scaling lever exists.

**Biggest risk: F2.** Word association and arithmetic are exactly the tasks with the strongest subvocalisation, jaw and number-space eye movements, and the dry cap has poor temporal contact. Clean cortical signal could fall to about 0.05 bits/s.

**Second risk: naturalness.** These are instructed mental tasks, and serial-7s for two hours is fatiguing. Choice and comfort gating help but do not remove this. A fair naturalness score is about 6/10.

## 10. What can be tested NOW on ds004362

ds004362 holds a 5-state mini-library per subject: imagined left fist, right fist, both fists, feet, and rest (T0). Using the 32-ch g.tec-matched subset and the existing `realdata/` cache:

1. **Personalisation hold-out.** Select each subject's best 2- and 3-class subset on runs 4/6/8/10 and test on runs 12/14. Compare against a fixed subset and a leave-subject-out population-best subset.
   - *Support:* per-person selection gains ≥3 pp in ≥55% of 106 subjects.
   - *Kill:* median gain ≤0.
2. **Cross-run confound.** L/R and fists/feet come from different runs. Use "rest-in-LR-run vs rest-in-FF-run" as a floor; cross-run pairs must exceed it by ≥5 pp.
3. **Display audit.** Targets are displayed (horizontal for L/R, vertical for fists/feet). Compare frontopolar-only, occipital-only and sensorimotor-only decoders, and imagined against executed. Flag any subset where the frontal or occipital decoder matches the full decoder. This dataset can screen for confound but **cannot prove purity**.
4. **L3 feasibility.** Predict each subject's best subset from baseline runs 1–2 (SMR-predictor and alpha-peak features), leave-subject-out. If r<0.2, drop L3.
5. **L2 curve.** Accuracy at 10/20/30/45 trials per class, fitted with a power law to set the Phase B pre-registration.

The repo already shows both-fists/feet (0.710) and idle/imagine (0.844) beating left/right (0.643). Test 1 decides whether choosing *per person* adds anything beyond choosing the right pair for everyone.
