# P_inner_speech: silent commands with a task-language prior

**Honest headline:** published scalp-EEG inner-speech decoding is barely above chance and often confounded. This is worth funding only as a data-scaling test with built-in purity controls and a clear kill switch.

## 1. One-line idea and signal basis

**Idea:** the user silently "says" one of five self-chosen commands (*left, right, up, down, grab*) or nothing. A 32-channel dry-EEG decoder outputs a posterior over these six states. A task-grammar prior and a goal-inference agent turn the sparse, noisy commands into continuous cursor or robot motion. The decoder is trained on 150 h per dense participant plus a 30-person cohort, and is audited by sEMG and an eye tracker that it never reads.

**Signal basis.** Inner speech is literal "thinking": people narrate intentions in words. Intracortically it is decodable:
- Kunz et al. 2025 (*Cell* 188(17), doi 10.1016/j.cell.2025.06.015), 4 participants: inner speech in motor cortex is a weaker, highly correlated version of attempted speech.
- Wandelt et al. 2024 (*Nat Hum Behav* 8:1136–1149), supramarginal gyrus: 8 internally spoken words at 79% and 23% in two participants (chance 12.5%).

The source is therefore **articulatory-motor and auditory imagery** in ventral sensorimotor cortex, IFG, STG and SMG. On the scalp it should appear as weak ERD/ERS over FC5/FC6, C5/C6 and T7/T8, plus syllable-rate theta (4–8 Hz) envelope structure. It should not appear as "lexical semantics".

**The scalp evidence is weak:**
- **Nieto et al. 2022** (*Sci Data* 9; OpenNeuro ds003626): 10 participants, 128 ch + 8 external channels, 4 Spanish direction words, 475–570 trials per participant across three conditions. Follow-up EEGNet work reports about **29.7%** for 4 classes (chance 25%). I rely on a secondary review for this number.
- **BCI Competition 2020 Track 3** (5 phrases including "stop", 64 ch): the CSBrain paper reports CBraMod at **0.537 balanced accuracy** (chance 0.20). Low trust as neural evidence: I could not verify EMG or order controls.
- **Porbadnigk et al. 2009** (21 subjects, 5 words): **45.5%** with block-ordered presentation fell to **chance** when the order was randomised. The "speech" signal was a temporal artefact.
- **Whitham et al. 2007** (*Clin Neurophysiol* 118:1877): neuromuscular blockade cut scalp power above 20 Hz by 10–200×. Most scalp EEG above 20 Hz is EMG.
- **AlterEgo** (Kapur et al. 2018, IUI): facial sEMG during *silent* internal articulation reached a 92% median word accuracy. Subvocal muscle activity carries exactly the label we want to attribute to cortex.

Prior: a small cortical signal under a large articulatory-EMG confound. Whether it grows with 100× more data per person is the untested question.

## 2. Label source: why no label is display-identifiable

All labels are the user's own covert choice, reported after the decoding window. No label is a threshold on anything displayed.

- **Free-choice blocks (main data):** the screen shows the same fixation cross for every class. The participant chooses a command, says it silently once at their own pace, and taps one "done" key with the left thumb (the same key for every class), which marks onset. **≥1.5 s after the window [−0.5, +2.0] s closes**, a report screen shows the 6 options at **randomised positions**, and the participant clicks one. Because positions are randomised, no class-specific motor plan, gaze target or lateralised readiness potential can start inside the window.
- **Memorised-sequence blocks:** a random 6-command sequence is shown, then a 3-s mask with a jittered blank, and the participant executes it on a blank screen at a self-paced rhythm. Sequences are randomised per block, which removes the Porbadnigk block-order artefact.
- **The semantics-to-eyes trap (an extension of the display trap).** Silently saying "left" may shift gaze or covert attention to the left. Direction words could then be decoded from eye signals with no display involved. Controls:
  - a parallel **pseudo-word vocabulary** (*bafe, tolu, mipa, kesh, rona*) mapped to the same 5 actions;
  - **eyes-closed blocks** (20% of the data);
  - a gaze-surrogate decoder (section 5).

  If direction words beat pseudo-words only through the gaze-correlated component, that advantage counts as an artefact.
- **Closed loop:** goals are **self-chosen** among 8 identical landmarks that are always visible, and the participant reports the goal after the trial. No target is ever singled out on screen. Training uses only windows that end before the effector moves.

## 3. Data engine and scaling mechanism

**Dense arm (the bitter-lesson test):** 3 participants × 75 sessions × 2 h = **150 h each**. About 70 min of command blocks per session at about 10 events/min gives **~700 labelled events per session and ~50k per participant**. (~100× public per-person counts). Chisco (Zhang et al. 2024, *Sci Data* 11:1265) shows that >900 min per subject of imagined-speech EEG is feasible.

**Cohort arm:** 30 participants × 2 sessions × 90 min, giving **~36k events**, for cross-subject pretraining and prevalence.

**Scaling mechanisms, each tested separately:**
1. **Within-person volume:** a fixed held-out test of each participant's last 10 sessions, never trained on. Training ladder: 250, 500, 1k, 2k, 4k, 8k, 16k, 32k events. Fit acc(n) = a − b·n^(−c) and report the asymptote *a* with a bootstrap CI.
2. **Pooling:** 0, 3, 10 or 30 cohort participants, plus 1k of the target's own events.
3. **Pretrained priors:**
   - (a) initialisation from LaBraM or CBraMod (modest gains in their own papers);
   - (b) contrastive alignment of EEG windows to HuBERT embeddings of the participant's *own overt recordings* of each word (the Défossez et al. 2023 *Nat Mach Intell* recipe moved from perception to imagery). The audio is only ever a training target and never a decoder input.

**Expected curve:** I expect fast saturation (CSP decoders plateau within hundreds of trials; in RLNF, volume helped only modestly). My forecast for 5-class EMG-clean accuracy is **24% at 500 events, 30–35% at 32k, plateau near 35%**, with 45–50% as the optimistic case. Pooling should add 2–4 points at low n and nothing at high n. A flat curve after 4k events means the bitter lesson failed for this signal.

## 4. Expected performance on dry 32-ch hardware

The Wolpaw ITR is computed per 3-s decision.

| Scenario | Accuracy | bits/decision | bits/s |
|---|---|---|---|
| Status quo, 4-class (Nieto-like) | 30% | 0.009 | 0.003 |
| Central forecast, 5-class | 35% | 0.088 | **0.03** |
| Optimistic, 5-class | 50% | 0.32 | 0.11 |
| Binary command-vs-rest detector (≈65%) | 65% | 0.07 | 0.02 |

These sit at the **bottom** of the project's established 0.03–0.3 bits/s band, below SMR cursor control (Wolpaw & McFarland 2004; Edelman et al. 2019). I estimate dry electrodes cost 3–5 points against gel, because the key sites (T7/T8, FC5/FC6) sit under temporalis muscle.

**Closed loop (8 self-chosen targets, 30-s limit, chance 12.5%):**
- Commands set persistent velocity modes, and the repo's Bayesian goal posterior blends motion toward the likely target.
- The repo simulation gave 0.49 success for a weak decoder alone and about 0.8 with the goal posterior. From that I estimate **success 0.35–0.55 and median time-to-target 15–25 s** at the central forecast, to be re-simulated with real posteriors in month 1.
- "Grab" at 35% accuracy is usable only with the grammar prior (allowed only within 5% of the workspace of an object, plus 2-of-3 confirmation). With it, I estimate fewer than 1 false grab per minute.

## 5. Purity proof

**Auditors (logged, never fed to the decoder):**
- 6 bipolar sEMG channels: suprahyoid, upper and lower orbicularis oris, masseter L/R, thyrohyoid;
- a throat microphone;
- a 250 Hz eye tracker;
- a head IMU.

All share LSL clocks.

**Pre-registered controls on identical trials, folds and n:**
1. **EMG surrogate:** the same architecture trained on the 6 sEMG channels. EEG must beat it. More decisively, accuracy is recomputed on **EMG-silent trials**: every auditor's RMS is inside the 95% rest band, *and* the EMG surrogate is at permutation chance on that subset. EEG must stay above chance there.
2. **Confound regression:** EEG-decoder correctness is modelled from EMG-decoder confidence, gaze-decoder confidence and EMG RMS. The partial dependence must be ≈0. If EEG wins only where EMG also wins, the claim is refuted.
3. **Band restriction:** the main decoder uses only **0.5–20 Hz** (Whitham). The 20–45 Hz decodability is reported separately as a contamination index.
4. **Channel ablation:** drop FP1, FP2, F7, F8, T7, T8. At least 50% of the above-chance margin must survive.
5. **Gaze surrogate** on eye-tracker x/y/pupil, and an **eyes-closed subset** where the effect must persist.
6. **Time and order surrogate:** a decoder from trial index, time in session and previous label must sit at chance.
7. **No-EEG matched control** (the RLNF lesson): phase-randomised EEG plus metadata.
8. **Condition gradient:** each session has 10 min each of overt, mouthed and inner. If EEG decodability falls in proportion to EMG decodability, the signal is EMG.
9. **Feedback circularity:**
   - 20% **frozen-feedback probes**, where the effector ignores the decoder for 3 s; accuracy must match non-frozen windows.
   - Training and adaptation use only pre-movement windows.
   - Closed-loop accuracy must not exceed open-loop blank-screen accuracy beyond its CI.

Permutation floors (1,000 shuffles within session) are computed at matched n on the fixed test set. There is **no performance-based exclusion**.

**Optional gel arm:** one dense participant does 30 sessions on 64-ch gel plus auditors, for source localisation (ventral sensorimotor/STG, not the rim) and the dry-gel gap. If no gel amplifier is available, the cost is an estimated €40–80k (unverified).

## 6. Protocol

- **Months 0–2:** section-10 analyses, auditor rig, re-simulation.
- **Months 2–9, open-loop engine.** P1–P3 do 3 sessions/week of 2 h. Each session:
  - 20 min setup;
  - 5 min rest, eyes open and closed;
  - 4 × 12 min free choice (2 direction-word, 2 pseudo-word);
  - 2 × 10 min memorised sequence;
  - 10 min overt/mouthed/inner gradient;
  - 8 min eyes-closed free choice;
  - per-block effort rating and strategy note.

  The cohort (n = 30) runs in months 3–8.
- **Months 9–15, closed loop** (P1–P3 plus 10 returning cohort members, 12 sessions each). Tasks: a 2-D 8-target cursor, a TurtleBot-class mobile robot, and a tabletop arm pick-and-place using "grab". Every session has frozen probes and open-loop calibration.
- **Months 15–18:** analysis. Total about **640 h** (450 dense, 90 cohort, ~100 closed loop).

## 7. Real-time architecture

- **Input:** LSL 500 Hz, 32 ch → 0.5–20 Hz → CAR → 128 Hz. Auditor streams are logged by a separate process with no socket to the decoder.
- **Windows:** 1.5 s, sliding every 100 ms.
- **Models:**
  - (a) baseline: Riemannian tangent space on filter-bank covariances (4–8, 8–13, 13–20 Hz) plus logistic regression;
  - (b) a 5M-parameter transformer initialised from CBraMod and fine-tuned per participant (AdamW, lr 3e-4, 30 epochs, early stopping on a fixed validation day).

  Processing latency is under 50 ms.
- **Evidence accumulation:** an HMM over {none, L, R, U, D, grab}, with a **task grammar** learned from logs (for example, P(grab | not near an object) = 0.02, P(repeat) = 0.3). A command is emitted when its posterior exceeds 0.8, giving an expected 2–4 s per decision.
- **Agent:** a command sets the velocity mode (0.15 workspace/s) until "stop", which is *none* held for 1.5 s, or until a new command arrives. The goal posterior blends assistance (α ≤ 0.5) and never reads the true goal.
- **Adaptation:**
  - unsupervised Riemannian re-centring every 60 s, which corrects dry-electrode drift as in the repo simulation;
  - a nightly supervised refit on self-reported open-loop labels;
  - closed-loop pre-movement windows labelled by post-trial goal report, at weight 0.5.
- **Effector interface:** a 10 Hz posterior plus a velocity setpoint, published to ROS2 `/cmd_vel` or the cursor API. Any robot that takes a velocity and a binary gripper works unchanged.

## 8. The Nature claim

**Headline figure:** per-participant learning curves on a log-n axis, from 250 to 32k self-chosen silent commands on a fixed held-out test. EEG is plotted against matched EMG, gaze, time, no-EEG and permutation floors on identical trials. An inset shows the EMG-silent curve and the overt→mouthed→inner gradient.

**The one sentence:** *"With 150 hours per person and muscle, eye and display auditors, scalp EEG decodes self-chosen silent commands above every matched non-neural surrogate, and this purified information grows with data to X% before saturating."* The alternative outcome is also a sentence: *"…and does not: published scalp-EEG inner-speech accuracies are accounted for by articulatory EMG, gaze and temporal confounds."*

**Why new:** I know of no inner-speech EEG study with >1k trials per word per person, an sEMG floor on identical trials, and self-chosen, post-hoc labels.

**Venue:** Nat Comms if positive; Nat Comms or J Neural Eng if negative. *Nature* needs positive plus the robot demo (~15%).

## 9. Falsifier and biggest risk

**Falsifier (pre-registered):** at ≥20k EMG-silent events, held-out-day 5-class EEG accuracy must exceed max(EMG, gaze and time surrogates, permutation 95th percentile) by at least 5 points in at least 2 of the 3 dense participants. The bootstrap CI of the learning-curve slope between 4k and 32k events must also exclude 0. If either fails, inner speech is dropped as a control channel.

**Biggest risk:** there is no usable cortical inner-speech information below 20 Hz on dry 32-ch, and everything decodable is subvocal EMG at the temporal rim, which is the RLNF failure in a new form. I put **≥50% probability** on ending below 33% 5-class accuracy.

**Second risk:** tedium; inner speech may weaken with automaticity (tracked against session index).

**Mitigation:** the agent, prior, auditor rig and data engine are signal-agnostic and can carry an SMR velocity decoder instead.

## 10. What can be tested now

**On ds004362 (109 subjects; imagined L/R fist; targets shown on screen).** It contains no speech, so it cannot test inner speech itself. It can test four things:
1. **The display trap:** decode target side from FP1/FP2/F7/F8 and occipital channels at 0–500 ms after the cue, and compare with C3/C4 mu/beta at 1–4 s. If the early window matches the late one, this shows the purity battery is needed and calibrates it.
2. **Cross-subject pooling of covert tasks:** 20 fixed test subjects, with training ladders of 1, 3, 10, 30 and 89 subjects. The slope and plateau decide whether the 90-h cohort arm is worth it.
3. **32-ch 10-20 subset and 20 Hz band restriction:** the accuracy cost bounds the hardware loss.
4. **Real posteriors through the agent:** feed held-out decoder posteriors into the repo's command→motion agent to replace the forecast success with an empirical one.

**Kill test on OpenNeuro ds003626 (Nieto; inner, pronounced and visualised conditions; 8 external channels including EMG and EOG; ~0.6 GB of epochs per participant).** Use the 32 10-20 positions and 0.5–20 Hz, decode the 4 inner-speech words within subject, train the identical model on the external channels as an EMG surrogate, and compare inner with pronounced.

- If EEG fails to beat its floors on these few hundred trials, the scaling hypothesis is **not** killed.
- If the external-channel surrogate *beats* EEG on inner-speech trials, the confound is confirmed. In that case the dense arm starts with the full auditor rig and switches to pseudo-words.

Not yet run; it is week one.
