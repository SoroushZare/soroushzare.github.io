# P_live_in: an N-of-1 co-adaptive data engine. Does EEG control keep improving with hours in one brain?

## 1. One-line idea and signal basis

**Idea.** Three people each spend about 200 hours in closed loop from session 1. The decoder retrains on every new minute, and the person finds whatever mental strategy works. The headline result is each person's **scaling law**: display-free accuracy against accumulated hours, split into a machine part (data) and a human part (learning).

**Signal basis.** We do not prescribe a strategy, so we name the signals we expect the loop to find, and we test for each:
- Lateralised mu/beta (8–30 Hz) desynchronisation (ERD) over C3/C4/CP, the substrate of trained SMR cursors (Wolpaw & McFarland 2004 PNAS; Edelman et al. 2019 Sci Robotics).
- Endogenous posterior alpha lateralisation toward a self-chosen goal while gaze stays fixed.
- Signatures of user-chosen mental tasks (Friedrich et al. 2013 PLoS ONE, "whatever works").

All come from deciding where to go; none is stimulus-evoked. Operant learning changes the EEG itself:
- Perdikis et al. 2018 PLoS Biol: two tetraplegic Cybathlon pilots over months, with learning at the application, output and EEG levels.
- Benaroch et al. 2021 Front Hum Neurosci: one user, 20 sessions over 3 months, adaptive Riemannian classifier, about +30% accuracy.

Intracortical co-adaptation shows the same pattern (ReFIT, Gilja 2012; CLDA, Orsborn 2014 Neuron), and a never-reset ECoG decoder kept improving over days (Silversmith et al. 2021 Nat Biotechnol). Our bet: this loop, run 10–20× longer per person than any published closed-loop EEG cursor study, still has slope left.

## 2. Label source: why no label is display-identifiable

ReFIT relabelling uses "the direction from the cursor to the target". When the target is drawn on screen, gaze, visual evoked responses and the cursor's own motion all encode that label. Four steps remove the trap.

1. **Private goals.** Eight identical, unmarked rings sit at 4° eccentricity around a fixation cross all session. The person picks a ring privately on each trial; the display never marks it.
2. **Randomised-mapping pre-commitment.** Before each trial the person taps the goal on a side tablet. The tablet rotates the ring layout by a random multiple of 45° on every trial, so finger and gaze movement is independent of the goal's screen direction. The cursor appears 1.5 s later.
   - The label is the committed goal, fixed before any cursor motion.
   - The tap location is logged as a muscle-leak surrogate.
3. **Hidden-cursor probes (the purity measurement of record).** On 15% of trials, randomly interleaved, the cursor is invisible for the 4 s decoding window and is revealed afterwards. With fixation enforced, the screen is pixel-identical across all eight goals, so **these labels are display-unidentifiable by construction**. The headline scaling law is fitted only on these trials.
4. **Visible trials (training only).** ReFIT-style labels: keep the decoded speed and rotate it toward the committed goal. They depend partly on the displayed cursor position, so they are used for training and never as purity evidence. If the loop learned to read its own visual consequences, the **visible − hidden gap** would widen; we track it every session.

## 3. Data engine, scaling mechanism and expected curve

**Two coupled processes.**
- **Machine.** More own-brain data lowers estimator variance and lets a small CNN overtake a linear decoder.
- **Human.** Operant learning reshapes the activity that drives the decoder (Ganguly & Carmena 2009 PLoS Biol; Orsborn 2014). Sadtler 2014 Nature limits fast learning to the existing neural manifold, which predicts quick early gains and a slow tail.

**Decomposition, measured every session.**
- **Human only:** a frozen-decoder probe. D_ref is fit on sessions 1–3, never updated, and run in closed loop for 5 min.
- **Machine only:** a data ladder on fixed test sets. Train offline on the first 1, 2, 4, …, 128 h. Evaluate on two fixed sets of hidden probes: T_early (sessions 4–6) and T_late (the last 10 h, never trained on).
- **Both:** the live co-adaptive curve, i.e. hidden-probe accuracy of the deployed decoder.
- **EEG-level change:** cross-validated Fisher separability of the 8 goals on a fixed small feature set.

**Pre-registered fit.** Per person: acc_hidden(h) − chance = A·h^α, compared with a version that breaks at h*. Bootstrap over sessions. Fit on 0–150 h and predict 150–200 h.

**Honest expectation.** Linear MI decoders usually saturate within a few hundred trials per class. Published per-person closed-loop data stop at about 10–20 h. Stieger et al. 2021 Sci Data has 62 people, about 600 h in total, up to 11 sessions each; Kaya et al. 2018 Sci Data has 60 h across 13 people. We predict:
- The linear machine ladder plateaus by 10–20 h.
- The CNN keeps rising to 50–100 h.
- The human curve rises steeply over about 10 sessions. Edelman 2019 reports nearly 60% faster centre-out learning; Benaroch reports about +30% over 20 sessions.
- The open question, and the novelty, is whether α > 0 beyond 50 h. We expect saturation somewhere between 30 and 150 h, set by dry-cap SNR and non-stationarity. The design measures where it happens.

RLNF (gains only from data volume) fits the machine half of the bitter lesson; live-in asks whether one brain also supplies the human half.

## 4. Expected performance on the dry 32-ch cap

Anchors:
- This project's established range for the cap: 0.03–0.3 bits/s.
- Repo simulation: r ≈ 0.4 gives 0.49 success on 8 targets alone and about 0.8 with the goal posterior.
- Vidaurre et al. 2011 Neural Comput: co-adaptive calibration gave 6 users good control within 3–6 min.
- Kaifosh et al. 2025 Nature: sEMG reaches 0.66 target acquisitions/s, a non-invasive ceiling we will **not** reach.

| Hours | Hidden 8-ring acc. (chance 12.5%) | Velocity r | Visible success, raw / assisted (6 s) | Bits/s, raw / assisted |
|---|---|---|---|---|
| 0–4 | 14–18% | 0.10–0.15 | ~0.20 / ~0.45 | ~0.02 / ~0.1 |
| 50 | 22–30% | 0.25–0.35 | ~0.40 / ~0.75 | ~0.07 / ~0.25 |
| 200 (optimistic) | 30–40% | ~0.4 | ~0.50 / ~0.80 | ~0.10 / ~0.30 |

Wolpaw ITR per 6 s selection (8 targets, P = 0.8: 1.72 bits, 0.29 bits/s). Time-to-target on hits: about 5 s falling to about 3 s. **Pessimistic case:** a plateau at about 0.15 bits/s assisted by 20 h.

## 5. Purity proof

**Auditors (recorded on LSL, never in the decoder's graph, neither as input nor as loss):**
- An eye tracker at ≥500 Hz.
- Six bipolar EMG channels: bilateral forearm, bilateral masseter/temporalis, trapezius.
- A cap IMU.
- A 2-min artifact calibration per session (blinks, saccades, jaw clench, head turns).

**Matched non-neural surrogates** (same model class, trials, CV folds and n): gaze (x, y, pupil); EMG envelopes plus 60–120 Hz power on FP1/FP2/T7/T8; IMU; display (rendered cursor kinematics and ring states); tap location.

**Controls.**
1. **Hidden probes.** The display surrogate is at chance by construction. EEG must beat the permutation floor (1,000 within-block shuffles that preserve goal frequencies, at matched n) and every surrogate.
2. **Partialling.** EEG stays above the floor after regressing out all surrogate predictions.
3. **Montage ablation.** Sensorimotor-only channels (FC/C/CP, 13 channels, no FP/F7/F8/T7/T8/O) keep at least 70% of the full montage's above-chance accuracy.
4. **Sham-feedback dissociation (5% of trials).** The cursor replays an unrelated trajectory. Decoder output must track the committed goal, not the displayed motion.
5. **Replay viewing (5 min per session).** While the person fixates and passively watches replays of their own trials, the frozen decoder must not decode the replayed goal.
6. **Gap alarm.** Pre-registered alarm if the visible − hidden gap grows by more than 10 points over 50 h.
7. **No exclusions.** Fixation breaks and artifact epochs are kept. Velocity is clamped, not rejected, during amplitude bursts (rejection cost killed RLNF's loop).

**Free-strategy drift.** A weekly auditor report tells the person if control tracks jaw EMG or gaze; flagged sessions stay in every analysis.

## 6. Protocol

**Live-in participants.** Three able-bodied adults. Target 200 h net closed loop each, minimum 100 h. Three 2-h sessions per week for about 40 weeks.

**Session (≈100 min net):** 3 min rest; 2 min artifact calibration; 5 min frozen-D_ref probe; 4 × 20 min co-adaptive blocks (80% visible, 15% hidden, 5% sham); 5 min replay viewing; 3 min rest; a 2-line written strategy note.

**Trials.**
- Up to 6 s per trial; a 0.5 s dwell selects a ring; 2 s inter-trial interval.
- About 400 trials/h, which gives about 80,000 trials and about 10,000 hidden probes per person.
- A 10-h bin holds about 400 probes, giving roughly ±4.5% (95% CI).

**Instruction.** Only this: "move the dot to the ring you chose, by thinking; eyes on the cross; don't move." The Friedrich task menu is offered only if the person is stuck at session 5.

**Short cohort.** 20 people × 3 sessions × 2 h (120 h in total) for:
- the population's early slope and between-person variance;
- a pooled prior;
- a **strategy-transmission test**: half receive the live-in participants' converged strategy notes, half receive nothing.

**Optional gel arm.** One live-in participant does every 10th session with a 64-ch gel cap under the same pipeline, to test whether the plateau is set by the sensor. Cost: one gel amplifier and caps (vendor quote needed; we expect a five-figure euro sum) plus about 10% of that person's session time.

**Team and timeline.** 1 postdoc, 2 RAs. Months 0–2 build and §10 tests; 2–12 live-in collection; 6–10 short cohort; 12–14 robot test and analysis; 14–18 writing.

## 7. Real-time architecture

**Input and preprocessing.** LSL at 500 Hz. Causal 1–40 Hz IIR filter, common average reference excluding FP1/FP2, then amplitude clamp.

**Decoders (output at 20 Hz, 1.0 s window, 50 ms hop):**
- **A, Riemannian.** Covariances in 4 bands (4–8, 8–13, 13–20, 20–30 Hz), tangent space, ridge regression to 2-D velocity. The covariance mean is re-centred without labels every 30 s (Zanini et al. 2018 IEEE TBME); in our simulation this recovered gain drift.
- **B, CNN.** A causal EEGNet-style network with about 50k parameters on the raw window.
- **Ensemble.** A and B are weighted by accuracy on the last 200 hidden probes.

**Adaptation:**
- **Fast:** SmoothBatch (Dangi et al. 2013 Neural Comput) every 60 s, W ← 0.9·W + 0.1·W_batch, on ReFIT-relabelled data from the last 5 min.
- **Slow:** nightly retraining on the full history, with per-session re-centring.
- **Deployment gate:** a new model goes live only if it is not worse by more than 1 SE on the latest 400 hidden probes. This selects models; it never excludes data.

**Output.** A Kalman velocity smoother plus a Bayesian goal posterior over the 8 rings. The posterior never reads the committed goal, and raw and assisted results are always reported separately. End-to-end latency under 80 ms.

**Effectors.** One LSL stream `intent` carries {v_x, v_y, p(goal_1..8), confidence}.
- Cursor: velocity × gain.
- Robot arm: planar end-effector velocity capped at 10 cm/s, plus a shared-autonomy reach planner over 8 physical objects.
- Month 12: robot runs from the frozen decoder, no retraining (2 sessions/person).

## 8. The Nature claim

**Headline figure.** Log–log plot of display-free accuracy above chance against cumulative hours (1–200) for three people, with α and h* marked.
- Overlaid: the frozen-decoder curve (human), the fixed-test-set ladder (machine), and the forecast from 0–150 h against the observed 150–200 h.
- Flat at chance throughout: the gaze, EMG, IMU, display and tap surrogates, and replay viewing.
- Inset: the visible − hidden gap over hours.

**The sentence.** "Purely cortical, stimulus-free EEG control keeps improving for at least N hours in a single brain, following a measurable per-person power law, with human learning and machine data each contributing, verified on trials where the screen carries zero information about the goal."

**Why it is new.**
- No published closed-loop EEG cursor study gives a per-person curve past about 10–20 h, separates the human and machine contributions on a fixed test set, or certifies display-free labels.
- Kaifosh 2025 shows bitter-lesson scaling across thousands of people with sEMG. This is the within-person counterpart for brain signals.
- A clean saturation result (α ≈ 0 after h*) still sets the field (likely Nat Commun).

## 9. Falsifier and biggest risk

**Falsifiers (pre-registered; tested in each person; the claim needs 2 of 3 people):**
1. Hidden-probe accuracy at 100 h is not above sessions 4–6 (bootstrap 95% CI).
2. The visible − hidden gap grows by more than 10 points, or the sham dissociation favours displayed motion: the loop reads the display.
3. Any surrogate reaches at least the EEG accuracy on hidden probes, or EEG falls to the floor after partialling.
4. Both the frozen-decoder curve and the ladder are flat after 8 h.

**Biggest risk.** A free-strategy co-adaptive loop is an optimiser. It will exploit anything that correlates with the goal, and with dry electrodes the easiest such signals are jaw tension, micro-saccades and head motion. Fixation, auditor feedback and the montage ablation *detect* this; nothing fully prevents it.

**Second risk.** A dry-cap SNR ceiling (about 0.15 bits/s by 20 h). Stopping rule: if α < 0.02 at 50 h, with a CI excluding 0.1, the remaining hours go to the short cohort and the gel arm.

**Third risk.** Attrition and dry-cap discomfort; mitigated by breaks, three participants, a 100-h minimum.

## 10. What ds004362 can test now

ds004362: 109 subjects, 64-ch gel EEG, 160 Hz. Imagined L/R runs 4/8/12 give about 45 trials per subject. Targets are displayed and there is no feedback. Loader: `neural-cursor/realdata/load.py`. Channels are subsampled to the g.tec 32 positions.

- **T1, per-person slope at small n.** Train chronologically on the first k = 5…30 trials and test on a fixed last 15. Use tangent-space + logistic regression and CSP-LDA, with a matched-k permutation floor. Fit acc(k) − 0.5 = A·k^α.
  - **Support:** median α > 0 with at least 40% of subjects' CIs excluding 0, on the sensorimotor montage.
  - **Kill:** a plateau by k = 10.
- **T2, display-identifiability audit.** Decode the same labels from occipital-only, frontopolar-only and sensorimotor-only channels, in an early window (0–0.5 s, visual evoked) and a late window (1–4 s, ERD).
  - **Support:** sensorimotor late beats both.
  - **Kill (for using such data as a prior):** occipital or frontopolar is at least as good.
- **T3, what a crowd is worth to one brain.** Pool N = 1, 4, 16, 64, 108 other subjects with re-centring, then fine-tune on k own trials. Report each prior's own-trial equivalent. A prior worth more than 45 own trials means live-in starts from the population model; a prior worth about 0 means N-of-1 is right.
- **T4, drift half-life.** A run-4 decoder tested on runs 8 and 12, with and without re-centring. This sets the SmoothBatch rate.
- **T5, no-feedback null.** Decodability should not rise across runs 4/8/12 without feedback, the baseline for claiming live-in learning.

**Limit.** ds004362 cannot test hours-scale co-adaptation; next, Stieger et al. 2021 (closed-loop SMR, up to 11 sessions/person, figshare) fixes the curve's first segment before participant 1.
