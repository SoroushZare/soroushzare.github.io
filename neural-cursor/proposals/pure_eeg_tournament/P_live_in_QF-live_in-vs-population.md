# P_live_in (amended): an N-of-1 co-adaptive data engine. Does sensorimotor EEG control keep improving with hours in one brain?

## 1. One-line idea and signal basis

**Idea.** Four people each spend 80–120 h in closed loop. The decoder retrains every minute and every night, and the person finds whatever strategy works inside fixed purity constraints. The headline is each person's **scaling law**: performance on held-out, display-free, fixation-held, microsaccade-free trials against hours, split into own-data volume at fixed recency (machine), brain-state separability at a fixed data budget (human), and drift.

**Signal basis.** Lateralised mu/beta (8–30 Hz) desynchronisation over FC/C/CP, the substrate of trained SMR cursors (Wolpaw & McFarland 2004 PNAS; Edelman et al. 2019 Sci Robotics), and user-chosen mental tasks (Friedrich et al. 2013 PLoS ONE). The decoder outputs **continuous 2-D velocity**. SMR gives about two independently trainable control axes, which is all a 2-D velocity needs. The 8 goal rings are a readout of direction, not 8 independent classes.

**Posterior alpha lateralisation toward a goal is no longer a target signal.** Covert attention brings goal-directed microsaccades (Hafed & Clark 2002; Engbert & Kliegl 2003), which produce scalp transients (Yuval-Greenberg et al. 2008 Neuron), and the link is not obligatory (Liu, Nobre & van Ede 2022 Nat Commun). It is measured as a confound family and partialled out (§5).

**Learning evidence.** Perdikis et al. 2018 PLoS Biol (months of mutual learning, two pilots); Benaroch et al. 2021 Front Hum Neurosci (one user, 20 sessions); intracortical ReFIT (Gilja 2012), CLDA (Orsborn 2014 Neuron) and a never-reset ECoG decoder (Silversmith et al. 2021 Nat Biotechnol).

## 2. Label source: why no evaluation label is display-, eye- or muscle-identifiable

1. **Private goals.** Eight identical, unmarked rings sit at 4° eccentricity around a fixation cross. The display never marks a goal.
2. **Sequence pre-commitment.** Every 4 trials, the person taps the next 4 goals on a side tablet whose layout is rotated by a random multiple of 45° per sequence, so finger and gaze movement is independent of screen direction. The last tap comes **≥3 s before the first decoding window**, keeping post-movement beta rebound and the gaze return out of it; taps are 4× rarer than before. Tap location and layout are logged as surrogates.
3. **Hidden trials (20%, interleaved).** The cursor is invisible during the 4-s window and revealed afterwards; the screen is pixel-identical across goals. **S-probes (8%)** serve model selection, ensemble weights and the deployment gate. **E-probes (12%)** are the evaluation stream of record: never used to train, select, weight or tune any deployed model, each scored prospectively by the model live at that moment.
4. **Clean windows of record.** Within E-probes, the headline uses 1-s windows with gaze inside 1° and no microsaccade (Engbert–Kliegl, 1 kHz tracker) within ±100 ms; we expect about 30% to qualify, up to about 7,000 per person. The criterion is auditor-based and pre-registered, never performance-based; all-window results are reported alongside.
5. **Visible trials (75%) are for training only.** They use ReFIT labels (keep the decoded speed, rotate it toward the committed goal). These labels depend on the displayed cursor, so they are never purity evidence.

## 3. Data engine, scaling mechanism and decomposition

**Mechanisms.** Machine: more own data lowers estimator variance and lets a small CNN overtake the linear decoder. Human: operant learning reshapes the driving activity (Ganguly & Carmena 2009 PLoS Biol); Sadtler 2014 Nature predicts fast within-manifold gains and a slow tail.

**Prior.** Decoder A starts from a re-centred central-channel model pooled across public data (ds004362; Stieger et al. 2021 Sci Data), so hour 0 is not chance. Every curve is reported both as "prior + own hours" and as "own hours only".

**Decomposition. All test sets come from E-probes only.**

| Component | Measure |
|---|---|
| Machine, volume at fixed recency | Training sets all end just before the fixed test set T_late (last 10 h) and extend backwards 1, 2, 4 … 64 h. |
| Machine, recency control | Fixed-size 8-h training windows at lags 0, 8, 16, 32 and 64 h before T_late. |
| Machine, random-subsample ladder | n hours drawn uniformly across the whole history. Also tested on T_early (sessions 4–6). |
| Human, fixed-budget separability | Every 3 sessions, a fresh decoder is cross-validated on exactly 300 hidden trials from that block, with committed-goal labels and no history (offline only). A rise means more separable brain states, independent of accumulated data. |
| Drift and co-adaptation | Frozen-decoder matrix: models frozen at 1, 5, 20, 50 and 100 h are scored offline on all later E-probes. This replaces the closed-loop D_ref probe, so a stale decoder costs no session time and gives no practice. |
| Both | Live curve of the deployed decoder on clean E-probe windows. |

**Hours versus calendar.** Two participants run 4 sessions/week and two run 2/week, decorrelating hours from weeks. Pre-registered: hours index the curve better than weeks (ΔAIC > 4). Impedance and cap-fit time are covariates.

**Pre-registered fit.** Per person: cosine(decoded velocity, committed direction) on clean windows = A·h^α, compared with a broken power law at h*. Session-block bootstrap; fit on each person's first two-thirds of hours, predict the last third.

**Honest expectation.** This project's ds004362 audit finds the per-person axis flat from 10 to 45 cued, no-feedback trials: pessimistic for the machine half at small n, silent on hours of closed loop. We expect the linear ladder to plateau by 10–20 h, the CNN to rise to about 50 h, and separability to rise over about 10 sessions (Benaroch: about +30% over 20). The open question is α > 0 beyond 50 h; we expect saturation at 30–100 h.

## 4. Expected performance on the dry 32-ch cap

Anchors: this project's range of 0.03–0.3 bits/s; the repo simulation (r ≈ 0.4 gives 0.49 success on 8 targets alone, about 0.8 with the goal posterior); Vidaurre et al. 2011 Neural Comput (co-adaptive calibration usable within minutes). The cosine is primary; 8-ring accuracy is a coarse readout.

| Hours | Cosine (clean E-probes) | Hidden 8-ring acc. (chance 12.5%) | Visible success, raw / assisted | Bits/s per 6-s trial, raw / assisted |
|---|---|---|---|---|
| 0–4 | 0.08–0.15 | 14–18% | ~0.20 / ~0.45 | ~0.01 / ~0.08 |
| 50 | 0.20–0.30 | 18–25% | ~0.35 / ~0.70 | ~0.04 / ~0.21 |
| 120 (extrapolated) | 0.25–0.35 | 20–30% | ~0.45 / ~0.78 | ~0.08 / ~0.27 |

- The 120-h row extrapolates the very curve the study measures and has no direct anchor.
- **Pessimistic case:** a plateau at about 0.1 bits/s assisted by 20 h.

## 5. Purity proof

**Decoder input.** 26 channels. FP1, FP2, F7, F8, T7 and T8 are **excluded from the decoder entirely**, not only from the reference. They are recorded as an in-cap edge-EEG surrogate.

**Auditors (on LSL; never decoder input or loss):** 1 kHz eye tracker; six bipolar EMG (forearms, masseter/temporalis, trapezius); cap IMU; 2-min artifact calibration per session.

**Matched surrogates** (same model class, trials, folds, n **and hours ladder**): gaze plus microsaccade-direction histograms; EMG envelopes and EMG power **in the decoder's own 4–30 Hz bands** (cranial EMG reaches beta and central sites; Goncharova et al. 2003 Clin Neurophysiol); edge-EEG 4–30 Hz covariances; IMU; rendered display; tap and layout.

**Controls:**
1. **Clean E-probe windows.** EEG beats the permutation floor (1,000 within-block shuffles at matched n) and every surrogate.
2. **Surrogate scaling.** Each surrogate gets its own α on the hours axis, and the EEG α must exceed every one of them. If EEG rises while a gaze or EMG surrogate rises with it, the gain is learned leakage, not brain control.
3. **Partialling.** EEG stays above the floor after regressing out all surrogate predictions, the microsaccade rate and direction, and posterior alpha lateralisation.
4. **Montage ablation.** A sensorimotor-only decoder (FC/C/CP, 13 channels) keeps **≥80%** of the 26-channel above-chance performance. Its curve is shown as a co-headline.
5. **Display-following coefficient.** Sham trials make up 5%, about 2,400 per person; on them the cursor replays an unrelated trajectory. We regress decoded velocity on displayed velocity lagged 100–500 ms. Pre-registered: β_display must stay below 0.05 (95% CI) and β_goal must be greater than 0.
6. **Replay viewing.** 5 min/session, about 7 h per person: frozen decoders must not decode the replayed goal.
7. **Label-provenance test.** We train on equal amounts of ReFIT-labelled visible data and of committed-goal S-probes, and test both on E-probes. A ReFIT advantage on visible trials that disappears on E-probes indicates circularity. The visible − hidden gap is now descriptive only.
8. **Goal-blind auditor veto (anti-shaping).** If the EMG envelope exceeds 3 SD of rest, gaze leaves 1.5°, or head rotation exceeds 2°, the cursor freezes and those samples get zero weight in adaptation. The veto carries no goal information, so it can only remove control and reward, never add them; auditors never steer. This extinguishes artifact strategies. Vetoed data stay in every analysis; veto rate is reported.
9. **No performance-based exclusion.**

## 6. Protocol

**Live-in participants.** Four able-bodied adults, drawn at random from consenting short-cohort volunteers, never screened on performance. Sessions net 80 min. **Pair A:** 4/week, 90 sessions (120 h) in about 23 weeks. **Pair B:** 2/week, 60 sessions (80 h) in about 30 weeks. Both keep ≥8 weeks of slack within months 2–11.

**Session (about 2 h):** 3 min rest; 2 min artifact calibration; 4 × 20 min co-adaptive blocks with breaks (75% visible, 20% hidden, 5% sham); 5 min replay viewing; 3 min free-gaze "natural use" block (descriptive only); strategy note; comfort rating.

**Trials.**
- Up to 6 s per trial, 0.5 s dwell, 2 s inter-trial interval: about 400 trials/h.
- At 120 h that gives 48,000 trials and 5,760 E-probes; each 10-h bin holds 480.

**Power, honestly.** Binomial SE about 2 points per bin; with an assumed 5-point between-session SD (pilot-measured), bin SE about 2.6. A 50→120 h gain of about 6 points is detectable per person at 80% power; the cosine adds power; smaller tails are reported as equivalence bounds.

**Instruction.** Only this: "move the dot to the ring you chose, by thinking; eyes on the cross; don't move." Fixation is a deliberate purity cost: with free gaze and a known goal, the decoder would learn gaze. The Friedrich task menu is offered at session 5 only if the person is stuck.

**Non-performers** are reported, not excluded. The claim is stated as "k of 4".

**Short cohort.** 16 people × 3 sessions: early slope, between-person variance, pooled prior, and a strategy-transmission test (half receive the live-in strategy notes).

**Optional gel arm.** One pair-A participant does every 10th session with a 64-ch gel cap, to test whether the plateau is set by the sensor. Cost: one gel amplifier and caps (vendor quote needed; we expect a five-figure euro sum).

**Team and timeline.** 1 postdoc, 2 RAs, one rig, about 12 sessions/week at peak. Months 0–2 build and §10; 2–11 live-in; 4–8 short cohort; 11–13 robot and analysis; 13–18 writing. A participant who withdraws contributes their curve up to withdrawal.

## 7. Real-time architecture

**Input and preprocessing.** LSL at 500 Hz, 26 channels. Causal 1–40 Hz IIR filter, common average reference over the 26 channels, then amplitude clamp.

**Decoders (output at 20 Hz, 1.0 s window, 50 ms hop):**
- **A, Riemannian.** Covariances in 4 bands (4–8, 8–13, 13–20, 20–30 Hz), tangent space, ridge regression to 2-D velocity. Initialised from the population prior. The covariance mean is re-centred without labels every 30 s (Zanini et al. 2018 IEEE TBME), which recovered gain drift in our simulation.
- **B, CNN.** A causal EEGNet-style network with about 50k parameters; it joins once there are 10 h of own data.
- **Ensemble.** Weights come from the last 200 **S-probes**.

**Adaptation.** Fast: SmoothBatch (Dangi et al. 2013 Neural Comput) every 60 s, W ← 0.9·W + 0.1·W_batch, on veto-weighted ReFIT data. Slow: nightly retraining on the full history excluding E-probes. Gate: a new model goes live only if not worse by >1 SE on the latest 400 S-probes (about 12 h).

**Output.** A Kalman velocity smoother plus a Bayesian goal posterior over the targets. The posterior never reads the committed goal, and raw and assisted results are reported separately. End-to-end latency under 80 ms.

**Effectors.** One LSL stream `intent` carries {v_x, v_y, p(goals), confidence}: cursor velocity × gain, or planar robot end-effector velocity (≤10 cm/s) with shared autonomy. **Robot test:** 4 sessions per person, frozen decoder, objects in 3 random layouts unlike the ring geometry, gaze free and audited, same surrogates.

## 8. The Nature claim

**Headline figure.** Log–log clean-window E-probe performance against hours for 4 people, with α and h*, and every surrogate curve with its own α on the same axis. Panels: fixed-recency volume against the recency control (machine); fixed-budget separability (human); frozen-decoder matrix (drift); forecast against observed.

**The sentence.** "In single brains, stimulus-free sensorimotor EEG control keeps improving for N hours following a per-person power law that eye, muscle and display surrogates do not follow, and the gain splits into own-data volume at fixed recency and growing separability of the person's brain states."

**Why it is new.** Long co-adaptive EEG training exists (Perdikis 2018; Benaroch 2021), but none reports a per-person curve on a held-out, display-free stream with auditor surrogates scaled on the same axis, or a recency-controlled volume/learning split. Kaifosh et al. 2025 Nature found the cross-participant axis primary for sEMG (personalisation about +16%); this study measures the per-person axis the PI's hypothesis concerns, on top of a population prior: complementary, not a mirror. A clean saturation result (α ≈ 0 after h*) is a Nat Commun-level answer to "keep collecting".

## 9. Falsifiers and biggest risk

**Falsifiers (pre-registered; reported per person as k of 4):**
1. Clean E-probe performance at 80 h is not above sessions 4–6 (session-block bootstrap 95% CI).
2. β_display exceeds its bound, or the label-provenance test shows a ReFIT advantage only on visible trials.
3. Any surrogate reaches the EEG level on clean E-probes, any surrogate α is at least the EEG α, or EEG falls to the floor after partialling.
4. The sensorimotor-only decoder retains less than 80%.
5. Both the fixed-recency volume ladder and separability are flat after 8 h.

**Biggest risk.** The loop is an optimiser. Edge-channel exclusion, the goal-blind veto, in-band surrogates and the surrogate-α test now act against artifact strategies, but sub-threshold in-band EMG at central sites is still only detected, not prevented.

**Second risk.** Dry-cap SNR ceiling. Stopping rule: if α < 0.02 at 50 h (CI excluding 0.1), remaining hours go to the gel arm and short cohort, and the paper is a saturation case series.

**Third risk.** Attrition and discomfort: 20-min blocks, comfort ratings, four participants, 80-h minimum.

## 10. What can be tested now

ds004362 (109 subjects, 64-ch gel, about 45 cued imagined L/R trials each, no feedback), subsampled to the g.tec positions minus the edge channels. This project's audit already shows a frontal eye proxy at 0.73 against 0.60–0.62 for cortex (hence the edge-channel exclusion), and zero-shot equal to within-subject accuracy (about 0.60).
- **T1, per-person slope.** Chronological within-subject k = 5…30, sensorimotor montage, matched-k floor. Fit acc(k) − 0.5 = A·k^α. **Kill:** a plateau by k = 10 together with a flat T6.
- **T2, identifiability.** Occipital, frontopolar and sensorimotor channels, early (0–0.5 s) and late (1–4 s) windows; only sensorimotor-late evidence is admissible as a prior.
- **T3–T5.** Crowd worth to one brain (sets the §3 prior); drift half-life (sets SmoothBatch); no-feedback null across runs 4/8/12 (the learning baseline).
- **T6, recency versus volume on Stieger et al. 2021** (closed-loop SMR, up to 11 sessions per person). Fixed-recency against fixed-size-at-lag ladders tests the §3 decomposition before participant 1.
