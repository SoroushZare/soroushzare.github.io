# P_inner_speech (amended): silent commands, tested at scale with a closed loop that cannot read the screen

**Honest headline:** published scalp-EEG inner-speech decoding is barely above chance and often confounded. Fund it as a decisive, auditor-controlled scaling test with a kill switch, not as a promised controller.

## 1. One-line idea and signal basis

**Idea:** the user silently says a short command word twice ("left, left") or nothing. A 32-ch dry-EEG decoder outputs a posterior over {none + 5 commands}; a goal-inference agent turns sparse commands into cursor or robot motion. Trained on 150 h per dense participant plus a 30-person cohort; audited by channels it never reads.

**Signal basis.** Intracortically, inner speech is decodable but weak and inconsistent:
- Kunz et al. 2025 (*Cell* 188(17)): inner speech in motor cortex is a weaker, correlated version of attempted speech.
- Wandelt et al. 2024 (*Nat Hum Behav* 8:1136–1149), SMG: 8 internal words at 79% and 23% in two participants (chance 12.5%).

These are mm-scale recordings. **Concession:** monosyllables share articulators and bilateral speech cortex, differing mainly in phoneme sequence over ~300–500 ms; there is no cm-scale contrast of the left/right-hand kind. Any scalp signal must be **temporal**: articulation-locked low-frequency waveform and ERD dynamics over FC5/FC6, T7/T8, CP5/CP6 (C5/C6 are not on a standard 32-ch cap; to be confirmed against the delivered montage). Hence (a) a time-domain decoder (§7), (b) the doubled form, a two-replicate rhythm spontaneous self-talk rarely has, and (c) a **speech-item library** varying syllable count (1–3) and dominant articulator (lips / tongue tip / tongue back / larynx), so temporal envelope can discriminate.

**Scalp evidence is weak:**
- Nieto et al. 2022 (*Sci Data* 9; ds003626), 10 participants, 4 words: EEGNet ≈ 29.7% (chance 25%; from a secondary review).
- BCI Competition 2020 Track 3: CBraMod 0.537 (chance 0.20, via the CSBrain paper); EMG and order controls unverified.
- Porbadnigk et al. 2009: 45.5% under block order fell to chance when randomised.
- Whitham et al. 2007: most scalp power above 20 Hz is EMG. Goncharova et al. 2003 (*Clin Neurophysiol* 114:1580): weak frontalis/temporalis EMG reaches the vertex from ~12 Hz (~8 Hz in some). Glossokinetic potentials sit in delta/theta.
- AlterEgo (Kapur et al. 2018): facial sEMG during silent articulation reached 92% median word accuracy.

**Prior:** a small cortical signal under an articulatory confound that reaches every band I use.

## 2. Label source: why no label is display-identifiable

All open-loop labels are the user's covert choice, reported after the window on a randomised-position menu. None is a threshold on anything displayed.

- **Tap-locked free choice.** Same fixation cross for every class; the participant taps one key (same for all classes), then says the word twice silently, so onset is fixed within a few hundred ms. Window [−0.5, +2.5] s; the random-position report comes ≥1.5 s after it closes.
- **Untapped free choice.** Blank screen, 4-s segments, no tap; one command or none per segment, reported afterwards; trained by multiple-instance learning. This matches the asynchronous online regime (conceded train/test mismatch).
- **Memorised sequences:** random 6-item sequences on a blank screen (no block-order artefact).
- **"None" includes self-talk:** 15% of open-loop time is *narration rest* (silently reading or narrating unrelated sentences) plus mind-wandering rest.
- **Semantics-to-eyes trap:** silently saying "left" may move gaze or covert attention. Controls: monosyllabic pseudo-words (*bef, tol, mip, kesh, ron*) **matched in syllable count** to the direction words; eyes-closed blocks (20%); a gaze surrogate (§5).
- **Closed loop (amended):**
  - Goals are **committed before movement**: the user keys the goal number into a sealed logger that the agent and decoder cannot read. Post-hoc goal reports are dropped, which removes the choice-blindness path (Johansson et al. 2005).
  - **Step-and-report mode:** the display shows the state, then blanks to fixation for a 2.5-s command window; the decoder acts; the user then reports the word said on a random-position menu. Every closed-loop window has its own blank-screen label.
  - **Only permuted-mapping blocks may train the decoder.** Using selected library items, the item→action mapping is re-drawn every block and taught verbally (an idea grafted from the opponent). Pooled over ≥24 blocks, the direction to the goal carries no information about the item label. **Direction-word closed loop, where "left" always means left, is evaluation only** and is never trained on, because its label is a function of goal geometry and therefore of fixation.

## 3. Data engine and scaling mechanism

**Cohort arm (months 2–8):** 30 participants × 2 sessions × 90 min: screen the 12-item library (4 direction, 4 pseudo-, 4 articulator/length-contrast words), population prior, prevalence, single-vs-doubled A/B.

**Dense arm:** 3 participants × 75 sessions × 2 h = 150 h each, about 50k labelled events per person. Each person's 5 items are chosen under nested held-out selection in sessions 1–6. Chisco (Zhang et al. 2024, *Sci Data* 11:1265) shows >900 min per subject of imagined-speech EEG is feasible.

**Mechanism:** at fixed single-trial SNR, data cannot raise the Bayes limit; it only reduces estimation error. Covariance models should saturate within a few thousand events; time-resolved models (~10⁵–10⁶ parameters) are variance-limited at small n and alone can use the temporal discriminant of §1. The scaling claim is a **crossover**: the time-domain model passes the covariance baseline at n\* and the gap grows to 32k. No crossover means volume buys only variance reduction and the bitter lesson fails here.

**Ladders (fixed test = last 10 sessions, never trained on):**
1. Volume: 250 → 32k events, **sampled uniformly at random across sessions 1–65, stratified by session**, so data volume is not confounded with recency. A separate *recency ladder* (most recent k sessions) and a session-index regressor measure drift and automaticity.
2. Pooling: 0/3/10/30 cohort participants + 1k of the target's own events.
3. Priors: LaBraM/CBraMod initialisation. *Exploratory only:* HuBERT alignment to the participant's own overt recordings. **Conceded** to be carried over from stimulus-locked perception (Défossez et al. 2023; EEG much weaker than MEG, as in Brain2Qwerty), so it is reported only against a matched no-alignment ablation on tap-locked data.

**Forecast (5-class, EMG-silent subset):** 24% at 500 events, 30–35% at 32k, plateau near 35%; 45–50% optimistic. Pooling adds 2–4 points at low n and nothing at high n.

## 4. Expected performance on dry 32-ch hardware

Wolpaw ITR per 3-s decision:

| Scenario | Accuracy | bits/decision | bits/s |
|---|---|---|---|
| Status quo, 4-class | 30% | 0.009 | 0.003 |
| Central, 5-class | 35% | 0.088 | **0.03** |
| Optimistic, 5-class | 50% | 0.32 | 0.11 |
| Command-vs-none | 65% | 0.07 | 0.02 |

These sit at the bottom of the project's 0.03–0.3 bits/s band, below SMR cursor control (Wolpaw & McFarland 2004; Edelman et al. 2019).

**Closed loop (8 goals, 30-s limit, chance 12.5%), corrected.** My earlier 0.35–0.55 success estimate borrowed the repo's continuous r≈0.4 velocity decoder, which carries far more information. That was a category error, and I withdraw it. **Fano bound:** selecting 1 of 8 needs 3 bits. At 0.03 bits/s × 30 s ≈ 0.9 bits, P(error) ≥ (3 − 0.9 − 1)/log₂7 ≈ 0.39, so success ≤ 0.61 for *any* decoder or agent. A goal posterior can only spend the bits efficiently; it cannot create them.
- **Central forecast: success 0.18–0.28** (chance 0.125).
- **Optimistic (0.11 bits/s, ~3.3 bits per trial): 0.45–0.65.**

Month 1 re-simulates the agent with **measured confusion matrices and overlap-corrected posteriors**.

**Grab** uses the grammar prior (only within 5% of an object), plus a doubled detection within 4 s and a 2-s cancel window. The false-grab rate is **not forecast**. The threshold is calibrated to ≤1 false emission per 5 min on held-out narration-rest blocks, and the achieved rate is reported.

## 5. Purity proof

**Auditors (logged by a separate process with no socket to the decoder):**
- 10 bipolar sEMG channels: temporalis L/R, masseter L/R, frontalis, suprahyoid, thyrohyoid, upper and lower orbicularis oris, and a chin–mastoid derivation for glossokinetic potentials;
- 2 EOG channels as auditors only;
- a 250-Hz eye tracker, throat microphone and head IMU.

**Per-session artefact calibration (4 min):** tongue sweeps and "la-la-la", jaw/forehead contraction at ~5% and 15% MVC, saccades, blinks → per-person artefact templates.

**Pre-registered tests (identical trials, folds and n; permutation floors of 1,000 within-session shuffles; no performance-based exclusion):**
1. **Auditor surrogate:** the same architecture trained on all auditor channels. EEG must beat it. **EMG-silent subset:** every auditor channel, including temporalis and glossokinetic, has RMS inside its 95% rest band, *and* the surrogate is at permutation chance on that subset. EEG must stay above chance there. I also report the fraction of trials that are naturally EMG-silent; users are told to "think the word as in silent reading" and are not trained to suppress.
2. **Artefact-subspace projection:** project the calibration templates out (SSP) and regress per-channel EMG spectra out of the EEG features, then re-decode. ≥60% of the above-chance margin must survive.
3. **Band report:** the main decoder uses 0.5–20 Hz. **Concession:** this does *not* exclude glossokinetic or low-frequency EMG, which is why tests 1–2 carry the burden. Decodability at 20–45 Hz is reported as a contamination index.
4. **Channel ablation, re-referenced:** drop FP1/FP2/F7/F8/T7/T8 **before** re-computing CAR over the remaining 26 channels, and separately use a surface Laplacian on interior sites. ≥50% of the margin must survive.
5. **Gaze:** a gaze surrogate on tracker x/y/pupil, and an eyes-closed subset where the effect must persist. Direction words vs syllable-matched pseudo-words: any direction-word advantage explained by the gaze-surrogate component counts as artefact.
6. **Time/order surrogate** and a **no-EEG matched control** (phase-randomised EEG plus metadata) must sit at chance.
7. **Condition gradient:** overt → mouthed → inner. If EEG decodability tracks EMG decodability proportionally, the signal is EMG.
8. **Feedback circularity (now computable):**
   - *Frozen probes:* in 20% of step-and-report steps the effector ignores the decoder. Accuracy against the per-window self-report must match unfrozen steps.
   - *Unannounced mapping reversal* (grafted): mid-block, the word→action mapping flips. A word decoder keeps reporting the word said; a decoder reading gaze, goal direction or the display follows the goal. This is scored against the per-window reports.
   - *Replay* (grafted): watching one's own session replay at rest must give chance output.
   - Closed-loop accuracy must not exceed open-loop blank-screen accuracy beyond its CI.

**Optional gel arm:** one dense participant, 30 sessions on 64-ch gel, for source localisation and the dry–gel gap (≈€40–80k if no amplifier exists; unverified).

## 6. Protocol

- **Months 0–2:** §10 analyses, the auditor rig, re-simulation with measured confusions.
- **Months 2–9, open loop.** P1–P3 do 3 sessions/week × 2 h. Each session:

| Block | Minutes |
|---|---|
| Setup and artefact calibration | 24 |
| Rest (eyes open/closed) and narration rest | 10 |
| Tap-locked free choice (alternating blocks: direction words, selected items) | 30 |
| Untapped free choice | 20 |
| Memorised sequences | 12 |
| Overt / mouthed / inner gradient | 10 |
| Eyes-closed free choice | 8 |
| Effort rating and strategy note | 2 |

  The cohort (n = 30) runs in months 2–8.
- **Months 9–15, closed loop:** P1–P3 plus 10 returning cohort members, 12 sessions each. Each session has three parts:
  step-and-report with permuted mapping (trainable); direction-word continuous control (evaluation only); frozen, reversal and replay probes. Effectors: a 2-D 8-goal cursor, a TurtleBot-class base, and a tabletop arm where planar commands plus "grab" trigger autonomous approach, descent and grasp (no depth command needed).
- **Months 15–18:** analysis. Total ≈ 640 h.

## 7. Real-time architecture

- **Input:** LSL 500 Hz, 32 ch → 0.5–20 Hz → CAR → 128 Hz.
- **Models:**
  - (a) Riemannian tangent space on filter-bank covariances (4–8, 8–13, 13–20 Hz) with logistic regression;
  - (b) **time-domain:** shrinkage-LDA on the 0.5–8 Hz waveform decimated to 32 Hz;
  - (c) a temporal CNN, initialised from CBraMod and fine-tuned with ±500 ms onset-jitter augmentation and MIL on untapped segments.

  Windows are 1.5 s every 100 ms; latency is under 50 ms.
- **Evidence accumulation (amended):** an HMM over {none, 5 commands} with a learned task grammar. Because consecutive windows overlap by 93%, each 100-ms log-likelihood is scaled by 1/15 (one effective sample per 1.5 s), and the emission threshold is **calibrated empirically** on held-out untapped and narration-rest data to a target false-emission rate. Reliability diagrams are reported.
- **Agent:** a command sets a velocity mode (0.15 workspace/s) until "none" is held for 1.5 s or a new command arrives. A goal posterior blends assistance (α ≤ 0.5) and never reads the sealed goal.
- **Adaptation:**
  - unsupervised Riemannian re-centring every 60 s (for dry-electrode drift);
  - a nightly supervised refit on open-loop labels plus **only** permuted-mapping step-and-report windows.

  No direction-word closed-loop data and no goal-derived label ever enters training.
- **Effector interface:** a 10-Hz posterior plus a velocity setpoint, published to ROS2 `/cmd_vel` or the cursor API, with a binary gripper.

## 8. The Nature claim

**Headline figure:** per-participant learning curves (log n, 250 → 32k, fixed test) for covariance and time-domain models against auditor, gaze, time, no-EEG and permutation floors on identical trials; insets: EMG-silent/SSP curves, condition gradient, mapping-reversal traces.

**The sentence:** *"With 150 hours per person and muscle, tongue, eye and display auditors, scalp EEG decodes self-chosen silent commands above every matched non-neural surrogate, and this purified information grows with data to X% before saturating"*, or the negative: *"…and does not: scalp-EEG inner-speech accuracies are accounted for by articulatory EMG, glossokinetic, gaze and temporal confounds."*

**Why new, and honest value.** No inner-speech EEG study has >1k trials per item per person, a full auditor floor on identical trials, and per-window self-labels in closed loop. **Concession:** the negative is likelier and extends what Porbadnigk 2009 and the EMG literature suggest; its value is being *decisive*, with the volume and controls earlier work lacked. The ~640-h audited dataset is a resource either way. Venue: Nat Comms; *Nature* ~10%, only if positive with a robot demo.

## 9. Falsifier and biggest risk

**Falsifier (pre-registered):** in at least 2 of the 3 dense participants, at ≥20k events and on the EMG-silent, SSP-projected data, held-out 5-class accuracy must exceed max(auditor, gaze and time surrogates, permutation p95) by ≥5 points, the bootstrap CI of the 4k→32k slope must exclude 0, and mapping-reversal traces must follow the word, not the goal. If any of these fails, inner speech is dropped as a control channel.

**Biggest risk:** everything decodable is subvocal EMG or glossokinetic activity (RLNF in a new form). **≥50%** on ending below 33% 5-class accuracy.

**Second risks:** self-talk false activations (measured on narration rest); tedium and automaticity (session-index regressor); naturalness: instructed covert articulation, about 6/10, closer to intention than arbitrary mental tasks but not free intention.

**Mitigation:** the agent, auditor rig, step-and-report and permuted-mapping machinery, and data engine are signal-agnostic. **Concession:** if inner speech fails, what survives is infrastructure, not an inner-speech result.

## 10. What can be tested now

**ds004362** (imagined L/R fist; targets on screen) has no speech. It can test:
1. **Display trap:** target side from FP1/FP2/F7/F8/occipital at 0–500 ms vs C3/C4 mu/beta at 1–4 s.
2. **Pooling:** 20 fixed test subjects; ladders of 1/3/10/30/89.
3. **Re-referenced ablation:** CAR before vs after dropping rim channels.
4. **Overlap-corrected HMM:** held-out posteriors through the agent; false-emission rate vs threshold; success vs Fano bound.

**Kill test on ds003626 (Nieto; inner/pronounced/visualised; 8 external channels incl. EMG and EOG).** Use 32 10-20 sites at 0.5–20 Hz. Run the covariance and time-domain decoders within subject, an external-channel surrogate on identical trials, a re-referenced ablation, and inner vs pronounced.
If the surrogate beats EEG on inner trials, the confound is confirmed and the library shifts to articulator-contrast items; a null on a few hundred trials does not kill the scaling hypothesis.

Not yet run; it is week one.
