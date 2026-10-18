# P_manifold (amended) — Intrinsic-manifold operant control, with a noise-matched manifold test

## 1. One-line idea and signal basis

**Learn each person's intrinsic EEG manifold from target-free recording, freeze a 2-D velocity decoder along two of its stable, controllable directions, and let the person learn to steer a cursor or robot by trial and error. No decoder weight is ever fitted to an intent label. The only supervision is a display-free screen that picks 2 of 10 latents and their signs (≈8.5 bits).**

*Signal.* Each 1-s window gives three 32-channel covariances (4–8, 8–13, 13–30 Hz) → Riemannian tangent vectors (3 × 528) → latent z ∈ R^10. The expected high-variance stable modes are sensorimotor mu/beta, posterior alpha and frontal-midline theta. These are the rhythms people have learned to regulate operantly: Wolpaw & McFarland 2004 PNAS (2-D), McFarland 2010 (3-D), Stieger et al. 2021 Sci Data (62 people, >600 h). **A 30-Hz ceiling does not remove EMG.** EMG peaks at 20–30 Hz frontally and reaches the vertex above ~12 Hz (Goncharova et al. 2003 Clin Neurophysiol). EMG is therefore handled by explicit exclusion and auditing (§2, §5), not assumed away.

*Instruction.* "Move it to the target, any way that works." After the controllability screen, each person is told which of their own strategies moved each axis (§6). Strategy reports are logged every session.

*Prior work.* Sadtler et al. 2014 Nature (monkeys relearn within- but not outside-manifold perturbations); Busch, Fincke, Lajoie, Krishnaswamy & Turk-Browne 2026 Nat Neurosci (the same in humans with fMRI, so the human non-invasive result exists); Oby et al. 2019 PNAS (long practice creates new patterns). Our narrower contribution: sub-second scalp-EEG control of cursor and robot; a manifold test that separates a **neural** from a **measurement** (SNR/volume-conduction) constraint (§6.2); and manifold expansion over ~80 h per person.

## 2. Label source and why it is not display-identifiable

| component | fitted on | label |
|---|---|---|
| manifold encoder f | **target-free windows only**: spontaneous, idle wear, free babbling (cursor visible, no target, D not yet built: random-walk cursor), and display-free blocks (identical fixation screen for every direction) | none |
| artefact exclusion | the instructed-artefact localiser (§5 C1) and EEG-only rules | artefact instruction (exclusion only) |
| decoder D (2 × 10) | stability (split-half principal angle < 30°) → controllability screen: audio-cued "raise / lower" for each surviving latent, eyes open, fixation enforced, pitch feedback of that latent only | cue (selects indices and signs; no weights fitted) |
| online adaptation | mean re-centring only (Zanini et al. 2018) | none |
| evaluation | target direction | never enters any fit |

**Correction from the previous draft.** "Unlabelled" is not "label-independent". PCA, FA or SSL fitted on task windows pooled over directions ranks between-direction variance into the top latents. That variance includes target-locked VEPs, saccade activity and hemifield-attention alpha. Our pilot's +0.034 gain from unlabelled *task* trials (§10) may be partly this. Visible-target windows therefore never enter f.

**EEG-only exclusion rules (pre-registered), applied before D is chosen:**
- (a) Rim rule: a latent is excluded when its squared Haufe loading on FP1/FP2/AF3/AF4/F7/F8/T7/T8/O1/O2/P7/P8 exceeds the chance share by 1.25×.
- (b) EMG rule: a latent is excluded when its time course correlates (|ρ| > 0.2) with 30–90 Hz EEG power at the rim channels. That power is computed from the EEG and used only for exclusion.
- (c) Localiser rule: a latent is excluded when it responds to the artefact localiser (§5 C1).

## 3. Data engine and scaling mechanism

Unlabelled volume buys a better **representation**; the person supplies the **skill**. Here is what our own evidence says.

**Loop A (machine): own target-free hours → manifold estimate.**
- Linear FA/PCA baseline vs per-person SSL (Banville et al. 2021); metric: split-half principal angles on a fixed held-out session.
- Pilot facts (§10): other people's rest did not help (population ladder n.s.), and doubling own rest did not help. Own task-state data helped but may be display-contaminated.
- Prediction: angles fall roughly as 1/√h and saturate at **5–10 h**. That prediction is pre-registered, so an early plateau is a finding, not a failure. The 40-h rung is kept offline only, as the saturation check.
- Online test: a crossover of D built from the 0.5-h manifold vs D built from the 10-h manifold (§6).

**Loop B (human): practice → skill** (Stieger 2021, Edelman 2019, Perdikis 2018); power law saturating at 20–40 h.

**Loop C (coupling), corrected.** The previous draft refitted f nightly and Procrustes-aligned latents "to keep D's meaning fixed". That would absorb the learning-induced change it claims to study. Degenhart et al. 2020 aligned latents to undo *recording* instability, not learning. Now:
- f and D are **frozen for 3-week epochs**.
- Nightly refits run in **shadow** only. They measure whether variance along the user's D directions rises in rank (Oby-style expansion).
- A refit is deployed only at the pre-registered epoch boundaries (weeks 9 and 17).

**Machine vs human decomposition** (adopted from smr_modern). Every test block is scored with the frozen session-1 map and with the current-epoch map. The first measures human gain only; the difference between the two is the machine gain.

**Where it saturates.** At the number of independently controllable, artefact-free dimensions the dry cap resolves. We expect 1–2. Pilot evidence supports one clearly (hands vs feet), and left/right is weak (§9).

## 4. Expected performance on this hardware

Score: r_int (produced velocity vs target unit vector), success, time-to-target. Anchors: repo simulation (r ≈ 0.4 → 0.49 success on 8 targets, ≈0.8 with goal posterior); Wolpaw & McFarland 2004 (gel, trained); pilot: the top-8 manifold keeps ~70% of the above-chance hands/feet margin of all 528 features, only ~29% for left/right.

| | session 1 | after 20 h practice (≥ half of long-arm users) |
|---|---|---|
| r_int | 0.05–0.15 | 0.20–0.35 |
| 4-target success (15 s) | ~35% (chance 25%) | 55–75% |
| median time-to-target | n/a | 6–10 s |
| bits/s, EEG alone | n/a | 0.05–0.2 |
| bits/s, with shared-autonomy posterior | n/a | 0.15–0.35 |

**Stated plainly:** this sits at the low end of the 0.03–0.3 bits/s envelope and is not expected to beat a well-run supervised SMR decoder on day 1. The cohort includes a supervised, display-free cued-MI reference, and if it beats the manifold map at matched practice hours we will report that. The contribution is the learning science, not throughput.

## 5. Purity proof

**Auditors** (separate amplifier, never fed to the decoder or to D's selection):
- 250 Hz eye tracker with pupil and eyelid aperture;
- bipolar HEOG/VEOG;
- EMG on bilateral temporalis, masseter, frontalis and splenius/upper trapezius, and on the forearm flexors/extensors;
- a cap accelerometer.

**Controls (pre-registered):**

- **C1 Instructed-artefact localiser** (sessions 1, 2 and every 10th session). Jaw clench, teeth grind, brow raise, squint, slow eye closure, blinks, saccades to 8 eccentric points, neck tension and swallowing: 3 × 10 s each. Any latent whose localiser response exceeds rest by 2 SD is excluded before D is chosen. This targets exactly the "cheapest levers" an operant learner finds.
- **C2 Matched non-neural surrogates.** These are linear **and** nonlinear lagged (gradient-boosted trees on auditor features at 0–2 s lags), predicting (a) target direction, (b) cursor velocity and (c) z. Requirements:
  - EEG r_int must exceed the best surrogate.
  - After nonlinear-lagged auditor residualisation of z, r_int must keep ≥80%.
  - The EMG gate (adopted from smr_modern): epochs with any EMG RMS > rest + 3 SD are scored separately, and their fraction is reported.
- **C3 Display-free, fixation-enforced test block** (one per session; merges the old blind and fixation blocks). A spoken cue is followed by a fixation cross; cursor and target are hidden. Gaze must stay within 1.5° and eyelid aperture at ≥70% of baseline.
  - A gaze break or lid narrowing counts as a failed trial in the end-of-trial tone and in the analysis, never as an exclusion. This removes the reward for ocular strategies. The tracker never touches velocity.
  - Criterion: purity is **claimed** if display-free/visible r_int ≥ 0.7 with lower 95% CI ≥ 0.5. It is **falsified** if the point estimate is < 0.5. Anything between is reported as "inconclusive" and no purity claim is made.
- **C4 Frame crossover** (adopted from smr_modern; weekly test blocks). An announced mapping M ∈ {identity, mirror-x, mirror-y, rotate-180} is applied after D and balanced within the block. We regress latent output on the D-frame command and on the screen-frame direction. Passive gaze, VEP or attention confounds predict β_screen > 0; volition predicts β_D ≫ β_screen ≈ 0.
- **C5 Rotation probe.** This tests volitional vs passive *only*. An ocular or muscular strategy would also adapt, so cortical provenance rests on C1, C2, C3 and C6.
- **C6 Topography.** Haufe patterns per latent are reported. Deleting the 12 rim channels must cost < 10% of r_int.
- **C7 Replay-watch.** The display → latent gain must be < 0.1 of the closed-loop gain.
- **Statistics.** Permutation and circular-shift floors at matched n, fixed test sessions, and no performance-based exclusion.

## 6. Protocol

### 6.1 Arms (15 months, OSF pre-registration)

- **Months 0–3:** rig, auditors, dry-cap phantom recording (sensor-noise covariance), offline ladders (§10).
- **Long arm L: 4 participants × 40 sessions × 1.5 h, plus 20 h passive wear each (≈80 h per person).**

  | weeks | content |
  |---|---|
  | 1–2 | target-free recording, localiser, controllability screen, D built |
  | 3–8 | practice on D: 3 × 15-min visible centre-out + one 12-min C3 block per session |
  | 9–10 | manifold-rung crossover (0.5-h vs 10-h D, ABBA, 4 sessions) |
  | 11–16 | perturbation epochs: WMP 3 sessions, washout 1, OMP-matched 3 sessions, washout 1; order counterbalanced |
  | 17–20 | robot arm and weekly C4 |

- **Cohort arm C: 24 participants × 4 sessions × 2 h (192 h), between-subject.** Twelve practise a within-manifold D and twelve an OMP-matched D (§6.2). Each gets about 4.5 h of centre-out on a single map, with no interference between mappings. The last 20 min is a display-free cued-MI supervised reference.
  - Power: Δr_int = 0.10 with SD 0.08 gives d = 1.25, so n = 12 per group has ≈0.8 power at α = 0.05.
- **Gel arm (optional):** 6 cohort participants do 2 sessions with 64-ch gel, to compare manifold dimensionality and the noise floor (€30–50k).

### 6.2 The manifold test, built so SNR cannot decide it

1. **WMP (within-manifold perturbation).** The latents feeding D are permuted. Per-direction SNR is identical by construction (Sadtler's design).
2. **OMP-matched.** These directions are outside the top-10 span and must pass two tests:
   - (i) **Expressible:** the projection onto the span of cortical leadfield outer products l_s l_sᵀ (fsaverage template, standard 10-20 positions) is ≥ the median of D's directions. Cortex *could* produce this covariance change at the scalp.
   - (ii) **Noise-matched:** sensor-noise variance (phantom covariance propagated to tangent space, plus within-session split-half residual) is within ±20% of D's.

   The difference from D is therefore spontaneous signal variance alone.
3. **Gains** are set so that *null-period* cursor jitter (replay-watch and rest) is equal across mappings. Noise is never amplified asymmetrically. Initial impairment is matched as in Sadtler 2014.
4. **Noise-injected-within control.** D plus synthetic noise matched to OMP-matched null jitter.
5. **Analysis.** A mixed model of learning rate on manifold membership + measured SNR + expressibility. The neural-constraint claim rests on the membership term at matched SNR.
6. **Measurement-limited outcome.** If fewer than 2 OMP-matched directions exist per person, the pre-registered conclusion is "the scalp manifold is measurement-limited", and the neural-constraint claim is dropped.

## 7. Real-time architecture

LSL 500 Hz → causal Butterworth bands → 1.0-s window, 50-ms hop → OAS covariance → re-centring (half-life 5 min) → tangent space → f (linear FA or 1–5M-parameter conv/transformer; frozen per epoch, shadow refits nightly) → z ∈ R^10 → v = g·M·D·z → smoothing (τ = 300 ms) with dead zone. Latency ≈0.7 s. Gain from null-period jitter (§6.2). One LSL stream `intent_velocity` (2-D, 20 Hz): the cursor integrates it; the robot takes it as Cartesian end-effector velocity via ROS 2; 1-s dwell selects. The same D drives both.

## 8. The Nature claim

**Headline figure:** (a) target-free manifold topographies with localiser exclusions; (b) learning curves for D, WMP, OMP-matched and noise-injected D **at matched noise**; (c) display-free vs visible, and C4 β_D vs β_screen; (d) manifold expansion along learned directions over 80 h; (e) machine vs human gain on fixed test sessions; (f) surrogates at floor.

**The sentence:** *On scalp EEG, humans learn to steer a cursor and a robot along directions of their own target-free EEG manifold; at matched measurement noise, cortically expressible directions outside it stay hard to learn, and with practice the manifold expands along the learned directions. Eyes, muscles and the display are ruled out.*

**Why new, honestly:** Busch 2026 established the human dissociation with fMRI at ~0.5 Hz. We add sub-second, effector-general EEG control; a test that separates neural from measurement constraints, which the field lacks for low-SNR modalities; and long-practice expansion. Nature Communications is likely; Nature needs (b) at matched noise plus (d).

## 9. Falsifiers and biggest risks

**Falsifiers:**
1. In the cohort, within − OMP-matched r_int after ≈4.5 h is TOST-equivalent (±0.05), or the membership term at matched SNR is n.s. in the long arm. → No neural manifold constraint on EEG.
2. C3 ratio point estimate < 0.5, or C4 β_screen ≥ β_D. → Display- or eye-driven.
3. Any auditor surrogate, linear or nonlinear, ≥ EEG.
4. Δr_int < 0.03 between the 0.5-h and 10-h D. → The machine-side scaling claim is dropped, and scaling rests on practice and expansion only.

**Biggest risk: variance is not controllability** (pilot: left/right top-8 is no better than bottom-8, p = 0.55). The controllability screen is now the primary selection step, not a fallback, and its ≈8.5 bits of supervision are reported. The likely outcome is one strong axis (hands/feet-like) and one weaker axis. Shared autonomy (secondary arm) compensates, and is always reported separately.

**Second risk: user burden.** An arbitrary map is operant neurofeedback (Wolpaw et al. 1991): slow, with non-learners. Mitigations: the strategy debrief, 1–2 axes, and NASA-TLX, time-to-first-control and non-learner rate reported for everyone.

## 10. What ds004362 can test now

Pilot (repo cached epochs; 106 subjects, 32 g.tec channels, 8–30 Hz; OAS tangent space re-centred on T0; PCA on T0 only, ≈3 min; logistic regression on k = 8, 5-fold × 3):

| imagined task | top-8 | random 8-D | bottom-8 | all 528 |
|---|---|---|---|---|
| hands vs feet | **0.636** | 0.562 | 0.548 | 0.694 |
| left vs right | 0.539 | 0.506 | 0.528 | 0.633 |

Hands/feet: top-8 > random +0.074 (p = 5.5e-9), > bottom-8 +0.088 (p = 1.8e-7). Left/right: > random +0.033 (p = 0.008), not > bottom-8 (p = 0.55). Own unlabelled task trials +0.034 (p = 1.7e-5); more rest −0.004; others' rest n.s. (p ≥ 0.37). Bottom-8 near random fits the SNR objection, hence §6.2.

**Tests to run next, with kill rules:**
- (i) **Display audit of the +0.034.** Refit with the 0–0.5 s post-target windows removed. If the gain vanishes, it was visual, which confirms the target-free rule in §2.
- (ii) **EMG index.** Correlate the top latents with 30–70 Hz rim power (160 Hz data allows it). If more than 2 of the top 8 exceed |ρ| = 0.2, the rim and EMG rules must be applied before any manifold claim.
- (iii) **OMP-matched existence.** On the full 64-ch data, with the fsaverage leadfield and split-half noise, count the directions outside the top-10 that are expressible and noise-matched. Kill the neural-constraint aim if the median subject has fewer than 2.
- (iv) **SSL encoders on 2–60 min of own data.** Kill if top-k never beats random-k by ≥ 0.05 for any contrast.

ds004362 is gel, open-loop and display-confounded. It can kill the idea; only the closed loop can confirm it.
