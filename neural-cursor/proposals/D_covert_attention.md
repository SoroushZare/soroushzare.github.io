# Proposal D — Endogenous covert visuospatial attention: "look at the cursor, attend to where it should go"

## 1. Paradigm and signal basis

**Signal.** When a person covertly attends to one visual hemifield while keeping the eyes still, parieto-occipital alpha (8–13 Hz) desynchronises over the hemisphere contralateral to the attended side and synchronises ipsilaterally (Worden et al. 2000, *J Neurosci*; Thut et al. 2006, *J Neurosci*; Kelly et al. 2005, *IEEE TNSRE*). The topography is retinotopic: Rihs, Michel & Thut (2007, *Eur J Neurosci*) showed that attention to each of eight peripheral locations produces a distinct alpha map, with an upper/lower-field axis that is weaker than the left/right axis. The lateralisation index LI = log P_right(PO/O) − log P_left(PO/O) is a continuous, sign-reversing control signal that appears within ~500 ms of an attention shift and persists while attention is held.

**Why it is natural.** Nothing flickers, no oddball, no imagined limb. The person looks at the cursor and attends to the direction they want it to go, the mechanism they use to monitor a doorway while reading. Global parieto-occipital alpha (high when disengaged, low when attending) gives a "go/idle" gate for free. Slow intention potentials (CNV / readiness potential) are the first casualty of dry-electrode low-frequency noise; they are an exploratory, pre-registered secondary feature only.

**What is not natural.** Covert attention requires the person NOT to look where they attend; fixation on the cursor is a mild instruction that the design must enforce (section 4), because the paradigm is defined by the absence of eye movement.

## 2. Expected performance on this hardware (honest range)

Literature (gel/MEG, trial-wise, mostly 2-class left/right):

- Kelly et al. 2005 (*IEEE TNSRE* 13:172): EEG, 2-class L/R from parieto-occipital alpha on ~2 s windows; mean single-trial accuracy ≈70%, best subjects ≈80–85%.
- van Gerven & Jensen 2009 (*J Neurosci Methods* 179:78): MEG, four directions; L/R around 70–77% in most subjects, up/down near chance for several subjects; van Gerven et al. 2009 (*Neural Networks*) reproduced ≈70% L/R with EEG.
- Treder et al. 2011 (*J NeuroEng Rehab* 8:24): EEG, 8 subjects, six target directions, 2 s windows; 6-class mean ≈31% (chance 16.7%), binary L/R ≈70%; online feedback runs were above chance but far from reliable.
- Tonin, Leeb, Sobolewski & Millán 2013 (*J Neural Eng* 10:056007): EEG, 8 subjects, online L/R with 2–3 s trials, no exogenous stimulation; online accuracy ≈70% (approximately 65–77% across subjects).
- Bahramisharif et al. 2010 (*J Neural Eng* 7:045004): MEG, continuous decoding of attended angle, demonstrating that the 2-D direction is linearly recoverable, not just L/R.

**Dry-electrode penalty.** Alpha is a large narrow-band rhythm above the 1/f floor, so it survives dry recording better than slow potentials or P300 (Mathewson, Harrison & Kizuk 2017, *Psychophysiology*: active dry electrodes had more low-frequency noise but comparable alpha-band task effects; Guger et al. 2012: g.SAHARA close to gel for P300). The problem is site-specific: PO7/PO8/O1/O2/P7/P8 sit under hair, where dry pins seat worst. I budget −3 to −6 pp and 10–20% of epochs rejected by amplitude gating.

**Realistic projected numbers on the g.Nautilus dry 32-channel cap:**

| Quantity | Pessimistic | Expected | Optimistic |
|---|---|---|---|
| L/R accuracy, 2 s epoch, session 1 | 56% | 62% | 68% |
| L/R accuracy, 2 s epoch, after 5 sessions | 60% | 68% | 75% |
| U/D accuracy, 2 s epoch | 52% | 57% | 63% |
| 4-direction accuracy, 2 s epoch | 30% | 36% | 44% |
| Wolpaw ITR (2-class, 2 s) | 0.01 bit/s | 0.05 bit/s | 0.09 bit/s |
| Time to a 1-D target (2 targets, 90% criterion) | 30 s | 14 s | 8 s |
| Time to a 2-D target (4 equidistant targets, 10° radius, 12° target) | 60 s | 30 s | 18 s |
| Fitts throughput (ID ≈ 2 bits) | 0.03 bit/s | 0.07 bit/s | 0.12 bit/s |

The ITR at 68% per 2 s is 0.05 bit/s, two orders of magnitude below SSVEP: the paradigm wins on naturalness and loses on raw rate, and the controller in section 5 makes weak evidence usable without hiding that gap. Successive 2 s epochs are not independent (attention lapses are autocorrelated); I assume 40% less effective evidence per epoch, which is why 14 s rather than 6 s is the expected 1-D decision time.

## 3. Scaling behaviour

**Mechanisms that scale:**

1. *Subject-specific alpha topography and individual alpha frequency (IAF).* A per-subject spatial filter (shrinkage LDA or Riemannian tangent space on 8–14 Hz covariances) typically adds +4–8 pp over a fixed PO7/PO8 index; with 6 sessions (~1,500 labelled epochs) and exponential forgetting the weights stop overfitting session noise.
2. *User learning (co-adaptation).* Alpha lateralisation is trainable: Okazaki et al. 2015 (*NeuroImage*) and Bagherzadeh et al. 2020 (*Neuron* 105:577) showed that real-time LI feedback increases lateralisation over sessions and shifts attention behaviourally. Unlike RLNF, the feedback is the quantity the decoder uses, so the user has a plausible path to raise the SNR.
3. *Cross-subject pretraining.* The feature is low-dimensional and anatomically stereotyped, so a population model (EEGNet-style CNN or tangent-space LR on all other participants, per-subject recentring) gives 58–62% at session 1 without calibration; self-supervised pretraining on public EEG (BENDR, Kostas et al. 2021; LaBraM, Jiang et al. 2024) should shorten calibration more than raise the ceiling, and I pre-register that expectation.
4. *Data engine.* Every second of free use yields a weak label (the eventual target, confirmed by delayed keypress, with its gaze trace); epochs with gaze deviation < 1.5° and a reached target enter the training pool at weight 0.5 versus 1.0 for cued epochs.

**Expected curve (2-class per 2 s, median subject):** session 1 generic 60% → +30 min calibration 64% → session 3 co-adapted 67% → session 6 70% → asymptote ≈74–78%. Time-to-target for a 2-D target: ≈35 s → ≈18 s.

**Where it saturates and why.** Attention modulates alpha by 10–30% on a rhythm with large trial-to-trial variability; single-epoch separability tops out near d′ ≈ 1.2 (≈80%) even with MEG and gel. IAF drift and electrode seating require continual recentring. Up/down is physiologically weaker (upper/lower-field alpha topographies overlap on the scalp) and will stay 5–10 pp below left/right. More participants improve the session-1 prior; more hours help up to ≈5 sessions; beyond that the limit is biology, not data.

## 4. Data-collection protocol

**Participants.** n = 16, no performance-based exclusion; two cohorts of 8 for a held-out cross-subject test. 6 sessions each, 75 min, on separate days (96 sessions, ~120 h of EEG).

**Hardware add.** A screen-mounted eye tracker (Tobii Pro Nano / Gazepoint GP3 class, 60–150 Hz, ~0.5–1° accuracy, over LSL) is mandatory; FP1/FP2 and the F7–F8 bipolar derivation give a secondary "virtual EOG".

**Display geometry.** 24-inch monitor at 60 cm. Validation mode: the cursor stays at screen centre and the world scrolls under it, so gaze stays central with targets at 8–10° eccentricity. Applied mode: the cursor moves freely and gaze follows it; logged separately because smooth pursuit produces slow EOG correlated with movement direction, a display-mediated confound.

**Session structure.**

- Block 0 (8 min): eyes-open rest, IAF estimate, impedance/quality log, eye-tracker calibration (9-point), amplitude baseline.
- Blocks 1–2 (2 × 10 min, cued, open-loop): a target appears at one of four directions (L/R/U/D, uniform random, 60 trials/block); the person fixates the cursor and attends to the target for 4 s; no cursor motion. Label = cued direction, randomised and not mediated by any display variable.
- Blocks 3–5 (3 × 12 min, closed-loop): targets uniform on an 8–10° annulus, quadrants balanced; cursor under decoder control with assistance randomised per block from {0, 0.3, 0.6}; a trial ends on dwell (1 s within 2°) or at 45 s. Label = target direction relative to the current cursor position, recomputed every 250 ms; a delayed keypress confirms "that was my target".
- Block 6 (8 min): free use with user-chosen targets; delayed keypress reports the intended target.

**LSL streams and markers.** EEG (32 ch, 500 Hz), eye tracker (gaze x/y, pupil, validity), cursor state (x, y, velocity, assistance, posteriors) at 60 Hz, markers (trial start, cue, target, dwell, keypress, gating). LSL clock throughout; sync check at session start and end.

**Gating.** Per-channel RMS > 5× session median (FP1/FP2 exempt) → epoch dropped; signal quality logged every 30 s; a channel bad for > 20% of a block is interpolated. Gaze > 1.5° from fixation for > 200 ms → epoch flagged "non-covert", excluded from the primary analysis, kept for the surrogate.

**Estimability audit (done before recording).** Positivity: every direction cued in every block of every session. The target is static after the first 200 ms. The label is never a threshold on a displayed variable. In closed-loop blocks the cursor's own motion is a visual mediator, so open-loop blocks carry the primary decoder test and closed-loop results are a separate pre-registered secondary.

**Controls and surrogates (all pre-registered).** (a) Ocular-only decoder: identical pipeline on gaze x/y/velocity plus FP1/FP2/F7/F8. (b) EEG-only on the ocular-flagged epochs. (c) Label-permutation floor at matched n (1,000 permutations). (d) A phase-randomised surrogate EEG with preserved spectra. (e) Assistance-only controller (decoder output replaced by noise) for the utility endpoint.

**Pre-registration list.** Primary endpoint; the ≥5 pp margin over the ocular surrogate; the closed-loop utility endpoint; the fixed test sessions (session 6 for every participant) and the held-out cohort; the scaling-curve prediction above; hyper-parameters in section 5; stopping rules; no exclusion.

**Power.** Primary contrast: within-subject EEG decoder − ocular-only decoder on fixation-verified epochs. With a true margin of 8 pp and SD of paired differences 8 pp (d = 1.0), n = 16 gives power 0.96 at α = 0.05 for margin > 0 and 0.75 for margin > 5 pp. Utility contrast (decoder vs assistance-only): assumed d = 0.6, power 0.62, stated as under-powered and treated as estimation with CIs.

## 5. Decoder and system

**Preprocessing.** 500 Hz → notch 50 Hz → band-pass 4–40 Hz (4th-order Butterworth, zero-phase offline, causal online with group delay ≈ 60 ms) → common average reference with bad channels interpolated → per-subject IAF ± 2 Hz band (default 8–13 Hz) → 2 s windows, hop 250 ms. Gaze regression: gaze x, y and their derivatives are regressed out of every channel in the 4–40 Hz signal before feature extraction (fitted on block 0, applied online).

**Features.** (i) IAF-band log-variance per channel; (ii) tangent-space projection of the 32 × 32 IAF-band covariance (shrinkage γ = 0.1, Riemannian recentring per block, Zanini et al. 2018); (iii) global PO/O log-alpha for the idle gate.

**Models.** Two logistic-regression heads (L2, C = 0.5, standardised on training data): H_lr → P(right), H_ud → P(up); optional ridge regression onto (cos θ, sin θ) of the attended angle (α = 10). Population model: EEGNet-8,2 on 2 s inputs, dropout 0.5, Adam lr 1e-3, 100 epochs, early stopping on a held-out session; its weights are the session-1 prior, fine-tuned at lr 1e-4.

**Splits.** Within subject: train on sessions 1–5, test on session 6 (fixed, never touched during development). Cross-subject: cohort A trains, cohort B tests. The resource ladder (10 min, 30 min, 1 h, 3 h, 5 h of training data) uses the same test set.

**Latency budget.** EEG window 2 s (inherent), hop 250 ms, filter delay 60 ms, feature + inference < 10 ms, render 16 ms: cursor reacts 330 ms after new evidence arrives; a true attention shift is reflected within ~1 s and fully within 2.5 s.

**Cursor dynamics (velocity control with evidence accumulation).** Every 250 ms the heads emit log-odds ℓ_x = logit P(right), ℓ_y = logit P(up). A leaky accumulator e_x ← λ e_x + (1 − λ) ℓ_x, λ = 0.85 (time constant ≈ 1.6 s), suppresses single-window noise. Velocity v_x = G · tanh(e_x / κ), κ = 1.0, G = 3°/s; the idle gate (global alpha above the block-0 70th percentile for > 1 s) sets v = 0. Assistance: v_final = (1 − a) v + a v_target toward the nearest candidate target, a ∈ {0, 0.3, 0.6} randomised per block and logged; assistance never enters a label. The payoff is the inverse of RLNF: moving is free, nothing is rejected, a wrong direction only slows the approach, and dwell-to-select fires only after ≥ 1 s of consistent evidence.

**Co-adaptation rule.** Between blocks, refit the heads on all epochs weighted by w = 0.5^(age in sessions / 2) × (1.0 if cued, 0.5 if free-use); within blocks, only Riemannian recentring is updated (every 30 s). The person sees a small LI bar beside the cursor (the quantity the decoder uses), which is the neurofeedback element supported by Bagherzadeh et al. 2020. No parameter changes inside the fixed test session.

## 6. Evaluation protocol

**Primary endpoint.** Session-6, open-loop, fixation-verified L/R accuracy of the EEG decoder minus the ocular-only decoder on the same epochs, pooled per participant, tested with a paired t-test and bootstrap 95% CI; success if the lower CI bound > 0 and the point estimate ≥ 5 pp. Permutation floor at matched n (1,000 label permutations per participant; for 240 epochs the 95th percentile is ≈ 55.3%), reported alongside.

**Co-primary utility endpoint.** Closed-loop time-to-target and Fitts throughput at assistance a = 0.3 for the decoder versus the assistance-only surrogate, same targets, session 6; success if median time-to-target is lower with a bootstrap CI excluding zero.

**Secondary endpoints.** U/D, 4-class and continuous-angle accuracy; resource-ladder scaling curve; cross-cohort transfer; closed-loop minus open-loop accuracy (the display-mediation estimate); EOG-only decoder on ocular-flagged epochs (expected above chance, reported); exploratory CNV/RP go-detection AUC with its own floor.

**Statistics.** Mixed-effects logistic regression (participant random intercept) for accuracy; effect sizes with CIs throughout; TOST equivalence (±3 pp) for any "candidate improves the decoder" claim.

**Held-out.** Session 6 of every participant and all of cohort B are untouched until the frozen, version-controlled analysis script is run once.

## 7. Failure modes and debugging

1. *Ocular leakage.* Microsaccades bias toward the attended location (Engbert & Kliegl 2003) and produce broadband spike potentials. If the ocular-only decoder approaches the EEG decoder: tighten gaze threshold to 1°, detect microsaccades (velocity > 15°/s) and exclude ±300 ms; if the margin collapses, the paradigm is not cortical on this rig and I say so.
2. *Dry pins fail under hair at PO/O.* > 30% bad epochs at PO7/PO8 → fall back to P3/P4/P7/P8 with a measured accuracy cost; worst case, report that the cap cannot record the paradigm.
3. *Up/down at chance.* Expected in some subjects; fall back to 1-D L/R with an idle-gate up/down toggle, and report 2-D only where the U/D CI excludes chance.
4. *Closed-loop drift.* Closed-loop accuracy exceeding open-loop → display mediation; use the scrolling world and never train on closed-loop epochs without the keypress label.
5. *Fatigue.* Alpha rises and lateralisation falls after ~40 min; blocks are capped at 12 min with mandatory 2 min breaks; time-of-block is a covariate.
6. *Non-stationarity.* IAF shifts by 0.5–1 Hz within session; re-estimate IAF every block and log it.

## 8. Honest self-score

1. Naturalness: **9** — no stimulus and no imagery; the only instruction is "look at the cursor, attend to where it should go", with fixation enforced by the tracker rather than by effort.
2. Achievable performance: **3** — honest expected numbers are 62–70% per 2 s epoch and ≈0.05 bit/s, with 15–30 s per 2-D target; usable with assistance, far below SSVEP or gel MI.
3. Scaling: **6** — subject-specific topography, trainable lateralisation (Bagherzadeh 2020) and population priors give a credible +8–10 pp over six sessions, but the biological ceiling (~75–80%) is close and clearly stated.
4. Validity: **8** — randomised cued labels with no displayed-variable threshold, mandatory eye tracking, pre-registered ocular-only and assistance-only surrogates, and open-loop/closed-loop separation of display mediation.
5. Feasibility and burden: **5** — eye tracker must be bought and calibrated each session, dry pins seat poorly over occipital hair, and 20 min of cued calibration per session is a real burden.
6. Rigour: **8** — fixed test session and held-out cohort, permutation floors at matched n, TOST for improvement claims, explicit hyper-parameters; power for the utility endpoint is admittedly marginal.

Weighted: (9×25 + 3×20 + 6×20 + 8×15 + 5×10 + 8×10)/100 = **6.6**.

## 9. Five-sentence pitch

This is the only paradigm in the competition where the person does nothing unusual: they look at the cursor, attend to where it should go, and the brain's own retinotopic alpha map says where that is. The honest cost is rate, roughly 0.05 bit/s and tens of seconds per target on a dry cap, so the design turns weak per-second evidence into usable control with a leaky accumulator, velocity integration, idle gating and logged assistance, never with a penalty that makes "ignore the decoder" the better policy. Its main confound, the eyes, is measurable and separable by design, so its first result is a yes/no answer to "is this cortical on this hardware", the question RLNF could not answer. It scales where it should: subject-specific topography, population priors and a trainable lateralisation signal the user sees as feedback, with the ceiling stated up front. If naturalness is the user's first priority, this proposal takes that priority seriously and tells the truth about the price.
