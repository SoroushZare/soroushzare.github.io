# P_generative (SF1): generative inverse decoding of a private goal from evaluative EEG, with a measured premise

## 1. One-line idea and signal basis

**Idea.** Fit one conditional generative model of a person's dry-cap EEG, p(EEG | context, candidate goal g). The context is the effector's move, its start and end points, the person's expectancy, and what the screen already predicts. Intent is decoded by Bayesian inversion: after each move, every candidate goal is scored by how well it explains the EEG, and the cursor or robot moves toward the posterior.

**Signal.** The evaluation / prediction-error response (error negativity and positivity, reward positivity, mid-frontal theta), frontocentral, 200–600 ms after a move. It answers one question: did the device just get closer to *my* goal, and how surprising was that? (Holroyd & Coles 2002; Chavarriaga et al. 2014; Iturrate et al. 2015 Sci Rep; Zander et al. 2016 PNAS.) Posterior alpha is demoted to exploratory status and excluded from every purity statistic.

**The grey zone, stated.** The user only wants the device somewhere and watches it. There is no imagery instruction and no added flash, tone or flicker; the only visual events are the device's own outputs. Still, those moves act as evoking events, and the component belongs to the P3 family. This is evaluation-driven control, not imagery. Information-seeking probes are removed (§7), so the device never selects a move for its diagnostic value.

**Not new:** Bayesian ErrP goal inference and implicit cursor control (Grizou 2014; Iturrate, Grizou et al. 2015 PLOS ONE; Iturrate 2015 Sci Rep; Zander 2016). All used displayed or instructed targets on gel caps.

**What I add:**
- (a) Labels with no display information, and a likelihood that partials out attention-gated sensory responses.
- (b) Expectancy and screen conditioning, so the model can be refit in closed loop.
- (c) A measured dry/gel answer to the premise.
- (d) Scaling over hours, sessions, people and capacity, with the saturation point reported.

## 2. Label source: never display-identifiable

1. **Private goal.** Eight distinct, always-present landmark shapes sit in a fixed, jittered, non-mirror-symmetric layout that is identical on every trial. The participant silently picks one, and a choice counter balances the picks. The display carries no goal information, and the distinct shapes make the pick memorable.
2. **Late report.** After a 1.5 s blank, the participant clicks the chosen landmark, outside every decoding window. 5% catch trials ask for the report mid-trial, which measures memory errors.
3. **Label unit.** One move every 1.2 s ± 150 ms. The label is r = Δ(distance to private goal)/step, which is graded.
4. **Open-loop blocks.** Moves are uniform random, or replays of another trial's path toward a landmark g_A, which varies expectancy. Each trial starts from a random position, which decorrelates r from the sensory nuisance d (§5). I(display; g) = 0.
5. **Closed loop.** 20% of moves are uniform-random ε-moves, which are the only moves that refit the goal term, with the screen-predicted r subtracted. Posterior-driven moves train only the goal-marginalized background.
6. **Label noise.** A confusion model comes from catch-trial errors (neighbor confusions included), plus a goal-switch hazard. A history-only prior is the floor.

**Effective n.** Goal-level confounds have one unit per trial (≈135 per session), not one per epoch (≈2,000). All statistics use trial-clustered bootstrap or mixed models.

## 3. Data engine, scaling mechanism, expected curve

**Model of record:** x = B(context) + Σ_k φ_k(r, s, d, a) W_k + ε.
- B: goal-marginalized background.
- W_k: low-rank spatio-temporal components (0.3–8 Hz, 32 ch × 40 samples).
- φ_k: smooth tunings over correctness r, expectancy s, post-move distance to the goal d (the attention/sensory nuisance), and move features a.
- ε: Student-t noise with low-rank-plus-diagonal covariance.

The model is fit by EM, which is iterative, not closed-form, and runs nightly in minutes.

**Why generative.** "Unlabeled hours make it scale" is withdrawn (Ng & Jordan 2001). What remains:
1. Explicit expectancy and screen conditioning, valid in open and closed loop.
2. Parameters shared across goals and effectors.
3. Calibrated likelihoods that combine by Bayes' rule.
4. A natural **hierarchical prior over people** (grafted from *population*).

**Conceded: a ~10⁴-parameter linear model saturates early on own-labeled hours.** Four axes are measured separately:

| Axis | Mechanism | Prediction |
|---|---|---|
| A. Own labeled hours, within-session test | estimating W, φ, noise | saturates at **5–10 h** (split-half ERP ceiling) |
| B. Sessions / cap placements, cross-session test | covers dry-contact and placement variance | still rising after A plateaus; ~20 sessions |
| C. People, hierarchical prior | shared component shapes | 15 min own data + prior ≈ 1–2 h own-only; ~20–30 people |
| D. Capacity: linear 10⁴ vs conditional neural generative 10⁶ (same interface) | nonlinear context effects | neural ≤ linear below 10 h; crossover open |

The bitter-lesson test is whether B, C or D keep rising after A plateaus. A flat curve is reported as the ceiling. Habituation is a time-on-task covariate.

**Volume vs recency.**
- Test sessions: every 6th session, frozen.
- Training data per rung (1, 2, 5, 10, 20, 40 h): sessions drawn randomly across the whole timeline, 20 draws per rung.
- Fit: I_goal ~ log(hours) + log(distance to test session) + (1|person), so volume is estimated net of recency.
- The past-only curve, which matches deployment, is reported separately.

**Baseline.** xDAWN + tangent space + shrinkage LDA, with identical re-centring, expectancy and d features, in the same Bayesian filter.

**Metric.** I_goal = E[log p(x | c, g*) − log Σ_g p(g) p(x | c, g)], with permutation floors at matched n.

## 4. Expected performance (dry 32-ch, fixation)

**Anchor.** RLNF is dropped as an anchor because it was display-confounded. The replacement is T4, run this round (§10).
- Gel, graded outcome evaluation, cortical channels only: median within-person AUC **0.62** (150 trials); pooled leave-one-person-out AUC **0.59**.
- The largest good–bad differences are at Fp1/Fp2 and TP9/TP10.
- Dry electrodes add ERP noise (Mathewson et al. 2017 Psychophysiology 54:74).

My dry, display-free prior is therefore **AUC 0.56–0.66**, lowered from 0.62–0.74. About 2,000 epochs per session (vs 150 in T4) should help, but by how much is unmeasured.

**Bits per evaluation** (equal-variance Gaussian, balanced label; computed): AUC 0.56 → 0.008, 0.60 → 0.023, 0.65 → 0.052, 0.70 → 0.093.

| Metric (deep, fixation) | Expected | Floor |
|---|---|---|
| I_goal | 0.007–0.05 bits/s | permutation, d-sensory, eye, EMG, screen, prior, dissatisfaction |
| 8 goals, success ≤90 s | 15–60% | 12.5% + prior |
| 8 goals, time to 80% posterior | 45–300 s | — |
| Broad cohort, 15 min own + prior | 0.003–0.04 bits/s | — |

**Conceded.** This is slow goal selection, at or below the low end of the brief's 0.03–0.3 bits/s. It does not compete with trained-user SMR continuous control (Wolpaw & McFarland 2004). The device moves smoothly toward the posterior mean, but the information rate is a selection rate. The claim is validity plus scaling, not bandwidth.

## 5. Purity proof

**Auditors (never inputs).** Eye tracker at 1,000 Hz (microsaccades, pupil), bipolar EOG, facial EMG (corrugator, frontalis, orbicularis oculi, zygomaticus), masseter/temporalis EMG, head accelerometer, photodiode.

**Fixation.** Gaze-contingent window **1.0°**, cursor step **2.0°**, so the permitted offset is half a step. Free viewing appears only in a frozen-decoder demonstration.

**Sensory and attention confounds.** Attending to the chosen landmark raises P1/N1 for moves landing near it, and a fixation offset changes retinal geometry. Defenses:
1. **Design:** random start positions keep corr(r, d) below 0.3, verified per session.
2. **Conditioning:** φ includes d, and the primary statistic is the partial I(r | d).
3. **Timing:** information at 50–180 ms occipital is reported. A goal term that passes only through that early window fails.
4. **Attend-ignore block:** same stimuli, but the cursor is declared irrelevant. Sensory gating should persist and evaluation should shrink. The ratio is reported, with no threshold.

**Matched surrogates.** The same pipeline, epochs and folds run on:
- (i-a) gaze, fixation offset × move geometry, microsaccades, pupil
- (i-b) facial and jaw EMG
- (ii) screen stream
- (iii) choice history
- (iv) dissatisfaction-only features (frontal theta, corrugator, blink, pupil)
- (v) d-only sensory model
- (vi) shuffled EEG, and EEG shifted by ≥10 moves

EEG must beat each surrogate and their union, the partial information given (i)–(v) must be > 0, and auditor regression must leave ≥70% of I_goal.

**Co-primary.**
- All-trials and saccade/blink-free analyses must both pass. Saccade-free means no microsaccade ≥0.1° and no blink from −200 to 800 ms. This is a pre-registered artifact criterion, not performance-based exclusion.
- Results are also reported without FP1, FP2, F7, F8, T7 and T8.
- The evidence must peak frontocentrally at 200–600 ms.

**Unannounced pixel replay.** From session 5, 25% of closed-loop trials are frame-exact replays of a past run toward g_A.
- **Statistic:** rank of the private pick g_B among the non-g_A goals, using the evaluation term only (alpha excluded).
- **Symmetry fix:** the layout is non-mirror-symmetric. Source runs are selected so the 7 non-g_A goals get distinct r sequences, with a minimum pairwise distance pre-set.
- **Power:** the number of trials needed for 80% power is simulated from pilot effects and pre-registered (minimum 300 per deep participant).
- Surrogates (iv) and (v) must fail.

## 6. Protocol

- **Pilot:** 6 people × 2 × 1 h. Two of them also do a paired gel session.
- **Deep cohort:** **6 × 60 h** (30 two-hour sessions), down from 4 × 100 h. One dropout now costs a sixth of the data, not a quarter.
- **Broad cohort:** 30 × 2 h, for the hierarchical prior and leave-one-person-out tests.

| Segment (120 min) | Min | Labeled |
|---|---|---|
| Setup, rest | 20 | — |
| Open-loop, fixation (3 × 13) | 39 | ≈1,950 epochs / ≈105 trials |
| Attend-ignore | 6 | ≈300 epochs |
| Closed loop, ε = 0.2, replay (2 × 14) | 28 | ≈280 epochs + ≈30 replays |
| Free viewing, frozen decoder | 7 | — |
| Breaks | 20 | — |

**Totals.** Per deep participant: ≈67k epochs and ≈4,000 trials. The "60 h" ladder holds ≈37 h of goal-term data, and is reported that way.

**Burden.** Wrong moves come from a random policy only. Fixation blocks last 13–14 min. Participants get per-session pay plus a completion bonus.

**Robot (months 10–14).** Planar tabletop with 8 distinct physical objects, 4-cm steps, and fixation on a central mark. No depth judgments.

**Timeline.**
- Months 1–3: build, T4 extension, pilot gate
- Months 3–12: deep cohort
- Months 5–11: broad cohort
- Months 14–18: analysis and writing

**Staff:** 2 RAs, 1 ML engineer.

## 7. Real-time architecture

- **Acquisition.** 500 Hz over LSL, photodiode-stamped onsets, causal 0.3–40 Hz filter, robust common average reference.
- **Inversion.** 800 ms after each move, the 8 goals are scored in under 5 ms (64 via a two-stage 8 × 8 hierarchy, cursor only), with goal-switch and confusion priors. Temperature is calibrated on held-out closed-loop moves.
- **Move policy.** Toward the posterior mean with probability 0.8; uniform ε-move with probability 0.2. **No information-seeking probes.** The convergence cost is to be quantified in simulation before the pilot.
- **Adaptation.** Re-centring every 10 s; nightly EM refit of φ, which tracks expectancy and habituation; hierarchical-prior update.
- **Exploratory, uninstructed term (grafted from *population*).** A goal-direction likelihood on inter-move 8–30 Hz sensorimotor power. It tests whether wanting to go somewhere spontaneously lateralizes. It is credited only beyond eye surrogates and is not part of the headline.
- **Effectors.** The output is a goal posterior, executed by shared autonomy (Dragan & Srinivasa 2013; Javdani et al. 2015) on a cursor or a planar arm. Goals outside the candidate set are inexpressible. Cursor→robot transfer is tested zero-shot, with a 20-min φ refit as the fallback.

## 8. The Nature claim

**One sentence.** *On a dry 32-channel cap, scalp EEG evaluating a device's own moves identifies which of eight places a person privately wants a cursor or robot to reach. This holds beyond sensory, attentional, ocular, facial and screen explanations, including when the screen is pixel-identical to a run aimed elsewhere, and the information scales with sessions and people up to a measured ceiling.*

**Headline figure.**
- (a) I_goal on axes A–D with surrogate floors, volume net of recency.
- (b) Replay rank vs the dissatisfaction, d-sensory, eye, EMG and screen surrogates.
- (c) Partial I(r | d) time course and topography; attend-ignore ratio; dry vs gel.
- (d) One posterior steering a cursor and a planar arm.

**Why it is new, honestly.** The paradigm exists (Zander 2016; Iturrate 2015). New here:
- a dry-cap, private-goal, display-free demonstration;
- separation of evaluation from attention-gated sensory responses;
- replay dissociation;
- multi-axis scaling.

**Nature Communications is the realistic venue.** Nature would require axis B, C or D to keep rising after A plateaus.

## 9. Falsifiers and biggest risk

**Pilot gate (rewritten; p95 + 0.03 was too weak).** In ≥4 of 6 pilots, at ≈2,000 epochs and ≈105 trials, the **saccade-free, frontocentral partial AUC for r given d** must have a trial-clustered 95% CI lower bound above all three of:
- the permutation p95;
- the d-only surrogate;
- the eye/EMG union.

The dry/gel ratio is reported. If the gate fails, the project stops, or switches to gel and drops the dry claim.

**Falsifiers.**
- **F1.** Replay rank ≤ null, or not above surrogates (iv)/(v).
- **F2.** Saccade-free partial I(r | d) at ≥30 h is not above the floors.
- **F3.** B, C and D are flat after A plateaus. This means no bitter-lesson effect, reported as the ceiling.
- **F4.** Generative ≤ discriminative at every rung → drop the generative framing.
- **F5.** Closed-loop/open-loop I_goal < 0.5.

**Biggest risk.** T4 already shows graded evaluation near AUC 0.6 on gel, with frontal channels matching it. Dry noise or habituation could push it to the floor. The gate is designed to find this out in month 3.

## 10. Tests now

**T4 (run this round) on OpenNeuro ds004262 (Hassall, Yan & Hunt).**
- **Data.** 20 usable people, 31-ch gel. Participants guess how high a bar will rise, then watch it; 150 scored trials each.
- **Pipeline.** Average reference, 0.3–8 Hz, outcome-locked 200–600 ms at 50 Hz. Shrinkage LDA. Label = absolute error below the person's median.
- **Channel sets.** Cortical: FC1, FCz, FC2, C3, Cz, C4, CP1, CPz, CP2, Pz. Frontal proxy: Fp1, Fp2, F7, F8.

| Test | Cortical | Frontal proxy | All 31 |
|---|---|---|---|
| Within-person 5-fold AUC, median (IQR) | 0.62 (0.53–0.67); 10/20 above own 20-perm p95 | 0.62; 12/20 above p95 | — |
| Leave-one-person-out AUC, trained on 1 / 4 / 10 / 19 people | 0.51 / 0.57 / 0.58 / 0.59 | 0.54 / 0.60 / 0.58 / 0.60 | 0.57 / 0.64 / 0.64 / 0.68 |
| ρ(score, −\|error\|), mean over people | 0.156 (p = 1e-4); 0.153 after partialling displayed bar height, guess and signed error | 0.163 | — |
| Display-only label (bar height) | AUC 0.63 | 0.58 | — |

**Reading.**
- A graded evaluative signal exists single-trial and survives the displayed-geometry partial.
- It is small, and frontal channels match it.
- The largest good–bad difference at 250–400 ms is at Fp1/Fp2 (+) and TP9/TP10 (−). This is the RLNF warning repeated on clean gel data.
- Sensory content (bar height) is decodable from the same channels, so d-conditioning and eye auditors are mandatory.
- The people axis rises: in miniature, axis C is real but small.

**Caveats.** Gel, no EOG, displayed outcome: physiology only, not purity. Permutation floors are coarse. Scripts: scratchpad `errp_scripts/` (ep.py, an.py, loso.py, partial.py).

**T1–T3 on ds004362** (secondary): purity-harness calibration and generative vs discriminative. ds004362 has no evaluative events.

**References:** Holroyd & Coles 2002 Psychol Rev 109:679 · Zander et al. 2016 PNAS 113:14898 · Iturrate et al. 2015 Sci Rep 5:13893 · Iturrate, Grizou et al. 2015 PLOS ONE 10:e0131491 · Grizou et al. 2014 AAAI · Chavarriaga et al. 2014 Front Neurosci 8:208 · Mathewson et al. 2017 Psychophysiology 54:74 · Hassall, Yan & Hunt, OpenNeuro ds004262 · Wolpaw & McFarland 2004 PNAS 101:17849 · Engbert & Kliegl 2003 Vision Res 43:1035 · Ng & Jordan 2001 NIPS · Dragan & Srinivasa 2013 IJRR · Javdani et al. 2015 RSS
