# P_generative (amended): generative inverse decoding with an expectancy-aware per-person model of EEG

## 1. One-line idea and signal basis

**Idea.** Fit one conditional generative model of a person's dry-cap EEG, p(EEG | context, candidate goal g). The context is the effector's move, the person's likely expectation, and what the screen already predicts. Decode intent by Bayesian inversion. After each effector move, score every candidate goal by how well it explains the EEG, then move the cursor or robot toward the posterior. While the goal is uncertain, the effector makes **information-seeking probe moves** whose evaluation is diagnostic.

**Signals.** One primary and one secondary.

| Relational feature | Phenomenon | Locus | Evidence |
|---|---|---|---|
| Was the last move toward or away from *my* goal, and how surprising was that? | Evaluation / prediction-error response (error negativity and positivity, reward positivity, mid-frontal theta), 200–600 ms | FCz/Cz | Holroyd & Coles 2002; Zander et al. 2016 PNAS (amplitude scaled with degree of expectation violation); Chavarriaga 2014; Iturrate 2015 Sci Rep |
| Hemifield of my goal (secondary) | Posterior alpha lateralization | PO/O | Thut et al. 2006; microsaccade-audited (§5) |

CNV and low-frequency "direction tuning" are no longer claimed. CNV indexes anticipation, not distance, and Kobler 2018 tuning was measured while people tracked displayed targets.

**"Thinking", and the grey zone.** The user only *wants* the effector somewhere and watches it. There is no imagery instruction and no added flash, tone or flicker. However, the effector's own moves are chosen to be diagnostic, so they work as evoking events. The claim (§8) says so.

**Not new:** Bayesian ErrP goal inference with planning (Grizou 2014; Iturrate, Grizou 2015 PLOS ONE; Zander 2016), all with displayed or instructed goals.

**What I add:**
- (a) Labels that carry no display information, verified by an unannounced pixel-replay dissociation and by eye, microsaccade and facial-EMG auditors.
- (b) A likelihood that conditions on expectancy and on the screen-predictable component, so it can be refit in closed loop without learning to read the screen.
- (c) Separately measured scaling laws for labeled and unlabeled hours.

## 2. Label source: never display-identifiable

1. **Private goal.** K identical, unlabeled placeholders (4 at first, up to 64) surround a fixation point. The participant silently picks one. A choice counter balances the picks.
2. **Late report.** After a 1.5 s blank, the participant clicks the chosen placeholder. This falls outside every decoding window.
3. **Label unit = probe evaluation.** One move every 1.2 s ± 150 ms, so 200–600 ms responses do not overlap. The label is r = cos(angle between move and private goal). It is graded.
4. **Open-loop label blocks (goal-blind, mixed expectancy).** Each trial uses one of two policies: uniform random moves, or a replayed closed-loop trajectory from another trial that drives steadily toward some placeholder g_A. When g_A happens to equal the private goal (1/K of replay trials), the person sees long runs of correct moves, as in closed loop. I(display; g) = 0 still holds. The base rate of "toward" moves now varies across trials, so the model learns how amplitude depends on expectancy instead of seeing only the error-rich random regime.
5. **Closed-loop blocks with ε-probes.** 20% of moves are uniform random probes, which the person cannot tell from posterior-driven moves. Here the screen history does carry goal information, so the goal term is refit on these probes only with the screen conditioned out. The background term receives the expected r under p_disp(g | screen history), the best screen-only posterior (surrogate ii, §5). The goal term gets credit only for EEG dependence on the true r beyond what the screen predicts. Posterior-driven moves train only the goal-marginalized background term.
6. **Label noise.** A history-only prior serves as a floor. Catch trials with a double report estimate changes of mind, and a goal-switch hazard of 0.01/s handles them.

## 3. Data engine, scaling mechanism, expected curve

**The model (structured, not a large transformer).** Each post-move epoch (0–800 ms) has two parts: 0.3–8 Hz amplitude at 50 Hz, and posterior 8–13 Hz power. Its likelihood is
x = B(context) + Σ_k φ_k(r, s, a) W_k + ε,
where:
- B is a goal-marginalized background;
- W_k are low-rank spatio-temporal evaluation components;
- φ_k are smooth tunings over graded correctness r, expectancy s (the probability this move is toward the goal under recent experience and under the screen) and move features a;
- ε is Student-t noise with low-rank-plus-diagonal covariance.

In effect this is a "generative xDAWN". A causal sequence model for B is added only if T2 (§10) shows a gain.

**Why generative: the revised argument.** The original claim was that background modeling of unlabeled hours lets the generative model scale where discriminative ones stalled. I withdraw it, for three reasons:
- Ng & Jordan 2001 show generative models converge faster, often to a worse asymptote.
- Most single-trial ERP noise is unpredictable innovation.
- Discriminative pipelines also get whitening and re-centring from unlabeled data.

The generative structure stays for three properties a per-epoch classifier lacks:
1. Explicit expectancy and screen conditioning, so one likelihood is valid in both open and closed loop (§2.5).
2. Graded, shared parameters: one response model scores all K goals and any effector, so 64 goals cost no more labels than 4.
3. Calibrated likelihoods combined across probes and signals by Bayes' rule.

**Two scaling axes, measured separately.**
- **Labeled probe epochs, the bitter-lesson axis.** About 2,500 free, display-independent labels per 2-h session. Prediction: AUC-equivalent rises from 0.62–0.66 at 1 h to 0.68–0.74 at 20–50 h. It then plateaus at the split-half ERP noise ceiling. Anchors: RLNF found volume-only gains; Kaifosh et al. 2025 Nature.
- **Unlabeled hours (background).** Prediction: ≤10% gain in I_goal, saturating by 5–10 h.

**Fair baseline.** xDAWN + Riemannian tangent space + shrinkage LDA in the same Bayesian filter, with the same unlabeled whitening, re-centring and expectancy features. Prediction: generative ≥ baseline at ≤5 h, and ≈ equal at ≥20 h.

**Figure of merit.** I_goal = E[log p(x | c, g*) − log Σ_g p(g) p(x | c, g)].
- Fixed test set: every 10th session, frozen from day 1.
- Rungs: 1, 2, 5, 10, 20, 50 and 100 h.
- Permutation floors at matched n.

## 4. Expected performance (dry 32-ch, fixation)

**Bits per probe.** For an equal-variance Gaussian decoder with a balanced binary label (computed): AUC 0.65 → 0.052 bits per evaluation; 0.70 → 0.093; 0.75 → 0.148. At a 0.25 base rate the values are 0.039, 0.071 and 0.113. Graded errors decode worse (Omedes et al. 2015), so the expected range is 0.04–0.09 bits per probe.

**Bits per second.** One probe per 1.2 s gives 0.03–0.08 bits/s. Alpha adds ≤0.02 bits/s. **Total: 0.04–0.10 bits/s.** The anchor is RLNF's dry-cap AUC of 0.66–0.69. The repo simulation's 0.68 is an assumption, not evidence.

| Metric (deep, ≥50 h, fixation) | Expected | Floor |
|---|---|---|
| I_goal | 0.04–0.10 bits/s | permutation, auditor, screen, prior and dissatisfaction surrogates |
| 4 goals, time to 90% posterior | 25–60 s | no-EEG probing does not converge |
| 8 goals, success ≤90 s | 50–75% | 12.5% + prior |
| Closed-loop / open-loop I_goal | ≥0.7 | measured |
| Broad cohort (population + 1 h own) | 0.02–0.06 bits/s | — |

This is goal selection, not fast steering. **The headline is validity plus measured scaling.** The simulation's "ErrP added nothing" was ErrP on top of an assumed r ≈ 0.4 velocity decoder that no dry-cap, display-free study has shown.

## 5. Purity proof

**Auditors (never inputs).**
- Eye tracker at 1,000 Hz, with microsaccade detection (Engbert & Kliegl 2003) and pupil.
- Bipolar EOG.
- **Facial EMG:** corrugator, frontalis, orbicularis oculi, zygomaticus.
- Temporalis, masseter, SCM/trapezius.
- Head accelerometer.
- Photodiode.

**Feature hygiene.** The goal term sees only 0.3–8 Hz amplitude and posterior alpha. Beta and gamma power, where EMG and saccadic spike potentials appear (Yuval-Greenberg et al. 2008), may enter only the background term, which cancels in the inversion. All results are also reported without FP1, FP2, F7, F8, T7 and T8.

**Fixation wherever it matters.** A 2° gaze-contingent pause applies to every block that trains the goal term or feeds a headline panel, robot blocks included: the participant fixates a mark on the table. Free viewing is used only in an exploratory demonstration with a frozen goal term.

**Matched surrogates.** The identical pipeline is run on the same epochs and folds with these inputs:
- (i-a) eye, microsaccade direction and pupil;
- (i-b) facial and jaw EMG;
- (ii) the screen stream;
- (iii) choice history;
- (iv) **dissatisfaction-only**, goal-agnostic features: frontal theta power, corrugator RMS, blink rate, pupil;
- (v) shuffled EEG, and EEG shifted by ≥10 probes.

EEG must beat each surrogate and their union, and the partial information of EEG given (i)–(iv) must be > 0. Microsaccades are biased toward the attended side, so I expect (i-a) above floor for the hemifield component. Alpha is credited only beyond that surrogate. "Saccade-free" means no microsaccade ≥0.1° and no blink from −200 to 800 ms. The primary analysis uses all trials, with no performance-based exclusion. Regressing out the auditors (lags ±200 ms, facial EMG included) must leave ≥70% of I_goal.

**Unannounced pixel replay.** From session 5 onward, 25% of closed-loop trials are photodiode-verified, frame-exact replays of a past run that headed to g_A. The participant has already picked a goal and cannot know the trial is a replay. A display reader recovers g_A. A dissatisfaction reader only down-weights g_A. An intent decoder recovers the pick g_B.
- **Statistic (pre-registered):** the rank of g_B among the K−1 goals other than g_A, against a uniform null. Equivalently: log P(g_B) − log mean_{g∉{g_A,g_B}} P(g) > 0.
- At least 300 replay trials per deep participant.
- Surrogate (iv) must fail this statistic.
- Picks equal to g_A are analyzed separately.

**Topography and timing.** Evidence must peak frontocentrally at 200–600 ms on saccade-free epochs. A frontopolar or temporal peak counts as a fail. All floors use 1,000 permutations at matched n, and every analysis is pre-registered.

## 6. Protocol

- **Pilot:** 6 × 2 × 1 h, with a gate (§9).
- **Deep cohort:** 4 × 100 h (50 two-hour sessions).
- **Broad cohort:** 24 × 2 × 2 h.

| Segment (120 min) | Min | Labeled epochs |
|---|---|---|
| Setup, quality check | 15 | — |
| Rest | 5 | unlabeled |
| A. Open-loop label blocks, fixation (3 × 15) | 45 | ≈2,250 |
| C. Closed-loop, fixation, ε = 0.2, unannounced replay (2 × 15) | 30 | ≈300 + ≈30 replay trials |
| Free use, fixation | 10 | unlabeled |
| Breaks | 15 | — |

**Accounting.** Per session: ≈2,500 labeled epochs and ≈135 trials. Per deep participant: ≈125k epochs and ≈6,750 trials. Goal-relevant data is 75 of 120 minutes, so the "100 h" ladder holds ≈62 h of goal-term data, and I report it that way.

**Robot.** Months 10–14: 10 of the 30 closed-loop minutes.

**Timeline.**
- Months 1–3: build, ds004362, pilot gate.
- Months 3–12: deep cohort.
- Months 6–12: broad cohort.
- Months 14–18: frozen evaluation and writing.

**Staff:** 2 RAs and 1 ML engineer. The model of record is closed-form and fits nightly on a workstation.

## 7. Real-time architecture

- **Acquisition.** 500 Hz over LSL, photodiode-stamped move onsets, causal 0.3–40 Hz filter, robust common average reference.
- **Inversion.** 800 ms after each move, all K ≤ 125 goals are scored in closed form in under 5 ms. A goal-switch hazard applies. Temperature calibration is fit on held-out closed-loop probes and checked with reliability diagrams.
- **Probe policy.** Maximize expected information gain over 8 or 26 directions, blended toward the posterior mean by α = max posterior. The ε = 0.2 probes are kept forever.
- **Adaptation.**
  - Every 10 s: unsupervised re-centring of gain, offset and covariance (the repo simulation showed this partly rescues drift).
  - Nightly: refit on open-loop epochs plus screen-conditioned closed-loop probes. The expectancy tuning φ is refit, so habituation is tracked rather than frozen.
- **Effectors.** The output is a posterior over Cartesian goal positions, executed by shared autonomy (Dragan & Srinivasa 2013; Javdani et al. 2015).
  - Cursor: 1.5° steps on a 64-point grid.
  - Robot arm: 3-cm steps over a 5 × 5 × 5 grid of known positions, with no perception system.
  - Paths and orientations outside the candidate set cannot be expressed.
- **Transfer.** Cursor→robot zero-shot transfer is a test. If it fails, the fallback is a 20-min refit of φ only.

## 8. The Nature claim

**One sentence.** *Scalp EEG alone, read through an expectancy-aware generative model of a person's evaluative responses to a device's own moves, identifies which of many locations they privately want a cursor or robot to reach. This holds when the screen is pixel-identical to a run aimed elsewhere, with eyes and face audited, and its information grows with hours of freely labeled data up to a measured ceiling.*

**Headline figure.**
- (a) I_goal vs log labeled hours (4 deep participants, noise ceiling), plus the unlabeled-hours curve and all surrogate floors.
- (b) Unannounced replay: rank of the private goal among the non-displayed goals, for EEG vs the dissatisfaction, facial-EMG, eye and screen surrogates.
- (c) Response amplitude vs expectancy, open vs closed loop, with topography.
- (d) One posterior, under fixation, steering the cursor and the arm.

**Why it is new.** Prior Bayesian ErrP goal inference (Grizou 2014; Iturrate 2015 PLOS ONE; Zander 2016) used displayed or instructed targets. None of it had replay dissociation, facial or microsaccade auditors, or scaling over tens of hours. The novelty is a validity standard plus a scaling measurement. Nature Communications is the realistic venue unless (a) shows a clean slope.

## 9. Falsifiers and biggest risk

- **Pilot gate.** Cortical-montage probe AUC at 2,000 epochs must exceed the permutation p95 + 0.03 in ≥4 of 6 pilots, or the project stops.
- **F1.** The replay rank statistic is ≤ null, or not above surrogate (iv) → no goal-specific signal.
- **F2.** At ≥50 h, I_goal does not exceed the permutation floor and the auditor union (95% CI).
- **F3a.** The labeled-hours slope from 10 to 60 h has a CI including 0 → no bitter-lesson effect (still publishable).
- **F3b.** Unlabeled hours add nothing → the background claim is dropped.
- **F4.** Generative ≤ discriminative at every rung → drop the generative framing; purity and replay still stand.
- **F5.** Closed-loop / open-loop I_goal < 0.5 → regime mismatch unsolved.

**Biggest risk.** Dry-cap graded-error AUC may stay at 0.62–0.68 (0.03–0.05 bits/s), and the prediction-error response may habituate over 100 h. Nightly expectancy refits make habituation a reported result rather than hidden drift. Larger probe angles are used when the posterior is flat.

## 10. Tests on ds004362 now

The dataset (109 subjects, 64-ch gel, 160 Hz) has displayed targets, so it calibrates the purity harness. I use 32 g.tec-equivalent channels, a fixed 19-subject test set, and a-priori exclusions only.

- **T1.** Generative inversion vs FBCSP+LDA, tangent space and EEGNet, at 10, 20 and 45 trials per class. *Kill:* generative is more than 5 points worse at every size.
- **T2.** Unlabeled background pretraining on 0–90 subjects, run **only in the T3-clean regime** (1–4 s, centro-parietal, frontal channels regressed out). *Kill:* the slope CI includes 0, in which case no sequence-model background is built.
- **T3.** Trap size: 0–400 ms vs 1–4 s, and frontal "EOG proxy" vs cleaned centro-parietal channels. This proves the harness detects leaks.

**Limit.** ds004362 has no evaluative events, so the pilot gate tests the core signal.

**References:** Holroyd & Coles 2002 Psychol Rev 109:679 · Zander et al. 2016 PNAS 113:14898 · Iturrate et al. 2015 Sci Rep 5:13893 · Iturrate, Grizou et al. 2015 PLOS ONE 10:e0131491 · Grizou et al. 2014 AAAI · Chavarriaga et al. 2014 Front Neurosci · Omedes et al. 2015 J Neural Eng · Thut et al. 2006 J Neurosci 26:9494 · Engbert & Kliegl 2003 Vision Res 43:1035 · Yuval-Greenberg et al. 2008 Neuron 58:429 · Ng & Jordan 2001 NIPS · Dragan & Srinivasa 2013 IJRR · Javdani et al. 2015 RSS · Kaifosh et al. 2025 Nature
