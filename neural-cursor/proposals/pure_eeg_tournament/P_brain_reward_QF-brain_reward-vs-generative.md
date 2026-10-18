# P_brain_reward (amended): the brain as the reward function. EEG evaluations teach a machine your own stable preferences; then the EEG is unplugged

*Amended: goal-blind label blocks; pixel replay, union surrogate and regress-out adopted from the opponent; stable natural preferences; RLNF anchor dropped; arithmetic fixed; interleaved test set; online Tier 2 cut; saturating scaling prediction.*

## 1. One-line idea and signal basis

**Idea.** A device (cursor or robot) acts by itself. The person only decides privately what they want and watches. Each action evokes an evaluative response. A reward model decodes that response, and the device's policy is trained on it, as in RLHF. The test: the policy is frozen, the **EEG is disconnected**, and the device must still do what this person wants.

**Signal.** Medial-frontal evaluation of an observed outcome: error negativity/positivity and reward positivity (Holroyd & Coles 2002; Proudfit 2015) and mid-frontal theta (Cavanagh & Frank 2014), 200–500 ms after the action at Fz/FCz/Cz. It is the brain judging whether the world moved toward its goal, with no imagery and no added stimulus.

**What this is not.** Moment-to-moment velocity control. At 0.03–0.3 bits/s I argue goal/preference-level control is the honest information-efficient route; this knowingly trades the PI's continuous steering for purity and amortisation.

**Precedents:** Iturrate 2015 Sci Rep (devices learned, then reached, targets from ErrPs); Zander 2016 PNAS (implicit cursor control); Kim 2017 Sci Rep (robot learned a gesture→action mapping); Batzianoulis 2021 Commun Biol (ErrP inverse RL); Lopes-Dias 2019 Sci Rep (asynchronous gel ErrP, TPR 70 %, TNR 86.8 %); Grizou 2014. Teach-then-execute is **not** new; the novelty is in §8.

## 2. Label source and why it is NOT display-identifiable

The label is the goal-relative value of an action, r = cos(θ_action − θ_goal), where the goal is committed privately before the trial.

**Conceded flaw.** Probes inside closed-loop trials were *not* display-clean: preceding policy moves came from a goal posterior, so cos(θ_probe − recent policy direction) predicts r from the screen (readable via motion-adaptation and looming VEPs), and the leak grows as the agent improves. Fix:

**A. Goal-blind label blocks (the only source of headline labels).** Every action is drawn uniformly from 8 directions, or replayed from another trial, and is independent of this trial's goal. The whole display stream (history, Δθ, nearest ring, adaptation state) is therefore independent of g, which gives I(display; g) = 0.
- Goals are not exactly uniform, so P(r | display) = P(r) is not assumed: a between-trial choice counter encourages balance, and a prior surrogate E[r | θ_action, choice history] is fitted and reported as a floor.
- 12-action trials: commit on a side tablet rotated by a random multiple of 45°, ending ≥1.5 s before the first action; late re-report after a 1.5 s blank; disagreements kept and flagged by a pre-set rule.

**B. Teaching/execution blocks (closed loop).** The policy uses the frozen reward model. Epochs from these blocks **never** train the headline reward model and never enter the headline AUC curve.

**C. Pixel-identical replay (adopted from the opponent; the decisive closed-loop control).** A teaching run aimed at g_A is replayed frame by frame to the same person, who now privately holds g_B. Trials where g_B = g_A are excluded by a goal-based rule, never by performance.
A display reader tracks cos(θ − g_A); a person reader tracks cos(θ − g_B). Statistic: partial correlation of EEG reward with cos(θ − g_B) controlling for cos(θ − g_A), ≥300 replay trials per deep participant, pre-registered > 0, beside a display-history surrogate on the same trials.

**D. Goal-switch catch trials (10 % of teaching trials).** These are logged on the tablet. Momentum then predicts the wrong label.

## 3. Data engine, scaling mechanism, expected curve

**Rates.** A label trial takes about 21 s (commit 3 s, 12 actions at 1.2 s, blank and report 3–4 s), which is about 34 clean epochs per minute, or **~2,000 per hour of label blocks**. A deep session devotes 60 of 120 min to label blocks.

**Three scaling axes**
1. **Within-person.** 3 deep participants × ~100 h, giving ≈100k clean label epochs each.
2. **Across people.** 16 people × 3 h give ≈60k epochs for a zero-calibration population model, tested on 6 held-out people.
3. **Amortisation.** A taught preference is reused at zero EEG cost; bits sent grow with distinct preferences, not uses. Preferences are now stable (§6), so this is measured within the study.

**Fixed test set (adopted).** Every 10th session, interleaved and never trained on. Ladder rungs (1–90 h) are drawn at random across remaining sessions (5 draws each), removing the recency confound; "first-N h" vs "random-N h" measures nonstationarity. Permutation floors at matched n per rung.

**Mechanism beyond "more data".** Templates saturate at single-trial SNR; more hours can still approach the noise ceiling via (i) per-trial latency realignment, (ii) graded coding across angle bins, (iii) state covariates (pre-action alpha/theta, habituation index), learned above ~20k epochs by a small CNN warm-started from the population model.

**Honest expected curve.** Dry cap, cortical-only, AUC for r > 0 versus r < 0:

| clean epochs | AUC (hypothesis) |
|---|---|
| 500 (15 min) | 0.58–0.64 |
| 2,000 (1 h) | 0.61–0.68 |
| 20k (10 h) | 0.63–0.71 |
| 100k (≈50–90 h) | 0.64–0.73, at the split-half noise ceiling |

I now predict a **saturating** curve (90 % of ceiling by 10–20 h; 10→100 h gain ≈0.01–0.03, on the old kill line). The within-person "scaling law" was over-sold and is withdrawn. Kept claims:
- (a) Within-person: the curve reaches the measured noise ceiling (pre-registered fit a − b·n^(−c)).
- (b) Across people: zero-calibration AUC comes within 0.03 of a 1-hour own model.
- (c) Amortisation: bits per executed choice fall without bound as reuse grows.

## 4. Expected performance on this hardware (re-anchored)

**Anchor.** RLNF AUCs measured agreement with the screen (a no-EEG control reached 0.655), so they **cannot** anchor anything, and I drop them.
Gel ErrP literature gives ~70–80 % balanced accuracy (Lopes-Dias 2019: 70 %/86.8 %; Zander ≈71 % offline). Subtract an unknown dry penalty (prior 0.03–0.08 AUC) and a penalty for observed, small, graded errors (Omedes 2015). Honest dry cortical-only prior: **0.60–0.70**, 0.58 possible; the gel arm measures the dry penalty.

**Teaching cost** (balanced binary, Gaussian decoder, computed; practical = 2× ideal; 1.2 s per action):

| AUC | bits/eval | evals for 24 bits (ideal) | practical minutes for 24 bits |
|---|---|---|---|
| 0.60 | 0.023 | 1,053 | 42 |
| 0.65 | 0.052 | 465 | 19 |
| 0.70 | 0.093 | 258 | 10 |
| 0.75 | 0.148 | 163 | 7 |

This corrects my earlier "10–20 min", which held only at AUC ≥ 0.70.

**Not fatal at low AUC.** The repo's 0.08 success at AUC ≈ 0.58 was **real-time steering** with a timeout. A *stable* preference has no timeout: at AUC 0.60, 24 bits take ~42 min over 2–3 sessions, then are reused for weeks. Real-time steering stays weak (≈0.03–0.12 bits/s); I do not claim otherwise.

**Predictions** (deep participants, frozen evaluation sessions):
- ≥85 % of the 8 contexts correct with EEG unplugged if AUC ≥ 0.65 (chance 12.5 %).
- ≥60 % correct if AUC is 0.60–0.65.
- Retention over 4 weeks without re-teaching (the policy is frozen, so it should be ~100 %; the measure is whether the person's preferences drifted).

Amortised throughput is reported separately and labelled as amortised. It is not a brain transfer rate.

## 5. Purity proof (strengthened)

**Auditors** (recorded, never inputs): eye tracker ≥500 Hz (gaze, microsaccades, pupil, blinks); bipolar EOG; corrugator/zygomaticus EMG (frowning at errors); temporalis EMG; head accelerometer; photodiode.

**Display.** No continuous translation (it invited pursuit/OKN): an action is a 150 ms, 1.5° glide of the ring world, then stillness. Label blocks **enforce fixation** (2° window; any exit pauses, whatever direction). Residual saccades, microsaccades and pursuit enter the union surrogate.

**Criteria** (all pre-registered, label blocks, on the interleaved test set):
1. **Union surrogate.** A nonlinear model (gradient-boosted trees) on eye, pupil, microsaccade direction relative to each candidate goal, EMG, display history and the choice prior, trained on the same folds. Required: EEG AUC − union AUC ≥ 0.03, and (EEG + union) − union ≥ 0.03, both with 95 % CIs excluding 0. This replaces "partial correlation > 0", which I concede was too easy at n ≈ 10^5.
2. **Regress-out (adopted).** Removing auditor regressors at lags −200 to +200 ms from the EEG, and separately dropping FP1/FP2/F7/F8/T7/T8, must each keep ≥70 % of bits per evaluation.
3. **Topography vs a measured artefact map.** A "frontocentral peak" was circular (F3/Fz/F4 carry residual vEOG). Each session records 20 blinks and 20 vertical saccades to map that person's ocular propagation; the decoder's forward pattern (Haufe 2014) must correlate |ρ| < 0.5 with it and show the negativity-then-positivity structure at FCz/Cz.
4. **Pixel replay** (§2C) for closed-loop data.
5. **Surrogate-reward policies.** Identical agents trained offline on logged teaching epochs with rewards from EEG, eye/pupil, EMG, display (allowed to exploit teaching-block leaks), union, shuffled, and ≥10-action-shifted EEG; frozen policies scored on held-out preference sets. Required: EEG − best surrogate ≥ 10 points.

**No rejection cost.** Nothing is vetoed online. Updates are weighted by calibrated confidence. No exclusion is based on performance.

## 6. Protocol

**Preferences are now natural and stable.** Previously each block had a new arbitrary icon→goal mapping; this is replaced.
A context is an everyday object at fixation (mug, keys, phone…); the person decides once where in the 8-location workspace *they* would put it, logged on the rotated tablet and re-confirmed each session. Self-generated and persistent, so memory load is low. Each deep participant has 3 sets of 8 objects (kitchen, desk, tools), taught at different times, giving leave-set-out folds. Genuine changes of mind are logged and serve as re-adaptation tests.

**Phases**
- **Phase 0, pilot (months 0–3).** 6 participants × 3 sessions × 1.5 h, with 40 min of label blocks per session (≈1,370 clean epochs per session). Three of them add one 64-channel gel session on an existing lab amplifier.
- **Phase 1 (months 3–11).** 3 deep participants at 2 h/day, 3–4 days/week, ≈100 h each. A cohort of 16 × 3 × 1 h.
- **Phase 2 (months 11–16).** Frozen evaluation sessions, 6 held-out participants on the population model with no calibration, and the robot arm for one deep participant (10 h).
- **Months 16–18:** writing.

**Deep session (120 min)**

| Segment | Minutes |
|---|---|
| Setup, impedance, artefact-map calibration | 18 |
| Rest | 4 |
| Goal-blind label blocks, fixation | 4 × 15 |
| Teaching/execution blocks (current context set) | 20 |
| Pixel replay (from session 10) | 10 |
| Breaks | 8 |

Free-goal trials have no 15-action cap any more. Label trials are fixed at 12 goal-blind actions. In teaching blocks the posterior carries over across trials and sessions, so a stable preference accumulates evidence until it is learned. The only instruction: "decide what you want; watch it."

**Staff:** 2 RAs and 1 ML engineer.

## 7. Real-time architecture

- **Acquisition.** LSL at 500 Hz. Photodiode-timestamped action onsets. Epochs 0–1,000 ms. Causal 1–10 Hz filter, decimated to 64 Hz. Cortical-only montage of record.
- **Reward model.**
  - xDAWN + Riemannian tangent space + shrinkage LDA; above ~20k epochs, a small CNN with latency alignment and state covariates.
  - Trained only on label-block epochs; Platt-calibrated.
  - Within-session covariance re-centring against gain drift; output 1.0 s after the action.
- **Agent.** A Bayesian posterior over 8 locations per context, using class-conditional likelihoods of the reward score given r, estimated on label blocks. Thompson sampling with ε = 0.2 exploration. *Cut for scope:* soft actor-critic.
- **Execution.** The frozen policy outputs a goal per context; a continuous-velocity controller drives the cursor, or the robot via a Cartesian impedance controller, smoothly to it. Teaching uses discrete 1.5°/3 cm steps (the ERP needs a time-locking event); the same reward model and goal output serve both effectors.
- **Tier 2: online REINFORCE cut** (its input window overlapped the previous evaluation by 800 ms, so it would learn lose-shift; no sample-complexity case). Replaced by an exploratory offline analysis in one 2.4 s-ISI label block per session: decode the goal from −1,000 to 0 ms pre-action (starting 1.4 s after the prior action), partialling out the previous r and reward output.

## 8. The Nature claim

**Headline figure**
- (a) Frozen-policy success with EEG unplugged against minutes of teaching, for the EEG reward and for the eye/pupil, EMG, display, union, shuffled and shifted surrogate rewards.
- (b) Pixel-replay dissociation: reward tracks the privately held g_B, not the displayed g_A.
- (c) Probe AUC against hours on the interleaved test set, saturating at the measured noise ceiling, plus population zero-calibration.
- (d) Forward-model topography against the person's own ocular map, plus one taught preference set executed by cursor and robot.

**One sentence.** "From unprompted brain evaluations of a device's actions alone, with goals the screen never reveals, eyes and face audited and a pixel-identical replay control, a machine learns a person's own stable preferences and then carries them out for weeks with the EEG disconnected."

**Why it is new.** Prior ErrP-RL showed teach-then-execute with displayed or instructed targets and no surrogate-reward or replay controls. New: display-unidentifiable labels; a pixel-replay dissociation for evaluative EEG; ~100 h per-person scaling to a measured ceiling on an interleaved test set; natural preferences executed for weeks.

Realistic venue: Nature Communications.

## 9. Falsifiers and biggest risk

**Falsifiers**
- **Pilot gate.** Cortical-only label-block AUC at n = 2,000 clean epochs (sessions 1–2), with r > 0 versus r < 0, must exceed the permutation p95 + 0.03 and the union surrogate in ≥4 of 6 participants. If gel passes and dry fails, the study switches to the gel arm before Phase 1. If both fail, the project stops.
- **Purity.** Any of §5 criteria 1–5 fails, or the replay statistic is ≤ 0, and the claim is dead.
- **Scaling.** The noise-ceiling fit is pre-registered. If the 90 h AUC is more than 0.03 below the split-half ceiling, model capacity was the limit. If zero-calibration population AUC is below the 15-minute own-data AUC, the population leg fails.

**Biggest risk.** Physiology. Dry, cortical-only, observed small graded errors may give AUC ≈ 0.58–0.62, and that makes teaching take ≥40 min per 24 bits.

**Probability and habituation.** The response is smaller when errors are common, but the reward positivity is robust at 50 % outcome probability (doors task; Proudfit 2015). Label blocks fix the base rate at 3:2:3 (positive : zero : negative r); habituation is tracked per session and spread evenly by the interleaved test set. **Oddball rule:** the event is the device's own action, not an added flash; a disclosed grey zone.

## 10. What can be tested now on ds004362

ds004362 has no evaluative events, so it cannot prove or kill the core claim. It tests scaffolding:
1. **Display-leak harness.** Decode target side (0–800 ms post-cue) from FP1/FP2/F7/F8, a cortical 32-ch subset, and occipital channels; apply §5 regress-out and ocular-map tests. The harness must *flag* this known leak, proving the criteria have power.
2. **Population ERP ladder.** Frontocentral cue-evoked single-trial detection, train n ∈ {5, 10, 20, 40, 80} subjects, fixed 29-subject test set; fixes the saturation-fit code.
3. **Teaching simulation.** AUC 0.58–0.75 into the `neural_cursor/` agent with *no timeout*: minutes to 24 bits and frozen-policy success, fixing the pilot gate numerically.
