# P_vocabulary (SF2): a purity-gated personal mental vocabulary that both steers continuously and selects goals from one EEG output

*Amended after the QF against smr_modern. Changes are marked* **[A]**.

## 1. Idea and signal basis

**Each person gets 4–5 comfortable mental acts, Rest included.** The acts are chosen from an 11-act library as the ones their cortex separates best once eye and muscle information is accounted for, judged on held-out data. Every 125 ms the decoder emits one posterior p(t) over the acts. That posterior drives continuous 2-D velocity, v = g·Σₖ pₖ(t)·uₖ (steer mode), and the same stream also selects goals (goal mode). [A]

**Signal.** Stimulus-free oscillatory change measured against rest: sensorimotor mu/beta ERD (motor imagery), frontal-midline theta and alpha–beta ERD (subtraction, word generation), parietal alpha (navigation, rotation) and temporal/occipital alpha (auditory, face imagery).

**Precedents.** Friedrich et al. 2013 (personal 4-class sets, 8 users, 44–84%) and Scherer et al. 2015 (brain-teaser plus imagery beat hands-vs-feet). Neither recorded auditors.

**SMR is a special case [A].** If the gate keeps only hand, foot and rest acts for a person, the decoder *is* an SMR decoder and steer mode *is* SMR continuous control. The extra claim is that some people gain a better or additional purified axis.

## 2. Label source (never display-identifiable)

Every label is **the act performed**. It is never a direction, a target or a threshold on a displayed variable.

**(1) Cue-remapped trials.** A 1-s abstract glyph is shown; the glyph→act key is redrawn each block and given by a spoken key. Then 1.5 s of blank fixation and 6 s of the act on a blank screen, with windows starting ≥1.5 s after the cue.

**(2) Free choice.** On a blank screen the person performs an act of their choosing and reports it afterwards, using randomised key positions.

**(3) Closed loop, with all outcomes kept [A; now the main source].**
- Each trial has an instructed target, spoken or shown as a glyph. The label is the act that the *current mapping* assigns to that target, whether or not the trial succeeds.
- *Display decorrelation (grafted from smr_modern).* Announced 12-min epochs alternate, 50/50, between the user's mapping M and its 180° rotation M′. M′ swaps the ±x acts and the ±y acts, so screen direction is uncorrelated with act identity.
- In goal mode nothing moves until the decision threshold is crossed, so every pre-decision window is labelled.
- Decoder output is never a label.

Yield: about 190 windows per act per day from closed loop and about 80 from cues. The QF version labelled no closed-loop time.

## 3. Data engine and scaling

**L1: library, purity-gated.** Each act must pass the §5 gate, which asks whether **EEG adds information beyond the auditors**. It does not require the auditors to be blind to the act. Ranking uses **Kelley (empirical-Bayes) shrinkage** toward the cohort atlas, with nested selection: sessions 1–2 select, session 3 tests. This removes the winner's curse of picking the maximum of noisy estimates.

In ds004148 (§10), shrunk selection gained +4.5 pp and naive selection +1.7 pp.

**L2: hours per person, the main mechanism [A].**
- Each day yields about 270 windows per act, or about 13,000 per act over 100 h.
- The last 8 sessions are a fixed test set and are never trained on.
- A ladder from 30 to 10,000 windows per act is fitted to acc(n) = a − b·n^(−c), and we report the asymptote *a* with a bootstrap CI.
- Volume vs recency (as in smr_modern): train on the last k, a random k, and all sessions.
- We pre-register an *expected* plateau at 300–1,000 windows per act (as in repo E2 and RLNF). This tests the PI's hypothesis rather than assuming it.

**L2b: vocabulary evolution, secondary and gated [A].**
- Challengers are recorded in the cue block only, so incumbents keep their closed-loop data.
- A challenger replaces the weakest act only if the one-sided 95% lower bound of the bits/decision difference is >0, measured on a fresh, never-reused day with ≥120 trials per act.
- At most one swap per 2 weeks (about 8 challengers in all). Retired acts keep their data, and the cost of a swap is measured as the change in the incumbent's L2 slope.

**L3: population priors [A].** (a) Rest plus a 15-min mini-screen predicts a newcomer's top acts (Blankertz 2010: r = 0.53); dropped if r < 0.2. (b) Grafted: a rest-re-centred pooled decoder per act. In repo E2b/E6 it was usable from trial 1, and own data caught up by about 40 trials. Pooled vs own is a pre-registered contest.

**L4: goal prior.** Engineering only. Primary tests use uniform goals.

## 4. Expected performance

**Steer mode, now simulated [A].**
- Setup: the repo's `TrialRunner`, unmodified (8 targets, 15-s timeout, 0.8-s dwell, D ∈ {0.5, 0.8}, W ∈ {0.08, 0.05}).
- Decoder model: 5-class posteriors (4 directions + Rest) from 1-s windows every 100 ms. AR(1) noise, ρ = 0.9 per 100 ms. 0.5-s ERD onset lag.
- User model: a simulated user who anticipates the lag and holds Rest at the target.
- 240 trials per cell. Ranges span gain 0.3–0.5.

| 5-class acc, 2 s (chance 0.20) | Unassisted | Assisted (0.5) | Median TTT, assisted |
|---|---|---|---|
| 0.20 (no information) | 0.00 | 0.09–0.11 (floor) | — |
| 0.50 | 0.07–0.15 | 0.39–0.40 | 5.8–7.3 s |
| 0.60 | 0.20–0.26 | 0.50–0.55 | 5.5–6.0 s |
| 0.70 | 0.30–0.32 | 0.59–0.63 | 5.0–5.6 s |
| 0.80 | 0.33–0.53 | 0.71–0.77 | 5.1–5.8 s |

**For the median user (0.50–0.60), continuous control is weak unassisted (7–26%)** and below the repo's r≈0.4 reference of 0.49. **Assisted, it reaches 39–55% at about 6 s.**

Lag is the binding cost: at full gain the cursor oscillated, so low gain and a Rest class are required. smr_modern's derived r (0.21–0.39) faces the same limit.

| Quantity | Median | Top quartile | Anchor |
|---|---|---|---|
| 4-class audited accuracy | 45–60% | 65–75% | Friedrich 2013 (gel, no auditors); dry and purification losses measured in Phase A |
| Steer, assisted / unassisted | 40–55% / 7–26% | 60–75% / 30–50% | simulation |
| Goal mode, 1 of 8, idle-safe | 40–90 s | 15–30 s | 3 bits ÷ 0.03–0.08 bits/s |
| Arm: planar steer + dwell grasp | ≥70% completion, 45–120 s per object | | Millán 2004; Leeb 2015; Meng 2016 |

## 5. Purity proof

**Auditors** (never connected to the decoder): a 250-Hz eye tracker with pupil, 4-electrode EOG, six EMG sites (temporalis, masseter, frontalis, submental/laryngeal, splenius, forearm), an accelerometer and a throat microphone. In the deep arm, laryngeal EMG and the microphone are used weekly only, so daily setup is about 20 min.

**Threats:** EMG above 20 Hz (Whitham 2007) and below it at the rim (Goncharova 2003); gaze tracking number generation (Loetscher 2010); imagery scanpaths (Brandt & Stark 1997); blink and pupil changes under load (Holland & Tarlow 1972).

**Conceded [A]: the eyes do change with these acts, so an auditor-only surrogate *will* decode them**, and ds004148 confirms it (§10). The QF gate ("surrogate ≤ permutation p95") rejected acts for being eye-*correlated*, not eye-*read*. It is replaced. All tests use matched n on fixed held-out sessions.

1. **Incremental information (gate).** Fit auditor-only and auditor+EEG models with the same learner (the better of gradient boosting and logistic regression). Pass: the increment's 95% lower bound is >0, and the increment is ≥70% of the EEG-only margin. Surrogate accuracy is reported, not disqualifying.
2. **Auditor-matched resampling (gate).** Propensity-match trials on blink count, pupil, gaze dispersion, saccade rate and EMG RMS until an auditor classifier is at chance. EEG keeps ≥70% of its margin.
3. **Silent subset.** EMG inside the rest band, gaze within 1.5°, no blink: EEG keeps ≥70% of its margin.
4. **Montage ladder [A].** *Audited* (operational): 4–30 Hz, 30 channels without FP1/FP2. *Strict*: 4–20 Hz, 26 non-rim. *Full-band*: contamination index. F7/F8/T7/T8 and 20–30 Hz stay in for an act only where tests 1–3 pass.
5. **Operational mimic [A].** *Gaze replay*: follow a dot replaying the act's recorded scanpaths, with cued blinks, without doing the act; the output must be Rest. *Muscle mimic*: EMG-feedback-matched contractions. Covert subvocal EMG is covered by tests 1–3 and the gradient.
6. **Gradient.** Overt → mouthed → covert for word association, subtraction, **song (humming)** and any verbal act [A]. Fail if EEG decodability tracks EMG decodability.
7. **Fixation and eyes-closed blocks:** EEG keeps ≥70% of its margin. **Glyph transfer:** train on key π₁, test on π₂.
8. **Feedback decoupling:** windows are frozen or replayed (15%), and the posterior must be unchanged. **M/M′ crossover:** β_act ≫ 0, β_screen ≈ 0 [A].
9. **No-EEG controller:** the pipeline is driven by phase-randomised, surrogate and label-permuted streams; brain-attributable success = success − max(surrogates); 10% blinded shams. **Replay:** output is Rest at the plain-rest rate. **Bookkeeping:** permutation and circular-shift floors, no performance-based exclusion.

**Claim: "auditor-bounded".** We report the upper 95% CI of the fraction of bits attributable to the recorded sources.

## 6. Protocol

**Library (11 acts):** serial-7 subtraction, word association, mental rotation, navigation, song imagery, face imagery, left hand, right hand, feet, one self-proposed act, and rest. Acts rated below 4/7 for comfort are ineligible. Users pick the mappings, with congruent pairs preferred (hands → ±x, feet → +y) [A].

**Phase A (months 1–6): 32 participants × 3 sessions × 2 h.**
- Sessions 1–2: 11 acts × 30 trials.
- Session 3: the top 6 candidates plus the canonical set (both hands, feet, rest), 45 trials each.
- 6 participants also do a same-day gel session.
- **Power [A].** Per-person SE is 0.05 and between-person SD ≤0.08, giving a cohort SE of about 0.017. Power to detect +4 pp (the ds004148 estimate) is about 0.78. The QF's n = 24 was too small.

**Phase B (months 3–14): 4 participants × about 100 h.**

A 2-h day (minutes, with active-act minutes in brackets): setup and rest 20 (0); cue trials 20 (12); closed loop with M/M′ 36 (28); distractor idle 10 (0); free choice, replay and mimic 14 (6); breaks 20.

- **Load cap, corrected [A].** The QF's "≤30 min" contradicted its own schedule. The cap is now ≤46 active minutes per day, in bouts of ≤8 min, with NASA-TLX and Karolinska Sleepiness Scale ratings every session.
- An act whose comfort stays below 4/7 for 2 weeks is retired.
- Fridays are frozen-decoder tests.
- **"Just intend" (grafted) [A].** From week 6, 10 min per week where only the command is named, not the act. We report retention relative to cued performance.

**Phase C (months 13–16).** 8 newcomers: L3-guided screening, then 4 sessions, the last a fixed test.

## 7. Real-time architecture

- **Signal path:** LSL at 500 Hz → causal filter bank (4–8, 8–13, 13–20, 20–30 Hz) on each act's admitted montage → covariances on 1-s (steer) or 2-s (goal) windows every 125 ms → tangent space re-centred on rest (Zanini 2018), then continuously (λ = 0.995) → multinomial logistic regression initialised from the pooled prior. EEGNet is used only if it wins on held-out days.
- **Steer:** v = g·Σₖ pₖuₖ (u_Rest = 0, g = 0.3–0.5), Kalman-smoothed; assistance a ∈ {0, 0.5}, with a = 0 always measured. **Goal:** leaky log-odds accumulation (τ = 3 s).
- **Idle safety [A]:** the system disarms after 20 s without a command; re-arming needs the most idle-distinct act held for 2 s. The armed threshold allows ≤1 false command per 5 min of distractor idle. The cost is counted at 5–9 s per decision.
- **Adaptation:** nightly refits on §2 labels only; test blocks frozen.
- **Effectors:** cursor (velocity + dwell); arm (planar ≤10 cm/s, grasp by a 1.5-s Rest-dwell within 3 cm, as in smr_modern, or goals via MoveIt); mobile base (steer). One device-blind output: 2 continuous DoF, a Rest-dwell event and discrete goals.

## 8. The Nature claim

**Figure.** (a) 32-person × 10-act atlas before and after the gate; (b) 100-h curves of bits per decision against hours and swaps on fixed test days, with surrogate, mimic and no-EEG controls at floor; (c) β_act vs β_screen, glyph transfer, decoupling; (d) one stream steering a cursor and an arm and selecting goals, above the no-EEG controller.

**Sentence.** *"Scalp EEG alone, without stimulus, movement or gaze, gives each person a self-discovered vocabulary of mental acts whose information survives eye and muscle auditing, improves with hours of use, and steers and selects on any device."*

**Novelty.** The components have precedents: Millán 2004, Friedrich 2012/2013, Scherer 2015, Leeb 2015, Royer 2011. What is new:
1. the first auditor-bounded test of mental-task BCIs, with a gate that separates "the eyes also change" from "the decoder reads the eyes";
2. shrinkage-based, purity-gated selection at cohort scale;
3. a 100-h engine whose closed-loop labels are not display-identifiable.

A refutation would be publishable, but it would not be control by thinking. **Venue: Nature Communications.** It becomes a Nature paper only if (a) shows idiosyncratic non-motor acts that survive the gate and (b) keeps rising.

## 9. Falsifiers and risks

- **F1:** the 95% CI of the personal-minus-canonical gain excludes +3 pp, or its lower bound is ≤0.
- **F2:** ≥50% of cognitive acts fail gate tests 1–2.
- **F3:** glyph transfer, the M/M′ crossover or decoupling fails.
- **F4:** the L2/L2b slope CI includes 0.
- **F5 [A]:** in ≥3 of 4 deep participants, assisted steer is not above the no-EEG controller, or unassisted steer is not above its floor.

**Biggest risk: F2.** If only motor acts and rest survive, the result is an auditor-bounded refutation plus an SMR steer-and-goal BCI, i.e. smr_modern's signal plus the atlas.

**Naturalness** is a fair 5/10: the acts are instructed and often unrelated to their commands. **Idle:** spontaneous thought uses the same operations; arming bounds false commands at the cost counted in §4.

## 10. Tested now [A]

**ds004362.** It has no cognitive acts (conceded), so it supports only the SMR case and the pooled prior (E2, E2b, E6).

**OpenNeuro ds004148 (run for this rebuttal).** The dataset has 60 participants, 3 sessions, 64-ch gel EEG at 500 Hz, and one 5-min block per condition: eyes-open rest, serial subtraction, song imagery and memory recall. Pipeline: the g.tec 32-ch subset, CAR, 2-s epochs, filter-bank tangent space re-centred per session, and logistic regression trained on sessions 1+2 and tested on session 3, for all 60 subjects. *Pure* means 4–20 Hz on the 26 non-rim channels. *Proxy* means EEG-derived eye and muscle stand-ins: frontopolar 1–13 Hz power plus 30–45 Hz power on all channels. The dataset has no real auditors.

| Contrast (median) | Pure | Proxy | Proxy + pure | Subjects >0.55 / <0.45 (pure) |
|---|---|---|---|---|
| rest vs math / memory / music | 0.98 / 0.96 / 0.96 | **0.92 / 0.91 / 0.90** | 0.98 / 0.97 / 0.96 | ≥0.95 / ≤0.02 |
| math vs music | **0.705** | 0.640 | 0.741 | 0.85 / 0.08 |
| math vs memory | 0.631 | 0.560 | 0.634 | 0.80 / 0.08 |
| music vs memory | 0.541 | 0.536 | 0.548 | 0.47 / 0.27 (≈ null) |
| 3-class cognitive (mean; chance 0.33) | 0.463 | 0.417 | 0.466 | |

**Findings.**
1. **The opponent's point 2 holds.** Occipital alpha is 3.3–5.0× higher during the tasks than during eyes-open rest, which suggests the tasks were done with eyes closed. The proxies decode task vs rest at 0.90–0.92, so a "surrogate at chance" gate would reject everything.
2. **Some cognitive contrasts carry information beyond the proxies.** Math vs music scores 0.705 pure, 0.640 proxy and 0.741 combined. Music vs memory carries none. This is the incremental gate's logic on real data, not a purity proof.
3. **Personalisation is small and needs shrinkage.** Each person's best cognitive pair was chosen on session 1→2 and tested on session 3 against the leave-one-out cohort-best pair. Naive selection gave +1.7 pp (CI −1.9 to +5.3). Kelley shrinkage, using reliability r from −0.10 to 0.51 estimated on the other subjects, gave **+4.5 pp (CI +2.1 to +7.1)**, and 50% of people switched pair. The data support a modest F1 effect, not large idiosyncratic gains.

**Caveats:** one block per condition confounds class with drift (cross-session testing mitigates this only partly); gel data; eye state may differ for music; proxies are not auditors. Code is in `scratchpad/ds4148_code/` and results in `scratchpad/ds4148_out/res_batch*.json`.

**Week-4 pilot kill rules (n = 6).**
- Drop the cognitive library if fewer than one cognitive act per person passes the incremental gate.
- Drop steer mode if the measured confusion matrix, run through the §4 simulator, gives assisted success ≤ floor + 0.15.
