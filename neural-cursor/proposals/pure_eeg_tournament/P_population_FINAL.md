# P_population (FINAL): a many-person central-montage EEG decoder on location-free labels, and the measured exchange rate between people and own hours

*Amended after the final against smr_modern. Changes are marked* **[F]**. *New analyses are in `scratchpad/pop_final/` (`central11.py/.json`, `noise.py/.json`).*

## 1. Idea and signal basis

**Idea.** One sensorimotor decoder is trained across ~100 people. A new person starts with **no per-person decoder fitting**, and the decoder is then co-adapted. Applied to dry EEG, this is the Kaifosh et al. 2025 recipe, with auditing added. **[F]** The measured object is now a **scaling surface**: people (N), windows per training person (k), and one person's own hours from a prior (depth arm, enlarged). The headline is the **exchange rate between these axes**, on labels that by construction carry no screen-location information.

**Signal.** Mu/beta (8–30 Hz) ERD/ERS with a 4-class body-part code: left hand, right hand, both hands, feet (Pfurtscheller et al. 2006). We use one joint multinomial, because the two-head readout is collinear (§10).

**[F] Montage: central 11 only** (FC5, FC1, FC2, FC6, C3, Cz, C4, CP5, CP1, CP2, CP6), re-referenced within those 11. P3, Pz and P4 are removed. No decoder channel overlaps the posterior-alpha attention surrogate. Cost on ds004362: 4-class zero-shot .436 vs .442 (§10).

**Instruction (conceded).** This is instructed motor imagery. "Zero calibration" means no labelled data from the new person fit the decoder, not "no instruction". Deployment burden: ~2 min instruction plus 3 min rest; the practice needed is *measured* (§6).

**Prior art.** Subject-independent SMR (Fazli et al. 2009); co-adaptation from a generic start (Vidaurre et al. 2011); re-centring (Zanini et al. 2018).

## 2. Label source: neither display- nor attention-identifiable [F]

**Conceded.** In SF1 the ring location equalled the class, inviting covert attention and goal-directed microsaccades; on ds004362 the eye proxy beats EEG on L/R (§10). Spatial goals are therefore **removed from every training label**.

**Training labels.** In every case the label is the **body part**, and the screen is blank except for a fixation dot. A gaze-contingent 1.0° window is enforced (grafted from generative): a break pauses the trial, and the trial is still kept.
1. **Cued body-part trials** (160 per person). A 250-ms luminance-matched glyph is shown at fixation. **The glyph→body-part key is redrawn every 40-trial block**, shown for 20 s beforehand, and counterbalanced across people. Windows run 1.25–4.25 s.
2. **Pre-committed body-part trials** (96). The person picks a body part on a side tablet whose layout is rotated at random on every trial. Then follow 1 s of blank screen and a 3-s window.
3. **Free-choice, report-after trials** (28): the gold test.

None of these involves a location to attend to. A residual body-centred bias (e.g., attending toward the left hand) is measured by S1/S7, not assumed away.

**Closed loop (co-adaptation and certification).** Rings are used. The screen mapping M ∈ {identity, rot90, rot180, rot270} from body parts to screen directions changes every block and is announced in advance. Training updates use {identity, rot180}; rot90 and rot270 are **held out** (grafted from smr_modern). The label is the body part that M assigns to the pre-committed ring, taken from a 2-s hidden-cursor phase. Visible-cursor segments never provide labels.

## 3. Data engine, scaling surface, expected curve

**Evidence (ds004362, central 11, 30 fixed test subjects, 76 pool subjects, fixed penalty).**

| | N=8 → 76 | gain/doubling [95% CI] |
|---|---|---|
| L/R broadband | .663 → .681 | +.0065 [.0024, .011] |
| Hands/feet broadband | .640 → .684 | +.0134 [.0075, .0195] |
| L/R eye proxy E2 | .720 → .766 | +.0141 [.0088, .0193] |
| Hands/feet eye proxy E2 | .554 → .607 | +.0165 [.008, .024] |
| EEG − E2 slope, L/R | | **−.0075 [−.0141, −.0015]** |
| EEG − E2 slope, hands/feet | | −.0031 [−.0143, +.007] |

**Reading (conceded).** On displayed-target data the eyes beat cortex on L/R in level and slope. The linear cross-person gain is ~1 point per doubling, consistent with E2's plateau at 32–48 people. ds004362 cannot show what happens once screen location leaves the label.

**[F] Why a 2-D surface, not "more of E2".** E2 saturates at **45 trials per person**. Its deep model crosses the linear plateau at 64 people × 45 ≈ 2,900 trials. Our cohort gives ~340 labelled windows per person (≈ 33k in total, about 11× that crossover volume).

We fit accuracy on a grid N ∈ {8, 16, 32, 64, 96} × k ∈ {45, 90, 180, 340}, for the linear model and the transformer (0.2M/2M/8M parameters). This separates **"saturates in people" from "saturated because each person gave 45 trials"**, which E2 cannot do.

Pre-registered statistics: gain per doubling along each axis, with a bootstrap over test participants. The a − b·N^(−c) form is descriptive only.

**[F] Power, from data.** In `noise.json`, at 45 test trials, 43–73% of the per-subject variance of the gain is trial-sampling noise. With ~340 test windows per person, the projected per-subject SD of the slope is ≈ .008–.012. With **32** test participants (up from 24), the SE is ≈ .0015–.0021.

At a dry slope of 50–100% of gel (+.003 to +.013), t ≈ 1.5–6: **the low end is underpowered (conceded)**. Pre-registered rule: if the M3 dry pilot projects power < 80%, up to 48 participants move to test before the cohort starts.

**[F] The increment is now well-conditioned.** In SF1 it was a difference of two noisy slopes, because the ds004362 eye proxy carries real display-driven signal. With location-free labels the surrogates should sit flat at floor; the increment is computed on paired per-trial log-loss (stacked minus surrogates-only) and then has roughly the EEG slope's variance. If surrogates are *not* at floor, G2 stops the ladder before it runs.

**[F] Own-hours axis (depth arm, enlarged; recency-controlled, grafted from live_in).** 4 participants × **40 h** from the N=64 prior. Volume ladder: every training set ends just before a fixed held-out test set and extends backwards; recency control: fixed 8-h windows at increasing lags; separability: a fresh decoder at n=300 every 5 h; human learning: a frozen session-1 decoder plus re-centring. Hidden probes split into **S-probes** (selection, adaptation) and **E-probes** (evaluation of record).

**Gates (amended).**
- **G1 (passed now, central 11):** EEG gain per doubling > 0 on both axes (table above).
- **G2 (prospective, M3; replaces the ds004362 eye gate, which cannot be passed on displayed-target data).** 12 dry pilots run the §6 session. Go only if all of these hold:
  - on L/R, hands/feet and 4-class separately, the upper 95% CI of the surrogate-union margin (S1 + S6 + S7) is below 25% of the EEG margin;
  - EEG is above its permutation p95;
  - the measured dry penalty leaves the projected power at ≥ 80% with ≤ 48 test participants.

  Otherwise the cohort shrinks to 40 and the depth arm doubles.

**Expected (dry, projected).**

| N | 2-class per axis | 4-class |
|---|---|---|
| 8 | .58–.63 | .33–.37 |
| 96 | .61–.67 | .36–.41 |
| Gel anchor (N=76) | .68 | .436 |

## 4. Expected performance (conceded: low)

**[F] Corrected throughput.** 4-class .36–.41 per 2-s decision gives 0.043–0.089 bits per decision, i.e. **0.02–0.045 bits/s**. The gel value (.436) is ~0.06 bits/s. This is the bottom of the established 0.03–0.3 range.

**Lag.** About 1 s, so "continuous" is nominal. A 1-s-window variant is reported.

**Assistance.** Near-chance posteriors make Σp_k·u_k small, so assistance would dominate: **unassisted performance is primary**; assisted is reported beside a shuffled-EEG surrogate through the same assistance (at chance in the repo simulation).

**[F] User learning.** The single-session cohort cannot show it; the 40-h depth arm measures it. Anchor: Wolpaw & McFarland 2004, ~0.8–0.9 bits/s for the two best trained gel users, 0.2–0.35 for the others. Pre-registered expectation for dry depth users at 40 h: 0.05–0.2 bits/s unassisted.

**Non-responders.** Blankertz et al. 2010 resting-mu predictor; results by tercile; no exclusions.

## 5. Purity proof

**Auditors** (never input or loss): 1,000-Hz eye tracker, with a chin rest **only in open-loop label blocks** (closed loop and robot use remote mode; conceded: a chin rest is not ecological); 12 EMG (forearms, masseter, tibialis anterior, gastrocnemius, trapezius, sternocleidomastoid); head and seat IMU.

**Surrogates, on the same ladder:** S1 gaze/microsaccades; S2 frontal/edge proxies E1–E3; S3 display/glyph/tablet; S4 label permutation (1,000, matched n); S5 circular shift; S6 EMG union; S7 posterior alpha lateralization (P3/Pz/P4/PO7/PO8/O1/O2, disjoint from the decoder); their union.

Positive control: S1 must decode 3 min of overt gaze to a ring at ≥ .90.

**[F] All criteria are stated on the above-chance margin, not on raw accuracy** ("auditor-bounded" wording, from vocabulary and smr_modern):
1. **Level, per axis and 4-class.** EEG is above S4's p95. The upper 95% CI of the fraction of the EEG margin explained by the surrogate union is **< 25%**.
2. **Increment (scaling claim).** The stacked-minus-surrogates paired log-loss has gain per doubling with a lower bound > 0.
3. **Clean-trial co-primary (graft).** Restrict to trials with gaze within 1°, no microsaccade ≥ 0.1° and no blink. Clean-margin / all-margin ≥ 0.8, with lower CI ≥ 0.6. The same rule applies to microsaccade-incongruent versus congruent trials. *At .40 accuracy (margin .15), incongruent trials must reach about .37, not .32.*
4. **Leak signature under held-out mappings.** On rot90/rot270 blocks, decoder output is regressed on the body label and on the label implied by the ring location. Pass: β_body ≥ 70% of the trained margin, and the upper CI of β_location < 10%. **This replaces the 45° diagonal control** (conceded: diagonals smear an attention reader into generic degradation).
5. **Frame-exact pixel replay (graft, from session 2 of the depth arm and in cohort revisits).** On 20% of trials, the visible-cursor trajectory of an earlier run toward ring A is replayed while the person has committed to B. Upper CI of β_replay < 10%.
6. **Saccade-locked output** shows no direction-specific deflection beyond S5 (Yuval-Greenberg et al. 2008).
7. **Glyph control.** A 0–1-s glyph-epoch decoder sets a floor. Because keys are redrawn, it cannot predict the body part across blocks.

**Co-adaptation veto (graft from live_in).** Adaptation weight is zero, and the cursor freezes, when EMG > 3 SD of rest, gaze leaves 1.5°, or the head turns > 2°. Vetoed trials are kept in all analyses.

**Circularity controls:** in closed loop rings are always present (no onsets); 5% sham feedback; a replay-without-intent block; S/E probe separation; technical exclusions only.

## 6. Protocol

**Participants.** 128: **96 training and 32 held-out test** (up to 48 test by the G2 rule), interleaved through recruitment. Plus 12 pilots and 4 depth participants.

**Session (88 min).**

| Min | Activity |
|---|---|
| 20 | Dry cap, eye tracker, 12 EMG, IMU |
| 5 | Rest (re-centring, non-responder predictor); artifact calibration |
| 6 | **Hint-free** ring block (24 trials, identity mapping): naturalness, scored only, never trained on |
| 2 | Instruction of the body-part code |
| 20 | 160 cued body-part trials (keys redrawn every 40) |
| 12 | 96 pre-committed body-part trials |
| 4 | 28 free-choice, report-after trials |
| 3 | Overt-gaze positive control |
| 14 | Closed loop, 4 mapping blocks: 4 min frozen zero-shot, then 10 min co-adaptive with veto (rot90/rot270 frozen) |
| 2 | Strategy questionnaire, NASA-TLX |

**Measured burden.** Test participants' zero-shot accuracy by trial quintile (practice curve), and the per-person hint-free versus instructed gap.

**[F] Closed-loop snapshots, reduced to 2.** Test participants revisit for 75 min and run N=8 versus N=96, blinded and counterbalanced, 30 min each (~180 hidden-probe windows). The projected SE of the paired difference is ≈ 1.2 points against an expected 1–3 points. **Panel (c) is secondary and descriptive (conceded).** The offline curve is the result of record.

**Dry penalty.** The M3 pilot (12 people) runs the hybrid electrodes dry and gelled in separate sessions.

**Vertex robustness.** 15% of sessions budgeted to lose ≥ 1 of Cz/FC1/FC2/CP1/CP2; channel-dropout training; hands/feet reported with and without them.

**Feasibility.** 128 sessions + 32 revisits + 12 pilots + 160 depth hours ≈ 450 rig-hours on 2 rigs (~13 sessions/week at peak). **Team: 1 postdoc, 3 RAs** (+1 RA, ~€45k).

**Timeline.** M1–2 rig and G2 tooling; M3 G2 pilot, pre-registration; M4–9 cohort; M5–12 depth arm (staggered); M10–11 revisits; M12 robot; M13–18 analysis.

## 7. Real-time architecture

**Input.** LSL 500 Hz → causal 4–40 Hz → 250 Hz, central 11 re-referenced within the 11. 2-s windows (1-s variant also reported), 100-ms hop.

**Alignment.** Riemannian re-centring to the person's rest, plus a running mean (5-min half-life).

**Models.** Primary: 4-band filter-bank tangent-space 4-class multinomial, penalty by grouped CV. Contender: transformer at 0.2M/2M/8M parameters, self-supervised pretraining on unlabelled cohort data. Public display-cued data is used only for pretraining.

**Output.** Body velocity v_body = Σ_k p_k·u_k, plus an intent-present head. Screen or robot velocity = g·M·v_body.

**Adaptation.** Zero-shot (headline): frozen decoder plus re-centring. Co-adaptive: SmoothBatch last-layer update (W ← 0.9W + 0.1W_batch) every 20 S-probe windows, with veto. Depth arm: nightly fine-tune, promoted only on S-probes.

**Effector.** Compute < 30 ms; 10 Hz output; optional goal-posterior assistance never reads the goal. One frozen decoder drives cursor or planar robot (≤ 5 cm/s). **Conceded:** a slow, near-discrete 4-direction selector; diagonals and speed come only from posterior mixing.

## 8. The Nature claim

**Figure.** (a) Scaling surface: zero-shot 4-class accuracy over N × k, linear vs transformer, with the increment and S1/S6/S7/S4 curves on the same axes. (b) Recency-controlled own-hours ladder against the prior, plus the human-learning curve. (c) Leak-signature and pixel-replay β's under held-out mappings. (d) Cursor and robot from one frozen decoder.

**Sentence.** *"On a dry EEG cap, with labels that neither the display nor gaze can identify and with eyes, muscles and spatial attention bounded by auditors, we measure what each additional person, each additional window per person, and each additional hour of one user buy, and find where each saturates."*

**Novelty (conceded: incremental in method).** New as a measurement: E2 shows one model saturating at 45 trials per person on display-confounded labels; the people × depth surface on labels the eyes cannot read is unmeasured. Realistic venue: **Nature Communications**.

## 9. Falsifiers and risks

**Falsifiers.**
1. The increment gain per doubling of N has a lower bound ≤ 0, at the G2-set power.
2. On any axis, the surrogate-union fraction of the margin is ≥ 50% (point estimate), or EEG is at or below S4's p95.
3. In leak-signature or pixel-replay tests, the upper CI of β_location or β_replay is > 10%.
4. On the own-hours ladder, the prior plus 40 own hours is not above the prior (reported as "people suffice").

**Risks.** G2 fails (body-centred leak survives → pivot to depth arm); dry slope ≈ 0 (a valid, powered "saturates" result); hint-free block at chance (expected, reported); low throughput (conceded).

## 10. ds004362 results (scripts and JSON in `scratchpad/pop_sf1/` and `pop_final/`)

**[F] Central 11** (P3/Pz/P4 removed; re-referenced within the 11 via the zero-sum subspace, so posterior channels cannot enter through the reference). See the table in §3. 4-class zero-shot at N=76: .436 broadband, .403 filter bank (14-ch: .442, .421).

**[F] Noise decomposition.** Per-subject variance of the 8→76 gain (single draw, 45 trials) splits into trial sampling and heterogeneity:

| Model, task | Trial-sampling SD | Heterogeneity SD | Sampling share of variance |
|---|---|---|---|
| L/R broadband | .060 | .041 | 68% |
| Hands/feet broadband | .061 | .070 | 43% |
| Filter bank (L/R / hands/feet) | | | 73% / 65% |

More test windows per person thus shrink the slope CI (basis of §3's power estimate).

**[F] Eye proxy, disclosed.** L/R EEG − E2 slope: −.0068 [−.0142, −.0006] (14-ch tuned), −.0075 [−.0141, −.0015] (central 11); E2 also wins in level by ~8 points. We read this as the display trap (targets drawn all trial), but that is *untestable here*; hence the prospective G2.

**Kept from SF1:** 1,000-permutation nulls (all real values above every null; 14–15/30 subjects after FDR); order surrogate .486/.517; run-confound 4-class .269; two-head collinearity.

**Pipeline note.** FINAL numbers are fixed-penalty and file-backed (tuned variants labelled). The old .599 gap is still unreconciled (conceded); no claim relies on it. ds004362 has displayed targets and no feedback, so the decisive purity tests are G2 and §5.
