# Design of record: provenance-tested SMR control on a dry cap, with one person's hours decomposed

*Chief scientist's design for the PI: the champion (smr_modern, final 5–0), with grafts from every contender and every verification and review correction applied. It governs where earlier proposals disagree.*

## 1. The claim and the headline figure

**Claim.** A 32-channel dry-cap decoder of sensorimotor mu/beta (8–30 Hz), trained only on labels neither screen nor eyes can identify, produces closed-loop velocity in body coordinates: y = feet vs both hands (primary), x = right vs left hand (earned per person, eye-confounded in public data, §3). Under screen rotations it never saw (±90°) and silent pixel replay it follows body intent, not the display, while a shadow decoder trained on the same person's on-screen (ReFIT) labels follows the screen in the same blocks. It also splits the PI's "sit, collect, retrain, adapt" into decoder volume (H1), robustness (H2), user skill (H3) and people vs own hours (H4). **The rate is not the claim.** 1-D y is the likely headline, at or below the bottom of the 0.03–0.3 bits/s band (§8): the expected outcome is *not* usable continuous control, which remains the PI's goal and is not promised.

**Why instructed imagery, honestly.** "Just intend" has no display-free label unless its goal is cued, which makes it instructed in all but name. Feet/hands imagery is the only stimulus-free cortical signal with gel precedent for continuous control and a topography auditors can check. Feet = up is arbitrary; the mapping M makes that explicit, and P2 rotates it. Instruction is a scaffold that is withdrawn; whether it becomes intention is measured (F4).

**Prior art.** Rotated or perturbed decoder mappings are established in intracortical BCI for studying adaptation (Jarosiewicz et al. 2008 PNAS; Chase, Kass & Schwartz 2012 J Neurophysiol). We know of no EEG study using held-out mappings plus a display-trained positive control as a *provenance* test; a week-1 search decides the novelty wording.

**Headline figure.**
- (a) **Provenance.** β_body and β_screen (95% CIs, pooled and per person) under rot ±90° and pixel replay, for the deployed decoder and two sighted-label shadows (central-11, all-32: the positive controls).
- (b) **Hours decomposed** for 6 deep participants (2 to 150 h): volume (random-k vs recency), robustness, user learning, each with gain per doubling. Inset: N × k surface with the public shadow at N ≥ 64.
- (c) **Naturalness.** Blind and sighted "just intend" relative to cued performance, and an intend-trained vs imagery-trained decoder on the same windows.
- (d) **Generality.** One frozen output drives a cursor and a planar arm.

**One sentence.** *"A dry-cap EEG decoder trained only on labels the display cannot identify follows body intent, not the screen, under mappings and replays it never saw, where one trained on on-screen labels follows the screen; and we measure what more hours buy."*

## 2. Why this won, honestly

Eight proposals, three judges per match (five in the final, won 5–0 over population, weighted 6.08–6.38 vs 5.30–5.93). smr_modern decodes the strongest scalp signal ds004362 offers, is the only design with a leak *positive control*, and fits 2 rigs, 1 postdoc and 2 RAs. All six load-bearing premises (C1–C6) were refuted *as stated*; they lower the numbers without sinking the purity architecture. live_in, the PI's hypothesis taken literally, lost 0–3 (a free-strategy loop on a dry cap finds jaw, eye and head levers); its recency-controlled ladder and separate selection/evaluation streams survive.

## 3. Signal basis and what the real data show

**Physiology.** Imagined hands suppress contralateral C3/C4 mu/beta; feet, the vertex. Gel precedents: Wolpaw & McFarland 2004 (2-D); McFarland 2010 (3-D); Edelman 2019 (pursuit, arm); Meng 2016 (reach-and-grasp).

**ds004362** (106 usable subjects, gel, untrained, targets drawn on screen). E1 within-subject CV (matched-n floor p95 = 0.644): imagined L/R 0.643 (38% above floor), imagined fists/feet 0.710 (65%), executed 0.832. E3: sensorimotor-11 0.629 (L/R) and 0.706 (fists/feet) vs occipital-8 0.604 and 0.661; frontal 8–30 Hz 0.507. Occipital channels decode the *displayed* target almost as well: the benchmark carries the display trap.

**Time course** (central 11, 1-s windows):

| Task | 0–1 s | 0.5–1.5 | 1–2 | 2–3 | 3–4 | 0.5–3.5 |
|---|---|---|---|---|---|---|
| L/R mean (frac > floor) | 0.605 (.37) | 0.616 (.42) | 0.588 (.40) | 0.574 (.33) | 0.537 (.19) | 0.631 (.48) |
| Feet/fists mean (frac) | 0.554 (.22) | 0.646 (.48) | 0.667 (.52) | 0.629 (.45) | 0.610 (.40) | 0.700 (.61) |

Feet/fists peaks at 1–2 s and fades (C1). Posterior with central regressed out decodes 0.529 (L/R) and 0.541 (feet/fists), so a third to a half of the central margin is shared with posterior channels.

**Eye proxy, levels not just slopes** (realdata/agent_analyses/population_final/central11.json, cross-subject, 8 → 76 training people). L/R: 1–8 Hz eye proxy (incl. VEOG/HEOG) 0.720 → 0.766, central-11 8–30 Hz 0.663 → 0.681, slope difference −.0075 [−.0141, −.0015]. Feet/fists: central-11 0.640 → 0.684 beats the eye proxy 0.554 → 0.607. **x is eye-confounded in the public data; y is not.** The earlier responder figure (top 35 keep L/R at 0.745/0.709/0.671) therefore cannot be read as sustained lateralised SMR, and is not used. x is admitted per person only if it also beats an eye-proxy decoder on the same windows (§5).

**People and own data.** E2 zero-shot L/R 0.583 (1 person) → 0.656 (32) → 0.660 (81); linear saturation at 32–48 people, EEGNet plateau by 64. Central-11 gain per doubling: L/R +.0065, feet/fists +.0134. noise.json: 43–73% of per-subject gain variance is trial sampling at 45 trials. E6: own-from-scratch 0.484 → 0.698 over 45 trials, catching the pooled prior by ~40 and **still rising** (C4).

**Provenance of these numbers.** E1–E6 are in EVIDENCE.md. The time course, posterior-regressed values, central-11 levels and noise.json live only in scratchpad files (realdata/agent_analyses/smr_sf2/res.json, realdata/agent_analyses/population_final/central11.json, realdata/agent_analyses/population_final/noise.json); week 1 commits them and their scripts to `neural-cursor/realdata/agent_analyses/`. **Unverified until reproduced:** a skeptic's display-reduced ≈ 0.62 (in no file) and the manifold pilot's +0.034.

## 4. Labels, and why none is display-identifiable

**Rule.** No label for a deployed decoder is computed from anything visible. Public display-cued data never supervise it; they serve self-supervised pretraining, offline analyses and a labelled *public shadow* (H4).

| Source | Screen and sound in the window | Label | Use |
|---|---|---|---|
| **L1** fixation probe | Fixation dot. 250-ms central glyph (key redrawn every session) or matched body-part word. Classes FEET, HANDS, LEFT, RIGHT and **REST** (20% of probes). Windows 1.0–4.0 s after cue onset. | Cued body vector; REST = (0,0) | Train; idle gate; *sel* probes (10%) select; *eval* probes (20%) evaluate |
| **L2b** blind closed loop (`blind_closed_loop`) | Cursor and goal hidden, fixation, label-symmetric progress tone, end flash after the window. Cue as L1, re-cued every 4 s, never a location. | Cued body vector | Train, supervised updates |
| **S** sighted, identity | Visible cursor and target | None (deployed) | User practice; shadow training |
| **Shadow** | — | ReFIT on S, mapped to body axes | Positive control, never deployed |
| **H** held-out (rot ±90°) / **I** identity, same block | Visible; diagonal targets | Test only | Provenance; I = margin |
| **R** pixel replay (in S) | Cursor replays toward A; no target drawn; goal B by auditory goal word | Test only | Circularity |
| **J** just intend | Blind: auditory goal word. Sighted: drawn target | J-decoder (blind only) | Naturalness |

**L2b window rule.** Windows ending 1.0–4.0 s after each cue or re-cue onset (the code's 1.0-s cue lockout); none after the hidden cursor is within the dwell radius. Decoders trained on early (1–2.5 s) vs late (2.5–4 s) windows are compared, since ERD decays.

**What remains.** Nothing spatial is drawn; keys are redrawn; glyph/word cues alternate; a per-session label audit (pixel log, audio, gaze, IMU, timing) must sit at matched-n p95. Covert attention, microsaccade bias and cue-locked signatures remain (C3) and are measured (§5).

**Gaze contingency.** No cue until gaze is within 1.0°; a fixation break aborts and repeats the trial (`fixation_break`). That is the only gaze-derived event in `events.csv`, and per-class abort rates enter the label audit; other gaze flags go to an auditor event file only `purity.py` reads, and a test asserts `BodyTrainer` reads no gaze field.

## 5. Purity proof

**Auditors are recorded, never inputs:** 1,000-Hz eye tracker, H/V EOG, bilateral EMG (forearm, tibialis, gastrocnemius, trapezius, masseter, temporalis, SCM, frontalis), head and seat IMU, photodiode, audio. `SessionStore` opens only EEG, events and cursor files; auditors reach only `purity.py` (`certify.py` sees decoder outputs and geometry). Floors are matched-n circular-shift p95; no performance-based exclusion.

**Blank-cue audit: a necessary sanity check, not evidence.** On held-out L1 eval and L2b windows, **structurally unable to fire:** DISPLAY_LEAK (display surrogate zero by construction), FEEDBACK_CIRCULAR (needs closed-loop vs blank-feedback windows) and gaze-*position* auditors (constant under fixation, skipped). **Can fire:** OCULAR_LEAK, NO_CORTICAL_EVIDENCE, and GAZE_TRACKS_LABEL once microsaccade features are auditors. The implemented rule is used as written: OCULAR_LEAK when a frontal or slow surrogate beats its floor at α/2 *and* reaches ≥ 50% of full skill; occipital is reported, not flagged. To build: `CERTIFYING_BLOCKS` extended from cued_blank to fixation_probe/blind_closed_loop; a **deployed row** (the BodyCoordinateDecoder pipeline, surrogates on frontal, occipital and slow-band subsets), since the audited full model is a 32-channel bandpower ridge; a CAR vs local-reference row.

**Primary purity family** (tests that can fail; fixed-sequence gatekeeping in this order, F1 fires only here):
1. **P2 β_screen bound** (y, pooled). 2. **P2 β_body**: absolute > 0 and ≥ 70% of the external margin. 3. **P3 β_replay bound.** 4. **P1 increment.**

**Supportive family** (Holm-corrected; a failure downgrades wording to the auditor-bounded share, not F1): clean-trial ratio (gaze < 1°, no microsaccade ≥ 0.1°, no blink: margin ≥ 0.8, lower CI ≥ 0.6); microsaccade-congruent vs incongruent; cue-epoch (0–1 s) decoder floor; glyph→word transfer ≥ 70%; label audit; localiser; EMG-silent subset; blank-cue sanity.

**P1, incremental information.** Auditor-only vs auditor + EEG, same learner, nested. Auditors include microsaccade direction and rate (gaze position is constant) and **S7**, 8–12 Hz posterior alpha lateralisation on P3/Pz/P4/PO7/PO3/PO4/PO8/Oz, never in the decoder montage. Pass: increment lower bound > 0 and ≥ 70% of the EEG margin, also after auditor-matched propensity resampling.

**P2, held-out mappings.** H trials use **diagonal targets** (±dx, ±dy, independent signs). Under rot ±90° body-y moves the cursor horizontally, so the body drives the reachable axis while the unreachable vertical offset dy, the screen regressor for the y output, varies orthogonally. Mirrors are dropped (collinear, TEST_INSENSITIVE). C2c: closed loop; C2b: 3 s with the cursor frozen, then a keyed report. Each H block opens with 6 unscored familiarisation trials; ~20 identity trials (**I**) on the same targets are interleaved.
- **External margin, always passed to `certify`:** the same frozen decoder's slope on I trials (same shift: moving eyes, visible cursor). It contains any screen component: conservative for β_body, < 1 point on β_screen at the 10% bound. Secondary: the L1 eval-probe slope. `certify`'s default (β_body + β_screen) would make the 70% criterion near-tautological.
- Pass: β_body > 0 (absolute), β_body ≥ 70% of margin with lower CI ≥ 50%, upper CI of β_screen < 10%; inconclusive zone declared.
- **Pooling:** per-person β (trial-cluster bootstrap) combined by a person-level random-effects model or person-then-trial bootstrap (to build; `certify` is trial-cluster only). Under no leak β_screen heterogeneity is ~0, so precision weighting is efficient; equal-weight and broad-only estimates are sensitivity analyses. Deep people contribute their first 33 sessions.
- **Positive controls:** central-11 ReFIT shadow (matched n, differs only in provenance) and all-32 shadow. P2 counts as certification only after shadow detectability is shown in `sim_body` and the G2 pilots; otherwise the claim is "screen-blind labels with β_screen bounded below 10%".

**P3, pixel replay.** On 15% of S trials no target is drawn, goal B is an auditory goal word, and the cursor silently replays a past trajectory toward A. A post-trial keyed report gives the held intent; windows after a reported switch are excluded and switching is modelled. Pass: β_replay upper CI < 10% of margin on trials confirming B. Passive-watching ERD (action observation, attention) is reported separately, not a falsifier.

**P4, artefact levers.** Instructed-artefact localiser (sessions 1, 2, every 10th; jaw, grind, brow, squint, eye closure, blinks, saccades, neck, swallow, 3 × 10 s): components > 2 SD projected out; output < 25% of margin. EMG/gaze-silent subset (rim index < rest p97.5, EMG RMS < rest p97.5, gaze < 1°) keeps ≥ 80% of margin; EMG reaches the vertex above ~12 Hz (Goncharova 2003).

**Secondary.** Haufe patterns; 8–20 Hz-only decoder; saccade-locked output (Yuval-Greenberg 2008).

## 6. Data plan

| Arm | n | Schedule | Dry cap h | Months |
|---|---|---|---|---|
| Tolerability (times setup) | 3 | 3 × 2 h | 18 | M1–2 |
| G2 pilots: dry vs gelled hybrid, order counterbalanced | 12 | 2 × 1.5 h | 36 | M2–3 |
| **Deep** pairs A, B, C | 6 | 33 sessions × 1.5 h, ~4/week | 300 | M4–9.5 |
| **Long** (pair A, by start order, not performance) | 2 | +67 sessions, to 150 h each | 200 | M6–10 |
| **Broad** (20 training, 12 fixed test) | 32 | 5 × 1.5 h | 240 | M4–10 |
| Gelled-hybrid checks, deep | 6 | monthly, 1 h | ~30 | M4–10 |

**Totals and calendar.** 824 cap hours; 9 + 24 + 198 + 134 + 160 + 30 = **555 sessions**, ~1,110 visit hours. Over M4–10 (~30 weeks): pair A 8/week for 25 weeks, pairs B (weeks 8–16) and C (16–24) 8/week, broad ~5.3/week, gel checks ~1/week: mean ~17, peak ~22/week against ~24 slots on 2 rigs, ~55 staff-hours/week. Robot sessions (session 20) start ~M5.5 (pair A) to ~M8 (pair C). At 3/week pair A ends by M11.5. A week-by-week rig and staff calendar is an M1 deliverable.

**Budget line.** Participants at €15–20 per visit hour (~€17–22k), dry-pin consumables, 1 postdoc + 2 RAs for 18 months; optional 64-ch gel arm (€20–40k) off the critical path.

**Session (90 min cap-on; auditors placed before the cap, ~20–25 min, timed in the tolerability pilot).**

| Min | Block |
|---|---|
| 0–8 | Cap, impedance, eye-tracker calibration, 3-min rest (re-centring reference) |
| 8–23 | L1: 120 probes (96 body + 24 rest) |
| 23–41 | L2b (~90 trials); first 20 on the frozen session-2 decoder (H3) |
| 41–51 | S, identity, 15% R trials |
| 51–63 | H/I (from session 2): 6 familiarisation, ~50 rot ±90° (C2b/C2c alternate), ~20 I |
| 63–73 | Just intend: half blind, half sighted |
| 73–76 | Idle: cursor live, instructed not to control during a listening task |
| 76–85 | Localiser (scheduled sessions) or more L2b |
| 85–90 | Strategy report, NASA-TLX |

L2b precedes S so sighted strategies do not carry into labelled windows; INTERFACES.md's "blank_feedback straight after closed_loop" rule belongs to the legacy block this design does not use. Instruction is withdrawn for S and J after session 10 (deep) or 3 (broad); L1/L2b stay cued.

**Volume, in trials.** Per deep person ~4,000 L1 + ~3,000 L2b; ~5,000 trainable after the 4 fixed test sessions and eval/sel probes; pair A ~18,000; broad ~1,050 each. Fixed tests: each deep person's last 4 core sessions plus the eval stream; the 12 broad test people.

**Regime limits.** 33 labelled min per 90, with L1 using 3 s of each 7.5-s probe: ~12–15 h of labelled EEG per deep person at 50 h (~35–45 h for pair A), comparable to one Stieger participant, not "hundreds of hours". Own-people N tops out at 26, below E2's linear saturation (32–48) and EEGNet's crossover (≥ 64). The design detects whether volume pays *early*; it cannot rule out gains at hundreds of hours or thousands of people (Kaifosh 2025, sEMG). Pair A's 150 h and the public shadow probe past that boundary.

**Expected curves (hypotheses; Stieger anchors them in week 1).** H1, Model A (~55–66 tangent features): a − b·k^(−c), gain per doubling < 0.005 r by ~1,000–2,000 trials, i.e. within the first 10–15 deep sessions. Model B: EEG foundation models (BENDR, LaBraM, CBraMod, EEGPT) report modest downstream gains, so the prior is a small advantage that may keep growing into pair A's 18,000. H4: linear saturates at 32–48 people; EEGNet keeps gaining past 64.

**Ladders** (fixed tests; Models A and B; units are trials).
- **H1.** Random-k vs last-k vs fixed 8-h windows at increasing lag; k = 45·2^j, j = 0…6 (45 … 2,880), plus full (~5,000; pair A ~18,000). EEGNet on every rung.
- **H2.** First-block r of each unseen day, re-centring only, vs training sessions (1, 2, 4, 8, 16, 29); pooled with broad sessions 2–5.
- **H3.** L1 separability at n = 120 with the decoder frozen at session 2; fresh CV decoder at n = 300 every 5 h; frozen-decoder L2b accuracy (Meng & He 2019).
- **H4.** N ∈ {2, 4, 8, 16, 20, 26} (26 = 20 broad training + 6 deep, deep truncated to the same k) × k ∈ {45, 90, 180, 360, full ≈ 1,000} trials, on the 12 fixed test people. **Public supervised shadow** at N = 64, 106 (ds004362) and 168 (+ Stieger), central-11, labelled display-confounded, scored on the same 12 clean test people; never deployed.

## 7. Decoder and real-time system

| Function | Module (status) |
|---|---|
| Causal 0.5–40 Hz filter, 50-Hz notch, amplitude gate, CAR over all good channels (FP1/FP2 gate-exempt, so ocular, saccadic and frontal EMG potentials enter central channels at ~1/n) | `signal` (implemented) |
| **Local reference**: central 11 re-referenced to their own mean, which cancels the global CAR term exactly (as in central11.json) | to build |
| **Model A**: central 11, 1-s window, per-axis 8–30 Hz OAS covariance → rest re-centring → tangent space → L2 logistic → 2p − 1 | `body_decoder.BodyCoordinateDecoder` (implemented; `fit` refuses non-L1/L2b) |
| x earned iff CV AUC ≥ 0.60 and > circular-shift p95; admitted only if purity confirms it beats the eye proxy (§3) | `earn_x` (implemented); eye-proxy check to build |
| **Model B**: EEGNet, SSL-pretrained on public and unlabelled data, fine-tuned on L1/L2b; deployed only if it passes the primary purity family alone | to build |
| Loop at 16 Hz (62.5-ms hop): `RealTimeLoop(rate_hz=16)` with the decoder's `loop_config` (config default is 10 Hz) | `realtime` (implemented) |
| Idle gate: logistic REST vs imagery from L1 REST probes; Kalman smoother; disarm after 20 s idle, re-arm after a 2-s hold | to build |
| v_screen = g·M·v_body; goal posterior never reads the goal | `mapping.Mapping`, `task.GoalPosterior` (implemented) |
| Labeler champion path (`source`, `body_label` from the cue); `BodyTrainer` (refuses non-L1/L2b); `ShadowTrainer` (S only). Legacy `Trainer` and blank_feedback unused | `data_engine` (implemented) |
| Promotion: body metric (per-axis AUC on sel probes, trial-cluster paired bootstrap); sel probes excluded from the candidate they select; current `ChampionChallenger` scores 2-D mean cosine on cued_blank | to build |
| Tests: end-to-end SessionStore → Labeler → BodyTrainer → `fit` → promotion; trainer refuses any window with `display_goal` not None; no gaze field read | to build |
| Block kinds fixation_probe, blind_closed_loop, sighted, heldout_mapping, pixel_replay | `protocol` (implemented); re-cue, R without drawn target, localiser, idle to build |
| Simulator: transient ERD, 20% x-responders, screen-leak pathway | `sim_body` (implemented); `simulate_champion` dry-penalty sweep to build |
| EMG rim index (30–90 Hz) from a raw 500-Hz tap before the 40-Hz low-pass | to build |

INTERFACES.md does not reference INTERFACES_CHAMPION.md (body_decoder, certify, mapping, sim_body), `source` provenance or champion block kinds; they are merged in M1, with the certify-window verdicts.

**Adaptation, per block.** Re-centring uses rest only (session-start 3-min rest plus L1 REST probes, class-free), updated between blocks, never within; this avoids removing class means on imbalanced stretches.

| Block | Weights | Re-centring | Supervised update |
|---|---|---|---|
| L1 (all probes) | frozen within block | frozen | nightly retrain, train probes only |
| L2b | updated every 3 min from a 24-min L2b buffer (SmoothBatch, Dangi 2013); frozen for the first 20 trials | frozen | yes |
| S, H/I, R, J, idle | frozen at block start | frozen | none |
| Fixed test sessions | frozen at session-start champion | today's 3-min rest only | none |

**EEG-only veto.** Zero update weight when `AmplitudeGate` trips or the rim index exceeds rest p97.5.

**Effectors.** Cursor, UDP, ROS2 and a simulated then real arm (≤ 10 cm/s, deep session 20, only after the idle test passes). With 1-D y the arm moves along one axis; the goal posterior picks the object; grasp = 2-s sustained y > θ within 2 cm of it; axis switch = 2-s sustained y < −θ at an end stop (θ = rest p99). Latency ~300–500 ms.

## 8. Expected performance

The dry penalty is an **unsourced planning range** (C2), measured at G2. Anchors are the sustained 2–3 s gel window.

| Quantity (dry, everyone reported) | Planning value | Basis |
|---|---|---|
| Per-1-s r on y, untrained | 0.10–0.25 | gel 0.629 → r 0.31; 20–70% loss |
| Per-1-s r on x | 0.11–0.15 | gel 2–3 s 0.574; eye-confounded |
| Fraction earning x | simulated | the old 10–35% used an n = 45 accuracy floor; `earn_x` at ~2,000 windows admits true AUC ≥ ~0.60 |
| r on y at 30–60 h | 0.15–0.35 | upper end only with learning; no dry/blind precedent (C5) |
| 1-D two-target accuracy at 10 s | 0.57–0.68 (independence bound 0.62–0.79) | integration of 1-s windows at r 0.10–0.25 with lag-1 noise autocorrelation 0.5 and ERD decay; `sim_body` replaces this |
| Time-to-target | most of the 10-s limit | same model; simulated |
| Throughput, 1-D y | ~0.003–0.03 bits/s | trial-based ITR at those accuracies; Gaussian bound 0.01–0.05 |
| Trained 2-D earners (r 0.35/axis) | ≤ 0.1–0.2 bits/s | bound; assisted reported beside a shuffled-EEG surrogate |

**Literature anchors.** Wolpaw & McFarland 2004, McFarland 2010 and Edelman 2019 report hit rates and movement times for gel-trained, sighted users; week 1 tabulates them from the papers (not from memory) as an upper reference this design does not expect to reach.

**Reading.** 1-D y plus a goal posterior is the expected headline. Expect 15–30% without usable y; none excluded; results by tercile of a rest-mu predictor (Blankertz 2010). Learning is documented only with gel and sighted feedback (Perdikis 2018; Meng & He 2019); auditory feedback is slower (Nijboer 2008).

## 9. Pre-registration: endpoints, falsifiers, power

Filed on OSF at the end of M3. H1–H3 directions are hypotheses with two-sided tests, revised beforehand if Stieger contradicts them. Each has a smallest effect of interest (SESOI): **supported** if the CI excludes 0 in the predicted direction; **falsified** only if the 90% CI lies inside ±SESOI; otherwise **inconclusive**, reported as such.

**Primary endpoints.**
1. **Provenance (y, pooled):** the primary purity family (§5), `certify` verdict with explicit margin.
2. **Control:** unassisted 1-D y accuracy on fixed test sessions above the shuffled-EEG surrogate.
3. **Hours.** H1: random-k gain per doubling, prior ≤ +0.03 r from 5 to 50 h, TOST ±0.03. H2: prior ≥ +0.05 r from 3 to 20 sessions, SESOI 0.03. H3: prior ≥ +0.05 accuracy by 30 h, SESOI 0.03. Group-level tests; per-person results descriptive. Asymptote a − b·n^(−c) with CI.
4. **H4:** gain per doubling along N and k.

**Secondary.** x-earners; blind J ≥ 70% of cued at 30 h; robot pursuit; idle false commands ≤ 1/min, tested on the idle blocks of the fixed test sessions.

**Falsifiers.**
- **F1a, screen reading:** β_screen upper CI > 10%. **F2, circularity:** β_replay upper CI ≥ 25% on trials confirming B.
- **F1b, no body signal under shift:** β_body not > 0, or lower CI < 50% of the I margin; or P1 increment ≤ 0. Distinct from F1a: the decoder may fail under covariate shift without reading the screen.
- **F3a:** random-k gain > +0.03 r falsifies our H1 prior *in the PI's favour*. **F3b/F3c:** H2/H3 CI inside ±0.03.
- **F4, not natural:** blind J < 70% of cued at 30 h. A J-decoder trained on blind-J windows (goal-word labels) is compared with the imagery decoder on the same windows, with EMG and strategy reports: if only the imagery decoder works, "natural" means habitual imagery, and is reported so.
- **F5:** < 25% earn x (expected).

**Power (planning approximations; `simulate_champion` with `sim_body` at dry penalties 0–0.7 replaces them before registration).**
- **P2.** SE(β_rel) ≈ 1/(ρ_trial·√n); 80% power for upper CI < 10% at a true 0 needs SE ≤ 0.10/2.80 = 0.036, i.e. n ≥ 784/ρ². ρ_trial ≈ r_window·√m_eff, m_eff 2–3 (transient ERD, autocorrelated windows): planning r 0.10–0.25 gives ρ 0.14–0.43. Identifiable y trials: deep 32 blocks × 50 × 6 = 9,600 + broad 4 × 50 × 32 = 6,400 = **~16,000**. Powered for ρ ≥ 0.22 (r ≥ ~0.13–0.16); at r = 0.10 power ≈ 0.4–0.6; equal weighting needs ρ ≥ 0.30. Absolute β_body > 0: > 99% per deep person, 50–99% per broad person.
- **Shadow sensitivity.** Sweep the `sim_body` screen-leak gain, calibrated to E3's occipital/central ratio; ≥ 80% power for shadow β_screen > 0, else lengthen H or downgrade (§5).
- **H1–H3** (noise.json: sampling SD ~0.06, heterogeneity 0.03–0.07; simulated before registration). H3: per-person SE of the 0→30 h change ≈ 0.038; between-person SD 0.05 gives group SE ≈ 0.026 at n = 6: power ≈ 0.5 for +0.05, 80% only for ≥ +0.073; "≥ 4 of 6" (~0.25 power per person) is descriptive. H2 is similar. H1: paired rungs give per-person SE ≈ 0.01 r; TOST power ≈ 0.9 at between-person SD 0.02, < 0.2 at 0.04. Inconclusive is a likely, pre-declared outcome.
- **H4:** if G2 projects < 80% power, the test set grows from 12 to 16.
- **G2 go/no-go** (12 pilots, M3; dry/gel order counterbalanced 6/6; both sessions include a 10-min H/I/R block scored by a within-session L1/L2b decoder): EEG above p95 on y; upper CI of the surrogate-union share (microsaccade + EMG + S7) < 25% on y and x separately; shadow β_screen detectable at the measured penalty (pilot data plus simulation); the penalty CI from 12 people leaves P2 powered. Pivot if G2 fails: broad 20, deep 8, the science arm gelled, dry secondary.

## 10. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Dry penalty 50–70%: y near floor, P2 underpowered (G2) | Gelled hybrids for the science; longer H blocks |
| x not earned or eye-driven | 1-D y + goal posterior; x-rescue pairs only through P1 |
| Shadows insensitive | Drop "certified"; keep bounded β |
| Attention, microsaccade, cue-locked or CAR-borne leak | Report the bounded share; local reference |
| Operant drift to jaw/eye levers | Project out localiser templates; EEG-only veto |
| Blind learning stalls; dry-pin attrition | Sighted practice; 3/week; 4 of 6 completers suffice |

## 11. Eighteen-month timeline

| Months | Work |
|---|---|
| M−2–0 | Ethics submitted and approved; budget costed |
| M1 | Week-1 analyses (§12); the §5/§7 "to build" items; INTERFACES merge; auditor rig; rig/staff calendar |
| M2 | Tolerability pilot (times setup); simulator power curves; G2 pilots begin |
| M3 | G2 decision; dry penalty measured; OSF filed |
| M4–10 | Broad cohort; deep pair A from M4 (long arm to M10), B from ~M6, C from ~M8; robot sessions M5.5–M9.5 |
| M7 | First pooled P2 interim (blinded, futility only) |
| M12 | Data lock |
| M13–15 | Primary analyses in order: purity → provenance → control → H1–H4 |
| M16–18 | Writing; Nature Communications submission; BIDS data and code release |

## 12. What to run first next week

1. **Stieger 2021** (figshare 10.6084/m9.figshare.13123148; the 14153504 cited earlier is unverified; 62 people, 598 sessions). H2: first-block accuracy vs prior sessions; H1: own-volume curves past 45 trials, random-k vs recency. Its vertical task is hands vs rest, not feet. Demote H2 if the per-doubling CI includes 0.
2. **ds004362 kill rules.** Train on 1–4 s, score sliding 1-s windows; add `cen_resid` to res.json. **Demote y and the certification claim** if central-with-posterior-regressed-out feet/fists on 2–4 s windows is at floor in > 70% of subjects. Remove x from the primary pre-registration if < 25% are above floor on sustained L/R, or if central-11 L/R does not beat the eye proxy within subject.
3. **Positive-control simulation.** `simulate_champion` with `sim_body` across the 0–0.7 penalty sweep, diagonal H geometry and the external margin: deployed vs both shadows through `certify`. **Demote certification** if P2 cannot separate deployed from shadow at the measured penalty; also simulate `earn_x`, hit rate and time-to-target.
4. **`nk_surface.py`** on ds004362 central 11, 25 fixed test subjects, plus the noise decomposition and the H1–H3 power simulation.
5. **Code and rig.** The §7 tests; auditor streams with no socket to the decoder; book tolerability pilots; prior-art search; draft OSF.
