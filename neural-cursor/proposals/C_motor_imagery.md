# Proposal C — Sensorimotor-rhythm (SMR / motor-imagery) continuous 2-D cursor control

## 1. Paradigm and signal basis

**Signal.** Mu (8–13 Hz) and beta (16–26 Hz) rhythms over sensorimotor cortex (C3/C4/Cz) desynchronise (ERD) when a limb is moved *or imagined moving* and resynchronise at rest. Left-hand imagery suppresses mu over C4, right-hand over C3, feet over Cz (Pfurtscheller & Lopes da Silva 1999). A 2-D controller maps a lateralised difference (C4 − C3) to horizontal velocity and a pooled/feet term (C3 + C4 or Cz) to vertical velocity (Wolpaw & McFarland 2004). The signal is endogenous, continuous, available every 50 ms, and needs no stimulus on the screen. It is the only EEG paradigm with a 25-year record of *continuous* 2-D and 3-D cursor control.

**Why it is not "natural" at first.** Reliable control starts with *instructed* kinaesthetic imagery (squeeze right hand, curl toes). This violates the user's stated preference, and I say so plainly.

**Why it can become natural.** The Wolpaw lineage treats SMR control as operant conditioning: feedback shapes the rhythm, and users consistently report that the imagery "drops away" after weeks and the cursor feels like a moved limb (Wolpaw & McFarland 2004; McFarland, Sarnacki & Wolpaw 2010; Kübler et al. 2005; Perdikis & Millán 2020, "a tale of two learners"). Imagery is the *bootstrap*, not the steady state. The honest caveats: (i) the transition is self-reported, not measured; (ii) it takes tens of sessions; (iii) 15–30 % of people never gain SMR control ("BCI inefficiency", Blankertz et al. 2010, n=80: ≈20 % below 70 % in session 1; Allison & Neuper 2010; Vidaurre & Blankertz 2010). This proposal keeps the imagery instruction visible for the first ~5 sessions, then explicitly withdraws it ("just move the cursor"), and measures whether control survives withdrawal — turning the user's objection into a pre-registered endpoint.

## 2. Expected performance on this hardware

Literature anchors (all wet gel, 64 ch unless noted):

| Study | Setup | Metric | Value |
|---|---|---|---|
| Wolpaw & McFarland 2004 PNAS | 4 users (2 SCI), 2-D 8-target, 64 ch, months of prior 1-D training | hit rate; movement time | ≈70–92 % across users; ≈1.9 s mean movement time best user |
| McFarland, Sarnacki & Wolpaw 2010 JNE | 2 users, 3-D 8-target | hit rate | ≈70–85 %; TTT 2–4 s |
| Edelman et al. 2019 Sci Robotics | 64 ch, source imaging, continuous pursuit | tracking improvement ESI vs sensor | ≈ +60 % in continuous-pursuit metric; ≈ +30 % learning rate |
| Stieger, Engel & He 2021 Sci Data (n=62, 7–11 sessions, 64 ch) | 1-D L/R; 1-D U/D; 2-D | hit rate (chance ≈ 50 % 1-D, ≈ 25 % 2-D) | 1-D LR ≈ 70–80 % by session 7; 2-D ≈ 55–65 % by session 7, still rising |
| Vidaurre & Blankertz 2010; Vidaurre et al. 2011 Neural Comput | co-adaptive calibration, 1-D binary, one session | accuracy of cat-III ("inefficient") users | most rose from chance to ≈ 60–75 % within the session; not all |
| Felton et al. 2009 JNE | Fitts-law SMR cursor, 1-D/2-D | throughput | ≈ 0.2–0.6 bit/s |
| Perdikis et al. 2018 PLoS Biol (Cybathlon) | 2 tetraplegic users, ≈ 1 yr mutual learning | command accuracy | ≈ 80–90 %, stable for months |

Per-user ceilings matter: Wolpaw's 2004 users had months of prior training; Stieger's ordinary volunteers after 7 sessions are the honest population figure.

**Hardware penalties (32 dry g.SAHARA-class, 500 Hz).**
- Channel count: only C3/Cz/C4 with FC1/FC2/FC5/FC6/CP1/CP2/CP5/CP6 neighbours in the g.Nautilus-32 layout → a small Laplacian is possible but with ≈ 4 instead of 8 neighbours; expect ≈ 0.1–0.2 lower r² for mu features than 64-ch large Laplacian (McFarland et al. 1997 spatial-filter comparison).
- Dry electrodes: 2–3× higher broadband noise at 8–30 Hz and more motion/impedance drift (Guger et al. 2012 showed dry g.SAHARA P300 ≈ 5 % lower than gel; Pinegger et al. 2016 ranked dry systems lowest among four). SMR is amplitude-modulation of a ≈ 2–10 µV rhythm; a 3-dB SNR loss costs roughly 5–10 hit-rate points and 10–25 % longer time-to-target in my simulation of Wolpaw-type normalised control (assumption, stated as such).
- No EOG: no feature penalty (Laplacian + 8–30 Hz removes most ocular power); a validity cost handled in §4/§6.

**Realistic range after the full protocol (≈ 15 sessions):**
- 1-D (4 targets): 65–80 % hits for ≈ 75 % of users; 1.5–3 s TTT.
- 2-D (8 targets, chance ≈ 12.5–25 % depending on timeout): **45–65 % hits, 3–6 s TTT, Fitts throughput 0.15–0.4 bit/s**; best quartile 70–80 %, 2–3 s, ≈ 0.5 bit/s. Worst quartile: no 2-D control.
- Versus a trackpad (≈ 4–6 bit/s) this is 10–20× slower: a usable, continuous, stimulus-free cursor, not a mouse.

## 3. Scaling behaviour (bitter lesson)

**Data engine.** Every centre-out trial yields a free, display-independent label: the *randomly drawn target*. The user generates ≈ 60 min of labelled, closed-loop data per session with no annotation. Unlike RLNF the label is not a threshold on a displayed variable; it is the sampled target identity (and, for the regression target, the target direction vector).

**Three learners, three time constants.**
1. *User (operant learning):* the dominant effect in the literature. Stieger 2021: group hit rate rises monotonically over 7–11 sessions; Perdikis 2018: still rising at month 10. This is the term that scales with *hours*.
2. *Within-subject decoder adaptation (co-adaptation):* adaptive mean/gain normalisation (Wolpaw/McFarland LMS, per trial) plus unsupervised adaptive LDA / Riemannian re-centring (Vidaurre et al. 2011; Zanini et al. 2018). Buys 5–15 points in sessions 1–3 (Vidaurre & Blankertz 2010), then saturates: the within-subject feature space is low-dimensional (≈ 3 independent SMR degrees of freedom).
3. *Cross-subject pretraining:* EEGNet (Lawhern 2018), FBCSP (Ang 2012), Riemannian transfer with re-centring (Zanini 2018; Rodrigues 2019), and self-supervised foundation models (BENDR, Kostas 2021; LaBraM, Jiang 2024; EEGPT, Wang 2024). Honest state of play: on BCI-IV-2a cross-subject MI, foundation models reach ≈ 55–65 % four-class versus ≈ 60–75 % within-subject with FBCSP. Pretraining mainly saves *calibration* (minutes), not ceiling. Stieger's 62-subject × 10-session corpus (≈ 600 session-hours) is the right pretraining set because it is the same task.

**Expected scaling curve.** Hit rate ≈ a − b·exp(−hours/τ_user) with τ_user ≈ 6–10 h (Stieger; Perdikis), plus a decoder term saturating within ≈ 1 h. Adding subjects shifts session-1 performance up by ≈ 5–10 points (fast start) but does not change the asymptote a.

**Saturation and why.** (i) SNR physics: SMR modulation depth is bounded (≈ 30–60 % ERD at best), dry electrodes raise the noise floor; (ii) only ≈ 2–3 independently controllable SMR sources exist non-invasively, so the information rate ceiling is ≈ 0.5–1 bit/s (Edelman 2019 with source imaging and 64 ch is the best ≈ 2-D continuous result and still well under 1 bit/s); (iii) non-stationarity (impedance drift, fatigue, arousal) puts a floor on decoder error that no architecture removes — RLNF's "nothing improved the decoder" is exactly what the MI literature finds (6 deep architectures ≈ FBCSP ≈ Riemannian; Lotte et al. 2018 review). More data turns an unreliable 2-D controller into a reliable one, not into a mouse.

## 4. Data-collection protocol

**Participants.** 24 healthy adults, no exclusion on performance (pre-registered). Stage A (screening, 3 sessions, 1-D) → all 24 continue to Stage B regardless of Stage A score (we *want* the inefficiency rate on dry hardware). Stage B: 12 further sessions (2-D), 2–3/week.

**Session (≈ 60 min).** Cap on, impedance log (10 min). Rest block: 2 min eyes-open, 1 min eyes-closed (mu/alpha baseline; alpha-peak predictor, Blankertz 2010). Then 6 runs × 24 trials (≈ 40 min).

**Trial (Wolpaw-type centre-out).** 1 s target shown, cursor at centre; 1 s pre-movement; ≤ 8 s movement with continuous velocity control; hit = cursor inside target ≥ 0.2 s; miss on timeout or wrong target (2-D: 8 targets; 1-D: 4). 1 s reward display; 1 s inter-trial. Target order: block-randomised Latin square per run (each target 3×), so the generator is *not* determined by previous trials.

**Instruction schedule (naturalness endpoint).** Sessions 1–5: explicit kinaesthetic imagery (right hand → right, left hand → left, both hands → up, feet/relax → down). Sessions 6–15: instruction reads only "move the cursor to the target". Self-report of imagery use each run (0–10).

**Ocular identifiability by design.**
- 25 % of trials per run are *memory* trials: target shown 1 s then hidden; the user must steer to its remembered location. Eye-movement surrogates that use the visible target fail here.
- 12.5 % of trials are *fixation-pursuit* trials: a fixation dot sits at the cursor; the target is shown peripherally; gaze instructed to stay on the cursor.
- Frontal EOG proxies: FP1/FP2 (vertical), F7/F8 (horizontal) 0.3–4 Hz are logged and modelled, never discarded.
- Recommended add-on (not assumed): a cheap webcam gaze tracker on LSL; gaze then becomes a covariate and a surrogate-decoder input.

**LSL streams and markers.** EEG (32 ch, 500 Hz, float32, with accelerometer if the amplifier provides it); `Markers`: trial_start, target_id (1–8), target_xy, trial_type (visible/memory/fixation), movement_start, hit/miss/timeout, cursor_xy at 20 Hz, feature vector + decoder output at 20 Hz, normaliser state (mean, gain) at each update, instruction_mode, impedance snapshot, run_id, session_id. Everything replays offline from the streams alone.

**Gating.** Per-channel RMS > 5× session median (FP1/FP2 exempt) → channel masked for that trial and logged; trials with > 25 % masked channels or EMG index (55–95 Hz power at T7/T8/FC5/FC6 > 3× median) flagged, never deleted, and analysed with and without.

**Controls and surrogates (pre-registered).**
1. *Non-neural surrogate decoder:* same pipeline trained on FP1/FP2/F7/F8 0.3–4 Hz + 55–95 Hz EMG index + accelerometer only. Must not exceed the permutation floor on memory and fixation trials.
2. *Shuffled-target permutation floor:* 1,000 label permutations per subject-session at matched n.
3. *Random-walk / zero-velocity cursor* with the same dynamics: empirical chance hit rate for each target layout and timeout.
4. *Replay control:* session-k participant watches their session-(k−1) cursor replay without control (sham feedback), 1 run per 3 sessions; decoder output during sham must be uncorrelated with the sham cursor's target.

**Pre-registration list.** Endpoints (§6), surrogates, floors, gating thresholds, the instruction-withdrawal test, the no-exclusion rule, analysis code hash, fixed test sessions (13–15) and 4 held-out participants.

**Power.** Primary: 2-D hit rate session 13–15 vs sessions 6–8, paired. Stieger 2021 gives between-session SD ≈ 15 points and ≈ 10-point gain over 5 sessions; d ≈ 0.67, α = 0.05 two-sided, power 0.80 → n = 20; 24 recruited for 15 % dropout. Secondary: naturalness non-inferiority (session 6 vs 5, margin 5 points) is under-powered at n = 24 and is reported with CI, not as a test.

## 5. Decoder and system

**Preprocessing (deterministic, pinned seeds).** 500 Hz → notch 50 Hz → 4th-order Butterworth 1–45 Hz → common-average reference over good channels → small Laplacian at C3, Cz, C4 using the available 4 neighbours. No ICA online (RLNF: no gain, latency cost).

**Features.** Every 50 ms: 16th-order Burg AR spectrum over the preceding 400 ms (Wolpaw/McFarland standard) → log power in 2-Hz bins 8–30 Hz at 3 Laplacian channels (36 features). Plus a Riemannian branch: 8–30 Hz covariance over 11 central channels (FC/C/CP), tangent-space at the session re-centred reference (Zanini 2018).

**Controller (stage 1, sessions 1–5): Wolpaw linear.** v_x = g_x·(w_x·f − m_x), v_y = g_y·(w_y·f − m_y). Weights w: ridge regression (λ = 1.0) of target direction on features, refit after every run; m, g: per-feature running mean/variance over the last 3 runs (adaptive normaliser; the co-adaptation rule) with the constraint that mean velocity over a run is zero (removes offset bias — the Wolpaw "intercept" fix). Cursor: velocity control, max speed 0.5 screen-width/s, 2-tap exponential smoothing (τ = 150 ms), no assistance. Gain g set so that mean |v| ≈ 0.3 screen/s in session 1 and then frozen per session.

**Controller (stage 2, sessions 6+): learned regressor.** Target-direction regression from AR + tangent-space features with a 2-layer MLP (128 units, dropout 0.25, Adam 1e-3, 30 epochs, early stopping on session k−1) warm-started from a cross-subject EEGNet-8,2 (Lawhern 2018 config, 8 temporal / 2 spatial filters) trained on Stieger 2021 (64→ our 32 channel subset) and fine-tuned per subject. The linear and learned controllers are compared in a pre-registered within-subject crossover (runs alternate) — if the MLP is TOST-equivalent to linear, we ship linear (the RLNF lesson).

**Validation splits.** Train: sessions ≤ k−1; validate: session k run 1; test: sessions 13–15 fixed for every model, never re-opened; 4 participants held out entirely for cross-subject claims.

**Latency budget.** Amplifier/LSL ≈ 20 ms; 400-ms window (effective lag ≈ 200 ms); feature + decoder < 5 ms (NumPy/pyRiemann); render 16 ms; total intent-to-cursor ≈ 240 ms at 20 Hz update — within the 300 ms the Wolpaw loop tolerates.

**Stack.** LSL + LabRecorder, a ≈ 400-line Python loop (MNE, pyRiemann, PyTorch, Pygame), DVC-pinned data, Docker image, config hash in every marker stream.

## 6. Evaluation protocol

**Co-primary endpoints.** (a) 2-D hit rate in sessions 13–15 vs sessions 6–8 (paired t / Wilcoxon, 95 % CI); (b) Fitts throughput (Shannon ID = log2(D/W + 1); Soukoreff & MacKenzie 2004) and time-to-target in sessions 13–15 against the random-walk floor.

**Secondary.** 1-D hit rate (Stage A); instruction-withdrawal test (session 6 vs 5 hit rate, CI); fraction of users above the 2-D floor + 20 points ("efficiency rate"); decoder r² of target direction on held-out sessions; cross-subject EEGNet accuracy on 4 held-out participants; memory-trial vs visible-trial performance ratio (ocular leak estimate); imagery self-report trajectory.

**Floors.** Shuffled-target permutation at matched n (1,000 draws) per subject-session; random-walk cursor empirical chance; non-neural surrogate (§4) must sit inside its own permutation band on memory + fixation trials — if it does not, the ocular/EMG channel carries intent information and the result is reported as "not attributable to cortex".

**Statistics.** Mixed model hit ~ session × trial_type + (session | subject); effect sizes with bootstrap CIs; TOST (margin ± 3 points) for decoder comparisons; no exclusion; held-out sessions fixed before unblinding; analysis code frozen at pre-registration.

## 7. Failure modes and debugging

1. **Inefficiency (≈ 20–30 % on dry hardware).** Predictor: resting mu peak amplitude (Blankertz 2010). Mitigation: co-adaptive calibration (Vidaurre 2011), overt-movement warm-up, source-space features (Edelman 2019). Report the rate; do not exclude.
2. **Dry-electrode drift.** Impedance rises over 60 min; mu amplitude appears to fall. Debug: per-run normaliser re-estimate; log impedance against feature mean; abort run if > 30 % channels masked.
3. **Vertical axis is weak** (C3 + C4 pooled term is noisier than lateral difference). Debug: switch vertical to a Cz-beta/feet term; report per-axis r².
4. **Ocular leakage** (eyes track the target → frontal potentials correlate with target). Detect via memory-trial drop and surrogate decoder; mitigate with the Laplacian/8–30 Hz band and, if still present, the gaze tracker.
5. **Imagery-withdrawal collapse** (session 6 drop). This is a legitimate finding: it means "natural" control had not emerged by 5 sessions; extend the imagery phase and re-test at session 10.
6. **Decoder beats nothing** (MLP ≡ linear). Expected; ship linear; invest in practice hours.
7. **Fatigue/arousal drift.** Track eyes-closed alpha per session as a covariate.

## 8. Honest self-score

1. Naturalness [25]: **4/10** — no stimulus, but instructed imagery is required for ≈ 5 sessions and the "fades to natural" claim is self-reported and tested here, not established.
2. Achievable performance [20]: **6/10** — the only paradigm with published continuous 2-D/3-D EEG control; on 32 dry channels I expect 45–65 % 8-target hits and 0.15–0.4 bit/s, not the 92 % headline.
3. Scaling with data [20]: **6/10** — scales genuinely with user hours (τ ≈ 6–10 h) and modestly with subjects via pretraining; saturates near 0.5–1 bit/s for physical reasons.
4. Validity [15]: **7/10** — target labels are display-independent, memory/fixation trials and a pre-registered non-neural surrogate make eye vs cortex identifiable without an EOG pair.
5. Feasibility & burden [10]: **5/10** — 240 ms latency and a tiny software stack are easy; 15 sessions × 1 h per user is the real cost, and 20–30 % of users get nothing.
6. Evaluation rigour [10]: **8/10** — fixed test sessions, held-out subjects, permutation floors, no exclusion, TOST decoder comparisons, instruction-withdrawal as an endpoint.

## 9. Pitch

SMR control is the only non-invasive paradigm that has actually put a cursor under continuous 2-D and 3-D volitional control, with no flicker, no oddball, and no proposal loop whose payoff structure can be gamed. Its cost is honest and front-loaded: five sessions of instructed hand/foot imagery, after which we withdraw the instruction and measure — rather than assume — whether control has become the "just move it" experience the user wants. Every trial produces a display-independent label from a randomised target, so ordinary use is training data and performance scales with the user's hours, which is the one axis the literature shows improves monotonically. On a 32-channel dry cap we should expect roughly half of 8-target hits at 0.15–0.4 bit/s for most users and a quarter who get real 2-D control, with a fifth who get none; that is a slow cursor, not a mouse, and we say so. What this proposal buys over the alternatives is a system that can only get better with practice and whose eye-versus-cortex question is settled by design before anyone records a session.
