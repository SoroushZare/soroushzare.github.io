# Proposal A — Stimulus-driven visual evoked control (SSVEP / c-VEP) for continuous 2-D cursor steering

## 1. Paradigm and signal basis

**Signal.** Steady-state (SSVEP) and code-modulated (c-VEP) visual evoked potentials: the occipital cortex (V1/V2, dominated by the foveal/parafoveal projection) phase-locks to the luminance modulation of whatever the eyes are pointed at. It is measurable in a single 0.5–1 s window at 5–20× the single-trial SNR of any endogenous rhythm, and decoding reduces to template correlation (CCA/TRCA) against K known codes with near-deterministic ground truth.

**Design.** A *cursor-locked ring*: 8 wedge sectors (radius 1.5–2.5° of visual angle) drawn around the cursor, each tagged with its own frequency/phase (SSVEP, 32–40 Hz) or a circular shift of one 63-bit m-sequence (c-VEP). The user looks at the wedge in the direction they want to go; velocity follows accumulated evidence; looking at the unmodulated centre (or away) yields symmetric/no evidence and the cursor stops. No imagery, no instructed mental task: the act is "look where you want to go".

**Honesty about naturalness.** This paradigm **violates the user's stated preference**: there is external stimulation on screen, continuously. The minimal-stimulation variant reduces, not removes, it: (i) frequencies above 30 Hz are at or beyond critical flicker fusion for most observers, so the ring looks like a faint shimmer rather than a flash; (ii) modulation depth 10–20 % instead of 100 % (Ladouce et al. 2022, Sci Rep, showed low-amplitude-depth and high-frequency stimuli preserve usable SSVEP while cutting reported discomfort); (iii) the ring is only present around the cursor, nothing else on the screen flickers; (iv) it fades out after 1.5 s of idle. 20 % depth costs ~2–3× in amplitude, which window length must compensate.

Second honest statement: overt SSVEP is **gaze-mediated**; an eye tracker would do the job faster. Our case: (a) no eye tracker exists today, (b) covert attention adds a non-gaze component (Kelly et al. 2005; Allison et al. 2008: 60–75 % covert accuracy, weak but real), (c) VEP control has, by one to two orders of magnitude, the strongest literature-backed performance of any EEG paradigm.

## 2. Expected performance on this hardware

**Literature anchors (gel, 8 occipital channels unless noted).**
- Chen, Wang, Nakanishi et al. 2015 (PNAS): 40-target JFPM SSVEP speller, 0.5 s stimulation, online ITR 267 bits/min (≈4.5 bits/s), 12 subjects.
- Nakanishi et al. 2018 (IEEE TBME): ensemble TRCA, simulated-online ITR 325 ± 38 bits/min on the 12-subject set, ~90 % accuracy with ≤1 s windows.
- Wang et al. 2017 (IEEE TNSRE) benchmark, 35 subjects; Liu et al. 2020 (Front Neurosci) BETA, 70 subjects: pretraining corpora.
- Bin et al. 2011 (J Neural Eng): 32-target m-sequence c-VEP, ITR ~108 bits/min. Spüler et al. 2012 (PLoS ONE): c-VEP with online unsupervised adaptation, 144 bits/min. Nagel & Spüler 2019 (PLoS ONE): EEG2Code CNN on 32-ch gel, ~700 bits/min EEG-only, 1237 bits/min with gaze fusion. Thielen et al. 2021 (J Neural Eng): zero-training c-VEP via reconvolution. Martínez-Cagigal et al. 2021 (J Neural Eng) reviews the field.
- Continuous steering: Trejo, Rosipal & Matthews 2006 (IEEE TNSRE), 1-D/2-D SSVEP cursor control with edge targets; Wilson & Palaniappan 2011 (J Neural Eng), online analogue SSVEP mouse pointer. Both are proof-of-principle, far below speller ITRs because steering is latency-limited.
- High-frequency asynchronous SSVEP: Diez et al. 2011 (J NeuroEng Rehab), 37–40 Hz; amplitudes at 32–40 Hz run 3–5× below 10–15 Hz.
- Dry electrodes: Xing et al. 2018 (Sci Rep), 12-target SSVEP with dry electrodes, ITR ≈ 92 bits/min, i.e. roughly half to two-thirds of comparable gel systems. Guger et al. 2012 (Front Neurosci): g.SAHARA pins match gel P300 accuracy with more noise and drop-outs.

**Penalties applied to this cap.**
1. *Occipital coverage.* The Chen/Nakanishi 8-channel set (PO3–PO6, POz, O1, Oz, O2) is mostly absent from a strict 10-20 32-channel montage; usable sites are O1, Oz, O2, P7, P3, Pz, P4, P8 (plus PO7/PO8 if the g.Nautilus montage has them). Going from 8 PO/O channels to 3 O + parietal costs ~5–10 % accuracy at 1 s in published channel-subset analyses. **Mitigation (strongly recommended):** the cap has spare holes; re-seat four of the frontal/temporal pins (e.g. F7, F8, T7, T8) at PO3, PO4, PO7, PO8. Keep FP1/FP2 for the ocular surrogate.
2. *Dry pins.* Occipital hair is the worst case for pins; expect 1.5–3× gel broadband noise, i.e. ~1.5–2× longer windows for equal accuracy.
3. *Low-contrast, high-frequency flicker.* Amplitude 3–5× below full-contrast 12 Hz; another ~1.5× in window length.

**Realistic range (8 sectors + idle, sliding 0.8–1.5 s windows, step 0.1 s).**
- Per-window direction accuracy: gel-PO reference ~95 %; this cap with relocated pins 80–90 %; strict 10-20 with O1/Oz/O2 only 65–80 %.
- Effective selection rate: 1.0–1.6 bits/s (relocated pins), 0.5–1.0 bits/s (strict 10-20); compare 4.5 bits/s for the gel speller.
- Fitts throughput for 2-D acquisition (ID 1.5–4 bits, 1 s dwell): **0.3–0.7 bits/s** with relocated pins, **0.15–0.4 bits/s** with strict 10-20. Time-to-target for ID ≈ 2.6 bits: 4–8 s and 7–14 s respectively. Best case (gel, PO coverage, full contrast) would be ~0.8–1.2 bits/s. For scale, a gaze pointer is 2–3 bits/s and a mouse 4–5 bits/s.
- Idle false positives (drift while not looking at the ring): target < 1/min.
- 5–10 % of participants will be low responders (Guger et al. 2012: most but not all of 53 subjects reached usable SSVEP accuracy); they stay in the analysis.

## 3. Scaling behaviour (bitter lesson)

This paradigm does not scale like a language model: the signal is low-dimensional and stimulus-driven, and per-subject TRCA templates saturate after **2–5 minutes** of calibration. Where more data does buy more:

- **More subjects → zero/short calibration.** User-independent CNNs (Waytowich et al. 2018, J Neural Eng, Compact-CNN; Ravi et al. 2020, J Neural Eng) reach ~80 % of per-subject accuracy; cross-domain transfer of templates (Chiang, Wei, Nakanishi & Jung 2021, J Neural Eng) recovers most of the remainder with 1 min of target-subject data. Expected LOSO curve on our channel subset: ~60 % (10 subjects) → ~72 % (30) → ~78 % (100) → plateau near the per-subject ceiling minus 5–8 %. Saturation cause: inter-subject variability is a handful of latency/topography parameters, learned quickly; the residual is SNR.
- **More hours → the controller, not the evoked response.** The steering policy (accumulation gains, idle threshold, velocity profile, per-sector bias) is a small Bayesian/RL problem fed by a data engine: in free steering, every window with posterior > 0.9 whose subsequent movement reached the eventual target becomes a self-labelled example (Spüler 2012 showed unsupervised c-VEP adaptation on exactly this logic). Expect 10–20 % throughput gain over sessions 1→3, then a physical (window-vs-SNR) plateau.
- **Pretraining corpus.** Benchmark (35 subj) + BETA (70 subj) + MOABB SSVEP sets give ~100 subjects of gel 64-ch 8–16 Hz data. Mismatch with our dry, 32–40 Hz, low-contrast setting is large; we expect modest transfer (+5–10 % initial accuracy over random init), measured not assumed. The EEG2Code-style arm (Nagel & Spüler 2019) is the one place a learned model clearly beat hand-crafted CCA, and is our scaling arm.
- **Co-adaptation.** The user learns stable wedge fixation and the centre brake; the decoder refits templates every 5 min from confident windows. Measured, not assumed.

**Where it saturates and why.** (i) SNR physics: amplitude at 36 Hz/20 % contrast is ~0.1–0.5 µV against 2–5 µV dry noise; accuracy grows with window length, and windows above ~1.5 s make steering sluggish. (ii) Non-stationarity: pin drift and head-fit changes between sessions shift spatial filters; 1-min re-calibration fixes most of it. (iii) Label noise: near-zero for cued blocks (the label is the stimulus the generator emitted), the single structural advantage of a stimulus-driven paradigm.

## 4. Data-collection protocol

**Participants.** N = 24, no performance-based exclusion, normal/corrected vision, screened for photosensitive epilepsy. Power: paired comparison of EEG controller vs non-neural surrogate on throughput, expected d ≥ 0.8, two-sided α = 0.05, 80 % power needs n = 15; N = 24 covers 25 % attrition and gives a 24-subject LOSO curve.

**Sessions.** 3 per participant (day 1; day 2–7; day 14–28), ~55 min cap time. Session 1 runs both variants (SSVEP, c-VEP) counterbalanced; sessions 2–3 use the per-subject better one (pre-registered rule: higher Block-B accuracy).

**Display/timing.** 120 Hz monitor, 60 cm viewing distance; SSVEP wedges 32.0–39.0 Hz in 1-Hz steps with sampled-sinusoid rendering (Wang et al. 2010, Electron Lett); c-VEP: one 63-bit m-sequence at 120 Hz (525 ms period), 8 shifts of 8 bits (67 ms) for the sectors. A photodiode on a corner patch feeds the g.Nautilus digital trigger input, giving frame-accurate stimulus timing inside the EEG stream.

**Blocks per session.**
- A. *Cued calibration*, 2 × 2 min: 9 classes (8 wedges + centre/idle) × 12 trials × 2 s, order from a seeded RNG; cursor static.
- B. *Cued steering*, 2 × 3 min: cue arrow switches among wedges with exponential dwell (mean 2.5 s, min 1 s); cursor moves under decoder control; label = cued wedge, which is causally upstream of the cursor and independent of its state.
- C. *Fitts acquisition*, 3 × 4 min: targets at 9 (distance, width) pairs, ID 1.5–4.0 bits, random position; cursor starts at previous target; select by 1 s dwell; 15 s timeout; no assistance.
- D. *Free use*, 5 min (maze or drawing) for the data engine.
- E. *Controls*, 3 min: eyes-closed with ring on (leakage check), centre fixation with ring on (idle floor), 1 min no-ring (spontaneous floor).

**Labels.** A/B: the generator-emitted code per frame. C: weak labels from the target direction, with confidence gating. **Estimability audit:** positivity (every code at every cursor position in A/B: yes by construction); collinearity (cue vs cursor state: cue is RNG-driven, cursor is decoder-driven, r checked < 0.1 pre-registered); visual mediator (the cue arrow is displayed, so a frontal-only decoder, FP1/FP2/F7/F8, must sit at the permutation floor — tests both cue-ERP and ocular leakage); generator-determined sequences (seeded RNG, seeds logged).

**LSL streams.** EEG 32 ch @ 500 Hz with photodiode trigger; `stim_frames` (per-frame code bits, 120 Hz); `cursor_state` (x, y, v, posterior, idle flag, 20 Hz); `events` (block/trial/cue/target/dwell/timeout); `quality` (per-channel RMS and g.tec quality index, 60 s); `eyetracker` if added (120 Hz).

**Gating.** Per-channel amplitude gate at 5× session median RMS (FP1/FP2 exempt); windows failing on >2 occipito-parietal channels are dropped and logged; dropped fraction is reported; electrode quality logged at setup and every 5 min.

**Pre-registered list.** Primary/secondary endpoints (Section 6), idle threshold rule, SSVEP-vs-c-VEP selection rule, surrogate definitions, floors at matched n, hyperparameters (Section 5), analysis code hash, falsifiers: (F1) EEG controller throughput ≤ frontal-only surrogate; (F2) eyes-closed decoding above floor (leakage); (F3) session-2 throughput with session-1 templates < 70 % of recalibrated.

## 5. Decoder and system

**Preprocessing (deterministic).** 500 Hz; 50 Hz notch (IIR, Q = 30); band-pass 3–95 Hz (4th-order Butterworth, causal online); no ICA or EOG regression (we test, not clean, the ocular channel); 8–10 occipito-parietal channels; frontal channels reserved for the surrogate.

**SSVEP decoder.** Filter-bank ensemble TRCA (Nakanishi 2018): 3 sub-bands [30–95], [62–95], [90–95→drop if empty] Hz, weights (n^-1.25 + 0.25); templates from Block A; sliding window 1.0 s (grid-searched 0.6–1.5 s on Block A CV, fixed before Block B), step 0.1 s; evidence ρ_k = weighted correlation per wedge; idle class = 9th template plus a null threshold from 1000 phase-scrambled correlations (reject if max ρ below the 95th percentile of the null; Pan et al. 2013 logic).

**c-VEP decoder.** CCA spatial filter + reconvolution template (Thielen 2021; pyntbci implementation), one filter shared across shifts; window 1.05 s (two periods), step 0.1 s. Scaling arm: EEG2Code-style 1-D CNN (3 conv layers, 16-32-32 filters, kernel 25 samples, dropout 0.25, Adam 1e-3, 50 epochs, early stop on a Block-A fold) predicting the stimulus bits, pretrained across subjects.

**Evidence accumulation and cursor dynamics.** Posterior p_t ∝ p_{t-1}^λ · softmax(β ρ_t), λ = 0.7 (≈300 ms memory), β = 20; velocity v = G · Σ_k p_k d_k · 1[max_k p_k > 0.6]; G per subject in 150–400 px/s chosen in Block B; first-order smoothing τ = 150 ms; velocity control (not position: gaze position is exactly what we cannot read). No target snapping in evaluation blocks.

**Latency budget.** Acquisition/LSL 20–40 ms; window end-to-decision 5 ms (TRCA) / 15 ms (CNN) on CPU; render 8 ms at 120 Hz; effective decision lag ≈ window/2 + 50 ms ≈ 0.55 s. Direction reversals take 0.6–1.0 s to show.

**Splits.** Block A trains; B/C/D are never used for template fitting within a session. Cross-session: session-1 templates applied to sessions 2–3 before recalibration (held-out session endpoint). Cross-subject: LOSO over 24 subjects for the CNN arm with a fixed test set across the resource ladder (10/30/60/100 % of training subjects, same test windows).

**Co-adaptation rule.** Every 5 min of free use, windows with posterior > 0.9 and congruent subsequent movement are appended to the template buffer (exponential forgetting, half-life 10 min); templates are refit; the update is logged so any step can be ablated offline.

## 6. Evaluation protocol

- **Primary endpoint (co-primary).** (P1) Fitts throughput (bits/s, effective-ID method) in Block C, sessions 2–3, vs the frontal-only non-neural surrogate controller run live on alternating trials (same pipeline, FP1/FP2/F7/F8 inputs). (P2) Block-B wedge accuracy vs the permutation floor at matched window count (1000 label permutations of the code-to-wedge map).
- **Secondary.** Time-to-target; path efficiency; idle false positives/min; eyes-closed decoding (must be at floor); session-1→2 transfer; LOSO ladder; SSVEP vs c-VEP within subject; visual-fatigue rating (0–10, every block); gated-window fraction.
- **Non-neural surrogates (pre-registered).** (S1) frontal-only EOG-proxy controller; (S2) photodiode-only "decoder" (must be 100 %: confirms the pipeline sees stimulus timing, and then is excluded from EEG channels by construction); (S3) if an eye tracker is added, a gaze-only controller — we expect it to **beat** EEG, and we will report that plainly as the paradigm's ceiling.
- **Statistics.** Paired Wilcoxon with bootstrap 95 % CIs; Cliff's δ; mixed model throughput ~ session + variant + (1|subject); TOST (±0.05 bits/s) for any "no difference" claim; Bonferroni across co-primaries; prediction interval for a new participant, as in RLNF.
- **Reproducibility.** Seeded generators, XDF archives, pinned environment and container hash, OSF pre-registration before participant 1.

## 7. Failure modes and debugging

1. *Occipital pin contact fails*: quality index and RMS gating flag it at setup; re-seat, part hair, or accept longer windows; report per-subject channel loss.
2. *Flicker is visible/annoying at 20 % depth*: raise frequency toward 40 Hz, reduce depth per subject to the lowest level keeping Block-A accuracy > 80 %; log the chosen depth.
3. *Frame drops/jitter*: photodiode vs `stim_frames` discrepancy > 1 frame aborts the block (c-VEP is the more sensitive variant).
4. *Mains/harmonic collisions*: 50 Hz notch and avoiding 25/50 Hz subharmonic stimulation frequencies; check the spectrum of the no-ring block.
5. *Leakage* (backlight/electrical coupling): eyes-closed control above floor invalidates the setup until shielding fixes it.
6. *Low responders*: fall back to 12–15 Hz full-contrast for that subject (explicitly a naturalness sacrifice), reported separately.
7. *Steering oscillation* (overshoot from lag): lower G, raise λ in Block B.
8. *Idle drift*: raise the null-percentile threshold; measured in the centre-fixation control.
9. *Gaze-only is the whole story*: if S3 beats EEG everywhere, say so; the product is then "add the eye tracker", with EEG contributing a covert-attention term worth measuring.

## 8. Honest self-score

1. **Naturalness: 3/10.** Continuous on-screen flicker, however faint and cursor-locked, is exactly what the user said they do not want; the only natural part is that the act is "look where you want to go".
2. **Achievable performance: 6/10.** Strongest EEG literature of any paradigm (4–5 bits/s spellers), but dry pins, thin occipital coverage and low-contrast high-frequency stimuli bring realistic steering throughput to 0.3–0.7 bits/s.
3. **Scaling: 4/10.** Per-subject decoders saturate in minutes; subjects buy zero-calibration and hours buy a better controller, with a hard SNR ceiling.
4. **Validity: 8/10.** Labels are generator-emitted codes causally upstream of the display state; ocular, cue-ERP and leakage surrogates are all pre-registered; the gaze-mediation ceiling is stated rather than hidden.
5. **Feasibility: 7/10.** 4-min calibration, trivial compute, mature stack (PsychoPy/pyntbci/LSL), but needs a 120 Hz monitor with photodiode, cap re-pinning, and visual fatigue limits session length.
6. **Evaluation rigour: 8/10.** Co-primary utility endpoint, matched-n floors, live surrogate controller, held-out sessions and LOSO ladder on a fixed test set, pre-registered falsifiers.

Weighted total ≈ 0.25·3 + 0.20·6 + 0.20·4 + 0.15·8 + 0.10·7 + 0.10·8 = **5.45/10**.

## 9. Five-sentence pitch

Visual evoked control is the only non-invasive EEG paradigm whose published numbers are in bits per second rather than tenths of a bit, and that margin survives a dry cap and a thin occipital montage. A cursor-locked ring of faint 32–40 Hz wedges turns steering into the most natural act there is, looking where you want to go, with no imagery and with idle emerging automatically when you look away. Its labels are emitted by the stimulus generator, not by anything the cursor displays, so the RLNF label trap cannot recur and ocular, leakage and gaze surrogates are cheap to run. It fails the user's no-stimulation preference, and I say so at 3/10 rather than disguise it; the mitigation is a faint, local, high-frequency shimmer that most users stop noticing. If the user will tolerate that shimmer, this is the design that will put a working cursor on screen in session one and tell us, through its own surrogates, exactly how much an eye tracker would add.
