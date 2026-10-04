# Proposal G — The Honest Hybrid: cap-EOG gaze pointing + EEG click/veto

**Claim.** The only signal on this cap that can drive a cursor *continuously, naturally and today* is the ocular dipole leaking into FP1/FP2/F7/F8. We decode it as a gaze proxy for pointing, let EEG add a small, pre-registered, surrogate-tested click/veto, and state up front that ≥ 90 % of the control information will be ocular, not cortical.

## 1. Paradigm and signal basis

**Signal.** The eye is a ~1 mV corneo-retinal dipole. Rotation produces a quasi-DC potential of roughly 14–20 µV/° periorbitally, linear to about ±30° horizontally and ±20° vertically (vertical suffers eyelid artefact). On a 10-20 cap, horizontal rotation appears as an F7–F8 (and AF7–AF8, if fitted) differential; vertical rotation and blinks appear in-phase at FP1/FP2 relative to F3/F4/Fz. Propagation factors to frontopolar/frontal sites are of order 0.3–0.5 of periorbital amplitude (Croft & Barry 2000 review), so we expect ~5–8 µV/° at the cap. Blinks (100–200 µV at FP1/FP2) are detected, not decoded.

**Why it is natural.** The person looks where the cursor should go. No flicker, no oddball, no imagery; fixation–saccade behaviour is what people already do with a mouse (gaze leads the pointer by 100–300 ms; Huang, White & Buscher 2012 found gaze and cursor tightly aligned at click time).

**Why it is not brain control, said plainly.** The decoded variable is eye orientation, mediated by oculomotor nuclei and cranial nerves, not cortical state. It is the confound RLNF could not resolve (FP1/FP2 carried the largest effects; an EOG-only decoder was above chance). We make the confound the controller and admit EEG only where it beats an ocular/behavioural surrogate:

- **Tier 0 (robust):** EEG engagement/artefact gate (occipital alpha, EMG band power, saturation) that freezes the cursor when the user disengages, limiting Midas touch.
- **Tier 1 (modest, pre-registered):** an *intent-to-click* decoder trained on real self-paced button presses (movement-related cortical potential and central mu/beta ERD, 0.5–1.5 s pre-press; Kornhuber & Deecke 1965; Niazi et al. 2011 report ~80 % true-positive detection of self-paced movement intention with gel, preceding EMG onset by ~0.1–0.3 s; Lew et al. 2012 similar for reaching). At deployment it shortens dwell. The label is a keypress, not a displayed variable.
- **Tier 2 (exploratory, with falsifier):** ErrP veto 200–600 ms after a dwell selection (Chavarriaga & Millán 2010: ~75–80 % single-trial accuracy with gel). RLNF found outcome-locked ErrP display-identifiable and unstable on this hardware, so Tier 2 ships only if it survives §6.

**Optional add-ons.** (a) A bipolar periorbital EOG pair (two spare channels or a 2-channel LSL amplifier) for SNR and clean attribution. (b) A consumer eye tracker (~0.5–1°) as a *teacher* for the gaze model, not the controller.

## 2. Expected performance on this hardware

**Reference numbers.**
- Barea et al. 2002 (IEEE TNSRE): EOG guidance with periorbital gel electrodes, angular resolution ≈ 2°.
- Usakli & Gurkan 2010 (IEEE TIM): EOG virtual keyboard, ~95 % selection accuracy, a 5-letter word in ~25 s (≈ 5 s/character, ≈ 0.5–0.9 bits/s); Usakli et al. 2010 (Comput. Intell. Neurosci.) report EOG faster than P300 on the same keyboard.
- Belkacem et al. 2015 (Biomed. Signal Process. Control; Comput. Intell. Neurosci.): online classification of 4–8 eye-movement directions from **two temporal EEG-cap sensors**, ≈ 90 %, used for real-time game control — the closest demonstration that cap sites, not periorbital pairs, suffice.
- Ma et al. 2015 (IEEE TBME): EOG/EEG hybrid (eye movements + ERPs) for robot control, hybrid beating either modality alone.
- Zander et al. 2010 (Int. J. HCI): eye-tracker pointing + BCI selection removes dwell-based Midas touch; Kosmyna & Tarpin-Bernard 2013 (IEEE TCIAIG): gaze + EEG on consumer hardware, gaze carrying the pointing load.
- Gaze pointing with dwell click: ≈ 3–4 bits/s under ISO 9241-9 (Zhang & MacKenzie 2007); mouse ≈ 4–5 bits/s (Soukoreff & MacKenzie 2004). This is the ceiling for any ocular pointer.
- Learned gaze from cap EEG without an EOG pair: EEGEyeNet (Kastrati et al. 2021, NeurIPS D&B; 128-ch gel, 356 participants, ~47 h with eye tracking): absolute-position RMSE ≈ 120 px for CNN baselines, ≈ 55 px for later transformer models (Yang & Modesitt 2023), roughly 3° and 1.5° at that geometry (verify px→deg against the dataset's screen specification); left/right saccade ≈ 98 %.
- Comparator, cortical continuous 2-D control: Wolpaw & McFarland 2004 (PNAS), sensorimotor rhythms, ~1.9 s movement time, ~90 % hits after months of training, ≈ 0.3–0.5 bits/s.

**Dry-electrode penalty.** Guger et al. 2012 (Front. Neurosci.) found g.SAHARA dry pins gave P300 accuracy comparable to gel with lower SNR; but EOG pointing lives at 0–5 Hz where dry electrodes are worst (polarisation drift, contact-pressure modulation). Two mitigations are design-specific: FP1/FP2/F7/F8 are hairless, where dry pins contact best; and the decoder fuses band-passed *saccade displacements* with slow absolute position, so drift degrades rather than destroys control. We budget a **2× angular-error penalty** versus gel periorbital EOG.

**Realistic range (24-in monitor, 60 cm, 1° ≈ 1.05 cm ≈ 40 px).**

| Configuration | Angular error (RMS) | Usable target width | Fitts throughput | Time-to-target |
|---|---|---|---|---|
| Cap only, linear calibration | 4–6° | ≥ 12° | 0.6–1.2 bits/s | 1.5–2.5 s |
| Cap only, learned model (≥ 10 h labels) | 2.5–4° | ≥ 8° | 1.0–1.8 bits/s | 1.2–1.8 s |
| + bipolar EOG pair (gel) | 1.5–2.5° | ≥ 5° | 1.5–2.5 bits/s | 1.0–1.5 s |
| + consumer eye tracker (cap = click only) | 0.5–1° | ≥ 2° | 2.5–3.5 bits/s | 0.8–1.2 s |

Derivation (row 2): D ≈ 25°, W ≈ 10° → ID ≈ 1.8 bits; saccade 0.05 s + settle 0.3 s + dwell 0.6 s + ~10 % misses ≈ 1.1–1.6 s per selection. The EEG click at best replaces 0.3 s of dwell (≤ 20 % of throughput); we expect 0–10 % after surrogate correction. Vertical error ~1.5× horizontal.

## 3. Scaling behaviour (bitter lesson)

**Mechanism.** The map from 32 scalp potentials to screen coordinates is a nonlinear, session-varying regression with abundant free labels: every mouse click gives a (1-s window, screen x,y) pair, and ordinary desktop use yields ~500–1,500 clicks/h; an eye tracker gives 60 labels/s. Linear calibration has a fixed error floor (electrode gain, head pose, illumination-driven corneo-retinal gain, eyelid nonlinearity); a learned model absorbs these because they are *observable* in the 32 channels, head pose and luminance.

**Data engine.** (i) Click labels in free use (strong, sparse); (ii) mouse-trajectory labels (weak, dense: 150 ms lead, robust loss); (iii) eye-tracker labels when available; (iv) self-paced keypresses for Tier 1; (v) undo/cancel events as weak error labels for Tier 2. Nightly retraining; the frozen test session never enters training.

**Pretraining and transfer.** Pretrain the gaze regressor on EEGEyeNet (channel-subset to the 32 cap positions, resampled to 500 Hz, dry-noise augmentation: 1/f drift, contact-gain jitter, pin pops), then fine-tune per user. Ocular geometry is similar across people, so expected calibration-free first-session error is 4–6°, falling to 2.5–4° after ~2 h of the user's clicks and ~1 h per later session.

**Co-adaptation.** The user learns to fixate steadily and exploit snapping; the decoder re-centres on every click (recursive least squares on the linear head, λ = 0.995) and fine-tunes nightly, with labels from *actions*, not the display.

**Expected curve.** Angular error vs labelled hours, cap-only: ~5° at 0 h (transfer), ~3.5° at 2 h, ~3° at 10 h, ~2.5° at 50 h; log-linear, then flat.

**Saturation and why.** (1) Physics: 5–8 µV/° against a dry 0–5 Hz noise floor of ~5–15 µV RMS over 1 s gives a floor near 1–2°, worse vertically. (2) Illumination: corneo-retinal gain changes up to ~2× over 10–20 min of light/dark adaptation; the model tracks it only through the clicks it sees. (3) Label noise: gaze is 1–2° from the click point, so click-only training saturates near 1.5–2°; the tracker lifts that floor. (4) The EEG click/veto saturates early: RLNF found 24 candidates and six deep architectures TOST-equivalent at ~11.5k trials; we expect the same plateau at a few thousand presses because single-trial MRCP/ErrP SNR, not capacity, limits it.

## 4. Data-collection protocol

**Participants.** N = 24, no performance-based exclusion; plus 2 long-haul users with ≥ 20 h of logged free use for the scaling curve. Three sessions (S1–S3) on separate days, 75 min each.

**LSL streams.** `EEG` (32 ch, 500 Hz; run with the lowest hardware high-pass available — DC if the amplifier configuration allows, else ≤ 0.01 Hz; log the setting), `EOGbip` (optional periorbital pair), `Gaze` (optional tracker, 60 Hz), `Mouse` (x, y, buttons, 120 Hz), `Cursor` (decoded x, y, state, 60 Hz), `Markers` (block/trial/target/selection/undo/keypress), `Head` (webcam 6-DoF pose, 30 Hz), `Lux` (1 Hz), `Impedance` (each block start). A 1-Hz sync pulse on a spare channel verifies alignment.

**Blocks per session (fixed order; within-block conditions randomised with logged seed).**
1. Calibration: 9-point grid × 2, 1.5 s/point; 1 min smooth pursuit.
2. Mouse Fitts (ISO 9241-9 multidirectional, 13 targets × IDs {1.5, 2.5, 3.5 bits} × 3), ~8 min; labels: gaze = click position. Then self-paced clicks with ≥ 2 s spacing (~150 presses, Tier 1 labels).
3. Free mouse use (web reading, casual game), 10 min.
4. Closed-loop Fitts, 3 × 6 min, randomised: **(a) EOG-dwell** (600 ms); **(b) EOG + EEG click** (dwell 300 ms when P(click) > θ); **(c) EOG + sham click**, the EEG decision replaced by a stochastic generator matched to (b)'s rate and latency. If (b) ≈ (c), the EEG click has no utility.
5. Free cursor use, Tier 2 veto on/off, 2 × 5 min; undo/cancel logged.

S3 is the frozen evaluation session: models trained after S2 only; only click re-centring runs online.

**Gating.** Per-channel amplitude gate at 5× session median RMS (FP1/FP2/F7/F8 exempt, but gated at saturation and > 1 mV non-blink excursions); impedance logged, blocks flagged (not excluded) above 500 kΩ; head excursion > 5 cm flagged.

**Controls and surrogates (pre-registered).**
- *Attribution decoders*: EOG-only (FP1, FP2, F7, F8 ± pair), EEG-only (channels posterior to Fz after regressing out the four ocular channels), combined; report R² and bits/s for each.
- *Non-neural surrogate for Tier 1*: predict click from cursor/fixation kinematics only (fixation duration, residual velocity, distance to nearest target). If its AUC ≥ EEG AUC − 0.03 (TOST), Tier 1 is declared behavioural and dropped.
- *Non-neural surrogate for Tier 2*: predict "error" from the display event (highlight timing, cursor–target distance); same rule.
- *Webcam-only surrogate* for pointing (head pose → screen): verifies the cap is not a head-tracker.
- Permutation floors at matched n by circular time-shift of labels (1,000 shifts).

**Pre-registration list.** Primary/co-primary endpoints; the table in §2 as prediction; TOST margins; surrogate rules; frozen S3; hyperparameters (§5); exclusion-free analysis; falsifiers: (F1) cap-only S3 throughput < 0.6 bits/s → paradigm fails on this hardware; (F2) EEG click not ≥ 0.1 bits/s better than sham → Tier 1 removed; (F3) EEG-only pointing R² > 0.1 after ocular regression → attribution model wrong, investigate leakage.

**Power.** Paired (b) vs (c) and (b) vs (a): d = 0.6, α = 0.05 two-sided, n = 24 → ≈ 80 %. Cap-only throughput vs the F1 threshold: SD ≈ 0.4 bits/s, expected mean 1.2 → power > 0.95.

## 5. Decoder and system

**Preprocessing (deterministic, seeded).** Ocular branch keeps the amplifier reference; EEG branch average-referenced after excluding FP1/FP2/F7/F8. Two paths: *slow* (zero-phase FIR 8 Hz low-pass, no high-pass or 0.01 Hz if DC unavailable) for absolute position; *fast* (0.5–40 Hz) for saccade detection and EEG features. Blink detector: FP1+FP2 > 80 µV for 100–400 ms; cursor frozen during blinks + 150 ms. With a 0.1 Hz high-pass the fixation level decays with τ ≈ 1.6 s, so absolute position is unrecoverable and the system falls back to saccade integration with click re-centring — hence the amplifier setting is logged and pre-registered.

**Gaze model.** Baseline: ridge regression (λ = 10) from {F7−F8, AF7−AF8, (FP1+FP2)/2 − Fz, FP1−FP2} slow-path 200 ms means to (x, y), fitted on calibration points. Learned: 1-D temporal CNN, input 32 ch × 500 samples, 3 blocks (64/64/128 filters, kernels 25/15/7, stride 2, GELU, BatchNorm, dropout 0.2), global average pool, linear head to (x, y, log σx, log σy); heteroscedastic Gaussian NLL; AdamW lr 3e-4, weight decay 1e-2, batch 128, 30 epochs, early stop on held-out S1 blocks; augmentations: 1/f drift 0.5–5 µV/s, per-channel gain ±30 %, pin pops, 50 Hz hum. Pretrained on EEGEyeNet (§3); linear head updated online by RLS at each click.

**State estimator.** Kalman filter on gaze: process = saccade displacement from the fast path (velocity threshold 30°/s); measurement = slow-path absolute estimate with the network's σ; reset to click position at each click. Drift-bounded absolute pointing even when the slow path is poor.

**Cursor dynamics.** Position control (gaze is absolute). Fixation-snapping: cursor moves only on a detected fixation ≥ 100 ms, with a 100 ms critically damped step; 0.3° dead-zone within fixation. Magnetic snap to the nearest selectable element within 1.5× current σ (logged; throughput reported with and without). Gain is identity in screen space.

**Click/veto.** Tier 1: logistic regression on Riemannian tangent-space features of C3/Cz/C4/FC1/FC2/CP1/CP2 (8–30 Hz covariance, 1-s window) plus 0.1–5 Hz MRCP means, positives −0.8 to 0 s before presses, negatives random fixations ≥ 2 s from any press, threshold at 5 % false positives per second. Tier 2: same features, 200–600 ms post-selection, veto if P(error) > 0.7.

**Latency budget.** LSL chunk 20 ms + filtering 2 ms + CNN inference 5 ms (CPU) + Kalman < 1 ms + render ≤ 16 ms → ~45 ms saccade-end to cursor (gaze interfaces tolerate < 100 ms). Tier 2 defers selection feedback by 600 ms.

**Co-adaptation rule.** Updates only from clicks not undone within 2 s; per-block throughput feedback; no mid-session goal changes.

**Software.** LSL + Python (MNE, pyRiemann, PyTorch), one YAML of config and seeds, XDF recording, bit-for-bit offline replay.

## 6. Evaluation protocol

**Primary.** ISO 9241-9 throughput (bits/s, effective width) in 4(a) on frozen S3, cap-only model; 95 % bootstrap CI over participants against the F1 threshold.
**Co-primary.** Angular error (deg RMS) on S3 mouse-Fitts clicks, cap-only vs cap + pair.
**Secondary.** (i) Throughput (b) − (c) and (b) − (a) with CIs; (ii) attribution: R² and bits/s for EOG-only, EEG-only, combined, and the ocular fraction = EOG-only bits / combined bits; (iii) Tier 1 AUC vs kinematic surrogate vs permutation floor at matched n; (iv) Tier 2 net benefit (errors corrected − correct selections vetoed); (v) scaling: error vs labelled hours for the long-haul users, and S1→S3 improvement across participants.
**Statistics.** Linear mixed models (participant random intercept); TOST margins 0.15 bits/s (utility), 0.03 AUC (surrogates); permutation floors beside every AUC/R²; effect sizes with CIs; no exclusions; leave-4-participants-out transfer; S3 fixed across the data ladder {0.5, 1, 2, 5, 10 h}.

## 7. Failure modes and debugging

- **Drift/illumination:** slow wander between clicks; diagnose from `Lux` and slow-path offset; fix with saccade-integration mode, re-centring, or the EOG pair.
- **Vertical inaccuracy, eyelid artefact:** wide-aspect targets; tracker teacher if needed.
- **Pin pops / contact loss at F7/F8 (hairline):** impedance and pop detector; fall back to AF/FP channels, then the pair.
- **Head movement:** mapping shifts; head pose as model input; if the webcam surrogate explains most variance, the cap is a head-tracker — report it.
- **Midas touch:** Tier 0 gate plus a "park" region; count unintended selections per minute.
- **EEG click equals surrogate (likely):** report, drop Tier 1, keep the honest ocular controller.
- **Ocular leakage into "EEG-only":** F3 falsifier; regress with the bipolar pair and re-test.

## 8. Honest self-score

1. Naturalness **8/10** — no stimulation, no imagery, the person just looks; off one for fixation discipline/Midas touch, one because the user asked for *brain* control and this mostly is not.
2. Achievable performance **7/10** — 1.0–1.8 bits/s cap-only is literature-anchored and 2–4× any cortical continuous controller on this hardware, but far below an eye tracker and dependent on DC behaviour.
3. Scaling **7/10** — a genuine data engine (free click labels, EEGEyeNet pretraining) with a predictable log-linear curve, but it saturates at a physics floor near 2° and the EEG tiers plateau early.
4. Validity **8/10** — labels are actions (clicks, keypresses, tracker); attribution decoders and non-neural surrogates are built in and pre-registered; the ErrP tier remains display-vulnerable.
5. Feasibility **8/10** — 45 ms latency, 3-min calibration, hairless frontal sites favour dry pins; the DC/0.1 Hz question and vertical drift are real risk.
6. Rigour **8/10** — frozen S3, permutation floors, sham-click utility control, fixed test set across the data ladder; the long-haul curve rests on two users.

## 9. Pitch

The only signal on this cap strong enough for continuous, stimulation-free, imagery-free cursor control is the eye; pretending otherwise is how RLNF ended with an unresolved ocular confound. We make the eye the controller, decode it from FP1/FP2/F7/F8 with a model that improves with every mouse click the user ever makes, and expect 1.0–1.8 bits/s on dry electrodes rising toward 2.5 bits/s with a bipolar pair. EEG may add a click and a veto only if it beats a kinematic surrogate and a sham generator, and we publish the ocular-versus-cortical information split whatever it turns out to be. Every label comes from an action, never from a displayed criterion, and the evaluation session is frozen before it is recorded. It is not brain control, and it is the strongest honest system this hardware can deliver.
