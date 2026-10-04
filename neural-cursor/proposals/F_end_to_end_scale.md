# Proposal F — The bitter-lesson, end-to-end approach: cursor control as large-scale supervised learning on naturally labelled EEG

## 1. Paradigm and signal basis

**Claim.** Stop designing features. Treat 2-D cursor control as sequence regression whose labels come free: record EEG while the person uses their computer with a real mouse; actual mouse velocity is the dense label. Pretrain a large EEG encoder self-supervised on public corpora, fine-tune to regress kinematics, then go closed-loop with ReFIT/CLDA-style intention relabelling (Gilja et al. 2012 Nat Methods; Orsborn et al. 2012 IEEE TNSRE, 2014 Neuron).

**What physiological signal is actually there.** The literature calls this "low-frequency kinematics decoding": 0.1–3 Hz time-domain EEG regressed onto hand velocity/position. Bradberry, Gentili & Contreras-Vidal (2010, J Neurosci) reported 3-D hand velocity with Pearson r ≈ 0.19–0.38 (per-axis means, 55 gel electrodes, delta band); Lv, Li & Gu (2010) and Robinson, Guan & Vinod (2015, J Neural Eng; Kalman tracking) r ≈ 0.3–0.5 for 2-D drawing; Paek et al. (2014) r ≈ 0.4 for finger velocity; Agashe et al. (2015) r ≈ 0.6 for grasp aperture. **The honest reading of this literature is poor.** Antelis et al. (2013, PLoS ONE) showed comparable r arises from linear regression on heavily low-pass-filtered signals carrying no movement information unless chance is estimated with trajectory-preserving surrogates; Kobler, Sburlea & Müller-Putz (2018, Sci Rep) showed that in visuomotor tracking the EEG "velocity tuning" is largely eye position/velocity via the corneo-retinal dipole (≈10–20 µV per degree versus 2–5 µV for movement-related cortical potentials: eyes outweigh cortex 5–10× in band). After subspace eye-artifact correction (SGEYESUB, Kobler et al. 2019, 2020 NeuroImage) a genuine but smaller centro-parietal/premotor component survives: Mondini et al. (2020, J Neural Eng) ran closed-loop robot-arm control from eye-corrected low-frequency EEG with online velocity correlations ≈0.3; Pulferer et al. (2022, J Neural Eng) decoded 2-D trajectories from attempted and executed movement across sessions, including one person with SCI, at r ≈ 0.2–0.4; Ofner et al. (2017 PLoS ONE; 2019 Sci Rep) classified executed/attempted upper-limb movements at ≈45–55% (chance 17–20%).

**Why this is "natural".** The overt phase asks nothing of the person: no flicker, no oddball, no imagery script. The transfer phase asks them to *attempt* the same hand movement with the hand resting (attempted movement, the strategy that works in SCI users, Ofner 2019). "Wishing the cursor there" with no motor attempt is a third, far weaker regime (covert visuospatial attention; Tonin et al. 2013 J Neural Eng ≈70% binary); I report expectations for all three.

**What a naïvely trained decoder will learn.** During real mouse use the gaze lands on the target 50–200 ms *before* the cursor (Smith, Ho, Ark & Zhai 2000, CHI), so gaze is a *leading* predictor of mouse velocity, 5–10× larger in EEG than cortex, maximal at FP1/FP2/F7/F8. A whole-cap regression becomes an EEG eye tracker plus a motor-execution detector (contralateral mu/beta ERD, readiness potential). Both vanish when the hand stops and the eyes fixate. Sections 4–5 make this failure impossible to hide.

## 2. Expected performance on this hardware (honest range)

Reference points: mouse ≈ 4–5 bit/s (ISO 9241-9, MacKenzie); intracortical ReFIT ≈ 2 bit/s (Gilja 2012; Pandarinath 2017 eLife); EEG sensorimotor-rhythm 2-D cursor control reached 89% hits at ≈1.9 s per target in trained users (Wolpaw & McFarland 2004, PNAS), roughly 0.3–0.5 bit/s; Edelman et al. (2019, Sci Robotics) improved continuous 2-D tracking with source imaging but remained far below hand performance.

**Dry-electrode penalty.** g.SAHARA dry electrodes matched gel for P300 (Guger et al. 2012, Front Neurosci: 90.7% vs 90.5%), but that is an averaged 0.5–10 Hz task. Our signal lives at 0.1–3 Hz, exactly where contact drift, cable motion and 100 kΩ–1 MΩ impedance hurt most; I assume a 1.5–2.5× higher in-band noise floor, i.e. 20–40% lower attainable r than the gel studies above.

| Regime | Velocity r vs. label (test, trajectory-preserving floor subtracted) | Fitts throughput, unassisted | with learned target prior (shared control) |
|---|---|---|---|
| Overt, eyes free (literature) | 0.3–0.5 — **eye-contaminated; reported only as surrogate ceiling** | n/a | n/a |
| Overt, eye-regressed, fixation-verified | 0.20–0.35 | — | — |
| Attempted (hand still, eyes free, gaze erased) | 0.15–0.30 | 0.10–0.30 bit/s | 0.3–0.6 bit/s |
| Covert, no motor attempt (attention only) | 0.05–0.15 | <0.1 bit/s | 0.1–0.3 bit/s |

Time-to-target in the attempted regime (ID 2–3 bits): 4–8 s unassisted, 2–4 s assisted; expect 20–40% timeouts at 10 s early in training. This is an order of magnitude below a mouse and at SMR-cursor level; the paradigm's case is scaling (Section 3), not day-one performance.

## 3. Scaling behaviour

**Data engine.** Every hour of ordinary computer use yields 72,000 labelled 50-ms steps at zero labelling cost, plus gaze, clicks, scroll, keys and UI-element geometry from the OS accessibility API. 24 participants × 10 h = 240 h of labelled EEG, ≈10× the largest published kinematics dataset.

**Pretraining corpus.** TUH EEG (TUEG, ≈27,000 h), MOABB, plus our own recordings. Candidate encoders: BENDR (Kostas et al. 2021, Front Hum Neurosci; wav2vec-style contrastive on TUEG), BIOT (Yang, Westover & Sun 2023, NeurIPS), LaBraM (Jiang, Zhao & Lu 2024, ICLR; ≈2,500 h, neural tokeniser + masked modelling, 5.8M–369M params), EEGPT (Wang et al. 2024, NeurIPS), Neuro-GPT (Cui et al. 2024), CBraMod (Wang et al. 2025, ICLR; criss-cross attention on TUEG). **Honesty check:** fine-tuned gains over strong from-scratch baselines are typically 1–5 points on classification benchmarks, larger in low-label regimes; none has been evaluated on continuous kinematic regression; LaBraM's 5.8M → 369M scaling bought ≈1–2 points. I expect pretraining to buy *sample efficiency at onboarding*, not a higher ceiling.

**Mechanism and expected curve** (pre-registered as a prediction, test set fixed across the ladder):

- *Hours within subject.* For a linear-Gaussian decoder with d effective dimensions, test r² ≈ r²∞ (1 − d/n); with d_eff ≈ 10³–10⁴ and n ≈ 7×10⁴ per hour the variance term vanishes after 1–3 h and what remains is bias (SNR) and label noise. Expected attempted-regime r(h) ≈ r∞·h/(h+h½) with r∞ ≈ 0.35, h½ ≈ 1.5 h: 1 h → 0.14, 5 h → 0.27, 20 h → 0.33, 100 h → 0.35. **Saturation at ≈0.35 is physics:** 2–5 µV cortical signal against a 5–10 µV rms dry-electrode floor in band, and 32 channels cannot separate the few relevant sources from eye and EMG dipoles the way 64–128 can; more hours fix neither.
- *Subjects.* Cross-subject zero-shot r ≈ 0.10–0.15 after Euclidean alignment (He & Wu 2020); pooled pretraining lowers h½: ≈40% less own data for the same r with 24 subjects, 2–3× less with ≈200, same r∞.
- *Sessions / co-adaptation.* The invasive precedent (ReFIT ≈2× throughput; CLDA converging in 10–20 min) depends on stable high-SNR units. RLNF saw no session-level adaptation; I expect co-adaptation to add 20–50% throughput via the *user* learning a consistent attempted-movement strategy and via assistance priors, not via the decoder. Per-session alignment handles non-stationarity.
- *Label noise.* Mouse velocity is a noisy proxy for intent (overshoots, idle jitter). ReFIT-style relabelling toward the eventual click location is where I expect the largest single jump (+0.05 r, +30% throughput): it removes label noise rather than adding data.

Where the bitter lesson *fails* here: sensor SNR and the overt→covert shift do not scale with data from this cap. The evaluation is designed to measure exactly that.

## 4. Data-collection protocol

**Participants.** N = 24 able-bodied (Phase 1), 12 continuing to Phase 2; no performance-based exclusion; two held-out participants recorded last, never used for model selection.

**Hardware additions.** A screen-mounted eye tracker (Tobii/Gazepoint class, ≥60 Hz, ≈£150–£2,000): the single most important purchase here. Optional 2-channel forearm sEMG on a separate LSL device.

**Phase 1 (overt data engine): 10 sessions × 60 min per participant.**
Each session: (i) 5 min setup: impedance log, 9-point gaze calibration, 2 min eyes-open rest; (ii) 40 min *free use* of ordinary tasks (browsing, email, drawing, a 2-D game, editing) with the real mouse; (iii) 4 × 3 min *identifiability blocks*, Latin-square order:
- **FIX-MOVE**: fixate a central dot (tracker-verified, <1.5°, saccade trials flagged) while moving the cursor to peripheral targets within 8°. Hand moves, eyes do not.
- **LOOK-STILL**: visual search without the mouse. Eyes move, hand does not; correct output is zero velocity.
- **ATTEMPT**: hand resting on a pad (sEMG < 2× rest if available), cursor driven by a replayed target sequence while the person *attempts* the corresponding hand movement; labels are the target trajectory.
- **REPLAY**: the cursor replays a previous trajectory while the person watches. Tests visual-evoked/pursuit leakage; correct output is zero.

**Phase 2 (closed loop): 6 sessions × 60 min, 12 participants.** Each session: 15 min overt re-calibration, then ISO 9241-9 multi-directional Fitts blocks (13 targets, ID 1.5–4 bits, seeded order, 10-s timeout), 6 blocks × 6 min: assistance ∈ {0, 0.3, 0.6} × decoder ∈ {neural, non-neural surrogate}, blind to the participant, Latin-square order, hand resting on the pad throughout. A 5-min fixation-verified block per session tests whether control survives with eyes still.

**Labels.** Mouse velocity from raw HID events (125–1000 Hz) resampled to 20 Hz; the label is the *device*, never a displayed quantity (RLNF rule). Secondary: eventual-click location (ReFIT relabelling), gaze, UI rectangles. No label is a threshold on anything displayed; the only generator-determined sequences (ATTEMPT/REPLAY) are randomised with logged seeds.

**LSL streams and markers.** EEG (32 ch, 500 Hz, impedance every 5 min), Mouse (raw HID), Gaze (native rate + validity), Screen (clicks, scroll, keys, UI-element rectangles at 10 Hz), EMG (optional), Decoder (velocity, gate, latency, model hash, 20 Hz), Cursor (60 Hz), Markers (block/trial/target IDs, seeds). All streams time-corrected via LSL `time_correction` every 5 s.

**Gating.** Per-channel RMS > 5× session median in 1-s windows → channel masked (training uses channel dropout, so masks are in-distribution); FP1/FP2 exempt (not decoder inputs, §5). Sessions with >8 masked channels are retained but flagged.

**Controls and surrogates (pre-registered).**
1. *Eye-only surrogate*: same architecture, inputs = gaze x/y + virtual EOG. Strong in free use; the neural decoder must beat it in FIX-MOVE and ATTEMPT by Δr ≥ 0.10.
2. *Frontal-only surrogate*: FP1/FP2/F7/F8 EEG only.
3. *Autoregressive surrogate*: past cursor velocity only (autocorrelated labels put this floor far above zero).
4. *EMG/temporal surrogate*: T7/T8 + sEMG.
5. Closed-loop *non-neural surrogate*: phase-scrambled EEG from another session through the identical gain/smoothing/assistance chain.

**Pre-registration list.** Endpoints (§6), floors, the r(h) prediction, surrogate margins, gating thresholds, held-out participants, hyperparameters (§5), code hash, and falsifiers: (F1) neural ≤ eye-surrogate + 0.05 in FIX-MOVE; (F2) decoder output in LOOK-STILL or REPLAY correlated with gaze/cursor at r > 0.1; (F3) closed-loop throughput ≤ non-neural surrogate; (F4) r(20 h) ≤ r(2 h) + 0.05 (no scaling).

**Power.** Phase 1 primary: paired (neural − eye-surrogate) FIX-MOVE r; expected Δ = 0.10, SD = 0.12, d = 0.83, one-sided α = 0.05 → 80% power at n = 11; N = 24 gives >0.95, and 80% down to d = 0.55. Phase 2 primary: throughput (neural − surrogate) at a = 0.3; Δ = 0.15 bit/s, SD = 0.15 → n = 12 gives 0.89.

## 5. Decoder and system

**Preprocessing (deterministic, causal).** 500 → 200 Hz (matches LaBraM/CBraMod); 0.1 Hz high-pass + 40 Hz low-pass (causal IIR); bad-channel masking; common-average reference; per-channel robust z-score from the first 2 min; Euclidean alignment by session covariance. **Virtual EOG**: hEOG ≈ F7 − F8, vEOG ≈ (FP1+FP2)/2 − (F3+F4)/2; these four channels are removed from the decoder input and, with gaze, regressed out of the remaining 28 (Schlögl et al. 2007; SGEYESUB subspace, Kobler 2019, fit on the gaze-calibration block).

**Architecture.** Encoder: LaBraM-base-style (or CBraMod): 1-s per-channel patches, temporal conv tokeniser, 12-layer Transformer, d = 200, 8 heads, 5.8M params, pretrained on TUEG + MOABB by masked-token prediction (100 epochs, batch 512, lr 5e-4, mask ratio 0.5). Head: causal GRU (256 hidden, 2-s context, 50-ms hop) → linear to [vx, vy, move-gate logit]. Losses: Huber on velocity (δ = 1 after unit-variance scaling) + BCE on gate + **gaze-erasure**: gradient-reversal adversary (Ganin et al. 2016) predicting gaze velocity from the embedding, λ = 0.3 (LEACE-style linear erasure as deterministic ablation). Fine-tuning: AdamW, lr 5e-5 encoder / 5e-4 head, layer-wise decay 0.65, weight decay 0.05, dropout 0.1, drop-path 0.1, 1-epoch warm-up then cosine, 20 epochs, batch 64, EMA 0.999; augmentation: channel dropout 0.1, amplitude ±20%, ±100 ms jitter, 1/f noise at 10 dB SNR. Seeds fixed; pipeline containerised.

**Splits.** Within subject: sessions 1–8 train, 9 validation, 10 test; identifiability blocks never trained on except ATTEMPT labels in Phase 2. Across subjects: leave-one-subject-out; the two held-out participants are touched once.

**Online latency budget.** Amplifier + LSL ≈ 20 ms; 50-ms hop; encoder+GRU inference ≈ 5 ms on a laptop GPU (≈ 25 ms CPU); velocity smoothing τ = 150 ms; render ≤ 17 ms → ≈ 250 ms perceived lag, tolerable for velocity control.

**Cursor dynamics.** Velocity control (position decoding from EEG is not credible): v_cursor = g · gate · ṽ, g calibrated so the 95th-percentile decoded speed maps to 400 px/s; gate = sigmoid with hysteresis (on > 0.6, off < 0.4) to kill idle drift; assistance a ∈ {0, 0.3, 0.6} blends decoded velocity with a vector toward the nearest UI element / Fitts target from a click-location prior learned by the data engine. Reported throughput is at a = 0 and 0.3; a = 0.6 is logged for usability.

**Co-adaptation rule.** Phase 2: every 60 s, relabel the last window ReFIT-style (direction rotated toward the eventual acquisition point, speed preserved, zero inside the target), then update the head only (encoder frozen online) with a CLDA-style smoothed batch update, step 0.1, forgetting half-life 10 min; full offline fine-tune between sessions. A 3-min probe block per session, decoded by the frozen session-start model, separates decoder drift from user learning.

## 6. Evaluation protocol

**Primary endpoints.** E1 (Phase 1): neural − eye-surrogate velocity r in FIX-MOVE and ATTEMPT (fixed test session, participant-level bootstrap 95% CI, paired test). E2 (Phase 2): Fitts throughput (bit/s, effective ID) neural vs. non-neural surrogate at a = 0.3; co-primary time-to-target and success rate.

**Secondary.** r(h) ladder at 1/2/5/10 h on the fixed test session; cross-subject zero-/few-shot r; decoder output energy in LOOK-STILL and REPLAY (must be ≈0); sEMG-gated ATTEMPT r; per-session probe-block r (drift); NASA-TLX.

**Floors.** Chance for continuous regression is the 95th percentile of 1,000 circular time-shifts of the label (≥ 60 s) and of block-wise permutation at matched n, never r = 0. The autoregressive surrogate accompanies every r.

**Statistics.** Linear mixed models (random intercept per participant), effect sizes with CIs; TOST bounds Δr = 0.05 / Δthroughput = 0.1 bit/s for decoder-variant comparisons; one-sided tests only where pre-registered. Held-out participants reported separately, never pooled.

## 7. Failure modes and debugging

1. **Decoder is an eye tracker.** Signature: frontal-only surrogate ≈ full decoder; collapse in FIX-MOVE. Fix: raise λ, re-verify tracker calibration, exclude more frontal channels, require SGEYESUB residual gaze r < 0.05.
2. **Decoder is a motor-execution detector.** Signature: ATTEMPT r ≪ FIX-MOVE r. Fix: weight ATTEMPT blocks ×5 from the start, add hand-still Phase 2 data; accept the attempted-regime number as the real number.
3. **Visual feedback loop.** Signature: REPLAY output correlated with cursor. Fix: interleaved 200-ms cursor blanking; report blanked-window performance.
4. **Dry-electrode drift.** Signature: r tracks impedance and masked-channel count. Fix: re-seat cap, pre-registered 0.3 Hz high-pass variant, accept lower r∞.
5. **No scaling (F4).** Compare learning-curve slopes for pretrained vs scratch; if flat, publish the null, which is itself the bitter-lesson answer for this cap.
6. **Payoff structure** (RLNF lesson): assistance must never make "do nothing" the best policy; the a = 0 arm is always measured.

## 8. Honest self-score

1. Naturalness **8/10** — the overt data engine is as natural as computing gets; the transfer regime needs an attempted-movement strategy, not instructed imagery but still a strategy.
2. Achievable performance **4/10** — 0.1–0.3 bit/s unassisted in the attempted regime on a dry cap is SMR-cursor level and an order of magnitude below a mouse.
3. Scaling **8/10** — the only paradigm with a free dense label and a falsifiable r(h) curve; penalised because I predict saturation near r ≈ 0.35 for physics reasons.
4. Validity **8/10** — eye tracker, virtual EOG, gaze-erasure, four identifiability blocks, five surrogates and four falsifiers make eye vs cortex identifiable by design; residual risk is EMG and the visual loop.
5. Feasibility **6/10** — needs an eye tracker, a GPU and 240 h of recording; latency is fine and per-session calibration is 15 min.
6. Rigour **8/10** — fixed test session across the ladder, trajectory-preserving floors, held-out participants, explicit hyperparameters, pre-registered falsifiers.

## 9. Pitch

Every other paradigm hand-designs a signal; this one lets 240 hours of ordinary computer use, labelled by the mouse itself, teach a pretrained EEG sequence model what cursor intent looks like in cortex. Because the EEG kinematics literature is mostly eye movements, the design exists to prove the decoder is not an eye tracker: gaze erasure, virtual EOG, fixation-verified and hand-still blocks, and surrogates that must fail where the brain decoder must succeed. The payoff is a system that improves with every hour and every user, with ReFIT relabelling and CLDA updates as the closed-loop engine. I predict honestly that 32 dry electrodes saturate near r ≈ 0.35 and 0.1–0.3 bit/s unassisted, with shared control doubling that. If the pre-registered scaling curve is flat, that null is the most valuable result this hardware can produce, measured correctly.
