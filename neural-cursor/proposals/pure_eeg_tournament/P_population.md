# P_population: one generic EEG decoder trained on many people, used by a new person with no labelled calibration

## 1. One-line idea and signal basis

**Idea.** We apply the Kaifosh et al. 2025 (Nature) sEMG approach to EEG. Many people each give one standardized 75-min session. One generic decoder is pretrained self-supervised on public EEG and then trained across all participants. A new person then steers on first contact with zero labelled calibration. The headline result is a cross-participant scaling law for closed-loop EEG control, measured with purity auditing.

**Signal.** The decoder reads the cross-person part of movement-intention activity: lateralized mu/beta (8–30 Hz) desynchronization and slow premotor preparation that come with *intending* a hidden cursor to move. There is no stimulus and no imagery script. Participants are told only "make the hidden cursor go there", and their self-reported strategy is logged and used to stratify analysis. Honestly, most people will fall back on hand-like imagery. That is acceptable because it is self-chosen, not trained.

**What Kaifosh shows (verified) and how much transfers.** A wristband decoder trained on thousands of participants worked for new people with no calibration: 0.66 target acquisitions/s, 0.88 gesture detections/s, 20.9 words/min handwriting. Performance followed a power law in data and model size, and personalization added about 16%. sEMG is a high-SNR peripheral signal; EEG is not. The open question is whether the **participant axis** of that scaling law exists for scalp EEG at all, and where it saturates.

## 2. Label source: never display-identifiable

I tested the trap first (§10). On ds004362 a 7-channel frontal low-frequency **eye proxy decodes imagined left/right better (0.73) than any cortical model (0.60–0.62)**, and it also improves with N. A naive population model on cued MI data would produce a scaling law for eye movements. So labels come from elsewhere:

- **Dark-intent trials (training labels).** A 250-ms central cue shows one of four arbitrary, luminance-matched glyphs; the glyph→direction mapping is re-randomized every 16-trial block. After a 1.0-s blank delay comes a **3-s intent window** with only a fixation dot and an invisible cursor. Labels use only 1.25–4.25 s after cue onset, after the visual evoked response. Within the window the pixels are identical across classes, and across blocks glyph identity carries zero information about the label, so a pixel decoder is at chance by design (checked empirically as S3, §5).
- **Free-choice, report-after trials (gold test labels).** There is no cue. The participant picks a direction, intends it for 3 s, and then reports it by keypress *after* the window closes. During the window the label exists only in the participant's head. A decoder trained on cued trials and tested on free-choice trials cannot be reading cue processing.
- **Closed-loop data is never used for training.** In closed loop, the visible cursor is the decoder's own output, so feedback is circular. Closed-loop blocks are used only for evaluation (§5).

## 3. Data engine, scaling mechanism, expected curve

**Mechanism.** In EEG, between-person variance (anatomy, cap fit, strategy) is larger than within-person variance. Pooling N people lets the model learn the invariant part. Unsupervised Riemannian re-centring on each person's own unlabeled data removes the per-person offset. Larger models can keep extracting more from larger N; Kaifosh reports that bigger models benefited more from more data.

**Evidence (I ran it on ds004362; §10).** A linear broadband model flattens by N≈16–32 at about 0.60; a 4-band filter-bank model is still rising (0.596 → 0.609 → 0.619 at N = 8/32/76); more than about 20 trials per person adds nothing. So the participant axis moves, the per-person axis does not, and the limit is model capacity. The PI's "bitter lesson" holds only partly: volume helps when it is volume across **people** with a large enough model.

**Data volume.**
- **Population arm:** 160 participants × one session, giving 288 cued + 64 free-choice intent windows each. That is 56k windows, about 47 h of labelled intent EEG and about 160 h of total recording.
- **Depth arm:** 4 participants × 50 h, to measure the personalization curve on top of the generic prior.
- **Self-supervised pretraining:** CBraMod-style masked-patch reconstruction (ICLR 2025) on TUEG, overlapping 10-20 channels (LaBraM used about 2,500 h). Foundation models give modest gains, so I expect +1–3 points.
- **Public supervised data:** ds004362 (106 usable), Stieger 2021 Sci Data (62 participants, about 600 h, >250k trials), Lee 2019 and Cho 2017 GigaScience (54, 52). These labels are display-identifiable, so they are used **only** for 8–30 Hz central-channel pretraining and never enter the purity evidence.

**Expected curve, fit as acc(N) = a − b·N^(−c) on a fixed test cohort.**

| N train | 2-class zero-shot (dark-intent, L/R) | 4-direction cosine-r of decoded velocity |
|---|---|---|
| 10 | 0.60 | 0.10 |
| 40 | 0.64 | 0.18 |
| 160 | 0.67–0.70 | 0.22–0.30 |

Saturation is declared when doubling N adds less than 0.5 points. I expect this between N = 100 and 300 on a dry cap. The 8× more trials per person and the cleaner labels compared with ds004362 justify the higher plateau, but that is an assumption.

## 4. Expected performance (honest)

- **Zero-shot alone:** 2-class 0.67 per 2-s decision is 0.085 bits, about **0.04 bits/s**, the low end of the established 0.03–0.3 bits/s (ds004362 zero-shot today: 0.62 per 3 s, about 0.014 bits/s).
- **With the shared-autonomy goal posterior already simulated in this repo:** a decoder at r≈0.4 reached 8 targets at 0.49 alone and about 0.8 with the posterior. At the r≈0.25 I expect at N=160, I project **success ≈ 0.6–0.7 with the posterior and time-to-target 15–25 s.** This is an extrapolation and will be re-simulated before pre-registration.
- **Personalized depth arm** (generic prior plus 30–60 min of own data): +3–6 points, roughly r≈0.35–0.4, where the repo simulation gives about 0.8 success with the posterior.
- **Anchors:** Wolpaw & McFarland 2004 and McFarland 2010 reached 2-D/3-D SMR control only after weeks of per-user training; Edelman 2019 did continuous robotic-arm pursuit with per-user training. Nothing comparable exists at zero calibration.

## 5. Purity proof

**Auditors (recorded, never decoder input):** eye tracker at ≥250 Hz; bipolar HEOG/VEOG; EMG on bilateral forearm flexor/extensor, splenius and masseter; accelerometers on the head and both wrists.

**Matched surrogates** (same architecture, trials, folds and **N-ladder**, so every headline point has a surrogate beside it): **S1** auditor-only; **S2** edge-EEG only (Fp1, Fp2, F7, F8, T7, T8); **S3** display/pixel model on rendered frames and glyph sequence; **S4** label permutation at matched n; **S5** circular-shift floor for continuous outputs.

**Claims that must all hold:**
1. The EEG curve is above S1–S5 at every N.
2. The **slope** in log N of the EEG curve is above the S1/S2 slope (on ds004362 the eye proxy has a slope too).
3. A central-only model (FC1/2/5/6, C3/Cz/C4, CP1/2/5/6) keeps ≥80% of above-chance accuracy.
4. Performance survives regressing all auditor channels out of the EEG.
5. Partial correlation of decoder output with the label, controlling for S1 output, stays significant. On ds004362 the corresponding cortical/eye-proxy correlation is already only r = 0.14.
6. Free-choice (report-after) accuracy is above the permutation 95th percentile.

**Closed-loop circularity controls:**
- **Hidden-cursor probes:** the cursor vanishes for 2 s in 25% of trials; decoded velocity must not fall.
- **Replay-without-intent:** the participant watches their own earlier trajectories while counting backwards; decoded-vs-replayed velocity correlation must sit at the S5 floor.
- **Memory-guided targets** (300-ms flash, then hidden; gaze free but audited), with metrics also reported on gaze-stable segments (<1.5° drift).

**Analysis rules:** no performance-based exclusion (technical criteria, computed before any decoding, are pre-registered); a fixed 30-person test cohort is frozen before the ladder is fitted.

## 6. Protocol

**75-min session.**

| Minutes | Activity |
|---|---|
| 10 | Dry cap and auditors |
| 3 | Rest (90 s eyes open, 90 s eyes closed), used only for unsupervised re-centring |
| 31 | 6 × 48 cued dark-intent trials (5.75 s per trial) |
| 7 | 2 × 32 free-choice report-after trials |
| 3 | Replay/no-intent |
| 10 | Zero-shot closed loop: 8 memory-guided targets, current population snapshot, shared autonomy a ∈ {0, 0.5}, hidden-cursor probes |
| 5 | Strategy questionnaire |

**Cohort.**
- 160 participants on 2 rigs, 2 sessions per rig per day, 4 days per week, which is about 16 per week. Recording takes 10–12 weeks plus recruitment slack. The fallback is 100 participants.
- Snapshot models are frozen at N = 10, 20, 40, 80 and 160. **Every new arrival is tested zero-shot with the snapshot trained on everyone before them**, so the closed-loop scaling curve is built prospectively.
- The 30 test-cohort members return for a second 60-min session and run all 5 snapshots in blinded counterbalanced order.

**Depth arm.** 4 participants × 40 sessions (about 50 h each).

**Optional gel arm.** 20 cohort members repeat with 64-ch gel for dry-to-gel transfer and source localization (cost to be quoted).

**Timeline:** months 1–3 build, 10-person pilot, pre-registration; 3–9 cohort; 9–14 depth arm and robot demo; 14–18 analysis and paper.

## 7. Real-time architecture

- **Input and preprocessing:** LSL 500 Hz, causal 1–40 Hz band-pass, decimation to 250 Hz, common average reference. A 2-s window slides with a 100-ms hop.
- **Unsupervised alignment:** Riemannian re-centring of the covariance stream to the person's 3-min rest reference, plus an exponential running mean with a 5-min half-life to absorb dry-electrode gain drift (the repo simulation showed that a frozen decoder collapses without this). No labels are used. "Zero calibration" means zero *labelled* calibration, and the 3-min rest is disclosed.
- **Model:** criss-cross spatial/temporal transformer encoder (32 channels × 1-s patches, 8 layers, d=256, about 8M parameters, masked-reconstruction pretrained). Head 1 gives a 2-D intended-direction vector (cosine loss); head 2 gives intent-present probability (BCE against no-intent and replay blocks). Training: AdamW, lr 3e-4, 40 epochs, subject-grouped validation. A filter-bank tangent-space linear model is the baseline and fallback.
- **Latency:** under 30 ms of compute on a laptop GPU, about 120 ms end-to-end including the window centre lag.
- **Adaptation:** in the generic arm, re-centring only. In the depth arm, fine-tune the last 2 layers plus head on the person's own dark-intent trials after 10, 30 and 60 min, and report the personalization curve in the same format as Kaifosh.
- **Effector:** the decoder emits (v, confidence) at 10 Hz. A Bayesian goal posterior over candidate goals (it never reads the true goal) blends v with assistance a. For the cursor, v becomes screen velocity; for the robot, the same v becomes planar end-effector velocity (≤5 cm/s) over objects from a tabletop detector. The decoder never sees the effector type, which is the generality claim.

## 8. The Nature claim

**Headline figure:**
- **(a)** Closed-loop zero-shot success on 30 frozen test people versus N (log axis). The EEG curve is shown with S1/S2/S3/permutation curves at matched N, with a power-law fit and a saturation point.
- **(b)** Free-choice report-after accuracy versus N.
- **(c)** Personalization gain versus size of the generic prior.
- **(d)** The same frozen decoder driving the cursor and the robot.

**One sentence:** *A single scalp-EEG decoder trained on 160 people lets a new person steer a cursor or robot by intention on first contact, and its performance follows a measured cross-participant power law that eye, muscle and display surrogates do not follow.*

**Why it is new:** every EEG cursor or robot result so far is calibrated or trained per user; EEG foundation models report offline benchmarks on display-cued labels; no one has published a purity-audited **closed-loop** scaling law over participants.

If the curve saturates early, the result is still new: the first measured ceiling on "more people" for non-invasive brain control, a quantitative answer to the PI's hypothesis (Nature Communications level).

## 9. Falsifier and biggest risk

**Pre-registered falsifiers. Any one kills the population claim on this hardware:**
1. On the frozen test cohort, closed-loop success at N=160 minus N=20 (a=0.5) is under 0.10, or the bootstrap 95% CI includes 0.
2. The offline EEG slope in log N is not greater than the S1 or S2 slope (TOST equivalence, ±0.01 per doubling).
3. Free-choice report-after accuracy is at or below the permutation 95th percentile.
4. Hidden-cursor probe velocity falls by more than 50% relative to visible periods (circularity).

**Biggest risk.** The cross-person intent signal on a dry cap may plateau too low. ds004362 already shows linear imagery decoding flattening at about 0.60 by N≈16–32, and zero-shot accuracy equal to within-person calibration (0.597). A transformer may only buy +2–4 points, which would leave the system around 0.03–0.05 bits/s, usable only with heavy shared autonomy.

**Second risk.** Dark-intent training (hidden cursor, fixation) is less natural than visible control and may transfer imperfectly to visible closed loop; the hidden-cursor probes measure that gap.

## 10. What can be tested NOW on ds004362 — already run

I ran this in memory on the local copy at `scratchpad/data/ds004362`, with no files written.

**Setup:** 106 subjects (S088, S092, S100 excluded technically: sampling rate ≠160 Hz); fixed test set of 30, training pool of 76; 45 trials per subject; window 0.5–3.5 s after cue; tangent-space logistic regression (C = 0.01) with per-subject unsupervised re-centring; N<76 is the mean of 6 draws (SD ≤ 0.03). Chance is 0.5; a test subject is individually significant at ≥ 29/45 (0.644).

**Results (accuracy at N = 4 / 8 / 16 / 32 / 76):**

| Model (imagined L/R) | 4 | 8 | 16 | 32 | 76 |
|---|---|---|---|---|---|
| 32-ch 10-20 subset, 8–30 Hz | .560 | .578 | .588 | .599 | .599 |
| Central 17-ch, 8–30 Hz | .566 | .598 | .588 | .598 | .595 |
| Central 17-ch, 4-band filter bank | – | .596 | – | .609 | .619 |
| Posterior 13-ch, 8–30 Hz | .532 | .531 | .538 | .539 | .546 |
| **Eye proxy** (Fp1/Fpz/Fp2/AF7/AF8/F7/F8, 0.3–4 Hz) | .656 | .691 | .707 | .728 | **.730** |
| Label-shuffled floor (EEG / eye proxy) | | | | | .504 / .470 |

**Other results:**
- **Individually significant test subjects (32-ch):** 3% at N=4, rising to 43% at N=76.
- **Personalized baseline:** within-subject 5-fold (about 36 training trials) gives 0.597. **Zero-shot equals per-person calibration**: the Kaifosh pattern at a much lower level.
- **Re-centring:** adds about 1 point (0.589 → 0.599).
- **Training trials per person (N=76):** 10/20/30/45 trials give .586/.599/.602/.595. The per-person axis is flat.
- **Adding executed-movement trials to training:** 0.606 central, 0.622 filter bank.
- **Executed L/R, 32-ch:** .590 → .676, still rising at N=76. The eye proxy for executed L/R reaches .739.
- **Independence from eyes:** the correlation between central and eye-proxy decoder outputs is 0.14. On the third of trials where the eye proxy is least confident, the central decoder still reaches 0.618. A combined model weights the eye proxy 1.17 against 0.36 for central EEG. **An unconstrained pooled decoder would mostly read eyes**, so surrogate curves on the same N-ladder are mandatory, not optional.

**Next tests (1–2 days, no new data):** paired subject-level bootstrap on the filter-bank N=8 → 76 gain (+0.023, about 1.5 nominal SE, not yet significant); EEGNet/CBraMod fine-tuning on central channels across the same ladder; pooling Stieger 2021 and Lee 2019 to reach N≈250 with the eye-proxy curve alongside; power-law fit with bootstrap CI on the asymptote.

**Kill condition now:** if a high-capacity central-channel model at N≈250 does not beat N=16 by ≥3 points (CI excluding 0), the participant axis is flat for scalp EEG, and the population approach should give way to a deep per-person design.
