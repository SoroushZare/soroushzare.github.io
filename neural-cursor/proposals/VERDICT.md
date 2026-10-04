# VERDICT — Continuous 2-D cursor control on a 32-channel dry g.tec cap

Scores re-derived from the rubric (weights 25/20/20/15/10/10), self-scores ignored; cells differing from the self-score by ≥2 carry a justification.

## 1. Scoring table

| Proposal | Natural [25] | Perf [20] | Scaling [20] | Validity [15] | Feasib [10] | Rigour [10] | **Weighted** |
|---|---|---|---|---|---|---|---|
| A SSVEP/c-VEP | 3 | 6 | 4 | 7 | 6 | 8 | **5.20** |
| B P300 | 2 | 5 | 4 | 8 | 7 | 8 | **5.00** |
| C MI/SMR | 3 | 5 | 6 | 6 | 5 | 8 | **5.15** |
| D Covert attention | 6 (self 9: must suppress saccades toward the target; "scrolling world" validation mode is not free cursor use) | 2 | 5 | 8 | 5 | 8 | **5.40** |
| E Shared autonomy ErrP | 6 (self 8: user is evaluator, not driver; cannot "move it around") | 3 | 6 (self 8: only the habit prior keeps scaling; dry ErrP plateau 0.75–0.78 is above anything RLNF saw on this cap) | 7 (self 9: display-only surrogate omits turn-angle Δθ, see §2) | 7 | 8 | **5.85** |
| F End-to-end | 4 (self 8: "attempted hand movement" is instructed hand imagery in all but name for an able-bodied user; the natural covert regime is <0.1 bit/s) | 3 | 7 | 8 | 5 | 8 | **5.50** |
| G EOG hybrid | 7 | 5 (self 7: 1.0–1.8 bits/s cap-only has no literature anchor; periorbital EOG anchors support 0.5–1.2) | 6 | 8 | 7 | 8 | **6.65** |

Other ≥2 deviations: none. Validity for A is 7 not 8 because a frontal-only surrogate cannot resolve eight wedges inside 2.5°, so falsifier F1 is nearly unfalsifiable and no eye tracker is mandated.

## 2. Factual problems per proposal

**A.** (i) Nagel & Spüler: the ~700 bits/min EEG-only speller is Sci Rep 2019; 1237 bits/min with eye tracking is PLoS ONE 2020 — two papers merged into one wrong venue/year. (ii) "32–40 Hz is at or beyond critical flicker fusion for most observers" is an overclaim; photopic CFF is ~50–60 Hz, and Ladouce 2022 shows reduced, not absent, perception. (iii) "Guger et al. 2012 … 53 subjects" is the SSVEP-demographics paper, distinct from the dry-electrode P300 paper cited under the same name in the same bullet. (iv) Re-seating four pins at PO sites assumes a free-hole cap; the g.Nautilus-32 montage may not permit it. (v) Gaze-vs-covert is left to an optional eye tracker. The 0.3–0.7 bits/s range is honest.

**B.** (i) Guger et al. 2009 tested 81 subjects, not 100 (the 72.8 %/≈89 % figures are right). (ii) 8-class Wolpaw bits at 80 % = 1.72, not 1.60. (iii) The "6 bits in 2 commits" zoom throughput conflates selection entropy with Fitts ID; 0.25–0.40 bits/s is inflated, 0.2–0.3 is the defensible range. (iv) Guger 2012 dry-vs-gel means are hedged ("verify") — acceptable. Citations are otherwise accurate; the covert/passive/sham blocks are the best gaze audit among the stimulus paradigms.

**C.** (i) "Laplacian + 8–30 Hz removes most ocular power" ignores saccadic spike potentials and microsaccade broadband energy (Yuval-Greenberg 2008); the eye tracker is only "recommended". (ii) Felton 2009 throughput 0.2–0.6 bit/s and Edelman 2019 "+60 %" are plausible but not precisely stated — treat as approximate. (iii) The protocol violates "no instructed imagery" for 5 sessions by design; the withdrawal test is a genuine endpoint, not a mitigation. (iv) The dry-penalty "simulation" is explicitly an assumption. (v) Assumes FC/CP neighbours exist in the 32-ch layout.

**D.** (i) Okazaki 2015 and Bagherzadeh 2020 are MEG neurofeedback studies; transfer to dry EEG under occipital hair is assumed. (ii) Mathewson 2017 used active dry electrodes on hairless/central sites, not pins through occipital hair; the "alpha survives dry" argument is extrapolated. (iii) Naturalness is overclaimed: the user must fixate the cursor and *not* look at the target for 15–60 s per target. (iv) Applied mode (cursor moves, gaze follows) reintroduces pursuit EOG, acknowledged but unsolved. Numbers (0.05 bit/s ITR, 55.3 % floor at n = 240) check out.

**E.** (i) The simulation assumes i.i.d. ErrP accuracy p independent of error magnitude; Omedes 2015 and Spüler & Niethammer 2015 show gradual/small errors evoke smaller, later ErrPs, so segments at 60–90° will be decoded worse than 180° reversals, and the 5–10 % label-noise budget is optimistic. (ii) Missing control: the display-only surrogate S3 lists direction, position, time and posterior but not the turn angle Δθ between consecutive segments; under a posterior-driven policy large turns are more often "errors" and evoke visual motion-onset ERPs — S3 must include Δθ or an ErrP claim is not separable from a visual-onset response. (iii) Planning value p = 0.70 and plateau 0.75–0.78 on dry FCz exceed RLNF's 0.66–0.69 on the same hardware; the pre-declared failure point (p < 0.62) is the right safeguard. (iv) The user never steers; this fails "continuously move it around" in spirit (admitted). Citations (Ferrez & Millán 2008, Chavarriaga 2010, Iturrate 2015, Javdani 2015/2018, Zander 2016) are correct.

**F.** (i) Attempted hand movement is an instructed motor strategy; the proposal under-acknowledges the constraint violation. (ii) 240 h of mouse-labelled free use is, per its own Kobler 2018 citation, mostly gaze; the gradient-reversal gaze adversary can erase genuine cortical oculomotor-coupled signal too, and no estimate of that loss is given. (iii) The d_eff/variance scaling argument is linear-Gaussian and does not apply to a 5.8M-parameter transformer; the SNR saturation argument is mechanistic, the pretraining gain story is not. (iv) "10–20 µV/°" is periorbital; frontal cap sites see ~5–10 µV/°. (v) Guger 2012 "90.7 vs 90.5 %" should be verified. Kinematics-decoding citations (Bradberry 2010, Antelis 2013, Mondini 2020, Pulferer 2022, Ofner 2017/2019) are accurate and honestly read.

**G.** (i) 1.0–1.8 bits/s cap-only is extrapolated from EEGEyeNet (128-ch gel, 356 subjects) and Belkacem 2015 (discrete 4–8 directions); the only pointing-type anchors are periorbital EOG (Barea 2002 ≈2°, Usakli 2010 ≈0.5–0.9 bits/s), which support 0.5–1.2 bits/s on dry frontal pins. (ii) Closed-loop drift correction "re-centres on each click" — but in EOG-driven use selections are dwells at the decoder's own estimate, so re-centring is self-referential; drift is bounded only by the eye-tracker teacher or EOG pair. (iii) Requires DC/≤0.01 Hz acquisition on dry pins where polarisation drift is largest; the fallback (saccade integration) is honest but untested. (iv) It is not brain control, stated plainly; the EEG tiers are correctly treated as increments with sham controls. Citations (Croft & Barry 2000, Niazi 2011, Belkacem 2015, Ma 2015, Kastrati 2021, Zhang & MacKenzie 2007) are correct.

## 3. Ranking and winner

**G 6.65 > E 5.85 > F 5.50 > D 5.40 > A 5.20 > C 5.15 > B 5.00.**

**Winner: G (EOG hybrid).** On the rubric as written it is the only proposal that meets all four user constraints (no stimulation, no imagery, continuous, this cap) *and* delivers throughput above 0.5 bits/s with controls that would survive the RLNF post-mortem. A and B are the fastest decoders in the room but fail criterion 1 by construction; C fails it for five sessions and 20–30 % of users never gain control. D is the most natural cortical paradigm but 0.05 bit/s is below usable even with assistance. E has the best label structure and the best scaling story, but the person does not steer and 20–35 s per target is RLNF-grade utility. F has the best identifiability design and the most mechanistic scaling argument, but its natural regime is the weakest signal of all. G wins because it makes the confound every other proposal fights into the controller, measures the cortical contribution as an increment, and is the only design that produces a cursor the user will actually move on day one. Recorded plainly: G is not a brain-computer interface and a €150 eye tracker beats its backbone by 2–3×; the user must decide whether "neural cursor" means cortex or "no stimulation, no imagery, from this cap".

## 4. Recommended composite

**"Gaze-anchored shared-autonomy cursor with an audited cortical term."** All elements are stimulus-free and imagery-free.

1. **Pointing backbone — from G.** Cap-EOG gaze model (FP1/FP2/F7/F8 ridge → CNN pretrained on EEGEyeNet, Kalman fusion of saccade displacement and slow position), position control, blink freeze, Tier-0 engagement gate. A consumer eye tracker is **mandatory** (D, E, F all require it) as teacher, drift anchor and attribution instrument — it fixes G's self-referential re-centring (§2 G-ii).
2. **Shared-autonomy layer — from E, reshaped.** A Bayesian posterior over selectable on-screen elements, updated every 250 ms by the *fused velocity vector* (not by an ErrP), blended into the cursor as assistance a ∈ {0, 0.3, 0.6} (D/F style). The user remains the driver; the agent never sees the goal; E's ε-probe positivity audit is kept. Mechanism: the orchestrator's simulation — decoder-only success 0.49 at r ≈ 0.40 vs 0.78 with the posterior, surrogates at 0.00–0.01, random 0.03 — i.e. the posterior roughly doubles usable control from a weak continuous decoder without reading the goal. This replaces E's 1-s ErrP channel (0.12 bits/decision) with a dense evidence stream and discards RLNF's outcome-locked ErrP entirely.
3. **Cortical velocity term — from D and F.** D's IAF-band parieto-occipital alpha lateralisation (tangent-space, Riemannian recentring) plus F's SGEYESUB/gaze-erased 0.1–3 Hz features, regressed onto target direction with F's ReFIT-style relabelling toward the eventual acquisition point. No instruction beyond "move the cursor". Its weight in the fused vector is learned, logged, and reported as a measured increment, never assumed.
4. **Click — from G Tier 1.** Self-paced MRCP/ERD press-intent decoder shortening dwell, with G's sham-click and kinematic surrogates. No ErrP veto (RLNF).
5. **Data engine — from F and G.** Clicks and eye-tracker samples as labels in ordinary use; nightly retraining; frozen final session.
6. **Identifiability and evaluation — from F, C, E, G.** F's FIX-MOVE / LOOK-STILL / REPLAY blocks and C's memory-target trials; E's time-shuffled-output surrogate and display-only surrogate *with turn angle added*; G's EOG-only / EEG-only / combined attribution split; permutation and circular-shift floors at matched n; a = 0 arm always measured (F); no exclusions; 4 held-out participants.

**Single biggest risk.** After gaze erasure the cortical term contributes nothing and the system is a mediocre eye tracker plus autocomplete; compounded by dry frontal DC drift pushing the backbone below usability.

**Pre-registered falsifier.** On fixation-verified epochs (tracker < 1.5°) in the frozen test session, (EEG-only + posterior) throughput minus (ocular-only + posterior) throughput is TOST-equivalent to zero (±0.05 bits/s) **and** EEG-only velocity r ≤ circular-shift floor + 0.05 → the cortical term is removed and the product is declared "eye tracker + shared autonomy". Secondary kill: cap-only backbone throughput < 0.6 bits/s on the test session (G's F1) → backbone replaced by the eye tracker.

## 5. Expected realistic performance on this hardware

| Configuration | Fitts throughput | Time-to-target (ID 2–3 bits) | Anchors |
|---|---|---|---|
| Full composite (cap-EOG backbone + posterior a = 0.3 + MRCP click), 8–13 targets | **0.8–1.5 bits/s** | **1.5–4 s**, success ≥ 90 % at 15 s | Usakli 2010 EOG ≈0.5–0.9 bits/s; Barea 2002 ≈2°; Belkacem 2015 ≈90 % cap-site directions; G's 2× dry penalty; Zhang & MacKenzie 2007 gaze 3–4 bits/s as ceiling |
| Cortical-only arm (eyes fixation-verified), with posterior | **0.1–0.3 bits/s** | **5–15 s**, success 0.4–0.7 on 8 targets at 15 s | Orchestrator sim 0.78 at r 0.40; expected r on this cap 0.2–0.3 (Tonin 2013 ≈70 % L/R; Mondini 2020 online r ≈0.3; Pulferer 2022 r 0.2–0.4), so re-run the sim at r = 0.2/0.3 and pre-register those cells |
| Cortical-only, unassisted | 0.03–0.15 bits/s | 10–40 s | D's ITR 0.05 bit/s; F's attempted regime 0.1–0.3 is unavailable without instructed movement |
| Reference | Wolpaw & McFarland 2004 gel SMR 0.3–0.5; mouse 4–5 | | |

Headline: on this cap a usable, natural, stimulus-free cursor at ~1 bit/s exists only because the eye does the pointing; the brain adds a small, measurable increment, and the design is built to say exactly how small.
