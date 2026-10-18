# Interface addendum: the champion design (modern SMR, body coordinates, certified purity)

Source of truth for the science: `proposals/pure_eeg_tournament/P_smr_modern_FINAL.md` (copied there by the
orchestrator) and the skeptic corrections in `proposals/pure_eeg_tournament/SKEPTIC_CORRECTIONS.md`, which
override the proposal where they disagree.  Everything in `INTERFACES.md` still holds; this file adds.
Signatures below are the code's; keyword arguments with defaults may be added, never removed.

Pure-EEG rule (unchanged): only EEG reaches decoders, trainers, controllers, effectors.

## Body coordinates
Classes: FEET, HANDS (both), LEFT, RIGHT.  Velocity in body coordinates: y = feet (+) vs hands (−),
x = right (+) vs left (−).  Diagonals are two-part cues.  The screen mapping M (2×2, orthogonal) turns
body velocity into screen velocity: identity for normal use; held-out test mappings rot+90, rot−90,
mirror_x, mirror_y (never trained on).  Central-11 montage on the g.tec 32: FC5, FC1, FC2, FC6, C3, Cz, C4,
CP5, CP1, CP2, CP6 (`body_decoder.CENTRAL11`, the one definition).  Every champion decoder uses **1-s
windows** and the design's **16-Hz** loop: build the labeler and the real-time loop with
`BodyCoordinateDecoder.loop_config(cfg)` (window 1 s, `task.control_hz` = `body_decoder.LOOP_HZ` 16; the
default config's 0.5-s windows are refused, not silently decoded).

## Label sources (provenance field "source")
- "L1"  fixation probe: central glyph (shown 0.5 s) cues a class, glyph→class map redrawn every session,
        gaze-contingent fixation; windows 1.0–4.0 s after cue onset.  LABEL = cued body vector.  Classes
        FEET, HANDS, LEFT, RIGHT only: DESIGN_PURE_EEG §4's REST probes (20 %) and sel/eval probe tagging
        are not implemented (deviation; the idle gate trains on L0 rest instead, and the simulator now
        adds a post-imagery beta rebound so that rest is not artificially clean).
- "L2b" blind closed loop: cue = central glyph or body-part word (≈ 1 s; never a location), cursor hidden,
        hidden goal, label-symmetric progress tone (pitch encodes distance decrease, identical for every
        direction); windows start 1.5 s after cue onset and end ≤ 4.0 s after it (`data_engine.CUE_LABEL_END_S`,
        DESIGN §4; re-cues, once they exist, restart it) and no later than the first logged hidden-cursor
        row inside the dwell radius of the goal — always before the central outcome flash.  LABEL = cued
        body vector.  y-only classes until x is earned.
  Both: windows start ≥ `CUE_LOCKOUT_S` (1.0 s) after cue ONSET and ≥ `CUE_OFFSET_LOCKOUT_S` (0.5 s) after
  cue OFFSET, so the class-specific offset response of the glyph/word is never in a training window
  (`parse_trials` enforces both for L1, L2b and blind J; the timing classes refuse a span that breaks them).
- "S"   sighted closed loop, identity mapping: NO LABEL for the deployed decoder (re-centring only).
        The SHADOW decoder (positive control, never deployed) trains on S with ReFIT labels
        u = (target − cursor)/‖·‖ mapped to body coordinates through M⁻¹.
- "H"   held-out mapping test blocks (rot±90, mirrors; C2b open loop, C2c closed loop): test only.
- "R"   pixel replay test trials (15 % of S and C2c trials): the screen replays a past trajectory toward
        goal A while the person holds goal B: test only.  A keyed `body_report` after the trial gives the
        held intent; certification keeps only trials whose report confirms it (`certify.confirmed_by_report`;
        the C2b report likewise).
- "J"   "just intend" (uninstructed, identity, frozen decoder; half blind, half sighted): scored only.
- "L0"  rest (setup rest, rest between fixation probes; nothing cued, nothing spatial drawn): trains only
        the idle gate (`BodyCoordinateDecoder.fit_idle`, `body_decoder.IDLE_SOURCE`); `fit` refuses it.
        Not yet emitted by `protocol` / `data_engine` (no L0 trial kind exists): callers build L0 windows
        from the setup rest themselves.
Trainers refuse S/H/R/J windows for the deployed decoder (`ValueError`, tested), and `BodyTrainer` also
refuses any window whose `display_goal` is set; the shadow refuses all but S.  Public data (ds004362, Stieger) carry display-cued, display-identifiable labels, so they can only
ever train a never-deployed shadow / reference model (as in DESIGN_PURE_EEG.md), never the deployed
decoder's prior: the only deployable prior is other people's L1 probes (as the simulation builds it).
The proposal's §3(a) "used from day 1" is superseded on this point.  Shared constants: `protocol.SOURCES`, `DEPLOYED_TRAINING_SOURCES`, `TEST_ONLY_SOURCES`,
`SOURCES_BY_KIND`, `BLANK_CUE_KINDS`; `data_engine.DEPLOYED_SOURCES`, `SHADOW_SOURCES`, `SOURCES_BY_BLOCK`,
`SELECTION_SOURCE` ("L1"); `body_decoder.TRAINABLE_SOURCES`, `IDLE_SOURCE`; `purity.CERTIFYING_BLOCKS`,
`CERTIFYING_SOURCES`.

## Block kinds, sources and provenance fields (one table)
| Block kind (`protocol`, = provenance `block`) | source | label (`labels` / `body_label`) | `display_goal` | role |
|---|---|---|---|---|
| setup, rest | — | never labelled | — | re-centring reference (setup rest), breaks |
| cued_blank (original) | — | start → cued goal | None | original training / held-out test; certifies |
| closed_loop, blank_feedback, free (original) | — | goal − displayed cursor (start_pos if hidden) | goal | original training |
| fixation_probe | L1 | cued body vector / `body_label` | None | deployed training, selection probes (promotion), x earning; certifies |
| blind_closed_loop | L2b (J if just intend) | cued body vector / `body_label` (J: start → goal) | None | deployed training (L2b); J scored only; L2b certifies |
| sighted | S (J if just intend) | goal − displayed cursor; shadow: ReFIT M⁻¹u | goal | shadow training, practice; J scored only |
| heldout_mapping | H | goal − displayed cursor (screen) | goal | certification (P2) only |
| pixel_replay (trial-level, inside S / C2c) | R | goal B − displayed cursor | goal B (`replay_goal` = A) | certification (P3) only |

Provenance keys per window (`data_engine.Labeler`): always `block`, `feedback_visible`, `session`, `t`,
`trial`, `display_goal`; champion windows add `source`, `mapping`, `instructed`; L1 / L2b add
`body_label`, `cue_class`, `cue_mode`; visible trials add `cursor_pos`, `cursor_pos_start`; H adds
`variant`, `body_report`; R adds `replay_goal`, `body_report`.  Who reads what: trainers read `source`, `body_label`,
`display_goal` (refusal only), `session` (re-centring), `mapping` (shadow ReFIT labels); `purity.py` reads
`block` / `source` (certifying windows), `display_goal` and `feedback_visible` (display surrogate),
`session` / `t` / `trial` (nulls, clusters); `certify.py` reads `source`, `session`, `trial` and
`body_report` (`confirmed_by_report`).  No provenance field is ever a decoder feature.

## Modules

### G `body_decoder.py`
`BodyCoordinateDecoder(cfg, fs, channels=CENTRAL11, window_s=1.0, band_hz=(8, 30), filter_order=4, C=0.1,
x_auc_min=0.60, floor_percentile=95, n_permutations=200, n_splits=5, seed=20261004, x_purity_retain=0.70,
alpha_band_hz=(8, 12), local_reference=True, idle_threshold=IDLE_THRESHOLD (0.7), idle_auc_min=0.70,
idle_C=0.1)`; `fs` must equal `cfg.hardware.fs` (`ValueError`).  `channels=cfg.hardware.channels` builds the
all-32 model (the second shadow: `ShadowTrainer(cfg, decoder_options={"channels": cfg.hardware.channels})`).
- Per window: last `window_s` of the montage → LOCAL re-reference to the montage mean (`local_reference`:
  cancels the real-time CAR exactly, so frontal / ocular potentials CAR spreads cannot reach the decoder;
  expressed in a Helmert basis, k − 1 dimensions) → zero-phase 8–30 Hz band-pass inside the window → OAS
  covariance → re-centred R^{-1/2} C R^{-1/2} on a rest reference → tangent space at the identity →
  standardise → L2 logistic per axis.  `predict(window, good=None) -> (2,)` (x, y) = 2p − 1 per axis, x = 0
  unless earned, exactly (0, 0) when the armed idle gate puts p(idle) > `idle_threshold`;
  `predict_many(windows) -> (n, 2)`; `features(window)`; `covariance(window)`; `ungated()` (a shallow copy
  without the idle gate: offline discriminability, e.g. the promotion body metric, never a command source).
- Idle gate: `fit_idle(windows, provenance, session_references=None) -> report` on L0 rest windows vs
  intent (L1 / L2b, or the last `fit`'s features), class-balanced logistic; armed only if its out-of-fold
  AUC ≥ `idle_auc_min` (`idle_report`: n_idle, n_intent, cv_auc, auc_min, armed, threshold, cv_idle_gated,
  cv_intent_passed); `idle_probability(windows)`.
- Validity: NaN (both axes; a NaN row in `predict_many`, which never aborts) when a central sample is
  non-finite, a central channel is bad in `good` (the real-time loop's gate/repair mask — which now flags a
  dead, constant-valued electrode too: `AmplitudeGate`'s low rule), an eighth of the
  window carries < 1 % of its mean 8–30 Hz power (`N_SEGMENTS`, `MIN_SEGMENT_SHARE`: an amplifier that just
  flat-lined), or the re-centred band power is > 30 dB (`POWER_LIMIT`) from the training median.  The loop
  turns NaN into a counted zero command; a flat line used to drive y to −1.000.
- `reference_from(rest_windows) -> cov` (Riemannian mean); `set_reference(rest_windows=None, *,
  reference=None)` (neither → identity of the decoder's own `n_dims` coordinates, i.e. no re-centring).
  `ungated()` copies get their own model table (fitting a copy never replaces the deployed models).  No online running re-centring: the caller
  re-references per session (the simulation uses each session's setup rest).
- `fit(windows, body_labels (n,2) in {-1,0,1}², provenance, session_references=None, *, x_mask=None,
  neutral_one_hand=False)`: refuses any source but L1/L2b; an axis trains on windows whose label is
  non-zero on it; `session_references = {session: rest cov}` re-centres each training session on its own
  rest.  `x_mask` restricts the x model to the windows `earn_x` tests (own L1: the deployed x model is the
  recipe the grant certified; trained on own L1+L2b+prior an "earned" x sat below its own pool floor);
  `neutral_one_hand` (x earners) enters one-hand windows into the y model as neutral (both classes at
  half weight → output 0), since y trained on feet/hands alone reads one-hand ERD as HANDS.  `fit` never
  changes x status; `neutral_one_hand` records the last fit's setting.
- `earn_x(windows, labels, provenance, session_references=None, *, sources=("L1",), previously_enabled=False,
  previous_grant_passed=False) -> dict` (auc, floor, p_value, auc_purity, purity_retained, rule, n_windows,
  n_sessions, reason, passed, grant_pending, enabled).  Confirm rule: the grant test runs after every
  retrain, so x is granted only when this test AND the previous one passed (`previous_grant_passed` = the
  last report's `passed`); a single pass is `grant_pending`.  Cumulative false-grant rate over 3 days
  (100 simulated non-responders, 3 × 3.5 min of L1 a day): 11 % with a single pass, 1 % with the rule.
  The statistic is the x AUC over 5 contiguous (session/time-ordered) folds on the windows of `sources`
  only (a fixed population: L1 probes; mixing in the far more numerous decayed L2b windows made a genuine
  responder's x flip daily).  grant: AUC > 0.60 and > the p95 of a within-session circular-shift null
  (`purity.circular_shift_permutations`) and purity; keep (`previously_enabled`): AUC > floor and purity, so
  an earned x stays unless a separate test rejects it; purity: the out-of-fold x scores regressed on the
  posterior (S7) alpha lateralisation index (8–12 Hz log power P4/PO8 − P3/PO7) keep ≥ 70 % of their AUC
  margin above 0.5 (P1's increment rule) — otherwise covert attention spreading onto CP5/CP6 could give a
  non-responder an attention axis.  Sets `x_enabled`, `x_report`.
- `loop_config(cfg=None) -> Config` with `decoder.window_s` = the decoder's window and `task.control_hz` =
  `LOOP_HZ` (16).

### H `mapping.py` + `certify.py`
- `Mapping(name)`: `.M` (read-only), `.heldout`, `body_to_screen(v)` (M v), `screen_to_body(u)` (Mᵀ u); rows
  of (2,) or (n, 2).  `MAPPING_NAMES`, `TRAINING_MAPPING = "identity"`, `HELDOUT_MAPPINGS`, `BODY_VECTORS`,
  `unit_directions(targets, cursors)`, `certification_regressors(targets, cursors, mapping,
  trained_on="identity") -> (body_intent = M⁻¹u, screen_dir_in_body = M_train⁻¹u)`.  Held-out geometry:
  `DIAGONAL_DIRECTIONS`, `diagonal_targets(distance) -> (4, 2)` (all four drawn every trial),
  `quadrant_schedule(n_trials, rng)` (every run of four trials a permutation of the quadrants, so the signs
  of dx and dy are independent and balanced in any block: under rot±90 a 1-D y person drives the reachable
  axis while the screen regressor on y varies orthogonally).
- `certify_mapping(decoder_outputs_body, body_intent, screen_dir_in_body, provenance, *, axes=("x","y"),
  margin=None, n_boot=2000, alpha=0.05, seed=20261018, min_clusters=8, cross_axis=True, margin_se=None,
  margin_df=None) -> dict`: per axis a, OLS of output_a on body_a, screen_a and the CROSS-AXIS body intent
  body_ā, with an intercept; per axis and "pooled" (one β set over both axes: reported, never decisive).
  Inference is cluster-robust over whole (session, trial) clusters: Bell–McCaffrey CR2 SEs with t(G − k)
  intervals (`df`), ≥ `min_clusters` trials.  (The earlier percentile trial-cluster bootstrap missed a
  true β_screen = 0 in 25 % of 8-trial and 10 % of 12-trial units; CR2-t misses 5.5–6.3 % at 8–30 trials.)
  `n_boot` / `seed` are accepted for old callers and unused.  Why the cross term: under rot+90 screen_y =
  body_x and screen_x = −body_y (signs flip under rot−90), so without it any cross-axis body coupling of
  the decoder lands in β_screen — at screen_leak 0 the old two-regressor test returned SCREEN_READING.
  With both rotations in a unit, β_screen is the rotation-symmetric part and β_cross the antisymmetric
  part; a single rotation (or a mirror, or the identity) leaves β_screen unidentified.  Each entry:
  beta_body/screen/cross, ci_*, se_*, df, intercept, scale (`margin`: a float or {axis: float}, the
  EXTERNAL margin; None → β_body + β_screen with `margin_source` "self", never CERTIFIED), scale_se /
  scale_df (`margin_se` / `margin_df`: the margin's own SE and df — every threshold test then accounts for
  it, Welch–Satterthwaite), rel_* / rel_se_* / rel_df_* (÷ scale, delta method) and rel_ci_* (Fieller: the ratios the threshold tests do not reject), regressor_corr,
  `block_corr` / `max_block_corr` per block (a provenance session) and `confounded` (|r| > `MAX_BLOCK_CORR`
  0.5 in a block or overall), collinearity (largest multiple correlation among the regressors),
  identifiable (collinearity ≤ 0.98 and df ≥ 1) and per-trial sufficient statistics (`cluster_stats`, for
  the person-then-trial bootstrap); plus `window_key` (hash of window ids and regressors).  Windows whose
  source is not "H" — a missing source included — are refused.
- `replay_test(outputs, private_intent, replayed_dir, provenance, *, ..., min_clusters=30, margin_se=None,
  margin_df=None)`: the same regression with the private intent toward B as the body regressor and the
  replayed direction toward A as the screen regressor; sources other than "R" refused; ≥ 30 trials (≈ 10
  a day: pooled over days, never tested per day).
- `confirmed_by_report(provenance, body_intent) -> bool mask`: keeps windows of trials without a keyed
  report and of trials whose `body_report` points along the trial's body intent (positive dot product).
- `cluster_robust_ols(X, y, clusters) -> (beta, cov)` (CR2) and `welch(terms) -> (variance, df)`: shared with
  `simulate_champion` (the identity-trial margin's SE).
- `combine_units(units: {id: result}, *, combination="fixed"|"random"|"bootstrap", alpha=0.05, n_boot=2000,
  seed=20261018) -> dict`: per-unit results (each fitted on its own windows with its own slopes, intercepts
  and margin) combined per axis in margin units (scale 1, each unit with its delta-method relative SE):
  "fixed" inverse-variance (a person's decoder versions: the claim is about them), "random"
  DerSimonian–Laird τ² with a Hartung–Knapp–Sidik–Jonkman t interval, "bootstrap" person-then-trial
  bootstrap of the equal-weight mean over people (each replicate also redraws each unit's margin from its
  SE; both: a population claim; with 2–3 people necessarily wide).  Entries carry `df_body` / `df_screen`.
  x combines over units that tested x (earners), y over all; confounded or unidentified units are listed
  in `excluded` / `excluded_confounded`.
- `holm(pvalues, alpha) -> {name: {p, p_holm, rejected}}`.
- `certify(deployed, shadow, *, axis="all", screen_tol=0.10, body_pass=0.70, body_lower=0.50, shadow_null=None)
  -> dict` {"verdict", "reason", "axis", "flags", "deployed", "shadow" (first shadow's entry), "shadows",
  "holm", "per_axis" (for "all"), "unit_verdicts", "vetoed_by" (combined results)}.  `shadow` is one result
  or {name: result} (the central-11 and the all-32 shadow); `shadow_null` the same structure: each shadow's
  result of the same test on data WITHOUT a display pathway (the simulation's screen_leak 0 twin; for
  combined results its `units` calibrate the unit verdicts).  The test is sensitive if EITHER shadow's
  β_screen is shown above its null (a Welch test on the difference, the null's SE included); without
  `shadow_null` the shadows are tested against 0, flagged `shadow_null_calibrated` False, and nothing is
  CERTIFIED (an uncalibrated shadow fired at Holm p 0.003 with no screen pathway).  Per tested axis the
  family {deployed β_body > 0, deployed β_body > body_lower·margin, deployed β_screen < +screen_tol·margin,
  deployed β_screen > −screen_tol·margin, each shadow β_screen > its null} is Holm-corrected at α/2
  one-sided ("shown" below); the screen bound is two-sided (TOST): a decoder that follows the display with
  a NEGATIVE sign carries display information too.  Per axis, the first that applies:
  1. TEST_INSENSITIVE: β_screen not identified on this axis;
  2. TEST_INSENSITIVE: body and screen regressors confounded (refused: a non-linear body response biases
     β_screen there with either sign, so no SCREEN_READING is read from it either);
  3. SCREEN_READING: deployed β_screen ≠ 0 in either direction (α/2 per side, uncorrected) and not shown
     inside ±screen_tol·margin (a positive finding needs no positive control; `deployed_screen_sign`);
  4. NO_BODY_SIGNAL: deployed β_body not shown > 0;
  5. TEST_INSENSITIVE: no shadow's β_screen shown above its null;
  6. INCONCLUSIVE (the proposal's declared zone): deployed β_screen not shown inside ±screen_tol·margin, or
     β_body < body_pass·margin, or β_body not shown > body_lower·margin (F1b: lower CI < 50 %), or no
     external margin, or no `shadow_null`;
  7. CERTIFIED.
  `axis="all"` (default) returns the worst verdict over the tested axes (`VERDICTS` order), so a clean y
  cannot dilute screen reading on x; a single axis or "pooled" reads that entry only.  For combined results
  an unconfounded unit whose β_screen ≠ 0 and is not bounded inside ±screen_tol·margin at the Bonferroni
  level over all units × axes vetoes the verdict to SCREEN_READING.  Deployed and shadow results must have
  the same `window_key` (`ValueError` otherwise).  Per-unit SCREEN_READING is calibrated: 5.0 % of 759
  synthetic unit-axes with the simulation's geometry, 17–35 trials, a significant β_body and cross-axis
  coupling, and no screen pathway (`tests/test_champion_null.py`).
  Real data: the screen_leak 0 twin does not exist there, and a label-permuted or rotation-swapped
  surrogate would not reproduce the regression misspecification the twin calibrates; a real-data
  no-display null for the shadows is still to build, so a real-data verdict is at most INCONCLUSIVE.

### I `protocol.py` + `data_engine.py`
- Block kinds `CHAMPION_BLOCK_KINDS` = fixation_probe (L1), blind_closed_loop (L2b; `just_intend` → J),
  sighted (S; `just_intend` → J), heldout_mapping (H), pixel_replay (R); `ALL_BLOCK_KINDS`.  `Block` gains
  `mapping`, `variant` ("C2b"/"C2c"), `instructed`, `just_intend`, `replay_fraction`, `localiser`,
  validated per kind; `.source`, `.is_test_only`, `.is_champion`, `.replay_block()`; `screen_items` adds
  the "outcome" phase (blind kinds draw only central, non-spatial items).  `BLANK_CUE_KINDS` = (cued_blank,
  fixation_probe, blind_closed_loop), derived from `screen_items`: the purity audit's certifying blocks.
- `session_plan_champion(day, seed, cohort="deep", x_earned=False) -> list[Block]`, deterministic, 85 min:
  setup 15 (3-min rest; artefact localiser in sessions 1, 11, 21, …), L1 15, L2b 20, S 15 (15 % replays),
  H 8 (two 4-min blocks every session: rot+90 and rot−90 in random order — both rotations in every
  certification unit; no mirrors, which identify nothing per axis; both blocks C2b in sessions 1–2, then
  one variant per session, C2c on odd and C2b on even sessions), just-intend 12 (3-min blocks alternating
  blind and sighted; on day 1 the blind ones open the session as the hint-free block).  `x_earned` no longer
  changes the plan (kept for callers): the held-out schedule must not depend on an x status the pre-S
  retrain can still revoke.  S instruction withdrawn
  after session 10 (deep) or 3 (broad).  `CHAMPION_SLOT_MIN`, `champion_plan_summary(plan)`.
- Cues and timing: `body_vector`, `BODY_CLASSES`, `L1_CLASSES`, `L2B_CLASSES_Y` / `_XY`,
  `glyph_map(day, seed)`, `BodyCue`, `BodyCueScheduler(seed, classes, modes, glyphs, reps_per_block=2, …)`,
  `hidden_goal(cue)`, `ProgressTone`, `FixationProbeTiming` (cue 0.5 s, labels 1.0–4.0 s),
  `ClosedLoopTiming` (cue 1.0 s, control from 1.5 s), `OpenLoopTiming`, `CUE_LOCKOUT_S`,
  `CUE_OFFSET_LOCKOUT_S`, `trial_timing(block)`, `ReplayScheduler(block)` (exactly 3 per 20 trials, never
  first, never adjacent; `REPLAY_PERIOD`, `replays_per_period(f)`: a Block refuses a replay_fraction that
  gives no replay per period),
  `champion_trial_start_payload(block, trial, *, cue=None, goal=None, start_pos=(0, 0), replay_goal=None)`.
  Events `fixation_break`, `outcome_flash`, `body_report` (C2b keyed report).
- `data_engine`: `CHAMPION_BLOCKS`, `CHAMPION_TARGET_BLOCKS` = (sighted, heldout_mapping, pixel_replay);
  the original `BLOCK_KINDS` / `Trainer` are unchanged and never select a champion window.  `parse_trials`
  validates source, mapping, cue timing (trial_start ≥ 1.0 s after cue onset and ≥ 0.5 s after cue offset),
  fixation breaks and aborts.
  `Labeler` adds to champion windows `source`, `mapping`, `instructed`; L1/L2b `body_label`, `cue_class`,
  `cue_mode` (label = cued body vector; every cue-locked window — L1, L2b, blind J — ends ≤ 4.0 s after
  its cue, `CUE_LABEL_END_S`, and an L2b window no later than the first logged hidden-cursor row inside the
  dwell radius of its goal); visible trials `cursor_pos` and `cursor_pos_start` (the displayed cursor at
  the window end / start); H `variant`, `body_report`; R `replay_goal`, `body_report` (`REPORTED_BLOCKS`);
  S/H/R `display_goal`.
- `require_sources(data, allowed, who)`, `body_labels(data)`, `refit_body_labels(data)` (M⁻¹ u).
- `BodyTrainer(cfg, window_s=1.0, decoder_options=None)`: `select(data)`, `fit(data,
  session_references=None, *, x_mask=None, neutral_one_hand=False) -> BodyCoordinateDecoder` (L1/L2b only;
  any window with `display_goal` set is refused; it reads no gaze field — tested with a read-only provenance
  Mapping that records every key observed through item access, `get`, `in`, iteration, `items`/`values`,
  and by fitting with and without the gaze fields to bit-identical outputs).
- `ShadowTrainer(cfg, window_s=1.0, axis_min=sin 22.5°, decoder_options=None)`: `select`, `axis_labels`,
  `fit(data, x_enabled=False, session_references=None) -> ShadowBodyDecoder` (S only, ReFIT body labels,
  sign per axis where |component| ≥ axis_min; x status copied from the deployed decoder).  NOT a matched-n
  twin of the deployed decoder: S windows only (≈ 1/3 of the person's own L1/L2b window count at the
  default allocation) and no population prior; its sensitivity is that smaller fit's.
- Promotion of the deployed decoder (the body metric): `ChampionChallenger.is_body(model)` selects it for a
  `BodyCoordinateDecoder`.  H′ = the newest 20 % of L1 probe windows (`candidate_heldout(data,
  body=True)`; ≥ 3 FEET and 3 HANDS probes), never trained on by the challenger they select; score =
  per-axis window AUC of the ungated axis outputs (`axis_outputs`, re-centred per session) against the cued
  class, mean over the scored axes (y; x when the probes hold RIGHT and LEFT and either decoder drives x);
  first champion: > 0.5 + `auc_margin` (0.02) and a whole-trial class-permutation p ≤ 0.05; afterwards the
  lower bound of a class-stratified trial-bootstrap CI of the paired difference > `auc_margin`.
  `DataEngine` runs it end to end with a `BodyTrainer` (store → Labeler(1-s windows) → BodyTrainer.select →
  fit → body-metric promotion; `population` = a `LabelledSet` of other people's L1 probes,
  `session_references` = {run name: rest covariance}); tested on simulated runs, including that a real
  improvement over a weak champion is promoted and the frozen probes are never trained on.
- Purity on champion data: `purity.SPATIAL_BLOCKS` is derived from `protocol.ALL_BLOCK_KINDS`, so it
  includes the champion target blocks and a purity audit of champion data sees their drawn goals; the
  certifying windows are L1 / L2b (`purity.is_certifying`, keyed on block or source; J excluded) beside
  cued_blank.  `PurityAudit.run(..., deployed=predictions)` adds the deployed decoder's own row (pass the
  out-of-sample, per-session re-centred outputs of the `BodyCoordinateDecoder`, e.g. on the fixed L1 eval
  pool) and DEPLOYED_NO_EVIDENCE to the certification verdict when it does not beat its floor;
  `gaze_fs={"eyetracker": Hz}` adds microsaccade rate and direction to a raw gaze auditor, the gaze
  signal that can still fire GAZE_TRACKS_LABEL while fixation holds (INTERFACES.md item 4).

### J `sim_body.py`
`BodySubject(SubjectEEG)` with `y_depth`, `x_depth`, `x_responder`, `dry_penalty` ∈ [0, 1), `screen_leak`,
`covert_leak`, `session_drift_sd`, `erd_peak_s`, `erd_decay_s`, `erd_sustain`, `x_time_course=None`,
`day_depth_sd=0`, `rebound_gain=REBOUND_GAIN` (1.0: post-imagery beta rebound — at an imagery offset the
desynchronised generators' beta rises above baseline, `rebound_envelope`, peak `REBOUND_PEAK_S` 0.6 s,
by the ERD at the offset; a new episode cancels it); control `cue(vx, vy)`
(restarts the ERD time course), `set_intent(vx, vy)` (continues it), `set_screen_direction(u | None)`.
Motor: AR(2) mu/beta resonators at C3 / C4 hand areas and the Cz foot area; FEET → vertex ERD, HANDS →
bilateral, RIGHT → C3 (0.3 at C4), LEFT → mirror; non-responders bilateral for L/R.  ERD peaks 1.5 s after
the cue and decays to 0.5: ds004362's FEET/FISTS time course.  By default x shares it, which is optimistic
for x (ds004362 L/R peaks at 0–1.5 s and falls to a d′ ratio of 0.42 by 3–4 s); `x_time_course = X_ERD_LR`
(0.3, 2.8, 0) is the L/R-fitted sensitivity case.  `dry_penalty` scales ERD depth; it is not the d′ loss.
Measured with the deployed pipeline on the L1 1–4-s windows it trains on (local reference, y_depth 0.25,
6 people × 40 min): 19 / 37 / 54 / 69 / 83 / ≈ 100 % d′ loss at 0.1 / 0.2 / 0.3 / 0.4 / 0.5 / 0.7 (y AUC
0.620 at 0), so the proposal's 20–40 % prior is ≈ 0.1–0.22, DESIGN §8's 20–70 % ≈ 0.1–0.4, and 0.7 is a
no-signal condition (the earlier 48 % / 85 % table predated the local reference).  Non-stationarity is per-channel gain drift (the best case for re-centring) plus,
with `day_depth_sd` > 0, a per-day ERD-depth factor that re-centring cannot remove.  Screen leak (only while a target is drawn): posterior
alpha (contralateral for x, dorsal/ventral for y, equal depth, spreading onto CP sites) and a frontal gaze
potential; `covert_leak` adds body-locked posterior alpha with nothing drawn.  `screen_leak = 0` makes the
EEG exactly independent of the screen direction.  `BodyPopulation(n, seed, hw, *, x_responder_prob=0.2,
y_depth_range=(0.1, 0.4), x_depth_range=(0.25, 0.5), dry_penalty, screen_leak, covert_leak, …)`; subject i
depends only on (seed, i).

### K `simulate_champion.py`
`ChampionSimConfig(days, hours, users, pop, seed, dry_penalty, screen_leak=1.0, covert_leak=0.0,
replay_attention="cursor", x_responder_prob, y_depth_range, x_depth_range, user_learning_rate, pop_minutes,
rest_s, eval_minutes, assist_gain=0.5, label_hop_s=0.5, x_permutations, n_boot, x_time_course=None,
day_depth_sd=0.0, slot_minutes=(), identity_fraction=0.25, null_calibration=True)`; `simulate(sim,
cfg=DEFAULT, workers=1) -> dict` (with the screen leak on and `null_calibration`, the same people also run
at screen leak 0 — same seeds, noise streams and prior, so the same decoder versions and units — and every
certification passes that twin's shadow results as `certify`'s `shadow_null`; at screen leak 0 a run is
its own null); `sweep(sim, dry_penalties, cfg=DEFAULT, workers=1, covert_leaks=None)` (every condition is
validated first and duplicates refused; a condition that raises `ValueError` is recorded as {"failed":
reason} and the sweep goes on); `write_report(res, prefix)`; `default_prefix(sim, dry_penalties,
covert_leaks)`; `champion_config(cfg)` (the real-time `Config` of the deployed decoder: 1-s windows, 16 Hz);
`SHADOWS = ("central11", "all32")`; `all_channels_shadow_options(cfg, options)`;
`identity_trial_margin(outputs, data, min_trials=MIN_IDENTITY_TRIALS) -> {axis: slope}` (the EXTERNAL
margin: a frozen decoder's slope on body intent in the identity trials of its own held-out blocks) and
`identity_trial_margin_stats(...) -> {axis: (slope, CR2 SE, df)}` (its CV is 0.3–0.4 at 7–12 trials; the
SE enters every threshold test; `MIN_IDENTITY_TRIALS` 8); `identity_margin(decoder, pool, refs)`
(secondary, the L1-pool slope, reported beside it); `leak_transfer(cfg, sim, pop, user, day, decoder,
reference, margin) -> {probe_leak, gain, rel, n_windows}` (a paired probe of a frozen deployed decoder:
identical noise streams at screen leak 0 and `PROBE_LEAK` 1, rot±90 imagery on diagonal targets; its output
slope on the screen direction ÷ the external margin: whether a display leak reaches the DEPLOYED
decoder); `recentred_outputs(decoder, data, refs)`; `l1_scores(decoder, pool, refs, …)`;
`idle_scores(decoder, pool, pool_idle, refs) -> {armed, cv_auc, rest_gated, intent_passed}`;
`idle_set(windows, provenance)`; `regressors(test, data)`; `record_fixation(cfg, sim, subject, day, block,
index, tag, root, t0) -> (L1 set, {session: reference}, L0 set)`; `load_prior(prior_dir)` and
`build_l1_pool(cfg, sim, pop, user, root)` return the same triple; `BlockRun(run_dir, outcomes,
identity_dir, idle)`; `ChampionRunner(subject, cfg, sim, t0, trajectories: {geometry: [Trajectory]})`
(replay pools "sighted" and "diagonal"; control every round(fs / control_hz) samples, the task clock at
exactly that rate) with `record_rest(day, index)` (the setup rest: re-centring reference and L0 windows);
`simulate_user(...) -> (result, {day: {test: CertificationLog}})` and `certify_user(result, day_logs,
null_logs, n_boot)` (the per-day and pooled-days certification against the twin's logs);
`CertificationLog` (`extra_shadows` {name: outputs} beside the central-11 shadow, `margin_se` / `margin_df`
per unit, `certify(test, n_boot, null=None)`, `certify_by_variant(n_boot, null=None)`).
Per deep participant and day: setup rest → re-centring reference and L0 windows; L1 (the rest between
probes is L0 too); L2b (day 1 on the population prior, L1-only prior of other simulated people); retrain on
all own L1/L2b + prior (the x model on own L1 probes only, `x_mask`; an x earner's y with neutral one-hand
windows), `fit_idle` on L0 (armed only above `idle_auc_min`), and `earn_x` on own data before the first S
block (x on own L1 probes, granted on two consecutive passing grant tests — one retrain a day, so day 2
at the earliest — and kept once earned unless the keep rule rejects it); S with GoalPosterior assistance
and 15 % replays (each followed by a keyed report of the held intent); H (frozen decoder) with
`identity_fraction` of its trials under the identity mapping (I trials: logged as a parallel
identity-mapping sighted run, kept out of every trainer, never certified, the source of the external
margin) and diagonal held-out targets with a balanced goal quadrant (`mapping.quadrant_schedule`; a replay
is decided before a quadrant is drawn and replays a past rotated C2c trajectory, so replays neither
unbalance the rotated trials nor change the layout); both shadows (central-11 and all-32, sighted ReFIT
labels, never deployed) score every certified window, which is kept only if its trial's keyed report
confirms the intent.  Just-intend blocks are not simulated.  Until x is earned the simulated person
imagines only the y component of M⁻¹u (1-D y control).  `covert_leak` acts only in blind (L1/L2b)
blocks, as covert attention toward the imagined screen direction.  Reports per day: fixed L1 test-set AUC
per axis with its floor (p95 of a trial-level label permutation, `l1_scores`; "above floor" = AUC >
floor) and r, x-earned (or pending) with AUC and floor, the idle gate, sighted success / TTT / throughput,
blind success, the external margin with its SE and the L1 margin, the leak-transfer gain, the β table
(deployed vs both shadows with their screen_leak 0 nulls, held-out — also C2b and C2c separately, since
C2c regressors come from a cursor the deployed decoder moved — and replay), the largest per-block
body–screen |r|, and verdict; combined over each person's days and over people (person-level bootstrap);
trials dropped by the report check; and the responder counts of the deep users and the prior against the
population rate.  Certification units: held-out = one deployed decoder version (one session), replay =
one person; regressors use the cursor at the window START (C2c's window-end cursor is endogenous) and, for
replay, directions matched to the whole window.  An axis whose identity-trial margin is missing or below
`MIN_MARGIN` (0.02) is not certified for that decoder version; without a y margin the version is skipped,
and the report says so.  A day whose sighted data cannot train a shadow is reported as skipped.
CLI: `python3 -m neural_cursor.simulate_champion --days D --hours H --users U --pop N
[--dry-penalty P …] [--screen-leak L] [--covert-leak C …] [--replay-attention cursor|goal]
[--y-depth-range LO HI] [--x-depth-range LO HI] [--x-time-course PEAK DECAY SUSTAIN] [--day-depth-sd S]
[--eval-minutes M] [--slot-minutes SLOT=MIN …] [--identity-fraction F] [--no-null-calibration] [--n-boot B]
[--workers W] [--out PREFIX]`.  A malformed or repeated `--slot-minutes`, and an invalid or repeated
(`--dry-penalty`, `--covert-leak`) condition, are usage errors raised before anything runs; the default
prefix names every non-default setting.
Power note (simulation result, not a protocol change).  The `results/champion_*` files are produced by the
current certification (CR2-t per-unit fits with the identity-trial margin's SE propagated, two-sided TOST
screen bound, body lower bound, both shadows calibrated against the screen_leak 0 twin, Holm family,
two-sided per-unit veto) and the current simulation (L2b windows ≤ 4 s after the cue, x model on own L1
with the two-pass grant, neutral one-hand y for earners, 16-Hz loop, post-imagery rebound, report check,
quadrant-safe replays): `champion_d3_h1_u3_pop10_dry0_leak{0,0.5,1,2}_covert0` (screen-leak sweep),
`…_leak1_covert1` (covert leak), `…_dry0-0.2-0.4_leak1_covert0` (dry sweep at the recalibrated points),
`…_leak1_covert0_seed7` (a second population draw: 2 of 13 responders against 7 of 13 for the default
seed), and `champion_d2_h1_u2_pop6_dry0_leak1_covert0_y0.3-0.4_eval8_slotshe24` (strong y responders, 24
held-out minutes; it replaces the stale `…_H24` file, whose CERTIFIED rows the current test does not give).
- Nothing is CERTIFIED in any condition.  Pooled over the 3 people the held-out verdict is NO_BODY_SIGNAL
  everywhere (t with 2 df on the person bootstrap, CR2-t per unit, the margin's SE in every threshold);
  per unit, y is INCONCLUSIVE or NO_BODY_SIGNAL at ≈ 34 held-out trials, and day 3 (C2c) is skipped for
  every person: its interleaved identity trials number < 8 (`MIN_IDENTITY_TRIALS`), so C2c sessions need
  more identity trials (a larger `identity_fraction` or longer H blocks) before they can be certified.
- Screen leak 0 (the null): no positive control fires (calibrated against itself) and no unit is
  SCREEN_READING (the uncalibrated run flagged two units and a firing shadow).  The per-unit SCREEN_READING
  flag is calibrated separately: 5.0 % of 759 synthetic unit-axes at the simulation's geometry and size.
- Shadow sensitivity vs screen leak (person-days on which a shadow reads the screen beyond its twin's null):
  all-32 0 / 7 at leak 0.5, 5 / 7 at leak 1, 5 / 7 at leak 2; the central-11 shadow never does.  This sweep
  measures the SHADOWS' sensitivity only: at covert leak 0 the deployed decoder's training data carry no
  display signal.  Whether a display leak reaches the DEPLOYED decoder is the paired leak-transfer probe
  (output slope on the screen direction ÷ margin, at screen leak 1): y ≈ 0 for the default seed's people
  (median 0.00, range −0.10 … +0.19), but +0.04 … +0.37 (median +0.12) for the seed-7 people, and the earned
  x of user 0 reads +0.25 … +0.44 (lateralised posterior alpha spreading onto CP5/CP6 aligns with its
  C3-vs-C4 pattern).  At covert leak 1 the probe reads +0.20 … +0.94 on y, and the certification flags user 2
  day 2 and the pooled replay test SCREEN_READING.
- Replay: at screen leak 1 the one person with ≥ 30 replay trials (user 0) is SCREEN_READING with a NEGATIVE
  β_screen (−0.60 of the margin; ≈ 0 at leak 0), which the old one-sided bound would have passed.  Its
  margin is too uncertain for a bounded ratio interval (Fieller: unbounded); the sign test does not
  depend on the margin.
- Strong y responders (`…_d2_…_y0.3-0.4_eval8_slotshe24`): per-unit y is INCONCLUSIVE (β_body ≈ 1 of the
  margin, β_screen not bounded inside ±10 %), and user 1 day 1 is SCREEN_READING (β_screen 0.36 [0.13, 0.75])
  while that decoder's paired leak transfer is −0.01: a held-out estimate the probe does not confirm, which
  vetoes the pooled verdict.  The earlier "≈ 770 pooled held-out trials certify y" does not survive.
- Dry penalty 0.2 / 0.4 (≈ 37 / 69 % d′ loss): the fixed-pool y AUC falls (user 2 0.81 → 0.75 → 0.70) and
  more units fail on the body test or lose their margin.
- x: the default seed's non-responder user 1, falsely granted x on day 1 before, is only `grant_pending`
  on day 1 and fails on day 2.
What carries over, with its conditions stated:
- Every CERTIFIED verdict is conditional on `covert_leak` = 0 and on the twin null; real data need their
  own no-display shadow null (to build).
- The shadow positive control's sensitivity is set by a tuned simulator constant (`ALPHA_VERT_DEPTH`); it
  shows that a LARGE leak is detectable on y by the all-32 shadow, not sensitivity at the 10 % bound (the
  deployed TOST's job).  The central-11 shadow learns the motor contrast on y and never reads the screen
  beyond its null; a y verdict that rests on it alone stays TEST_INSENSITIVE.
- A power curve at the default depths is owed before any trial count is quoted: at ≈ 34 held-out trials a
  unit cannot bound |β_screen| inside ±10 % of a margin estimated from ≈ 8 identity trials.

### L `realdata/nk_surface.py`
`realdata/run.sh nk_surface --cache C --feat F --nkfeat K [--out results/realdata] [--tasks imag_FF,imag_LR]
[--Ns …] [--ks …] [--draws 4] [--boot 2000] [--jobs 3]`: people (N ∈ 2…81) × first-k trials per person
(k ∈ 10…45) surface on ds004362, central-11, re-centred on own rest, fixed 25 test subjects (E2's split),
acc = a + b·log2 N + c·log2 k with test-subject bootstrap CIs (and a draw-resampling CI); writes
`results/realdata/E8_nk_surface_<task>.json`.

### M `realtime_demo.py`
The deployed chain live on the wall clock: `DemoConfig(seconds=30, effector="both"|"arm"|"cursor", seed=2,
y_depth=0.4, x_depth=0.5, x_responder=True, dry_penalty=0.0 (in [0, 1)), calibration_minutes=24 (≥
`MIN_CALIBRATION_MINUTES` 2), rest_s=30, label_hop_s=0.5, x_permutations=200, segment_s=5, chunk_s=0.010,
screen_px, cursor_gain_px_s=80, arm_max_speed=0.04, arm_start_xy=(0, 0.30))`; `run_demo(demo,
cfg=DEFAULT) -> dict`; `write_report(res, path)`; `calibrate(demo, cfg, subject) -> (decoder, report)`
(setup rest + L1 probes through `simulate_champion.record_fixation`, `BodyTrainer` on L1 only, `earn_x`
twice — on the first half of the probes and on all of them: the confirm rule — an earned x refitting y
with neutral one-hand windows, `fit_idle`, re-centring); `live_run(...)` (a `BodyLSLStreamer` —
`SimLSLStreamer` plus `cue(vx, vy)`, which restarts the ERD episode — publishes a new block over LSL in
10-ms chunks; `LSLEEG` → `RealTimeLoop` (16 Hz) → `DecoderTap`, which passes window and validity mask to
the decoder unchanged and records each output → `TeeEffector`, one command to every device in the same
tick); `score_segments(ticks, …)` (per imagery segment, from the 1.5-s control onset to the next cue: mean
decoded velocity, device displacement, sign agreement on the cued axis, and the decoded mean on the OTHER
axis); `summarise` adds `cross_axis_drift` (y while imagining one hand, x while imagining feet/hands);
`arm_joint_angles(l1, l2, xy)`.  Reports latency p50/p95/max, update rate, drops / overruns / invalid ticks,
per-axis sign agreement for each device and the cross-axis drift.  The demo person is a top-of-range
responder (disclosed in the report): the run shows the chain working, not a population estimate.
Pure-EEG: the decoder sees only (500, 32) EEG windows and the loop's (32,) good-channel mask.
CLI: `python3 -m neural_cursor.realtime_demo --seconds S --effector arm|cursor|both [--seed N] [--segment-s
S] [--dry-penalty P] [--calibration-minutes M] [--out results/realtime_demo.md]`; an invalid argument is a
usage error.
