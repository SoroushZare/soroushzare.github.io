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
windows**: build the labeler and the real-time loop with `BodyCoordinateDecoder.loop_config(cfg)` (the
default config's 0.5-s windows are refused, not silently decoded).

## Label sources (provenance field "source")
- "L1"  fixation probe: central glyph cues a class, glyph→class map redrawn every session, gaze-contingent
        fixation; windows 1.0–4.0 s after cue.  LABEL = cued body vector.
- "L2b" blind closed loop: cue = central glyph or body-part word (never a location), cursor hidden, hidden
        goal, label-symmetric progress tone (pitch encodes distance decrease, identical for every direction);
        windows start ≥ 1.0 s after cue and end at trial_end, before the central outcome flash.
        LABEL = cued body vector.  y-only classes until x is earned.
- "S"   sighted closed loop, identity mapping: NO LABEL for the deployed decoder (re-centring only).
        The SHADOW decoder (positive control, never deployed) trains on S with ReFIT labels
        u = (target − cursor)/‖·‖ mapped to body coordinates through M⁻¹.
- "H"   held-out mapping test blocks (rot±90, mirrors; C2b open loop, C2c closed loop): test only.
- "R"   pixel replay test trials (15 % of S and C2c trials): the screen replays a past trajectory toward
        goal A while the person holds goal B: test only.
- "J"   "just intend" (uninstructed, identity, frozen decoder; half blind, half sighted): scored only.
Trainers refuse S/H/R/J windows for the deployed decoder (`ValueError`, tested); the shadow refuses all
but S.  Shared constants: `protocol.SOURCES`, `DEPLOYED_TRAINING_SOURCES`, `TEST_ONLY_SOURCES`;
`data_engine.DEPLOYED_SOURCES`, `SHADOW_SOURCES`; `body_decoder.TRAINABLE_SOURCES`.

## Modules

### G `body_decoder.py`
`BodyCoordinateDecoder(cfg, fs, channels=CENTRAL11, window_s=1.0, band_hz=(8, 30), filter_order=4, C=0.1,
x_auc_min=0.60, floor_percentile=95, n_permutations=200, n_splits=5, seed=20261004)`
- Per window: last `window_s` of the central-11 → zero-phase 8–30 Hz band-pass inside the window → OAS
  covariance → re-centred R^{-1/2} C R^{-1/2} on a rest reference → tangent space at the identity →
  standardise → L2 logistic per axis.  `predict(window) -> (2,)` (x, y) = 2p − 1 per axis, x = 0 unless
  earned; `predict_many(windows) -> (n, 2)`; `features(window)`; `covariance(window)`.
- `reference_from(rest_windows) -> cov` (Riemannian mean); `set_reference(rest_windows=None, *,
  reference=None)` (neither → identity, i.e. no re-centring).  No online running re-centring: the caller
  re-references per session (the simulation uses each session's setup rest).
- `fit(windows, body_labels (n,2) in {-1,0,1}², provenance, session_references=None)`: refuses any source
  but L1/L2b; an axis trains only on windows whose label is non-zero on it; `session_references =
  {session: rest cov}` re-centres each training session on its own rest.  `fit` never changes x status.
- `earn_x(windows, labels, provenance, session_references=None) -> dict` (auc, floor, p_value, n_windows,
  n_sessions, reason, enabled): x enabled iff the x-axis AUC over 5 contiguous (session/time-ordered)
  folds exceeds both the p95 of a within-session circular-shift null (`purity.circular_shift_permutations`)
  and the pre-declared 0.60.  Sets `x_enabled`, `x_report`.
- `loop_config(cfg=None) -> Config` with `decoder.window_s` = the decoder's window.

### H `mapping.py` + `certify.py`
- `Mapping(name)`: `.M` (read-only), `.heldout`, `body_to_screen(v)` (M v), `screen_to_body(u)` (Mᵀ u); rows
  of (2,) or (n, 2).  `MAPPING_NAMES`, `TRAINING_MAPPING = "identity"`, `HELDOUT_MAPPINGS`, `BODY_VECTORS`,
  `unit_directions(targets, cursors)`, `certification_regressors(targets, cursors, mapping,
  trained_on="identity") -> (body_intent = M⁻¹u, screen_dir_in_body = M_train⁻¹u)`.
- `certify_mapping(decoder_outputs_body, body_intent, screen_dir_in_body, provenance, *, axes=("x","y"),
  margin=None, n_boot=2000, alpha=0.05, seed=20261018) -> dict`: OLS of output on both regressors with an
  intercept per axis, per axis and "pooled"; 95 % trial-cluster bootstrap CIs (whole (session, trial)
  clusters, ≥ 1000 draws, ≥ 8 trials).  Each entry: beta_body, beta_screen, ci_body, ci_screen, intercept,
  scale (margin; None → β_body + β_screen), rel_* (÷ margin), regressor_corr, identifiable (|corr| ≤ 0.98:
  the identity and a mirror's unflipped axis are not).  Windows with a source other than "H" are refused.
- `replay_test(outputs, private_intent, replayed_dir, provenance, *, ...)`: the same regression with the
  private intent toward B as the body regressor and the replayed direction toward A as the screen
  regressor; sources other than "R" refused.  (`provenance` is required: the bootstrap needs trial ids.)
- `certify(deployed, shadow, *, axis="pooled", screen_tol=0.10, body_pass=0.70) -> dict` {"verdict",
  "reason", "axis", "flags", "deployed", "shadow"}; first that applies:
  1. TEST_INSENSITIVE: regressors not separable on this axis;
  2. SCREEN_READING: deployed β_screen CI > 0 and not bounded below screen_tol·margin (a positive
     finding needs no positive control);
  3. NO_BODY_SIGNAL: deployed β_body CI does not exclude 0;
  4. TEST_INSENSITIVE: shadow β_screen CI not > 0;
  5. INCONCLUSIVE (the proposal's declared zone): deployed β_screen CI reaches screen_tol·margin, or
     β_body < body_pass·margin;
  6. CERTIFIED.
  Deployed and shadow results must come from the same test on the same windows.

### I `protocol.py` + `data_engine.py`
- Block kinds `CHAMPION_BLOCK_KINDS` = fixation_probe (L1), blind_closed_loop (L2b; `just_intend` → J),
  sighted (S; `just_intend` → J), heldout_mapping (H), pixel_replay (R); `ALL_BLOCK_KINDS`.  `Block` gains
  `mapping`, `variant` ("C2b"/"C2c"), `instructed`, `just_intend`, `replay_fraction`, `localiser`,
  validated per kind; `.source`, `.is_test_only`, `.is_champion`, `.replay_block()`; `screen_items` adds
  the "outcome" phase (blind kinds draw only central, non-spatial items).
- `session_plan_champion(day, seed, cohort="deep", x_earned=False) -> list[Block]`, deterministic, 85 min:
  setup 15 (3-min rest; artefact localiser in sessions 1, 11, 21, …), L1 15, L2b 20, S 15 (15 % replays),
  H 8 (two 4-min blocks, every session; the first is rot±90; mirror_x only once x is earned; C2b only in
  sessions 1–2, C2b/C2c alternate from session 3), just-intend 12 (3-min blocks alternating blind and
  sighted; on day 1 the blind ones open the session as the hint-free block).  S instruction withdrawn
  after session 10 (deep) or 3 (broad).  `CHAMPION_SLOT_MIN`, `champion_plan_summary(plan)`.
- Cues and timing: `body_vector`, `BODY_CLASSES`, `L1_CLASSES`, `L2B_CLASSES_Y` / `_XY`,
  `glyph_map(day, seed)`, `BodyCue`, `BodyCueScheduler(seed, classes, modes, glyphs, reps_per_block=2, …)`,
  `hidden_goal(cue)`, `ProgressTone`, `FixationProbeTiming`, `ClosedLoopTiming`, `OpenLoopTiming`,
  `trial_timing(block)`, `ReplayScheduler(block)` (exactly 3 per 20 trials, never first, never adjacent),
  `champion_trial_start_payload(block, trial, *, cue=None, goal=None, start_pos=(0, 0), replay_goal=None)`.
  Events `fixation_break`, `outcome_flash`, `body_report` (C2b keyed report).
- `data_engine`: `CHAMPION_BLOCKS`, `CHAMPION_TARGET_BLOCKS` = (sighted, heldout_mapping, pixel_replay);
  the original `BLOCK_KINDS` / `Trainer` are unchanged and never select a champion window.  `parse_trials`
  validates source, mapping, cue timing (trial_start ≥ 1.0 s after cue onset), fixation breaks and aborts.
  `Labeler` adds to champion windows `source`, `mapping`, `instructed`; L1/L2b `body_label`, `cue_class`,
  `cue_mode` (label = cued body vector; L1 clipped at 4.0 s after the cue); visible trials `cursor_pos`;
  H `variant`, `body_report`; R `replay_goal`; S/H/R `display_goal`.
- `require_sources(data, allowed, who)`, `body_labels(data)`, `refit_body_labels(data)` (M⁻¹ u).
- `BodyTrainer(cfg, window_s=1.0, decoder_options=None)`: `select(data)`, `fit(data,
  session_references=None) -> BodyCoordinateDecoder` (L1/L2b only).
- `ShadowTrainer(cfg, window_s=1.0, axis_min=sin 22.5°, decoder_options=None)`: `select`, `axis_labels`,
  `fit(data, x_enabled=False, session_references=None) -> ShadowBodyDecoder` (S only, ReFIT body labels,
  sign per axis where |component| ≥ axis_min; x status copied from the deployed decoder).
- `purity.SPATIAL_BLOCKS` is derived from `protocol.ALL_BLOCK_KINDS`, so it includes the champion target
  blocks and a purity audit of champion data sees their drawn goals.

### J `sim_body.py`
`BodySubject(SubjectEEG)` with `y_depth`, `x_depth`, `x_responder`, `dry_penalty` ∈ [0, 1), `screen_leak`,
`covert_leak`, `session_drift_sd`, `erd_peak_s`, `erd_decay_s`, `erd_sustain`; control `cue(vx, vy)`
(restarts the ERD time course), `set_intent(vx, vy)` (continues it), `set_screen_direction(u | None)`.
Motor: AR(2) mu/beta resonators at C3 / C4 hand areas and the Cz foot area; FEET → vertex ERD, HANDS →
bilateral, RIGHT → C3 (0.3 at C4), LEFT → mirror; non-responders bilateral for L/R.  ERD peaks 1.5 s after
the cue and decays to 0.5 (ds004362 time course).  Screen leak (only while a target is drawn): posterior
alpha (contralateral for x, dorsal/ventral for y, equal depth, spreading onto CP sites) and a frontal gaze
potential; `covert_leak` adds body-locked posterior alpha with nothing drawn.  `screen_leak = 0` makes the
EEG exactly independent of the screen direction.  `BodyPopulation(n, seed, hw, *, x_responder_prob=0.2,
y_depth_range=(0.1, 0.4), x_depth_range=(0.25, 0.5), dry_penalty, screen_leak, covert_leak, …)`; subject i
depends only on (seed, i).

### K `simulate_champion.py`
`ChampionSimConfig(days, hours, users, pop, seed, dry_penalty, screen_leak=1.0, covert_leak=0.0,
replay_attention="cursor", x_responder_prob, y_depth_range, x_depth_range, user_learning_rate, pop_minutes,
rest_s, eval_minutes, assist_gain=0.5, label_hop_s=0.5, x_permutations, n_boot, slot_minutes=())`;
`simulate(sim, cfg=DEFAULT, workers=1) -> dict`; `sweep(sim, dry_penalties, …)`; `write_report(res, prefix)`.
Per deep participant and day: setup rest → re-centring reference; L1; L2b (day 1 on the population prior,
L1-only prior of other simulated people); retrain on all own L1/L2b + prior and `earn_x` on own data before
the first S block; S with GoalPosterior assistance and 15 % replays; H (frozen decoder); just-intend blocks
are not simulated.  Until x is earned the simulated person imagines only the y component of M⁻¹u (1-D y
control).  `covert_leak` acts only in blind (L1/L2b) blocks, as covert attention toward the imagined screen
direction.  Reports per day: fixed L1 test-set AUC / r per axis, x-earned with AUC and floor, sighted
success / TTT / throughput, blind success, β table (deployed vs shadow, held-out and replay) and verdict;
pooled over each person's days and over people.
CLI: `python3 -m neural_cursor.simulate_champion --days D --hours H --users U --pop N
[--dry-penalty P …] [--screen-leak L] [--covert-leak C] [--replay-attention cursor|goal]
[--slot-minutes SLOT=MIN …] [--n-boot B] [--workers W] [--out PREFIX]`.

### L `realdata/nk_surface.py`
`realdata/run.sh nk_surface --cache C --feat F --nkfeat K [--out results/realdata] [--tasks imag_FF,imag_LR]
[--Ns …] [--ks …] [--draws 4] [--boot 2000] [--jobs 3]`: people (N ∈ 2…81) × first-k trials per person
(k ∈ 10…45) surface on ds004362, central-11, re-centred on own rest, fixed 25 test subjects (E2's split),
acc = a + b·log2 N + c·log2 k with test-subject bootstrap CIs (and a draw-resampling CI); writes
`results/realdata/E8_nk_surface_<task>.json`.
