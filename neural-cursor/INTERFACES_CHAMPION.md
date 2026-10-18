# Interface addendum: the champion design (modern SMR, body coordinates, certified purity)

Source of truth for the science: `proposals/pure_eeg_tournament/P_smr_modern_FINAL.md` (copied there by the
orchestrator) and the skeptic corrections in `proposals/pure_eeg_tournament/SKEPTIC_CORRECTIONS.md`, which
override the proposal where they disagree.  Everything in `INTERFACES.md` still holds; this file adds.

Pure-EEG rule (unchanged): only EEG reaches decoders, trainers, controllers, effectors.

## Body coordinates
Classes: FEET, HANDS (both), LEFT, RIGHT.  Velocity in body coordinates: y = feet (+) vs hands (−),
x = right (+) vs left (−).  Diagonals are two-part cues.  The screen mapping M (2×2, orthogonal) turns
body velocity into screen velocity: identity for normal use; held-out test mappings rot+90, rot−90,
mirror_x, mirror_y (never trained on).  Central-11 montage on the g.tec 32: FC5, FC1, FC2, FC6, C3, Cz, C4,
CP5, CP1, CP2, CP6.

## Label sources (provenance field "source")
- "L1"  fixation probe: central glyph cues a class, glyph→class map redrawn every session, gaze-contingent
        fixation; windows 1.0–4.0 s after cue.  LABEL = cued body vector.
- "L2b" blind closed loop: cue = central glyph or body-part word (never a location), cursor hidden, hidden
        goal, label-symmetric progress tone (pitch encodes distance decrease, identical for every direction);
        windows start ≥ 1.0 s after cue and end before the end flash.  LABEL = cued body vector.
- "S"   sighted closed loop, identity mapping: NO LABEL for the deployed decoder (re-centring only).
        The SHADOW decoder (positive control, never deployed) trains on S with ReFIT labels
        u = (target − cursor)/‖·‖ mapped to body coordinates through M⁻¹.
- "H"   held-out mapping test blocks (rot±90, mirrors): test only, never trained on.
- "R"   pixel replay test trials: the screen replays a past trajectory toward goal A while the person holds a
        pre-committed goal B: test only.
Trainers must refuse S/H/R windows for the deployed decoder (assert, with a test).

## New modules and owners
G `body_decoder.py`: BodyCoordinateDecoder(cfg, fs, channels=CENTRAL11, window_s=1.0):
   fit(windows, body_labels (n,2) in {-1,0,1}², provenance) — refuses non-L1/L2b; per axis: 8–30 Hz OAS
   covariance → re-centred on rest reference (set_reference(rest_windows)) → tangent space → L2 logistic;
   predict(window) -> (2,) body velocity in [−1,1] (2p−1 per axis, x = 0 unless x earned);
   earn_x(windows, labels, provenance) -> dict: x enabled iff the x-axis CV AUC beats a time-block
   (circular-shift) permutation p95 AND exceeds 0.60, pre-declared; returns the numbers.
H `mapping.py` + `certify.py`: Mapping(name) with .M, body_to_screen, screen_to_body;
   certify_mapping(decoder_outputs_body, body_intent, screen_dir_in_body, provenance) -> β_body, β_screen
   with trial-cluster bootstrap 95% CIs (OLS of output on both regressors, per axis and pooled);
   replay_test(outputs, private_intent, replayed_dir) -> same; certify(deployed, shadow, ...) -> verdict:
   CERTIFIED (deployed β_body CI > 0 and β_screen CI contains 0 or < 0.1, AND shadow β_screen CI > 0),
   TEST_INSENSITIVE (shadow β_screen not > 0), SCREEN_READING (deployed β_screen CI > 0), NO_BODY_SIGNAL.
I `protocol.py` + `data_engine.py` (one owner): new block kinds fixation_probe, blind_closed_loop, sighted,
   heldout_mapping, pixel_replay with the provenance "source" above; session_plan_champion(day, seed) per the
   proposal's session structure; data_engine.Labeler emits "source" and body labels; Trainer for the deployed
   decoder uses only L1/L2b; a ShadowTrainer uses S with ReFIT labels.  Keep all existing behaviour and tests.
J `sim_body.py`: BodySubject (extends the existing simulator): SMR topographies per class (feet = vertex
   mu/beta ERD; hands = bilateral C3/C4; left = C4-dominant; right = C3-dominant), transient time course
   (ERD peaks 1–2 s after cue, decays), per-subject gains, x-responder flag (probability 0.2 default),
   dry penalty d′ loss parameter (sweep 0–0.7), gain drift, and a SCREEN-LEAK pathway active only when
   targets are visible: posterior alpha lateralisation + frontal gaze potential toward the screen-direction
   of the target (gain parameter), zero in L1/L2b by construction unless covert_leak > 0.
K `simulate_champion.py` (after G–J): days of the champion protocol for deep participants with the
   population prior; reports per day: L1 fixed-test-set decoder AUC/r per axis, sighted success/TTT/TP,
   blind success, x-earned status, certification β table for deployed vs shadow under held-out mappings and
   replay, verdict.  CLI small default (< 10 min).  Also a dry-penalty sweep.
L `realdata/nk_surface.py`: people × trials-per-person scaling surface on ds004362 (central-11, feet/fists
   and L/R), fixed 25 test subjects, log-linear fit with bootstrap CIs.
