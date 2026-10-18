# Interface contract for the pure-EEG runtime

Pure-EEG rule: **only EEG samples ever reach a decoder, trainer, controller or effector.** Eye tracker,
EMG and accelerometer streams may be recorded (e.g. beside the EEG in a run directory) and passed to
`purity.py` for auditing; they are never inputs to `decoder`, `body_decoder`, `realtime`,
`data_engine.Trainer` / `BodyTrainer` / `ShadowTrainer` or any effector.
`data_engine.SessionStore` opens only `RUN_FILES = ("eeg.f32", "eeg_ts.f64", "events.csv", "cursor.csv")`
in a run directory, and every trainer rejects any window that is not `(n_samples, 32)`. The only
gaze-derived event a trainer's data ever passes through is `fixation_break`, which removes a trial in
`parse_trials` and is never a feature. Tests assert this where it applies (audit hook on file opens; spy
decoders recording every input shape; provenance dicts recording every field a trainer reads).

All signatures below are the code's; keyword arguments with defaults may be added, never removed.
This file covers the base runtime and the original (screen-coordinate) protocol, plus the parts of
`data_engine`, `purity` and `protocol` that serve both designs. The champion design (body coordinates,
certified purity) is in [`INTERFACES_CHAMPION.md`](INTERFACES_CHAMPION.md), which adds to this file.

## Module index
| Module | Role | Contract |
|---|---|---|
| `config` | frozen hyperparameter dataclasses (`DEFAULT`) | Base modules |
| `acquisition` | `SimulatedEEG`, `LSLEEG`, `RingBuffer` | Base modules |
| `signal` | causal filter, CAR, amplitude gate | Base modules |
| `features` | log band power, slow shift, shrinkage covariance, tangent space, `OnlineRecenter` | Base modules |
| `decoder` | `VelocityDecoder` (screen-coordinate ridge) | Base modules |
| `task` | cursor dynamics, goal posterior, trial runner, Fitts targets | Base modules |
| `logger` | `RunLogger` run directories | Base modules |
| `evaluation` | Fitts throughput, ITR, success / time-to-target, permutation floors, bootstrap CIs | Base modules |
| `effectors` | screen cursor, UDP, ROS2, planar arm | Runtime 1 |
| `realtime` | `RealTimeLoop`, `SimLSLStreamer` | Runtime 2 |
| `data_engine` | store → label → train → promote (original and champion) | Runtime 3; CHAMPION I |
| `purity` | surrogate-decoder audit, blank-cue certification, deployed row, microsaccades | Runtime 4 |
| `protocol` | block kinds, plans, cue schedulers, timings, payloads | Runtime 5; CHAMPION I |
| `sim_population`, `simulate_engine` | population simulator, multi-day engine simulation | Runtime 6 |
| `simulate`, `simulate_composite`, `scaling_ladder` | round-1/2 closed-loop simulations and resource ladder (`DESIGN.md`, kept for the record) | Legacy simulations |
| `body_decoder` | `BodyCoordinateDecoder` (deployed central-11 Riemannian decoder, idle gate) | CHAMPION G |
| `mapping`, `certify` | body ↔ screen mappings; held-out-mapping / replay certification | CHAMPION H |
| `sim_body` | body-coordinate SMR subjects with screen and covert leak pathways | CHAMPION J |
| `simulate_champion` | days of the champion protocol in simulation | CHAMPION K |
| `realtime_demo` | the deployed body decoder live over LSL driving a cursor and a planar arm | CHAMPION M |
| `realdata/*` | ds004362 loader, experiments, deep scaling, eye proxy, N × k surface | Real data; CHAMPION L |

## Base modules
- `acquisition.SimulatedEEG(hw, seed, noise_uv, intent_gain_uv, eye_gain_uv, errp_uv, drift)`: `start()`,
  `set_intent(vx, vy)`, `evoke(wrong)`, `generate(n) -> (n, 32) float32 µV`, `pull() -> (x, ts)`, `.ring`.
  `generate` is **chunk-invariant**: `generate(a)` then `generate(b)` equals `generate(a + b)`. Blinks
  are a Poisson process (`BLINK_RATE_HZ`) at absolute sample indices, drawn from their own RNG and
  queued until fully emitted, and drift is applied per sample. Real-time pushes of a few samples
  therefore see the same signal as offline blocks.
- `acquisition.LSLEEG(hw, timeout_s=10.0, name=None, source_id=None)`: `start()` resolves
  `type=hw.lsl_stream_type` and, if given, `name` / `source_id` (`predicate()` returns the XPath).
  It calls `inlet.open_stream()` so the first samples are kept, and checks the stream: channel count
  and rate, and — when the stream describes its channels — labels equal to `hw.channels` in order and
  unit microvolts (`RuntimeError` otherwise; a stream without labels is accepted with a warning).
  `pull() -> (x, ts)` returns time-corrected timestamps; `.ring` is also available. `close()` is
  idempotent and is also the context-manager exit; a failed or repeated `start()` closes the previous
  inlet first. Pin `source_id` whenever more than one EEG stream can be on the network.
- `signal.CausalFilter(hw, pre)(x) -> filtered` (stateful, chunk-invariant; starts in the steady state
  of its first sample, so a DC offset causes no onset transient); `signal.common_average(x, good)`;
  `signal.AmplitudeGate(hw, pre).update(window) -> good_mask` (bad: RMS > `gate_rms_factor` × the session
  median channel RMS, FP1/FP2 exempt; or sample-to-sample RMS < `gate_low_factor` (0.05) × its session
  median, every channel: a dead electrode held at a constant value, which the decoder's pooled checks
  cannot see, leaves the CAR and makes a body-decoder window invalid).
- `decoder.VelocityDecoder(cfg, fs, channels=None, feature_set="bandpower", adapt=False)`:
  `fit(windows: list[(n_samples, 32)], targets: (n, 2)) -> self`, `predict(window) -> (2,)` in [−1, 1],
  `predict_many(windows) -> (n, 2)`, `features(window)`.
- `task.Cursor(cfg).step(command, dt) -> CursorState(pos, vel, t)` (velocity smoothing defined per 0.1 s,
  `task.SMOOTHING_STEP_S`, rescaled to `dt`: the same time constant at 10 and 16 Hz); `task.GoalPosterior(targets, cfg)`;
  `task.TrialRunner(cfg, rng).run(...) -> TrialOutcome`; `task.fitts_targets(n, distance, width, rng)`.
- `logger.RunLogger(root, cfg, tag="", clock=time.time)`: `eeg(x, ts)`, `event(kind, payload, t)`,
  `cursor(t, pos, u, p_goal_max, inside, dwell)`, `close()`, `RunLogger.load_eeg(run_dir, n_channels)`.
  It logs raw µV before any filtering, with one clock for EEG, events and cursor rows. Every write is
  flushed (EEG and timestamps together), so a crash loses at most the chunk being written;
  `load_eeg` truncates a cut-off run to the samples both files hold, with a `RuntimeWarning`.
- `config.DEFAULT`: frozen dataclasses (hardware, pre, decoder, task, eval).
- `features`: `log_band_power(window, fs, bands)`, `slow_shift(window, frac=0.2)` (signed late-minus-early
  mean), `shrink_cov(window, shrinkage)`, `tangent_space(c, ref)`, `OnlineRecenter(train_ref, buffer, prior)`.
- `evaluation`: `fitts_throughput(outcomes)` (ISO 9241-9 effective form), `wolpaw_itr_bits_per_trial`,
  `success_rate`, `time_to_target`, `permutation_floor(statistic, labels, n_draws, rng, percentile=95.0)
  -> Floor(observed, floor_p95, p_value, n_draws)`, `bootstrap_ci`, `pearson_r`, `surrogate_table`.

## Runtime modules

1. `effectors.py`
   ```
   class Effector(dim, max_speed):         # base; context manager
       send(v, dt) -> None                 # dt validated first (a rejected call changes no state);
                                           # any NaN/inf entry → the whole command is 0 (stop); clipped; × max_speed
       state() -> dict                     # last command, max_speed, units, plus device state
       task_position() -> array | None     # displayed cursor in task units ([−1, 1]², +y up); None if none drawn
       close() -> None                     # idempotent; velocity devices get a final zero command
   NullEffector(dim=2)                     # replay / blank-feedback probes
   ScreenCursor(width_px, height_px, gain_px_s, backend="auto"|"log", edge_margin_px=1)
                                           # pyautogui if it imports (any import error → log-only)
   UDPVelocity(host, port, scale=1.0, dim=2)   # one JSON datagram {"t", "seq", "v"} per send
   ROS2Twist(topic="/cmd_vel", scale=0.2, angular_scale=0.8, mode="diff_drive"|"holonomic")
                                           # rclpy imported lazily; clear ImportError if absent
   PlanarArm2Link(l1=0.3, l2=0.25, max_speed=0.25, damping=0.05, max_joint_speed=3.0,
                  joint_limits=((−π, π), (0, 2.8)), workspace=(−0.55, 0.55, 0, 0.55))
                                           # damped-least-squares Jacobian inverse; joint + workspace limits;
                                           # dt > 10 ms integrated in substeps; a step leaving the workspace
                                           # (> 1 mm) is rejected; a q0 outside the workspace raises ValueError
   ```
2. `realtime.py`
   ```
   SimLSLStreamer(sim, chunk_s=0.010, stream_name="NeuralCursorSim", source_id=None)   # threading.Thread
       start() waits until the outlet exists; stop() joins and re-raises thread errors; set_intent(vx, vy)
   RealTimeLoop(source, decoder, effector, cfg, logger=None, assist=None, rate_hz=None,
                warmup_s=1.0, sample_clock=None, stale_s=None (2 periods), stale_abort_s=2.0,
                mapping=None (identity))
       run(duration_s, intent_fn=None) -> dict   # n_updates, n_expected, n_overruns, n_gated, n_samples,
                                                 # dropped_samples, nonfinite_samples, n_stale, n_invalid,
                                                 # aborted (None or reason), elapsed_s,
                                                 # latency_ms / jitter_ms {p50, p95, max, mean}, decoded_mean
       stop(); .decoded (decoder frame, e.g. body), .sent (screen command), per decoded update
   ```
   Each update runs pull → non-finite repair → CausalFilter → last window_s → AmplitudeGate → CAR
   over good channels → decoder.predict (with `good=` the gate/repair mask when the decoder's predict
   accepts it) → screen mapping `u = M·decoded` (`mapping.Mapping`; the champion's held-out blocks run
   live this way, and `realtime_start` logs it) → optional assist blend `(1 − g)·u + g·assist`
   (`g = cfg.task.assist_gain`, which must be > 0 if an assist is given) → effector.send. Latency is
   measured on the sample clock, from the newest sample's timestamp to the return of effector.send.
   Missed deadlines are skipped and counted, not burst, and the dt handed to the effector is capped at
   2 periods. Non-finite samples are held at the channel's last finite value (logged as repaired, the
   channel gated while the repair is in the window: excluded from CAR, and a decoder that takes `good`
   refuses a window with a bad channel it reads); a non-finite window or output sends 0 (`n_invalid`). No new sample for `stale_s` → the tick sends 0 (`n_stale`); for `stale_abort_s` →
   the run ends (`aborted`). However `run` ends it sends a final zero command; the caller owns, and
   closes, the effector and the logger. A `stop()` before `run()` is kept (that run returns at once).
   `cursor.csv` holds the displayed cursor (`effector.task_position()`, else the internal model), and
   `realtime_start` records `cursor_source` ∈ {"effector", "model"}; the assist uses the same position.
   `intent_fn`'s return value is only logged.
3. `data_engine.py`
   ```
   SessionStore(root, n_channels=32)        # list_runs(), load(run_dir) -> Session (EEG, ts, events, cursor, trials)
   parse_trials(session, events) -> list[Trial]   # Trial(session, trial, t_start, t_end, goal, block,
                                            #       feedback_visible, start_pos, source=None, info={})
   Labeler(cfg, hop_s=0.25, settle_s=2.0, min_goal_dist=None, filter_chunk_s=60.0)
       label(session) -> LabelledSet(windows, labels (n, 2), provenance); signature()
   ModelRegistry(root)                      # register(model, meta) -> version, load, meta, set_champion,
                                            # champion_record, champion_history; append-only
   Trainer(cfg, blocks=BLOCK_KINDS, feature_set="bandpower", pop_ref_windows=300,
           recenter_visible=True, max_free_share=0.2)
       fit(data, population=None) -> WeightedVelocityDecoder   # w_own = max(1, n_pop / 300), weights mean 1
   BodyTrainer / ShadowTrainer               # champion trainers (INTERFACES_CHAMPION I)
   ChampionChallenger(margin=0.02, heldout_frac=0.2, min_heldout=40, min_heldout_trials=8,
                      n_resamples=2000, alpha=0.05, seed=20261004, auc_margin=0.02, min_class_trials=3)
       candidate_heldout(data, body=False); training_set; subset; is_body(model);
       score(model, data, session_references=None);
       challenge(champion, challenger, data, candidate=None, champion_sessions=None,
                 session_references=None) -> dict; freeze(keys)
   mean_cosine(pred, labels); cosines(pred, labels); constant_ceiling(labels)   # |mean unit label|
   body_outputs(model, data, session_references=None); axis_outputs(...); body_auc(...) -> {axis: AUC}
   DataEngine(store, labeler, trainer, registry, judge=None, population=None, min_new_windows=200,
              cache_dir=None, session_references=None)
       ingest(run_dir) -> n_new_windows ;  maybe_retrain(force=False) -> {"promoted": bool, ...}
   ```
   - **Blocks.** `BLOCK_KINDS = ("cued_blank", "closed_loop", "blank_feedback", "free")` (the original
     protocol; the only blocks `Trainer` selects), `CHAMPION_BLOCKS = ("fixation_probe",
     "blind_closed_loop", "sighted", "heldout_mapping", "pixel_replay")`, `ALL_BLOCKS`; drawn-goal blocks
     `TARGET_BLOCKS = ("closed_loop", "blank_feedback", "free")` and `CHAMPION_TARGET_BLOCKS = ("sighted",
     "heldout_mapping", "pixel_replay")`. An unknown block raises `ValueError`.
   - **Events.** A `trial_start` payload is `{trial, goal | null, block, feedback_visible, start_pos}`; a
     champion block adds `source`, `mapping`, `body_cue`, `cue_class`, `cue_mode`, `variant`,
     `replay_goal`, `instructed` (`protocol.champion_trial_start_payload`, validated per source:
     INTERFACES_CHAMPION I). A `trial_end` payload is `{trial, goal?, aborted?}`, where `goal` is a
     hindsight reveal used only when the start's goal was null (a selection the person undid is not a
     reveal: write null) and `aborted: true` discards a champion trial. A trial with no end is discarded.
     A cued_blank `trial_start` after a `cue` with no `cue_offset` in between raises `ValueError`; so does
     labelling a run with visible-cursor trials whose `realtime_start` says `"cursor_source": "model"`.
     Champion runs also use `cue` (onset, optional `body`), `cue_offset`, `fixation_break` (discards the
     open L1 / L2b / blind-J trial), `outcome_flash` (must follow `trial_end` in blind blocks) and
     `body_report` (C2b keyed report, after its trial's end).
   - **Label.** The label is the unit vector from the cursor *as the person knows it* to the revealed
     goal at the window's end. With a visible cursor that is the last logged cursor position; with a
     hidden cursor (`feedback_visible` false) it is `start_pos` for the whole trial. For L1 / L2b windows
     it is the normalised cued body vector (body = screen under the identity; nothing is drawn).
   - **Provenance.** Every window carries `{"block", "feedback_visible", "session", "t", "trial",
     "display_goal"}`: `t` is the window's last timestamp, `session` the run directory's name, and
     `display_goal` the goal as drawn ([x, y] for `TARGET_BLOCKS + CHAMPION_TARGET_BLOCKS`; None for
     cued_blank, fixation_probe and blind_closed_loop). `display_goal` only reaches `purity.py` (and
     `BodyTrainer`, which refuses any window where it is set). Champion windows add `source` (L1, L2b, S,
     H, R, J), `mapping`, `instructed`; L1 / L2b add `body_label` ([bx, by] ints in {−1, 0, 1}: the
     deployed decoder's label), `cue_class`, `cue_mode`; visible trials add `cursor_pos` /
     `cursor_pos_start`; H adds `variant`, `body_report`; R adds `replay_goal`. Original-protocol windows
     keep exactly the six original keys. `(session, t)` is a window's identity.
   - **Training labels.** Visible-cursor labels are re-centred per (session, block) (they point
     against the deployed decoder's bias); free-use windows carry at most `max_free_share` of own
     weight (success-selected hindsight labels). Once n_pop ≥ `pop_ref_windows` the prior is a fixed
     pseudo-count of that many windows: `--pop N` changes its composition, not its strength.
   - **Promotion.** A challenger trains on everything except the frozen held-out set H and the
     candidate H′, which is the most recent 20 % of cued_blank windows cut at a trial boundary (≥
     `min_heldout` windows and `min_heldout_trials` trials). Before the first champion it must beat
     the null decoder on H′: mean cosine > margin and above a whole-trial label-permutation floor
     (p ≤ alpha). Afterwards it is scored with the champion on H ∪ (windows of H′ from sessions the
     champion never trained on), and promoted iff the lower bound of a 95 % trial-cluster paired
     bootstrap CI of (challenger − champion) mean cosine exceeds the margin. On promotion H′ becomes
     the new frozen H.
   - **Body metric** (a `BodyCoordinateDecoder` challenger, `is_body`). H′ is the most recent 20 % of
     **L1** fixation-probe windows (`candidate_heldout(data, body=True)`: the selection probes, never in
     the training set of the candidate they select), with ≥ `min_class_trials` FEET and HANDS probes.
     The score is the per-axis window AUC of the decoder's axis outputs (`axis_outputs`: re-centred per
     session with `session_references`, idle gate off, an invalid window = 0) against the cued class:
     y always, x when the probes hold RIGHT and LEFT and either compared decoder drives x; the statistic
     is the mean over scored axes. First champion: mean AUC > 0.5 + `auc_margin` and above a whole-trial
     class-permutation floor (p ≤ alpha). Afterwards: lower bound of a 95 % class-stratified trial
     bootstrap CI of the paired mean-AUC difference > `auc_margin`. A test set with any non-L1 window
     raises. Result keys: `metric` ("body_auc"), `axes`, `auc` {challenger, champion: {axis}},
     `challenger_auc`, `champion_auc`, `n_class_trials`, `n_heldout`, `n_heldout_trials`, `n_fresh`,
     `promoted`, and `floor` / `p_value` or `ci95`.
   - **DataEngine with a `BodyTrainer`** (`engine.body`): H′ from L1 probes, training on
     `BodyTrainer.select` (L1 / L2b) of everything outside H ∪ H′ plus `population` (then a `LabelledSet`
     of other people's L1 probes), the body metric, and `session_references` ({run name: rest
     covariance}, filled by the caller as sessions arrive) for training and scoring. Label with
     `Labeler(decoder.loop_config(cfg))` (1-s windows).
4. `purity.py`
   ```
   PurityAudit(cfg, fs=None, feature_set="bandpower", n_permutations=None (cfg, ≥ 200), n_splits=5, seed=None,
               ocular_fraction=0.5, min_group_n=20)
       run(windows, labels, provenance, auditors: dict | None = None, cursor_logs=None,
           deployed=None, gaze_fs: dict | None = None) -> dict
   is_certifying(provenance_record) -> bool;  CERTIFYING_BLOCKS;  CERTIFYING_SOURCES;  SPATIAL_BLOCKS
   auditor_features(stream, n, gaze_fs=None);  microsaccades(gaze, fs);  microsaccade_features(gaze, fs)
   circular_shift_permutations(provenance, n_permutations, rng)
   ```
   - **Reports.** Each surrogate gets a permutation floor (p95) at matched n, scored with the same
     scaler and ridge on contiguous folds. The null circularly shifts the labels within each session
     (`circular_shift_permutations`), keeping their autocorrelation and per-session distribution; a
     window-level shuffle is anti-conservative on trial-structured labels and drifting features.
     The surrogates are:
     - **intercept:** no EEG at all (the training-fold mean label): the baseline skill is measured above.
     - **full:** all channels (a 32-channel bandpower ridge: the same model for every surrogate, so
       surrogate-vs-full differences are differences of inputs).
     - **deployed** (when `deployed` is given): the DEPLOYED decoder's own out-of-sample predictions per
       window (an (n, k) array, (n,) for binary labels, or a callable `windows -> predictions` that
       receives the EEG windows only), scored by the same metric against the same permuted labels with
       no refit; a non-finite row scores as the zero command the loop sends (`n_zero_rows`). frontal,
       slow and occipital then also report `share_of_deployed` = (surrogate − intercept) / (deployed −
       intercept): reported, never flagged.
     - **frontal:** FP1, FP2, AF3, AF4, F7, F8.
     - **slow:** 0.1 s binned means, which keep the 1–4 Hz signal and null 10, 20 and 50 Hz.
     - **occipital:** PO7, PO3, PO4, PO8, Oz (reported; not flagged, because occipital channels also
       carry alpha-lateralised attention — visual leakage outside the blank-cue blocks is listed as
       `untested`).
     - **display:** everything spatial on screen, with no EEG: the displayed goal (provenance
       `display_goal`, required for target blocks, `SPATIAL_BLOCKS` = the provenance blocks whose
       `protocol.Block.screen_items` hold a spatial item) and, with a visible cursor, its position and
       its velocity over the previous 1 s (`cursor_logs`, an `(m, ≥3) [t, x, y]` array or
       `{session: array}`, required then). Zero only where nothing spatial was drawn.
     - **auditor:\<name\>:** one per auditor stream, when auditors are given. Raw segments (n,
       n_samples, d) become mean, late-minus-early shift and log SD per channel; for an auditor named
       in `gaze_fs` ({name: Hz}; x, y in the first two columns) microsaccade rate and mean direction are
       appended (velocity-threshold events, Engbert & Kliegl 2003: 5-sample velocity, λ = `MS_LAMBDA` = 6
       median-based SDs per axis, ≥ `MS_MIN_DURATION_S` = 6 ms) and the report adds
       `microsaccade_rate_hz`. These are the gaze features that still vary while fixation holds.
   - **Feedback test.** closed_loop vs blank_feedback (free use excluded) skill of the full decoder's
     out-of-fold predictions, skill = per-window score minus the intercept-only score, with a
     stratified bootstrap CI over *trials* `(session, trial)`, not windows, because windows of one
     trial are dependent.
   - **Result.** `metric`, `n`, `n_permutations`, `floor_percentile`, `null`, `feature_set`, `reports`,
     `by_block` (full and intercept-only mean score per block), `feedback`, `untested`, `flags`,
     `certification`, `verdict`.
   - **Flags** (whole window set).
     - **OCULAR_LEAK:** frontal or slow beats its floor with a Bonferroni-corrected permutation
       p ≤ α/2, and reaches ≥ 50 % of the full decoder's skill above chance (= the intercept-only score).
     - **DISPLAY_LEAK:** the display surrogate beats its floor (on a set with drawn targets or a visible
       cursor this says "some windows are display-identifiable", expected by design).
     - **FEEDBACK_CIRCULAR:** the lower bound of the 95 % CI of (visible − blank) is above 0.
     - **PASS:** none of the above.
   - **Certification (the verdict of record).** The same surrogates refit on the certifying windows only:
     `is_certifying` is true when the window's block (if recorded) is in `CERTIFYING_BLOCKS` = the
     provenance blocks of `protocol.BLANK_CUE_KINDS` (cued_blank, fixation_probe, blind_closed_loop) and
     its source (if recorded) is in `CERTIFYING_SOURCES` = ("L1", "L2b"); blind just-intend (J) is
     excluded. Needs ≥ max(2·n_splits, `min_group_n`) such windows (else `skipped`). Returns `n`,
     `blocks`, `sources`, `reports` (incl. "deployed" and `share_of_deployed` when given), `decodes`,
     `deployed_decodes` (None without deployed predictions), `flags`, `verdict`. Verdict tokens:
     **NO_CORTICAL_EVIDENCE** (full does not beat its floor), **DEPLOYED_NO_EVIDENCE** (deployed
     predictions given and they do not beat theirs), **OCULAR_LEAK** / **DISPLAY_LEAK** (as above),
     **GAZE_TRACKS_LABEL** (an auditor beats its floor and the intercept although the person fixates;
     an auditor constant on these windows is `skipped`), else **PASS**. On blank-cue windows
     DISPLAY_LEAK and FEEDBACK_CIRCULAR cannot fire and gaze position is (near) constant, so the
     checks that can fail there are OCULAR_LEAK, the microsaccade auditors and the evidence verdicts.
5. `protocol.py`
   ```
   Block(kind, duration_s, cue=None, feedback=None, seed=0, mapping=None, variant=None, instructed=None,
         just_intend=False, replay_fraction=None, localiser=False)
       # kinds BLOCK_KINDS: setup, cued_blank, closed_loop, blank_feedback, free_use, rest;
       #       CHAMPION_BLOCK_KINDS: fixation_probe, blind_closed_loop, sighted, heldout_mapping, pixel_replay
       # cue/feedback (and the champion options) default per kind and are validated
       .feedback_visible, .provenance_block (free_use → "free"; setup/rest → None; champion kinds → kind),
       .source, .is_validity, .is_test_only, .is_champion, .replay_block(), .screen_items(phase=None |
       "cue" | "intent" | "outcome")
   ALL_BLOCK_KINDS; BLANK_CUE_KINDS; PROVENANCE_BLOCK; CUE_MODES; FEEDBACK_MODES; SOURCES;
   DEPLOYED_TRAINING_SOURCES; TEST_ONLY_SOURCES; SOURCES_BY_KIND
   session_plan(day: int, seed: int) -> list[Block]           # deterministic; plan_summary, render_plan_markdown
   CueScheduler(seed, reps_per_block=2, max_run=2, logger=None)   # balanced, randomized; .next(t, mode)
   CuedTrialTiming(cue_s=1.0, settle_s=0.5, intent_s=4.0, iti_s=1.5)   # trial_start logged at intent onset
   trial_start_payload(block, trial, goal, start_pos=(0, 0)) ; cued_goal(cue) ; ZipfTargets(seed)
   session_plan_champion(...) and the champion cue/timing API      # INTERFACES_CHAMPION I
   ```
   - **Cued intent.** Cued-intent blocks present the direction by an auditory word or a central
     symbol and show NOTHING at a spatial position, so the label cannot be identified from the
     display. The stimulus lasts `cue_s` and its end is logged as `cue_offset` (`EVENT_CUE_OFFSET`)
     before `trial_start`; during the intent phase only the fixation mark is drawn.
   - **Blank-cue kinds.** `BLANK_CUE_KINDS` = the labelled kinds whose `screen_items` hold nothing at a
     trial-dependent position in any phase, with the cursor hidden: cued_blank, fixation_probe,
     blind_closed_loop (derived from the renderer's contract, so `purity.CERTIFYING_BLOCKS` cannot drift
     from what is drawn).
   - **Blank-feedback probes** (original protocol only). These run the decoder with the cursor hidden.
     They come straight after a closed_loop block, and every original session has one; the champion
     design does not use blank_feedback (its goal is drawn, so its labels are display-identifiable).
   - **cued_blank share.** cued_blank keeps at least 20 % of task time in the original protocol.
6. `sim_population.py` + `simulate_engine.py`
   - `PopulationSim(n_subjects, seed, hw, ..., ocular_leak_fraction=0, user_learning_rate=0)`. Its
     `SubjectEEG` instances are `SimulatedEEG` subclasses with:
     - topography = normalise(shared·(1 − h) + individual·h), with h ~ U[0.3, 0.7];
     - per-channel day gain drift;
     - optional gaze-driven ocular leak;
     - optional user learning (default 0).

     `begin_block(day, b)` makes a block's EEG independent of earlier blocks. Subject i depends only
     on (seed, i).
   - `simulate_engine.py` runs days × hours of simulated use through protocol → RunLogger →
     DataEngine → an offline-paced copy of the realtime chain (no wall clock). It produces learning
     curves (decoder cosine [trial CI] / r on a FIXED evaluation pool of `eval_minutes` auditory
     cued_blank per simulated day that no model trains on, with the pool's constant-output ceiling;
     closed-loop success, time-to-target, throughput), with and without a population prior, plus a
     purity audit per day. Closed-loop intent equals the label unless `intent_angle_sd_deg` /
     `intent_lapse_prob` are set (the report says so). The simulated eye tracker is written as
     `eyetracker.f64` and reaches only `PurityAudit`.
     CLI: `python3 -m neural_cursor.simulate_engine --days D --hours H --pop N [--users U] [--seed S]
     [--pop-minutes M] [--ocular-leak F] [--learning-rate R] [--min-new-windows W] [--min-heldout W]
     [--min-heldout-trials T] [--eval-minutes M] [--intent-angle-sd DEG] [--intent-lapse P]
     [--fixation-break F] [--heterogeneity LO HI] [--pop-ref-windows W] [--audit-permutations P]
     [--workers W] [--out PREFIX] [--keep-runs DIR]`.

## Legacy simulations (the first design round, `DESIGN.md`; kept for the record)
- `simulate.py`: closed-loop Fitts simulation of one decoder per simulated user with oracle, decoder,
  decoder + assist, EOG-only and no-neural conditions. `simulate(cfg, users, trials, gain, eye, seed, ...)`,
  `to_markdown`; CLI `python3 -m neural_cursor.simulate --users U --trials T --gain G --eye E [--seed S]
  [--json PATH] [--adapt] [--features bandpower|bandpower+slow]`.
- `simulate_composite.py`: round 2, velocity decoder + shared-autonomy agent + error-evidence decoder.
  CLI adds `--errp UV --drift D`.
- `scaling_ladder.py`: decoder quality vs labelled volume on a fixed held-out set. CLI `python3 -m
  neural_cursor.scaling_ladder --users U --gain G --errp UV [--seed S] [--json PATH]`.

## Real data (`realdata/`, run isolated through `realdata/run.sh <module> ...`)
- `load.py` (ds004362 → per-subject epoch caches), `experiments.py` (E1–E6 subcommands: features,
  within, confound, scaling, fewshot, online, transfer), `deep.py` (EEGNet people-scaling, E2-deep),
  `eye_proxy.py` (eye proxy vs motor cortex, cross-subject), `nk_surface.py` (people × trials surface,
  INTERFACES_CHAMPION L). Public data carry display-cued labels: they never supervise a deployed decoder.
