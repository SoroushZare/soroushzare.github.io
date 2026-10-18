# Interface contract for the pure-EEG runtime

Pure-EEG rule: **only EEG samples ever reach a decoder, controller or effector.** Eye tracker, EMG and
accelerometer streams may be recorded (e.g. beside the EEG in a run directory) and passed to `purity.py`
for auditing; they are never inputs to `decoder`, `realtime`, `data_engine.Trainer` or any effector.
`data_engine.SessionStore` opens only `RUN_FILES = ("eeg.f32", "eeg_ts.f64", "events.csv", "cursor.csv")`
in a run directory, and `Trainer` rejects any window that is not `(n_samples, 32)`. Tests assert this
where it applies (audit hook on file opens; spy decoders recording every input shape).

All signatures below are the code's; keyword arguments with defaults may be added, never removed.

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
  `signal.AmplitudeGate(hw, pre).update(window) -> good_mask`.
- `decoder.VelocityDecoder(cfg, fs, channels=None, feature_set="bandpower", adapt=False)`:
  `fit(windows: list[(n_samples, 32)], targets: (n, 2)) -> self`, `predict(window) -> (2,)` in [−1, 1],
  `predict_many(windows) -> (n, 2)`, `features(window)`.
- `task.Cursor(cfg).step(command, dt) -> CursorState(pos, vel, t)`; `task.GoalPosterior(targets, cfg)`;
  `task.TrialRunner(cfg, rng).run(...) -> TrialOutcome`; `task.fitts_targets(n, distance, width, rng)`.
- `logger.RunLogger(root, cfg, tag="", clock=time.time)`: `eeg(x, ts)`, `event(kind, payload, t)`,
  `cursor(t, pos, u, p_goal_max, inside, dwell)`, `close()`, `RunLogger.load_eeg(run_dir, n_channels)`.
  It logs raw µV before any filtering, with one clock for EEG, events and cursor rows. Every write is
  flushed (EEG and timestamps together), so a crash loses at most the chunk being written;
  `load_eeg` truncates a cut-off run to the samples both files hold, with a `RuntimeWarning`.
- `config.DEFAULT`: frozen dataclasses (hardware, pre, decoder, task, eval).

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
                warmup_s=1.0, sample_clock=None, stale_s=None (2 periods), stale_abort_s=2.0)
       run(duration_s, intent_fn=None) -> dict   # n_updates, n_expected, n_overruns, n_gated, n_samples,
                                                 # dropped_samples, nonfinite_samples, n_stale, n_invalid,
                                                 # aborted (None or reason), elapsed_s,
                                                 # latency_ms / jitter_ms {p50, p95, max, mean}, decoded_mean
       stop(); .decoded, .sent (per decoded update)
   ```
   Each update runs pull → non-finite repair → CausalFilter → last window_s → AmplitudeGate → CAR
   over good channels → decoder.predict → optional assist blend `(1 − g)·u + g·assist`
   (`g = cfg.task.assist_gain`, which must be > 0 if an assist is given) → effector.send. Latency is
   measured on the sample clock, from the newest sample's timestamp to the return of effector.send.
   Missed deadlines are skipped and counted, not burst, and the dt handed to the effector is capped at
   2 periods. Non-finite samples are held at the channel's last finite value (logged as repaired, the
   channel gated while the repair is in the window); a non-finite window or output sends 0
   (`n_invalid`). No new sample for `stale_s` → the tick sends 0 (`n_stale`); for `stale_abort_s` →
   the run ends (`aborted`). However `run` ends it sends a final zero command; the caller owns, and
   closes, the effector and the logger. A `stop()` before `run()` is kept (that run returns at once).
   `cursor.csv` holds the displayed cursor (`effector.task_position()`, else the internal model), and
   `realtime_start` records `cursor_source` ∈ {"effector", "model"}; the assist uses the same position.
   `intent_fn`'s return value is only logged.
3. `data_engine.py`
   ```
   SessionStore(root, n_channels=32)        # list_runs(), load(run_dir) -> Session (EEG, ts, events, cursor, trials)
   Labeler(cfg, hop_s=0.25, settle_s=2.0, min_goal_dist=None)   # label(session) -> LabelledSet
   ModelRegistry(root)                      # register(model, meta) -> version, load, meta, set_champion,
                                            # champion_record, champion_history; append-only
   Trainer(cfg, blocks=BLOCK_KINDS, feature_set="bandpower", pop_ref_windows=300,
           recenter_visible=True, max_free_share=0.2)
       fit(data, population=None) -> WeightedVelocityDecoder   # w_own = max(1, n_pop / 300), weights mean 1
   ChampionChallenger(margin=0.02, heldout_frac=0.2, min_heldout=40, min_heldout_trials=8,
                      n_resamples=2000, alpha=0.05, seed=20261004)
   mean_cosine(pred, labels); cosines(pred, labels); constant_ceiling(labels)   # |mean unit label|
   DataEngine(store, labeler, trainer, registry, judge=None, population=None, min_new_windows=200)
       ingest(run_dir) -> n_new_windows ;  maybe_retrain(force=False) -> {"promoted": bool, ...}
   ```
   - **Events.** A `trial_start` payload is `{trial, goal | null, block, feedback_visible, start_pos}`
     with `block ∈ {"cued_blank", "closed_loop", "blank_feedback", "free"}`. A `trial_end` payload is
     `{trial, goal?}`, where `goal` is a hindsight reveal used only when the start's goal was null.
     A selection the person undid is not a reveal: write null. A trial with no end is discarded.
     A cued_blank `trial_start` after a `cue` with no `cue_offset` in between raises `ValueError`;
     so does labelling a run with visible-cursor trials whose `realtime_start` says
     `"cursor_source": "model"`.
   - **Label.** The label is the unit vector from the cursor *as the person knows it* to the revealed
     goal at the window's end. With a visible cursor that is the last logged cursor position; with a
     hidden cursor (`feedback_visible` false) it is `start_pos` for the whole trial.
   - **Provenance.** Each window carries `{"block", "feedback_visible", "session", "t", "trial",
     "display_goal"}`; `display_goal` is the goal as drawn (None for cued_blank) and only reaches
     `purity.py`.
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
4. `purity.py`
   ```
   PurityAudit(cfg, fs=None, feature_set="bandpower", n_permutations=None (cfg, ≥ 200), n_splits=5, seed=None)
       run(windows, labels, provenance, auditors: dict | None = None, cursor_logs=None) -> dict
   ```
   - **Reports.** Each surrogate gets a permutation floor (p95) at matched n, scored with the same
     scaler and ridge on contiguous folds. The null circularly shifts the labels within each session
     (`circular_shift_permutations`), keeping their autocorrelation and per-session distribution; a
     window-level shuffle is anti-conservative on trial-structured labels and drifting features.
     The surrogates are:
     - **intercept:** no EEG at all (the training-fold mean label): the baseline skill is measured above.
     - **full:** all channels.
     - **frontal:** FP1, FP2, AF3, AF4, F7, F8.
     - **slow:** 0.1 s binned means, which keep the 1–4 Hz signal and null 10, 20 and 50 Hz.
     - **occipital:** PO7, PO3, PO4, PO8, Oz (reported; not flagged, because occipital channels also
       carry alpha-lateralised attention — visual leakage outside cued_blank is listed as `untested`).
     - **display:** everything spatial on screen, with no EEG: the displayed goal (provenance
       `display_goal`, required for target blocks) and, with a visible cursor, its position and its
       velocity over the previous 1 s (`cursor_logs`, an `(m, ≥3) [t, x, y]` array or
       `{session: array}`, required then). Zero only where nothing spatial was drawn (cued_blank).
     - **auditor:\<name\>:** one per auditor stream, when auditors are given.
   - **Feedback test.** closed_loop vs blank_feedback (free use excluded) skill of the full decoder's
     out-of-fold predictions, skill = per-window score minus the intercept-only score, with a
     stratified bootstrap CI over *trials* `(session, trial)`, not windows, because windows of one
     trial are dependent.
   - **Result.** Also `by_block` (full and intercept-only mean score per block) and `untested`.
   - **Flags.**
     - **OCULAR_LEAK:** frontal or slow beats its floor with a Bonferroni-corrected permutation
       p ≤ α/2, and reaches ≥ 50 % of the full decoder's skill above chance (= the intercept-only score).
     - **DISPLAY_LEAK:** the display surrogate beats its floor.
     - **FEEDBACK_CIRCULAR:** the lower bound of the 95 % CI of (visible − blank) is above 0.
     - **PASS:** none of the above.
5. `protocol.py`
   ```
   Block(kind, duration_s, cue=None, feedback=None, seed=0)   # kinds: setup, cued_blank, closed_loop,
       # blank_feedback, free_use, rest; cue/feedback default per kind and are validated
       .feedback_visible, .provenance_block (free_use → "free"; setup/rest → None), .is_validity,
       .screen_items(phase=None | "cue" | "intent")
   session_plan(day: int, seed: int) -> list[Block]           # deterministic
   CueScheduler(seed, reps_per_block=2, max_run=2, logger=None)   # balanced, randomized; .next(t, mode)
   CuedTrialTiming(cue_s=1.0, settle_s=0.5, intent_s=4.0, iti_s=1.5)   # trial_start logged at intent onset
   trial_start_payload(block, trial, goal, start_pos=(0, 0)) ; cued_goal(cue) ; ZipfTargets(seed)
   ```
   - **Cued intent.** Cued-intent blocks present the direction by an auditory word or a central
     symbol and show NOTHING at a spatial position, so the label cannot be identified from the
     display. The stimulus lasts `cue_s` and its end is logged as `cue_offset` (`EVENT_CUE_OFFSET`)
     before `trial_start`; during the intent phase only the fixation mark is drawn.
   - **Blank-feedback probes.** These run the decoder with the cursor hidden. They come straight
     after a closed_loop block, and every session has one.
   - **cued_blank share.** cued_blank keeps at least 20 % of task time.
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
     CLI: `python3 -m neural_cursor.simulate_engine --days D --hours H --pop N [--out PREFIX]
     [--eval-minutes M] [--intent-angle-sd DEG] [--intent-lapse P] [--heterogeneity LO HI]
     [--pop-ref-windows W] [--min-heldout-trials T]`.
