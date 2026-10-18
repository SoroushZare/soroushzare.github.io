# Interface contract for the pure-EEG runtime (builders must follow exactly)

Pure-EEG rule: **only EEG samples ever reach a decoder or controller.** Eye tracker, EMG and
accelerometer streams may be recorded and passed to `purity.py` for auditing; they are never inputs
to `decoder`, `realtime`, `data_engine.Trainer` or any effector. Tests must assert this where it applies.

Existing modules (do not change their public behaviour; additive changes only, owned as listed):
- `acquisition.SimulatedEEG(hw, seed, intent_gain_uv, eye_gain_uv, errp_uv, drift)`: `start()`,
  `set_intent(vx, vy)`, `generate(n) -> (n, 32) float32 µV`, `pull() -> (x, ts)`, `.ring`.
- `acquisition.LSLEEG(hw)`: `start()`, `pull() -> (x, ts)`, `.ring`.  (Owner: realtime builder; must call
  `inlet.open_stream()` in `start()` so the first samples are not dropped.)
- `signal.CausalFilter(hw, pre)(x) -> filtered`, stateful; `signal.common_average(x, good)`;
  `signal.AmplitudeGate(hw, pre).update(window) -> good_mask`.
- `decoder.VelocityDecoder(cfg, fs, channels=None, feature_set="bandpower", adapt=False)`:
  `fit(windows: list[(n_samples, 32)], targets: (n, 2)) -> self`, `predict(window) -> (2,)` in [−1, 1],
  `predict_many(windows) -> (n, 2)`.
- `task.Cursor(cfg).step(command, dt) -> CursorState(pos, vel, t)`; `task.GoalPosterior(targets, cfg)`;
  `task.fitts_targets(n, distance, width, rng)`.
- `logger.RunLogger(root, cfg, tag)`: `eeg(x, ts)`, `event(kind, payload, t)`, `cursor(t, pos, u, p_goal_max, inside, dwell)`,
  `close()`, `RunLogger.load_eeg(run_dir, n_channels)`.  Raw µV is logged before any filtering.
- `config.DEFAULT` (frozen dataclasses: hardware, pre, decoder, task, eval).

New modules and owners (one owner per file; tests in `tests/test_<module>.py`):

1. `effectors.py` (owner A)
   ```
   class Effector:                      # base
       def send(self, v: np.ndarray, dt: float) -> None   # v in [−1, 1]^k, k = 2 for planar
       def state(self) -> dict                             # e.g. {"pos": [...]} for logging
       def close(self) -> None
   ScreenCursor(width_px, height_px, gain_px_s, backend="auto"|"log")   # pyautogui if importable, else log-only
   UDPVelocity(host, port, scale=1.0)   # JSON {"t":..., "v":[...]} per send; any robot can listen
   ROS2Twist(topic="/cmd_vel", scale=...)  # optional rclpy; raises a clear ImportError message if absent
   PlanarArm2Link(l1=0.3, l2=0.25)      # simulated 2-link arm; v = end-effector velocity → joint velocity
                                        # via damped least-squares Jacobian inverse; joint limits enforced
   NullEffector()
   ```
2. `realtime.py` (owner B)
   ```
   class SimLSLStreamer(threading.Thread)  # publishes a SimulatedEEG over an LSL outlet at real rate
   class RealTimeLoop:
       __init__(source, decoder, effector, cfg, logger=None, assist=None, rate_hz=None)
       run(duration_s, intent_fn=None) -> dict   # returns latency stats (p50/p95/max ms), loop jitter,
                                                 # n_updates, n_gated, dropped samples
   ```
   Per update: pull → CausalFilter → AmplitudeGate → CAR over good channels → last window_s → decoder.predict
   → optional assist blend → effector.send.  Latency = time from newest sample's arrival to effector.send.
3. `data_engine.py` (owner C)
   ```
   class SessionStore(root)                 # lists run dirs, loads EEG + events + cursor logs
   class Labeler(cfg)                       # windows + labels from logs: label = unit vector from the cursor
                                            # to the REVEALED goal at the window end; each window carries
                                            # provenance {"block": "cued_blank"|"closed_loop"|"blank_feedback"|"free",
                                            #             "feedback_visible": bool, "session": id, "t": float}
   class ModelRegistry(root)                # versioned models (joblib) + metadata json; append-only
   class Trainer(cfg)                       # fits a decoder on selected provenance; population prior +
                                            # personal up-weighting (weight w_own = max(1, n_pop / 300))
   class ChampionChallenger                 # new model is promoted only if it beats the champion on a
                                            # FROZEN held-out set of the most recent cued_blank windows
   class DataEngine(store, labeler, trainer, registry)
       ingest(run_dir) -> n_new_windows ;  maybe_retrain() -> {"promoted": bool, ...}
   ```
4. `purity.py` (owner D)
   ```
   class PurityAudit(cfg)
       run(windows, labels, provenance, auditors: dict | None) -> dict
   ```
   Reports, each with a label-permutation floor at matched n: decoder on all channels; frontal-only
   (FP1, FP2, AF3, AF4, F7, F8) decoder; slow-potential-only (1–4 Hz binned means) decoder; occipital-only
   (PO7, PO3, PO4, PO8, Oz); display-only surrogate (predict label from the displayed cursor velocity of the
   previous 1 s, no EEG); auditor-only decoder (eye tracker / EMG features) when auditors are given;
   blank-feedback vs feedback-visible accuracy. Flags: "OCULAR_LEAK", "DISPLAY_LEAK", "FEEDBACK_CIRCULAR",
   "PASS".
5. `protocol.py` (owner E)
   ```
   class Block(kind, duration_s, cue="auditory"|"central_symbol", feedback="none"|"cursor"|"delayed")
   def session_plan(day: int, seed: int) -> list[Block]   # deterministic
   class CueScheduler(seed)  # balanced, randomized directions; cue never spatial on screen in cued_blank
   ```
   Cued-intent blocks present the direction by an auditory word or a central symbol and show NOTHING at a
   spatial position, so the label is not display-identifiable; blank-feedback probe blocks run the decoder
   with the cursor hidden.
6. `sim_population.py` + `simulate_engine.py` (owner F)
   - `PopulationSim(n_subjects, seed)`: SimulatedEEG subclasses with subject-specific random topographies
     (shared component + individual deviation), per-day gain drift, optional ocular leak per subject,
     optional user-learning rate (default 0: do not assume users learn).
   - `simulate_engine.py`: days × hours of simulated use through protocol → data engine → realtime-like
     closed loop (offline-paced, no wall clock), producing the learning curve (success, throughput, decoder r)
     vs cumulative hours, with and without population prior, and the purity audit per day.
