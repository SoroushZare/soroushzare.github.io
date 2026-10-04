"""neural_cursor — reference implementation of the continuous EEG cursor-control system.

Modules
-------
config       every tunable in one frozen dataclass (hardware, timing, decoder, task, evaluation)
acquisition  LSL inlet with a ring buffer, plus a deterministic simulated EEG source for offline tests
signal       causal filtering, re-referencing, per-channel amplitude gating, EOG derivation
features     windowed features (log band power, covariance → tangent space)
decoder      the decoder interface and the reference decoders
task         the cursor task: targets, cursor dynamics, shared-autonomy agent, Fitts blocks
logger       append-only logging of EEG, markers and cursor state with one clock
evaluation   throughput, time-to-target, permutation floors, surrogate controls, bootstrap CIs

Nothing here imports a GUI; rendering is a thin optional layer so every component runs headless in CI.
"""
__version__ = "0.1.0"
