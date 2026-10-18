"""All tunables in one place.  Everything else derives from this file.

The dataclass is frozen so a run cannot mutate its own configuration; copy with
``dataclasses.replace`` to create a variant and log the variant with the run.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json


@dataclass(frozen=True)
class Hardware:
    """g.tec 32-channel dry hybrid cap, LSL stream."""
    fs: int = 500                                  # Hz, amplifier sampling rate
    n_channels: int = 32
    # 10-20 layout as the cap labels it (FP1/FP2 spelling is the cap's, not MNE's).
    channels: tuple = (
        "FP1", "FP2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
        "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
        "PO7", "PO3", "PO4", "PO8", "Oz",
    )
    eog_proxy: tuple = ("FP1", "FP2", "F7", "F8")  # vertical ≈ FP mean, horizontal ≈ F8 − F7
    lsl_stream_type: str = "EEG"
    lsl_marker_stream: str = "neural_cursor_markers"
    expected_latency_s: float = 0.040               # amplifier + LSL, measured not assumed (log it)


@dataclass(frozen=True)
class Preprocessing:
    hp_hz: float = 0.5           # causal Butterworth high-pass (drift from dry electrodes)
    lp_hz: float = 40.0          # causal Butterworth low-pass
    notch_hz: float = 50.0       # mains; set 60.0 in the Americas
    filter_order: int = 4
    reference: str = "car"       # "car" or "none"; CAR excludes gated-out channels
    gate_rms_factor: float = 5.0 # channel is bad if trial RMS > factor × session median channel RMS
    gate_exempt: tuple = ("FP1", "FP2")  # a blink is 100+ µV there and that is physiology
    gate_low_factor: float = 0.05  # channel is bad (dead / flat) if its sample-to-sample RMS < factor ×
                                 # the session median of that RMS: −26 dB, far below any live electrode
                                 # (amplifier noise alone sets a floor), far above a filtered constant (~0)
    decimate_to_hz: int = 100    # features need ≤40 Hz; 100 Hz keeps the pipeline cheap


@dataclass(frozen=True)
class Decoder:
    """Velocity decoder (continuous) and evidence classifier (discrete) hyperparameters."""
    window_s: float = 0.5        # feature window for the continuous velocity decoder
    hop_s: float = 0.1           # update period of the velocity estimate (10 Hz control loop)
    bands_hz: tuple = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30))
    ridge_alpha: float = 100.0   # ridge on log-band-power features (inner-CV default; 160 features)
    tangent_shrinkage: float = 0.1  # Ledoit-Wolf style shrinkage of per-window covariance
    evidence_window: tuple = (0.2, 0.8)  # s after an agent action, for the accept/veto classifier
    evidence_C: float = 0.01     # logistic inverse regularisation (528 tangent features, ~10² trials)
    recenter_buffer: int = 64    # trials in the online covariance re-centring buffer
    recenter_prior: float = 5.0  # blend weight n/(n+prior): the reference moves only once n ≫ prior
    adapt_windows: int = 600     # running-mean length of the velocity decoder's feature re-centring
                                 # (60 s at 10 Hz): long enough that a trial's persistent intent
                                 # averages out, short enough to follow dry-electrode gain drift


@dataclass(frozen=True)
class Task:
    """Cursor dynamics and target geometry; screen units are normalised to [−1, 1]."""
    control_hz: float = 10.0
    max_speed: float = 0.6       # screen widths per second at full decoder output
    velocity_smoothing: float = 0.7  # exponential smoothing of the decoded velocity per 100 ms (rescaled
                                     # to the step: a**(dt / 0.1) per control step, any loop rate)
    target_radius: float = 0.08
    dwell_s: float = 0.8         # dwell to select (confirm) a target
    n_targets: int = 8           # candidate targets on screen per trial
    trial_timeout_s: float = 15.0
    inter_trial_s: float = 1.0
    assist_gain: float = 0.0     # 0 = pure decoder; agent assistance blends toward the posterior goal
    # Shared-autonomy agent
    posterior_temperature: float = 1.0
    rejection_cost_s: float = 0.0  # a veto must be cheap (RLNF's 5.25 s rejection killed utility)


@dataclass(frozen=True)
class Evaluation:
    n_permutations: int = 2000
    n_bootstrap: int = 2000
    seed: int = 20261004
    floor_percentile: float = 95.0  # "floor" means the null's 95th percentile — say so in every table


@dataclass(frozen=True)
class Config:
    hardware: Hardware = field(default_factory=Hardware)
    pre: Preprocessing = field(default_factory=Preprocessing)
    decoder: Decoder = field(default_factory=Decoder)
    task: Task = field(default_factory=Task)
    eval: Evaluation = field(default_factory=Evaluation)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


DEFAULT = Config()
