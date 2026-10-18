# neural-cursor

Pure-EEG continuous control of a cursor, and through the same output any robot, from a 32-channel dry
EEG cap (g.tec, 500 Hz): no external stimulation, no eye or muscle signals as inputs, built so that more
data buys more control and so that every claim can be shown to come from cortex.

## Start here
- [`DESIGN_PURE_EEG.md`](DESIGN_PURE_EEG.md) — the design of record: provenance-certified sensorimotor-rhythm
  control in body coordinates, the data plan that tests "collect, retrain, adapt", purity proof,
  pre-registration, 18-month timeline.
- [`proposals/pure_eeg_tournament/`](proposals/pure_eeg_tournament/) — how it was chosen: eight pure-EEG
  concepts, debate matches judged by three to five judges, the bracket, the final, skeptic corrections.
- [`results/realdata/`](results/realdata/) — experiments on 106 people from OpenNeuro ds004362 on the cap's
  32 channels: within-person decoding, cross-person scaling (linear and deep), people × trials surface,
  collect-retrain-adapt replay, eye/vision confound audits.
- [`SYNAPTRIX_ANALYSIS.md`](SYNAPTRIX_ANALYSIS.md) — what the Synaptrix cursor demo does and does not show.
- [`DESIGN.md`](DESIGN.md) — the first design round (EOG allowed), kept for the record.

## Code
- `neural_cursor/` — acquisition (LSL, simulator), causal filtering and amplitude gating, decoders
  (velocity, evidence, body-coordinate SMR), real-time loop, effectors (screen cursor, UDP, ROS2, simulated
  arm), data engine (collect → label → retrain → promote), purity audit and certification, protocol,
  population and body simulators, multi-day simulations.
- `realdata/` — loader, experiments, deep scaling, eye-proxy replication, N × k surface; `run.sh` runs
  them isolated. Agents' own tournament analyses are archived in `realdata/agent_analyses/`.
- `tests/` — headless tests (`python -m pytest -q tests`).
- [`INTERFACES.md`](INTERFACES.md), [`INTERFACES_CHAMPION.md`](INTERFACES_CHAMPION.md) — module contracts.

```
pip install -r requirements.txt          # plus pyriemann, mne, pylsl, torch for real data and real time
python -m pytest -q tests
python -m neural_cursor.simulate_engine --days 2 --hours 0.5 --pop 5
python -m neural_cursor.simulate_champion --days 3 --hours 1 --users 3 --pop 10
```

Real data: download OpenNeuro ds004362 (S3 bucket `openneuro.org`), then
`realdata/run.sh load --data <dir> --cache <cache>` and
`realdata/run.sh experiments features --cache <cache> --feat <feat>`; each experiment is a subcommand
(see the docstring of `realdata/experiments.py`).
