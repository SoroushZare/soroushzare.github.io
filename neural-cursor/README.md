# neural-cursor

Design and reference implementation for continuous 2-D cursor control from a 32-channel dry EEG cap
(g.tec, 500 Hz) without external stimulation or instructed motor imagery, built to improve with data.

- `DESIGN.md` — the design of record: competition verdict, the chosen composite system, two rounds of
  simulation, data-collection protocol, implementation, evaluation protocol, failure modes.
- `proposals/` — the seven competing proposals (SSVEP/c-VEP, P300, motor imagery, covert attention,
  shared autonomy + error evidence, end-to-end scaled, EOG hybrid), the brief, and the judge's verdict.
- `SYNAPTRIX_ANALYSIS.md` — what the Synaptrix cursor demo does and does not show.
- `neural_cursor/` — the system (acquisition, filtering, gating, features, decoders, task, agent,
  logging, evaluation) and the simulation harness.
- `results/` — simulation outputs; regenerate with the commands in `DESIGN.md` §7.
- `tests/` — headless regression tests (`python -m pytest -q tests`).

```
pip install -r requirements.txt
python -m pytest -q tests
python -m neural_cursor.simulate --users 6 --trials 24
python -m neural_cursor.simulate_composite --users 6 --trials 24
```
