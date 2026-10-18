# Analyses written by tournament agents

During the pure-EEG tournament the contender agents ran their own analyses to support or attack
proposals. Their scripts and outputs are archived here unchanged so that every number a proposal cites
can be traced. They were written under time pressure by agents arguing a side: treat them as leads,
not results of record. The results of record are `realdata/experiments.py`, `realdata/deep.py` and
`realdata/eye_proxy.py`, whose outputs are in `results/realdata/`.

- `population_sf1/` — the population finalist's cross-subject scaling ladders on ds004362, including the
  eye-proxy comparison (E1–E3 proxies) that `realdata/eye_proxy.py` independently replicated.
- `vocabulary_ds004148/` — the vocabulary semifinalist's tests on OpenNeuro ds004148 (mental-task EEG).
- `smr_sf2/` — the motor-rhythm finalist's analysis script.
- `misc/` — smaller probes (an ErrP/eye comparison, diagnostics); their provenance is the proposal that
  cites them.

Paths inside these scripts point at the session scratchpad where the cached data lived; rerunning them
needs `realdata/load.py` and `realdata/experiments.py features` to rebuild the caches first.
