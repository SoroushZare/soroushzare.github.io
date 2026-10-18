import numpy as np
from tests.test_purity import _eeg, _prov, _unit, N_WINDOWS, DEFAULT, PurityAudit
for seed in range(10, 16):
    rng = np.random.default_rng(seed)
    labels = _unit(rng, N_WINDOWS)
    leak = PurityAudit(DEFAULT, feature_set="bandpower+slow", n_permutations=2000).run(
        _eeg(labels, seed=seed + 100, eye_gain_uv=1.0), labels, _prov(N_WINDOWS, ["cued_blank"], 0.5))
    clean = PurityAudit(DEFAULT, n_permutations=2000).run(
        _eeg(labels, seed=seed + 200), labels, _prov(N_WINDOWS, ["cued_blank"], 0.5))
    print(seed, leak["flags"], leak["reports"]["slow"]["p_value"], clean["flags"])
