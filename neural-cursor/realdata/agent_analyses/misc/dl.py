import sys; sys.path.insert(0, "tests")
import numpy as np
from test_purity import _eeg, _prov, _unit, N_WINDOWS, DEFAULT, PurityAudit
for seed in (5, 15, 25, 35, 45, 55):
    rng = np.random.default_rng(4)
    labels = _unit(rng, N_WINDOWS)
    windows = _eeg(np.zeros((N_WINDOWS, 2)), seed=seed)
    prov = _prov(N_WINDOWS, ["closed_loop"], 1.0)
    res = PurityAudit(DEFAULT, n_permutations=2000).run(windows, labels, prov)
    r = res["reports"]
    print(seed, res["flags"], {k: (round(r[k]["score"],3), round(r[k]["floor"],3), round(r[k]["p_value"],3)) for k in ("full","frontal","slow","occipital")})
