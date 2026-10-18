import sys, numpy as np
sys.path.insert(0, "/home/user/soroushzare.github.io/neural-cursor/tests")
from test_purity import _unit, _eeg, _prov
from neural_cursor.config import DEFAULT
from neural_cursor.purity import PurityAudit
a = PurityAudit(DEFAULT)
for s in range(10, 16):
    rng = np.random.default_rng(s); l = _unit(rng, 300)
    clean = a.run(_eeg(l, s), l, _prov(300, ["closed_loop", "blank_feedback"], 0.5))["flags"]
    leak = PurityAudit(DEFAULT, feature_set="bandpower+slow").run(_eeg(l, s, 1.0), l, _prov(300, ["cued_blank"], 0.5))
    print(s, clean, leak["flags"], {k: round(v["score"], 3) for k, v in leak["reports"].items() if "score" in v})
