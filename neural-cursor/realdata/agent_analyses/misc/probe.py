import sys, time, numpy as np
sys.path.insert(0, "/home/user/soroushzare.github.io/neural-cursor")
sys.path.insert(0, "/home/user/soroushzare.github.io/neural-cursor/tests")
from test_purity import _unit, _eeg, _prov
from neural_cursor.config import DEFAULT
from neural_cursor.purity import PurityAudit
for n in (300, 600):
    rng = np.random.default_rng(6)
    labels = _unit(rng, n)
    prov = _prov(n, ["closed_loop", "blank_feedback"], 0.5)
    vis = np.array([p["feedback_visible"] for p in prov])[:, None]
    intents = np.where(vis, labels, 0.0)
    t=time.time()
    res = PurityAudit(DEFAULT, n_permutations=500).run(_eeg(intents, seed=7), labels, prov)
    print(n, time.time()-t, {k: round(v["score"],3) for k,v in res["reports"].items() if "score" in v}, res["feedback"])
