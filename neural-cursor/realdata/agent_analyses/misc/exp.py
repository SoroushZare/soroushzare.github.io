import sys, time, tempfile; sys.path.insert(0, "/tmp/claude-0/-home-user-soroushzare-github-io/097178de-1116-520a-ba7e-ee5094385c63/scratchpad")
from gen import make_run
from neural_cursor.config import DEFAULT as cfg
from neural_cursor.data_engine import *
import numpy as np
for seed in range(int(sys.argv[1]), int(sys.argv[2])):
    root = tempfile.mkdtemp(); t=time.time()
    a = make_run(root+"/runs", cfg, "a", seed, 25, 1000.0, 4, 2)
    b = make_run(root+"/runs", cfg, "b", seed+50, 150, 9000.0)
    tg=time.time()-t
    store = SessionStore(root+"/runs"); reg = ModelRegistry(root+"/reg")
    eng = DataEngine(store, Labeler(cfg), Trainer(cfg), reg, ChampionChallenger(), min_new_windows=100)
    n1=eng.ingest(a); r1 = eng.maybe_retrain()
    n2=eng.ingest(b); r2 = eng.maybe_retrain()
    data = eng.data; tr = eng.judge.training_set(data, None)
    rng=np.random.default_rng(0); sh = LabelledSet(tr.windows, tr.labels[rng.permutation(len(tr))], tr.provenance)
    m = Trainer(cfg).fit(sh); r3 = eng.judge.challenge(eng.champion, m, data)
    print(seed, f"gen {tg:.1f}s total {time.time()-t:.1f}s", n1, n2, {k: (round(v,3) if isinstance(v,float) else v) for k,v in r1.items()}, "\n  ", {k: (round(v,3) if isinstance(v,float) else v) for k,v in r2.items()}, "\n  shuf", {k: (round(v,3) if isinstance(v,float) else v) for k,v in r3.items()})
