import os, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from common import recentre
F="/tmp/claude-0/-home-user-soroushzare-github-io/097178de-1116-520a-ba7e-ee5094385c63/scratchpad/feat"
SEED=20261018
subs=[]
for f in sorted(os.listdir(F)):
    if not f.startswith("sub-"): continue
    d=np.load(os.path.join(F,f))
    if list(d["sfreq"])==[160.0] and "y_imag_LR" in d.files and len(d["y_imag_LR"])>=20 and "Crest32" in d.files: subs.append(int(f[4:7]))
rng=np.random.default_rng(SEED); test=sorted(rng.choice(subs,25,replace=False).tolist()); pool=[s for s in subs if s not in test]
def get(s):
    d=np.load(os.path.join(F,f"sub-{s:03d}.npz")); return recentre(d["C32_imag_LR"],mean_riemann(d["Crest32"])),d["y_imag_LR"]
P=[get(s) for s in pool]; T=[get(s) for s in test]
out={"n_subs":len(subs),"pool":len(pool)}
for C in [1.0,0.1,0.01]:
    m=make_pipeline(TangentSpace(metric="riemann"),LogisticRegression(C=C,max_iter=5000)).fit(np.concatenate([p[0] for p in P]),np.concatenate([p[1] for p in P]))
    out[f"C={C}"]=round(float(np.mean([np.mean(m.predict(x)==y) for x,y in T])),3); print(out,flush=True)
json.dump(out,open("reconcile.json","w"),indent=1)
