# How much of the per-test-subject variance of the scaling gain is trial-sampling noise (shrinks with more test windows)
# versus true between-person heterogeneity? Same pipeline as pop_sf1/scaling.py (bb/fb, fixed C).
import sys, json, numpy as np
sys.path.insert(0,"/tmp/claude-0/-home-user-soroushzare-github-io/097178de-1116-520a-ba7e-ee5094385c63/scratchpad/pop_sf1")
from common import *
D,subs,test,pool=load_all()
out={}
for kind in ["bb","fb","E2"]:
  for t in ["imag_LR","imag_FF"]:
    C=1e-2 if kind in("bb","fb") else 1e-1
    def fit(pick):
        return model(kind,C).fit(np.concatenate([D[s][f"{kind}_{t}"] for s in pick]),np.concatenate([D[s]["y_"+t] for s in pick]))
    big=fit(pool)
    gains=[];impl=[]
    for rep in range(10):
        rng=np.random.default_rng(SEED+8000+rep); pick=[pool[i] for i in rng.choice(len(pool),8,replace=False)]
        small=fit(pick)
        g=[];v=[]
        for s in test:
            X=D[s][f"{kind}_{t}"]; y=D[s]["y_"+t]
            cb=(big.predict(X)==y).astype(float); cs=(small.predict(X)==y).astype(float)
            dd=cb-cs; g.append(dd.mean()); v.append(dd.var(ddof=1)/len(dd))
        gains.append(g); impl.append(v)
    gains=np.array(gains); impl=np.array(impl)
    obs_var=gains.var(axis=1,ddof=1).mean()   # across test subjects, per draw
    samp_var=impl.mean()                      # mean within-subject sampling variance
    het=max(obs_var-samp_var,0)
    n=len(D[test[0]]["y_"+t])
    out[f"{kind}|{t}"]=dict(mean_gain=round(float(gains.mean()),4),obs_sd=round(float(np.sqrt(obs_var)),4),
        sampling_sd=round(float(np.sqrt(samp_var)),4),heterogeneity_sd=round(float(np.sqrt(het)),4),
        frac_sampling=round(float(samp_var/obs_var),2),n_trials=int(n))
    print(kind,t,out[f"{kind}|{t}"],flush=True)
json.dump(out,open("noise.json","w"),indent=1)
