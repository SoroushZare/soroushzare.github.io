import json, os, numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from common import *
D,subs,test,pool=load_all()
out={}
def fitp(kind,t,seed):
    rng=np.random.default_rng(max(seed,0)+7)
    X=np.concatenate([D[s][f"{kind}_{t}"] for s in pool])
    y=np.concatenate([rng.permutation(D[s]["y_"+t]) if seed>=0 else D[s]["y_"+t] for s in pool])
    m=LogisticRegression(C=1e-2,max_iter=3000).fit(X,y)
    return [float(np.mean(m.predict(D[s][f"{kind}_{t}"])==D[s]["y_"+t])) for s in test]
NP=int(os.environ.get("NP","1000"))
for t in ["imag_LR","imag_FF"]:
    for k in ["bb","fb"]:
        real=fitp(k,t,-1)
        null=np.array(Parallel(n_jobs=4)(delayed(fitp)(k,t,i) for i in range(NP)))  # (NP, ntest)
        cm=null.mean(1)
        p_cohort=(1+np.sum(cm>=np.mean(real)))/(1+NP)
        ps=np.array([(1+np.sum(null[:,j]>=real[j]))/(1+NP) for j in range(len(test))])
        # BH
        o=np.argsort(ps); q=ps[o]*len(ps)/np.arange(1,len(ps)+1); q=np.minimum.accumulate(q[::-1])[::-1]
        sig_bh=int(np.sum(q<0.05))
        out[f"{t}|{k}"]=dict(real_mean=round(float(np.mean(real)),3),null_mean=round(float(cm.mean()),4),
            null_p95_cohort=round(float(np.percentile(cm,95)),4),null_max_cohort=round(float(cm.max()),4),
            p_cohort=float(p_cohort),
            null_p95_single_subject=round(float(np.percentile(null,95)),4),null_max_single_subject=round(float(null.max()),4),
            frac_subj_p05=round(float(np.mean(ps<0.05)),3),n_sig_BH05=sig_bh,n_test=len(test),n_perm=NP)
        print(t,k,out[f"{t}|{k}"],flush=True)
# trial-order surrogate (run index one-hot + position within run), cross-subject
for t in ["imag_LR","imag_FF"]:
    def F(s):
        o=D[s]["o_"+t]; p=D[s]["pos_"+t]
        return np.column_stack([o==0,o==1,o==2,p,p**2,(p%2)]).astype(float)
    X=np.concatenate([F(s) for s in pool]); y=np.concatenate([D[s]["y_"+t] for s in pool])
    m=model("E1",0.1).fit(X,y)
    a=[np.mean(m.predict(F(s))==D[s]["y_"+t]) for s in test]
    out[f"order_surrogate|{t}"]=round(float(np.mean(a)),3)
    # baseline (pre-cue) window floor, sensorimotor bb
    m=LogisticRegression(C=1e-2,max_iter=3000).fit(np.concatenate([D[s]["base_"+t] for s in pool]),y)
    out[f"precue_window|{t}"]=round(float(np.mean([np.mean(m.predict(D[s]["base_"+t])==D[s]["y_"+t]) for s in test])),3)
    print(t,out[f"order_surrogate|{t}"],out[f"precue_window|{t}"],flush=True)
json.dump(out,open("nulls.json","w"),indent=1)
