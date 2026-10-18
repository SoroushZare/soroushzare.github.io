import os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
FEAT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"feat")
SEED=20261018
BANDS=["b1","b2","b3","b4"]
def _isqrt(C):
    w,v=np.linalg.eigh(C); return (v/np.sqrt(w))@v.T
def _logm(C):
    w,v=np.linalg.eigh(C); return (v*np.log(w))@v.T
def tvec(C):
    n=C.shape[-1]; iu=np.triu_indices(n); wt=np.where(iu[0]==iu[1],1.0,np.sqrt(2))
    L=np.array([_logm(c) for c in C]); return L[:,iu[0],iu[1]]*wt
def rmean(Cs,it=20):
    from pyriemann.utils.mean import mean_riemann
    return mean_riemann(Cs)
def recentre(C,ref):
    w=_isqrt(ref); return w@C@w
def load_all():
    subs=sorted(int(f[4:7]) for f in os.listdir(FEAT) if f.startswith("sub-"))
    D={}
    for s in subs:
        d=dict(np.load(os.path.join(FEAT,f"sub-{s:03d}.npz")))
        e={}
        for t in ["imag_LR","imag_FF"]:
            if "y_"+t not in d: continue
            e["y_"+t]=d["y_"+t].astype(int); e["o_"+t]=d["o_"+t].astype(int)
            o=e["o_"+t]; pos=np.zeros(len(o)); 
            for r in np.unique(o): idx=np.where(o==r)[0]; pos[idx]=np.arange(len(idx))
            e["pos_"+t]=pos
            ref=rmean(d["C_bb_rest"]); e["bb_"+t]=tvec(recentre(d["C_bb_"+t],ref))
            fb=[]
            for b in BANDS:
                rb=rmean(d[f"C_{b}_rest"]); fb.append(tvec(recentre(d[f"C_{b}_{t}"],rb)))
            e["fb_"+t]=np.concatenate(fb,1)
            e["base_"+t]=tvec(recentre(d["Cbase_"+t],ref))
            e["E1_"+t]=d["E1_"+t]; e["E2_"+t]=d["E2_"+t]
            E3=d["E3_"+t]; e["E3_"+t]=tvec(recentre(E3,rmean(E3)))
        D[s]=e
    subs=[s for s in subs if "y_imag_LR" in D[s] and "y_imag_FF" in D[s]]
    rng=np.random.default_rng(SEED); test=sorted(rng.choice(subs,30,replace=False).tolist())
    pool=[s for s in subs if s not in test]
    return D,subs,test,pool
def model(kind,C):
    if kind in ("bb","fb","base","E3"): return LogisticRegression(C=C,max_iter=3000)
    return make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000))
