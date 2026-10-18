# Re-run the population ladder on the CENTRAL 11 dry-cap positions (P3/Pz/P4 removed; re-referenced within the 11
# via projection onto the zero-sum subspace so posterior channels cannot enter through the reference).
import sys, json, os, numpy as np
sys.path.insert(0,"/tmp/claude-0/-home-user-soroushzare-github-io/097178de-1116-520a-ba7e-ee5094385c63/scratchpad/pop_sf1")
import common
from common import tvec, rmean, recentre, SEED, BANDS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
keep=list(range(11))  # SM order: Fc5 Fc1 Fc2 Fc6 C3 Cz C4 Cp5 Cp1 Cp2 Cp6 | P3 Pz P4
Q,_=np.linalg.qr(np.column_stack([np.ones(11),np.eye(11)[:,:10]])); U=Q[:,1:11]  # orthonormal, orthogonal to ones
def proj(C): S=C[...,keep,:][...,:,keep]; return np.einsum('ij,...jk,kl->...il',U.T,S,U)
FEAT=common.FEAT
subs=sorted(int(f[4:7]) for f in os.listdir(FEAT) if f.startswith("sub-"))
D={}
for s in subs:
    d=dict(np.load(os.path.join(FEAT,f"sub-{s:03d}.npz"))); e={}
    if "y_imag_LR" not in d or "y_imag_FF" not in d: continue
    for t in ["imag_LR","imag_FF"]:
        e["y_"+t]=d["y_"+t].astype(int)
        ref=rmean(proj(d["C_bb_rest"])); e["bb_"+t]=tvec(recentre(proj(d["C_bb_"+t]),ref))
        e["fb_"+t]=np.concatenate([tvec(recentre(proj(d[f"C_{b}_{t}"]),rmean(proj(d[f"C_{b}_rest"])))) for b in BANDS],1)
        e["E2_"+t]=d["E2_"+t]
    D[s]=e
subs=sorted(D)
rng=np.random.default_rng(SEED); test=sorted(rng.choice(subs,30,replace=False).tolist()); pool=[s for s in subs if s not in test]
print("subs",len(subs),flush=True)
def model(kind): return LogisticRegression(C=1e-2,max_iter=3000) if kind!="E2" else make_pipeline(StandardScaler(),LogisticRegression(C=1e-1,max_iter=3000))
NS={8:10,16:8,32:6,64:4,76:1}
def run(k,t,N,rep):
    r=np.random.default_rng(SEED+1000*N+rep); pick=[pool[i] for i in r.choice(len(pool),N,replace=False)]
    m=model(k).fit(np.concatenate([D[s][f"{k}_{t}"] for s in pick]),np.concatenate([D[s]["y_"+t] for s in pick]))
    return k,t,N,rep,[float(np.mean(m.predict(D[s][f"{k}_{t}"])==D[s]["y_"+t])) for s in test]
jobs=[(k,t,N,r) for k in ["bb","fb","E2"] for t in ["imag_LR","imag_FF"] for N,R in NS.items() for r in range(R)]
res=Parallel(n_jobs=4)(delayed(run)(*j) for j in jobs)
out={}; rb=np.random.default_rng(1); Ns=sorted(NS); x=np.log2(Ns); x=x-x.mean()
cur=lambda k,t: np.array([np.mean([q[4] for q in res if q[:3]==(k,t,N)],0) for N in Ns])
for t in ["imag_LR","imag_FF"]:
    for k in ["bb","fb","E2"]:
        A=cur(k,t); s=(A.T@x)/(x@x); bs=[s[rb.integers(0,30,30)].mean() for _ in range(2000)]
        out[f"{t}|{k}"]=dict(acc={N:round(float(A[i].mean()),3) for i,N in enumerate(Ns)},slope=round(float(s.mean()),4),
            ci=[round(float(v),4) for v in np.percentile(bs,[2.5,97.5])],subj_sd=round(float(s.std(ddof=1)),4))
        print(t,k,out[f"{t}|{k}"],flush=True)
    for k in ["bb","fb"]:
        dd=((cur(k,t).T@x)-(cur("E2",t).T@x))/(x@x); bs=[dd[rb.integers(0,30,30)].mean() for _ in range(2000)]
        out[f"{t}|{k}-E2"]=dict(diff=round(float(dd.mean()),4),ci=[round(float(v),4) for v in np.percentile(bs,[2.5,97.5])]); print(t,k,"-E2",out[f"{t}|{k}-E2"],flush=True)
# 4-class zero-shot at N=76
for k in ["bb","fb"]:
    XY=lambda s: (np.concatenate([D[s][f"{k}_imag_LR"],D[s][f"{k}_imag_FF"]]),np.concatenate([D[s]["y_imag_LR"],2+D[s]["y_imag_FF"]]))
    m=LogisticRegression(C=1e-2,max_iter=5000).fit(np.concatenate([XY(s)[0] for s in pool]),np.concatenate([XY(s)[1] for s in pool]))
    a=[np.mean(m.predict(XY(s)[0])==XY(s)[1]) for s in test]
    out[f"4class|{k}|N76"]=round(float(np.mean(a)),3); print("4class",k,out[f"4class|{k}|N76"],flush=True)
json.dump(out,open("central11.json","w"),indent=1)
