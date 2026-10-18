import json, sys, numpy as np
from joblib import Parallel, delayed
from common import *
D,subs,test,pool=load_all()
print("subs",len(subs),"test",len(test),"pool",len(pool),flush=True)
CS=[1e-3,3e-3,1e-2,3e-2,1e-1]
def run(kind,t,N,rep,tuned):
    rng=np.random.default_rng(SEED+1000*N+rep); pick=[pool[i] for i in rng.choice(len(pool),N,replace=False)]
    X=[D[s][f"{kind}_{t}"] for s in pick]; y=[D[s]["y_"+t] for s in pick]
    if tuned:
        g=np.arange(N)%4; best=None
        for C in CS:
            acc=[]
            for f in range(4):
                tr=[i for i in range(N) if g[i]!=f]; te=[i for i in range(N) if g[i]==f]
                m=model(kind,C).fit(np.concatenate([X[i] for i in tr]),np.concatenate([y[i] for i in tr]))
                acc+= [np.mean(m.predict(X[i])==y[i]) for i in te]
            a=np.mean(acc)
            if best is None or a>best[0]: best=(a,C)
        C=best[1]
    else: C=1e-2 if kind in("bb","fb","base","E3") else 1e-1
    m=model(kind,C).fit(np.concatenate(X),np.concatenate(y))
    return kind,t,N,rep,tuned,C,[float(np.mean(m.predict(D[s][f"{kind}_{t}"])==D[s]["y_"+t])) for s in test]
NS={8:10,16:8,32:6,64:4,76:1}
jobs=[(k,t,N,r,tu) for k in ["bb","fb","E1","E2","E3"] for t in ["imag_LR","imag_FF"] for N,R in NS.items() for r in range(R) for tu in (False,True)]
res=Parallel(n_jobs=4,verbose=0)(delayed(run)(*j) for j in jobs)
json.dump(dict(test=test,pool=pool,res=res),open("scaling.json","w"))
# summary
B=2000; rb=np.random.default_rng(1)
def curve(k,t,tu):
    M={N:np.mean([r[6] for r in res if r[:3]==(k,t,N) and r[4]==tu],0) for N in NS}
    return M
out={}
for tu in (False,True):
  for t in ["imag_LR","imag_FF"]:
    for k in ["bb","fb","E1","E2","E3"]:
        M=curve(k,t,tu); Ns=sorted(M); A=np.array([M[N] for N in Ns])  # (nN, ntest)
        x=np.log2(Ns); x=x-x.mean()
        slope=(A.T@x)/(x@x)
        bs=[]; d76=[]
        for b in range(B):
            i=rb.integers(0,len(test),len(test)); bs.append(slope[i].mean()); d76.append((A[-1,i]-A[0,i]).mean())
        Cs=[r[5] for r in res if r[:2]==(k,t) and r[4]==tu]
        out[f"{'tuned' if tu else 'fixed'}|{t}|{k}"]=dict(acc={int(N):round(float(M[N].mean()),3) for N in Ns},
            slope_per_doubling=round(float(slope.mean()),4),slope_ci=[round(float(v),4) for v in np.percentile(bs,[2.5,97.5])],
            gain_8_76=round(float((A[-1]-A[0]).mean()),3),gain_ci=[round(float(v),3) for v in np.percentile(d76,[2.5,97.5])],
            Cs_chosen=sorted(set(Cs)))
        print(f"{'tuned' if tu else 'fixed'}|{t}|{k}",out[f"{'tuned' if tu else 'fixed'}|{t}|{k}"],flush=True)
  # EEG - eye slope differences
  for t in ["imag_LR","imag_FF"]:
    for k in ["bb","fb"]:
      for e in ["E1","E2","E3"]:
        Mk=curve(k,t,tu); Me=curve(e,t,tu); Ns=sorted(Mk); x=np.log2(Ns); x=x-x.mean()
        sk=(np.array([Mk[N] for N in Ns]).T@x)/(x@x); se=(np.array([Me[N] for N in Ns]).T@x)/(x@x)
        dd=sk-se; bs=[dd[rb.integers(0,len(test),len(test))].mean() for _ in range(B)]
        out[f"{'tuned' if tu else 'fixed'}|{t}|{k}-{e}"]=dict(diff=round(float(dd.mean()),4),ci=[round(float(v),4) for v in np.percentile(bs,[2.5,97.5])])
        print(f"{'tuned' if tu else 'fixed'}|{t}|{k}-{e}",out[f"{'tuned' if tu else 'fixed'}|{t}|{k}-{e}"],flush=True)
json.dump(out,open("scaling_summary.json","w"),indent=1)
