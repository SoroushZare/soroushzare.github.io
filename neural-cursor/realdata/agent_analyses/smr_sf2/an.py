import sys, os, json, numpy as np
from scipy.signal import butter, sosfiltfilt
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ProcessPoolExecutor
C = sys.argv[1]
CEN = ['Fc5','Fc1','Fc2','Fc6','C3','Cz','C4','Cp5','Cp1','Cp2','Cp6']
POS = ['P7','P3','Pz','P4','P8','Po7','Po3','Po4','Po8','Oz']
FS=160
WINS=[(0.0,1.0),(0.5,1.5),(1.0,2.0),(2.0,3.0),(3.0,4.0)]
BANDS={'mu_beta':(8,30),'alpha':(8,12),'beta':(16,30)}
def ts_feats(X):
    cov = Covariances('oas').fit_transform(X.astype(np.float64))
    return cov
def cv(Ftr_fn, y, seed=0):
    skf=StratifiedKFold(5,shuffle=True,random_state=seed); acc=[]
    for tr,te in skf.split(np.zeros(len(y)),y):
        Ftr,Fte=Ftr_fn(tr,te)
        m=LogisticRegression(C=1.0,max_iter=2000).fit(Ftr,y[tr]); acc.append((m.predict(Fte)==y[te]).mean())
    return float(np.mean(acc))
def run(sub):
    d=np.load(os.path.join(C,f'sub-{sub:03d}.npz'))
    ch=[str(c) for c in d['ch_names']]
    ic0=[ch.index(c) for c in CEN]; ip0=[ch.index(c) for c in POS]; ic=ic0; ip=ip0
    out={}
    for task in ['imag_LR','imag_FF']:
        if f'X_{task}' not in d.files: continue
        X=d[f'X_{task}'][:,ic0+ip0]; y=d[f'y_{task}'].astype(int)
        if len(y)<20 or min(np.bincount(y))<8: continue
        r={}; ic=list(range(len(ic0))); ip=list(range(len(ic0),len(ic0)+len(ip0)))
        for bn,(lo,hi) in BANDS.items():
            sos=butter(4,[lo,hi],btype='band',fs=FS,output='sos')
            Xf=sosfiltfilt(sos,X,axis=-1)
            wins = WINS if bn=='mu_beta' else [(0.5,3.5)]
            if bn=='mu_beta': wins = WINS+[(0.5,3.5)]
            for (a,b) in wins:
                s0=int((a+1)*FS); s1=int((b+1)*FS)
                Cc=ts_feats(Xf[:,ic,s0:s1]); Cp=ts_feats(Xf[:,ip,s0:s1])
                def mk(Cm):
                    def f(tr,te):
                        t=TangentSpace().fit(Cm[tr]); return t.transform(Cm[tr]),t.transform(Cm[te])
                    return f
                key=f'{bn}_{a}-{b}'
                r[key+'_cen']=cv(mk(Cc),y); r[key+'_pos']=cv(mk(Cp),y)
                if (a,b)==(0.5,3.5):
                    def resid(tr,te):
                        tc=TangentSpace().fit(Cc[tr]); tp=TangentSpace().fit(Cp[tr])
                        fc_tr,fc_te=tc.transform(Cc[tr]),tc.transform(Cc[te])
                        fp_tr,fp_te=tp.transform(Cp[tr]),tp.transform(Cp[te])
                        # ridge-regress posterior features on central features (train fold only)
                        from sklearn.linear_model import Ridge
                        R=Ridge(alpha=1.0).fit(fc_tr,fp_tr)
                        return fp_tr-R.predict(fc_tr), fp_te-R.predict(fc_te)
                    r[key+'_pos_resid']=cv(resid,y)
                    # permutation floor (one shuffle) with posterior
                    rng=np.random.default_rng(sub); yp=rng.permutation(y)
                    r[key+'_cen_perm']=cv(mk(Cc),yp)
        out[task]=r
    return sub,out
if __name__=='__main__':
    subs=[int(f[4:7]) for f in sorted(os.listdir(C)) if f.endswith('.npz')]
    res={}
    with ProcessPoolExecutor(4) as ex:
        for s,o in ex.map(run,subs): res[s]=o
    json.dump(res,open(sys.argv[2],'w'))
    for task in ['imag_LR','imag_FF']:
        ks=sorted({k for s in res for k in res[s].get(task,{})})
        print(task, 'n=',sum(task in res[s] for s in res))
        for k in ks:
            v=np.array([res[s][task][k] for s in res if task in res[s] and k in res[s][task]])
            bs=[np.mean(np.random.default_rng(i).choice(v,len(v))) for i in range(1000)]
            print(f'  {k:28s} {v.mean():.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]')
