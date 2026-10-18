"""SF1 feature extraction: 14-pos dry-cap sensorimotor montage (local avg ref), bb + 4-band FB covs,
baseline-window covs, rest covs, and stronger eye proxies. Reads cache npz, writes feat npz."""
import os, sys, numpy as np
from scipy.signal import butter, sosfiltfilt
from joblib import Parallel, delayed
from pyriemann.estimation import Covariances
CACHE, OUT = sys.argv[1], sys.argv[2]
FS=160; T0=-1.0
SM=["Fc5","Fc1","Fc2","Fc6","C3","Cz","C4","Cp5","Cp1","Cp2","Cp6","P3","Pz","P4"]
OC=["Fp1","Fpz","Fp2","Af7","Af8","F7","F8"]
BANDS={"bb":(8,30),"b1":(8,12),"b2":(12,16),"b3":(16,24),"b4":(24,30)}
def sl(a,b): return slice(int(round((a-T0)*FS)), int(round((b-T0)*FS)))
def bp(x,lo,hi):
    return sosfiltfilt(butter(4,[lo,hi],"bandpass",fs=FS,output="sos"),x,axis=-1)
def cov(x): return Covariances("oas").fit_transform(x.astype(np.float64))
def one(f):
    out=os.path.join(OUT,f)
    if os.path.exists(out): return f
    d=np.load(os.path.join(CACHE,f),allow_pickle=False)
    if list(d["sfreq"])!=[160.0]: return f+" skip"
    names=[str(n).lower() for n in d["ch_names"]]
    ix=[names.index(c.lower()) for c in SM]; io=[names.index(c.lower()) for c in OC]
    res={}
    for t in ["imag_LR","imag_FF","exec_LR","exec_FF","rest"]:
        k="X_"+t
        if k not in d.files: continue
        X=d[k].astype(np.float64)
        if t!="rest" and len(X)<20: continue
        S=X[:,ix]; S=S-S.mean(1,keepdims=True)
        for b,(lo,hi) in BANDS.items():
            res[f"C_{b}_{t}"]=cov(bp(S,lo,hi)[:,:,sl(0.5,3.5)])
        if t=="rest": continue
        res[f"Cbase_{t}"]=cov(bp(S,8,30)[:,:,sl(-1.0,0.0)])
        res["y_"+t]=d["y_"+t]; res["o_"+t]=d["o_"+t]
        O=X[:,io]
        sm=X[:,ix].mean(1)
        veog=O[:,:3].mean(1)-sm; heog1=O[:,5]-O[:,6]; heog2=O[:,3]-O[:,4]
        O2=np.concatenate([O,veog[:,None],heog1[:,None],heog2[:,None]],1)
        # E1 (original): 1-4 Hz, 8 bins 0-4 s, 7 ocular channels
        a=bp(O,1,4)[:,:,sl(0,4)]; n,c,tt=a.shape
        res["E1_"+t]=a.reshape(n,c,8,tt//8).mean(-1).reshape(n,-1).astype(np.float32)
        # E2: 1-8 Hz, 16 bins of 0.25 s, 10 channels incl. derived VEOG/HEOG
        a=bp(O2,1,8)[:,:,sl(0,4)]; n,c,tt=a.shape
        res["E2_"+t]=a.reshape(n,c,16,tt//16).mean(-1).reshape(n,-1).astype(np.float32)
        # E3: ocular 1-30 Hz covariance (eye + frontal EMG power)
        res["E3_"+t]=cov(bp(O,1,30)[:,:,sl(0.5,3.5)])
    np.savez(out,**res)
    return f
fs=sorted(x for x in os.listdir(CACHE) if x.startswith("sub-"))
r=Parallel(n_jobs=4)(delayed(one)(f) for f in fs)
print(len(r),[x for x in r if "skip" in x])
