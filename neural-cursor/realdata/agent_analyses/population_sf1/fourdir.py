import json, numpy as np
from sklearn.linear_model import LogisticRegression
from common import *
D,subs,test,pool=load_all()
out={}
for k in ["bb","fb"]:
    fit=lambda t,feat=k: LogisticRegression(C=1e-2,max_iter=3000).fit(np.concatenate([D[s][f"{feat}_{t}"] for s in pool]),np.concatenate([D[s]["y_"+t] for s in pool]))
    mLR=fit("imag_LR"); mFF=fit("imag_FF")
    r={}
    # cross-axis: hands/feet head on single-hand trials (y_FF: 0 = both fists (up), 1 = feet (down))
    phands_LR=[1-mFF.predict_proba(D[s][f"{k}_imag_LR"])[:,1] for s in test]
    phands_FFh=[1-mFF.predict_proba(D[s][f"{k}_imag_FF"][D[s]["y_imag_FF"]==0])[:,1] for s in test]
    phands_FFf=[1-mFF.predict_proba(D[s][f"{k}_imag_FF"][D[s]["y_imag_FF"]==1])[:,1] for s in test]
    fr=lambda L: round(float(np.mean([np.mean(p>0.5) for p in L])),3)
    r["HF_head_frac_called_hands"]=dict(single_hand_trials=fr(phands_LR),both_hands_trials=fr(phands_FFh),feet_trials=fr(phands_FFf))
    pr_h=[mLR.predict_proba(D[s][f"{k}_imag_FF"][D[s]["y_imag_FF"]==0])[:,1] for s in test]
    pr_f=[mLR.predict_proba(D[s][f"{k}_imag_FF"][D[s]["y_imag_FF"]==1])[:,1] for s in test]
    pr_l=[mLR.predict_proba(D[s][f"{k}_imag_LR"][D[s]["y_imag_LR"]==0])[:,1] for s in test]
    pr_r=[mLR.predict_proba(D[s][f"{k}_imag_LR"][D[s]["y_imag_LR"]==1])[:,1] for s in test]
    r["LR_head_frac_called_right"]=dict(left=fr(pr_l),right=fr(pr_r),both_hands=fr(pr_h),feet=fr(pr_f))
    # two-head 4-direction readout: vx=2P(right)-1, vy=2P(hands)-1; true dirs L,R,U,D
    accs=[];coss={c:[] for c in "LRUD"}
    conf=np.zeros((4,4))
    for s in test:
        for t,yy in [("imag_LR",D[s]["y_imag_LR"]),("imag_FF",D[s]["y_imag_FF"])]:
            X=D[s][f"{k}_{t}"]
            vx=2*mLR.predict_proba(X)[:,1]-1; vy=2*(1-mFF.predict_proba(X)[:,1])-1
            lab=np.where(yy==0,0,1) if t=="imag_LR" else np.where(yy==0,2,3)
            tv=np.array([[-1,0],[1,0],[0,1],[0,-1]])[lab]
            cos=(vx*tv[:,0]+vy*tv[:,1])/np.maximum(np.hypot(vx,vy),1e-9)
            pred=np.where(np.abs(vx)>=np.abs(vy),np.where(vx<0,0,1),np.where(vy>0,2,3))
            for a,b in zip(lab,pred): conf[a,b]+=1
            for c,i in zip("LRUD",range(4)): coss[c]+=list(cos[lab==i])
    r["twohead_4dir_acc"]=round(float(np.trace(conf)/conf.sum()),3)
    r["twohead_confusion_rows_true_LRUD"]=(conf/conf.sum(1,keepdims=True)).round(3).tolist()
    r["twohead_mean_cosine"]={c:round(float(np.mean(v)),3) for c,v in coss.items()}
    # 4-class multinomial
    def XY(s,feat=k):
        X=np.concatenate([D[s][f"{feat}_imag_LR"],D[s][f"{feat}_imag_FF"]])
        y=np.concatenate([D[s]["y_imag_LR"],2+D[s]["y_imag_FF"]]); return X,y
    for feat in [k,"base"]:
        Xp=np.concatenate([XY(s,feat)[0] for s in pool]); yp=np.concatenate([XY(s,feat)[1] for s in pool])
        m=LogisticRegression(C=1e-2,max_iter=5000).fit(Xp,yp)
        cf=np.zeros((4,4)); a=[]
        for s in test:
            X,y=XY(s,feat); p=m.predict(X); a.append(np.mean(p==y))
            for u,v in zip(y,p): cf[u,v]+=1
        r[f"multinomial4_{'task' if feat==k else 'precue_runconfound'}"]=dict(acc=round(float(np.mean(a)),3),confusion=(cf/cf.sum(1,keepdims=True)).round(3).tolist())
        # run-type (LR run vs FF run) decodability
        yr=(yp>=2).astype(int)
        mr=LogisticRegression(C=1e-2,max_iter=3000).fit(Xp,yr)
        r[f"runtype_decode_{'task' if feat==k else 'precue'}"]=round(float(np.mean([np.mean(mr.predict(XY(s,feat)[0])==(XY(s,feat)[1]>=2)) for s in test])),3)
    out[k]=r; print(k,json.dumps(r),flush=True)
json.dump(out,open("fourdir.json","w"),indent=1)
