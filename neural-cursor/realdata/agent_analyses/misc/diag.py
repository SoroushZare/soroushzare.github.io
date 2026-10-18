import numpy as np, time
from neural_cursor.config import DEFAULT
from neural_cursor.decoder import VelocityDecoder
from neural_cursor.simulate import SimUser, calibration
from neural_cursor.acquisition import SimulatedEEG
from neural_cursor.realtime import RealTimeLoop
from neural_cursor.effectors import NullEffector
u=SimUser(DEFAULT,seed=11,gain=8,eye=0)
ws,ys=calibration(u,240,np.random.default_rng(11))
d=VelocityDecoder(DEFAULT.decoder,500).fit(ws,ys)
for wu in (1.0,3.0):
  for s in (23,24):
    sim=SimulatedEEG(DEFAULT.hardware,seed=s); sim.set_intent(1,0); sim.start()
    l=RealTimeLoop(sim,d,NullEffector(),DEFAULT,warmup_s=wu)
    st=l.run(3.0)
    print(wu,s,np.round(st['decoded_mean'],3), st['n_gated'], np.round(l.decoded[:10,0],2))
