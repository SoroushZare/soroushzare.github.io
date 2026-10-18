import numpy as np, dataclasses, uuid
from neural_cursor.config import DEFAULT
from neural_cursor.decoder import VelocityDecoder
from neural_cursor.simulate import SimUser, calibration
from neural_cursor.acquisition import SimulatedEEG, LSLEEG
from neural_cursor.realtime import RealTimeLoop, SimLSLStreamer
from neural_cursor.effectors import NullEffector
u=SimUser(DEFAULT,seed=11,gain=8,eye=0)
ws,ys=calibration(u,240,np.random.default_rng(11))
d=VelocityDecoder(DEFAULT.decoder,500).fit(ws,ys)
for s in (23,24,25):
  for chunk in (0.01,0.1):
    hw=dataclasses.replace(DEFAULT.hardware,lsl_stream_type="T"+uuid.uuid4().hex[:6]); cfg=dataclasses.replace(DEFAULT,hardware=hw)
    sim=SimulatedEEG(hw,seed=s); st=SimLSLStreamer(sim,chunk_s=chunk); st.set_intent(1,0); st.start()
    src=LSLEEG(hw); src.start()
    l=RealTimeLoop(src,d,NullEffector(),cfg)
    r=l.run(3.0); st.stop()
    print(s,chunk,np.round(r['decoded_mean'],3), r['latency_ms']['p95'])
