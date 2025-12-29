import numpy as np
from app.preprocess import preprocess

time=np.linspace(0,10,500)
signal=5+0.2*time+10*np.exp((-0.5*(time-4)/0.2)**2)+ np.random.normal(0, 0.2, size=time.shape)

sm, base, corr=preprocess(time, signal)
print("OK, preprocess: ", sm.shape, base.shape, corr.shape, "min corr: ", corr.min())
