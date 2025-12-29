import numpy as np
from app.preprocess import preprocess
from app.peaks import detect_peaks, pick_anchor_peak

#analytical signal HPLC type

time= np.linspace(0,10,500)
signal=(
    5
    + 0.2 * time
    +10*np.exp(-0.5*((time-4)/0.2)**2)
    +6*np.exp(-0.5*((time-7)/0.25)**2)
    +np.random.normal(0,0.2, size=time.shape)
)
sm, base, corr= preprocess(time, signal)

peaks=detect_peaks(time, corr, prominence=1.0, distance_pts=20)
anchor= pick_anchor_peak(peaks)

print(f"Detected peaks: {len(peaks)}")
for p in peaks:
    print(f"RT{p.rt:.3f} height={p.height}:.2f area{p.area:.2f} [{p.left},{p.right}]")

    if anchor:
        print(f"ANCHOR:RT={anchor.rt:.3f} height={anchor.height:.2f}")
    else:
        print("No anchor peak found")

