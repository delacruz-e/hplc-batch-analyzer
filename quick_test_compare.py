import numpy as np

from app.preprocess import preprocess
from app.peaks import detect_peaks
from app.compare import compare_batches

def make_signal(time, shift=0.0, scale=1.0,extra_peak=False):
    sig = (
        5
        + 0.2 * time
        +(10*scale) * np.exp(-0.5 * ((time - (4 + shift)) / 0.2) ** 2)
        +(6*scale) * np.exp(-0.5 * ((time - (7 + shift)) / 0.25) ** 2)
        + np.random.normal(0, 0.2, size=time.shape)
    )
    if extra_peak:
        sig += 3.5*np.exp(-0.5*((time-(2.5+shift))/0.12)**2)

    return sig
time = np.linspace(0, 10, 500)
ref_raw = make_signal(time, shift=0.0, scale=1.0, extra_peak=False)
_ , _, ref_corr = preprocess(time, ref_raw)
ref_peaks = detect_peaks(time, ref_corr, prominence=1.0, distance_pts=20)


test_raw = make_signal(time, shift=0.08, scale =0.95, extra_peak=True)
_ , _, test_corr = preprocess(time, test_raw)
test_peaks = detect_peaks(time, test_corr, prominence=1.0, distance_pts=20)

res= compare_batches(
    ref_peaks,
    test_peaks,
    rt_tolerance=0.12,
)

print(f"RT shift estimated: {res.rt_shift:+.3f} min")
print(f"Matches:{len(res.matches)}| New: {len(res.new_peaks)} | Lost: {len(res.lost_peaks)}")
print()

for m in res.matches:
   print(
       f"REF RT={m.ref.rt:.3f} TEST RT={m.test.rt:.3f} dRT(aligned)={m.delta_rt:+.3f}"
       f"area%={m.area_pct_change:+.1f}%"
   )
if res.new_peaks:
    print("\nNew Peaks:")
    for p in res.new_peaks:
        print(f" RT={p.rt:.3f} height={p.height:.2f} area={p.area:.2f}")