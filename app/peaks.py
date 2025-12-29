from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks, peak_widths

@dataclass
class Peak:
    rt: float
    height: float
    area: float
    left: int
    right: int

def integrate_trapz(x: np.ndarray, y: np.ndarray, left: int, right: int)->float:
    """
    Trapezoidal integration between indices [left, right].
    """
    return float(np.trapezoid(y[left:right+1],x[left:right+1]))

def detect_peaks(
        time : np.ndarray ,
        signal: np.ndarray,
        prominence: float= 1.0,
        distance_pts: int =10,
)-> list[Peak]:
    """
    Detect peaks on a baseline-corercted signal.
    Returns a list of Peak objects sorted by retention time (rt).
    """
    time= np.asarray(time, dtype=float)
    signal= np.asarray(signal, dtype=float)

    idx, props= find_peaks(signal, prominence= prominence, distance=distance_pts)
    if len(idx) == 0:
        return[]
    
    widths= peak_widths(signal, idx, 
    rel_height=0.5)
    left_ips=widths[2].astype(int)
    right_ips=widths[3].astype(int)

    peaks: list[Peak]=[]
    for i,p in enumerate(idx):
        left= max(0, left_ips[i])
        right=min(len(signal)-1, right_ips[i])
        area= integrate_trapz(time, signal, left, right)
        peaks.append(Peak(
        rt=float(time[p]),
        height= float(signal[p]), 
        area=area,
        left=int(left), 
        right=int(right),
    ))

    peaks.sort(key=lambda pk: pk.rt)
    return peaks
def pick_anchor_peak(peaks: list[Peak]) -> Peak| None: 
    """ Anchor peak= the tallest peak (MVP)."""
    if not peaks:
        return None
    return max(peaks, key=lambda p: p.height)

