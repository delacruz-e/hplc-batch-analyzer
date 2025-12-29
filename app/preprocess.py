"""
Signal processing for HPLC chromatograms.

Goal: 
-reduce noise
-Estimate baseline drift
-Return baseline-corrected signal suitable for peak detection/integration
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

def smooth(signal : np.ndarray, window: int =21, poly : int = 3)->np.ndarray:
    """
    Savitzky-Golay smoothing

    windos: must be odd and < len(signal)
    poly: polynomial order

    """

    signal=np.asarray(signal, dtype=float)
    if signal.ndim != 1 or len(signal) <5 :
        raise ValueError("signal must be a 1D array with at least 5 points")
    window = max(5, int(window))
    if window % 2 ==0:
        windows +=1
    if window >= len(signal):
        window=max(5,(len(signal)//2)*2-1)
    
    poly=int(poly)
    poly=max(2, min(poly, window-2))

    return savgol_filter(signal, window_length=window, polyorder=poly)

def baseline_rolling_min(signal: np.ndarray,window:int=200)->np.ndarray:
    """
    Simple baseline estimator (MVP): rolling minimum.

    This is not the BEST baseline method, but it's stable and easy.
    Later we can replace it with ALS baseline without changing other modules

    """
    signal=np.asarray(signal, dtype=float)
    n= len(signal)
    w= max(10, int(window))

    baseline=np.empty(n, dtype=float)

    half= w//2
    for i in range(n):
        a=max(0,i-half)
        b=min(n,i+ half +1)
        baseline[i]=float(np.min(signal[a:b]))
    return baseline 

def preprocess (time:np.ndarray,
                signal: np.ndarray, smooth_window: int = 21,
                smooth_poly: int=3, baseline_window:int=200):
    """
    Returns:
    smoothed_signal, baseline, corrected_signal

    corrected_signal is clipped at 0 to avoid negative 
    values after substraction
    """
    time=np.asarray(time, dtype=float)
    signal= np.asarray(signal, dtype=float)

    if len(time) != len(signal):
        raise ValueError("time and signal must have the same length")
    if len(time) <5:
        raise ValueError("chromatogram too short")
    
    sm=smooth(signal, window=smooth_window, poly=smooth_poly)
    base=baseline_rolling_min(sm,window=baseline_window)
    
    corrected=sm-base
    corrected[corrected<0]=0.0

    return sm, base, corrected