from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from app.peaks import Peak, pick_anchor_peak

@dataclass
class PeakMatch:
    ref: Peak
    test: Peak
    delta_rt: float
    area_pct_change: float
    height_pct_change: float

@dataclass
class CompareResult:
    matches: list[PeakMatch]
    new_peaks: list[Peak]
    lost_peaks:list[Peak]
    rt_shift: float
    anchor_ref:Optional[Peak]
    anchor_test:Optional[Peak]

def _pct_change(new:float, old:float) ->float:
    if old==0:
        return float("inf") if new != 0 else 0.0
    return(new-old)/old*100

def estimate_rt_shift(ref_peaks: list[Peak], test_peaks:list[Peak])->float:
    """
estimate RT shift usin another peaks (tallest peak).
rt_shift=test_anchor_rt-ref_anchor_rt
So to align test to ref: aligned_test_rt= test_rt-rt_shift
"""
    a_ref=pick_anchor_peak(ref_peaks)
    a_test=pick_anchor_peak(test_peaks)
    if(a_ref is None) or (a_test is None):
        return 0.0
    return float(a_test.rt-a_ref.rt)

def match_peaks_by_rt(
        
        ref_peaks: list[Peak],
        test_peaks: list[Peak],
        rt_tolerance: float= 0.10,
        rt_shift: float=0.0,
) ->tuple[list[PeakMatch], list[Peak], 
list[Peak]]:
    """
    Greedy matching by nearest RT after shifting test peaks.
    Returns (matches, new_peaks, lost_peaks).
    """
    test_adj=[(p, p.rt-rt_shift) for p in test_peaks]
    used_test=set()
    matches: list[PeakMatch]=[]

    for ref in sorted(ref_peaks, key=lambda p: p.rt):
        best_j=None
        best_abs=None
        best_adj_rt=None
        for j,(tp, adj_rt) in enumerate(test_adj):
            if j in used_test:
                continue
            d=adj_rt - ref.rt
            ad=abs(d)
            if ad <= rt_tolerance and (best_abs is None or ad < best_abs):
                best_abs=ad
                best_j=j
                best_adj_rt=adj_rt 
        if best_j is not None:
            tp, adj_rt= test_adj[best_j]
            used_test.add(best_j)

            matches.append(
                PeakMatch(
                    ref=ref,
                    test=tp,
                    delta_rt= float(best_adj_rt - ref.rt),
                    #afer alignement
area_pct_change=float(_pct_change(tp.area, ref.area)),
height_pct_change=float(_pct_change(tp.height, ref.height)),
                )
            )

        new=[test_adj[j][0] for j in range(len(test_adj)) if j not in used_test]
        lost=[r for r in ref_peaks if all(m.ref is not r for m in matches)]

    return matches, new, lost

def compare_batches(
        ref_peaks: list[Peak],
        test_peaks: list[Peak],
        rt_tolerance: float=0.10,
) -> CompareResult:
    anchor_ref= pick_anchor_peak(ref_peaks)
    a_test= pick_anchor_peak(test_peaks)
    rt_shift=estimate_rt_shift(ref_peaks, test_peaks)
    matches, new_peaks, lost_peaks= match_peaks_by_rt(
        ref_peaks,
        test_peaks,
        rt_tolerance=rt_tolerance,
        rt_shift=rt_shift,

    )
    
    anchor_ref= pick_anchor_peak(ref_peaks)
    anchor_test= pick_anchor_peak(test_peaks)

    return CompareResult(
        matches=matches,
        new_peaks=new_peaks,
        lost_peaks=lost_peaks,
        rt_shift=rt_shift,
        anchor_ref=anchor_ref,
        anchor_test=anchor_test,
    )
#sort outputs for nice reporting
    matches.sort(key=lambda m: m.ref.rt)
    new_peaks.sort(key=lambda p: p.rt)
    lost_peaks.sort(key=lambda p: p.rt)

    return CompareResult(
        matches=matches,
        new_peaks=new_peaks,
        lost_peaks=lost_peaks,
        rt_shift=rt_shift,
        anchor_ref=anchor_ref,
        anchor_test=anchor_test,

    )
        

