from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi

def dice_per_class(pred: np.ndarray, gt: np.ndarray, c: int) -> float:
    P = (pred==c)
    G = (gt==c)
    inter = np.count_nonzero(P & G)
    denom = np.count_nonzero(P) + np.count_nonzero(G) + 1e-8
    return float(2.0*inter/denom)

def boundary_4n(binary: np.ndarray) -> np.ndarray:
    b = binary.astype(bool)
    up = np.roll(b, -1, axis=0)
    dn = np.roll(b, 1, axis=0)
    lf = np.roll(b, -1, axis=1)
    rt = np.roll(b, 1, axis=1)
    interior = b & up & dn & lf & rt
    return b & (~interior)

def hd95_per_class(pred: np.ndarray, gt: np.ndarray, c: int) -> float:
    P = (pred==c)
    G = (gt==c)
    BP = boundary_4n(P)
    BG = boundary_4n(G)
    h,w = pred.shape
    if BP.sum()==0 and BG.sum()==0:
        return 0.0
    if BP.sum()==0 or BG.sum()==0:
        return float(max(h,w))
    # compute EDT of complement boundary
    dist_to_BG = ndi.distance_transform_edt(~BG)
    dist_to_BP = ndi.distance_transform_edt(~BP)
    dPG = dist_to_BG[BP]
    dGP = dist_to_BP[BG]
    p95_PG = float(np.percentile(dPG, 95))
    p95_GP = float(np.percentile(dGP, 95))
    return float(max(p95_PG, p95_GP))

def dice_macro(pred: np.ndarray, gt: np.ndarray) -> tuple[float,float,float,float]:
    d1 = dice_per_class(pred, gt, 1)
    d2 = dice_per_class(pred, gt, 2)
    d3 = dice_per_class(pred, gt, 3)
    return float((d1+d2+d3)/3.0), d1, d2, d3

def hd95_macro(pred: np.ndarray, gt: np.ndarray) -> tuple[float,float,float,float]:
    h1 = hd95_per_class(pred, gt, 1)
    h2 = hd95_per_class(pred, gt, 2)
    h3 = hd95_per_class(pred, gt, 3)
    return float((h1+h2+h3)/3.0), h1, h2, h3
