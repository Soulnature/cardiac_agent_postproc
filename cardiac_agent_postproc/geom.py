from __future__ import annotations
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.measure import regionprops
from dataclasses import dataclass

def boundary_4n(binary: np.ndarray) -> np.ndarray:
    # boundary pixels where any 4-neighbor is 0
    b = binary.astype(bool)
    up = np.roll(b, -1, axis=0)
    dn = np.roll(b, 1, axis=0)
    lf = np.roll(b, -1, axis=1)
    rt = np.roll(b, 1, axis=1)
    interior = b & up & dn & lf & rt
    return b & (~interior)

def label_cc(binary: np.ndarray) -> tuple[np.ndarray, int]:
    # 8-connectivity
    structure = np.ones((3,3), dtype=np.int8)
    lab, n = ndi.label(binary.astype(np.uint8), structure=structure)
    return lab, int(n)

def keep_largest_cc(binary: np.ndarray) -> np.ndarray:
    lab, n = label_cc(binary)
    if n <= 1:
        return binary.astype(bool)
    areas = [(lab==i).sum() for i in range(1, n+1)]
    k = int(np.argmax(areas) + 1)
    return (lab == k)

def remove_small_cc(binary: np.ndarray, small_thr: int) -> np.ndarray:
    lab, n = label_cc(binary)
    if n == 0:
        return binary.astype(bool)
    out = np.zeros_like(binary, dtype=bool)
    for i in range(1, n+1):
        comp = (lab == i)
        if comp.sum() >= small_thr:
            out |= comp
    return out

def fill_holes(binary: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(binary.astype(bool))

def perimeter(binary: np.ndarray) -> float:
    # use contour length
    b = binary.astype(np.uint8)
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return float(sum(cv2.arcLength(c, True) for c in cnts))

def compactness(binary: np.ndarray) -> float:
    area = float(binary.sum())
    per = perimeter(binary)
    if per <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (per*per))

def eccentricity_largest_cc(binary: np.ndarray) -> float:
    lab, n = label_cc(binary)
    if n == 0:
        return 0.0
    areas = [(lab==i).sum() for i in range(1, n+1)]
    k = int(np.argmax(areas) + 1)
    props = regionprops((lab==k).astype(np.uint8))
    if not props:
        return 0.0
    return float(props[0].eccentricity)

def centroid(binary: np.ndarray) -> tuple[float,float]:
    ys, xs = np.nonzero(binary)
    if len(xs)==0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))

def bbox(binary: np.ndarray) -> tuple[int,int,int,int]:
    ys, xs = np.nonzero(binary)
    if len(xs)==0:
        return (0,0,0,0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
