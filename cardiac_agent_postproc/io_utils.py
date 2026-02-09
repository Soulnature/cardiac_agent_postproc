from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import cv2
import numpy as np

VALID_MASK_VALUES = {0,1,2,3}

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img

def to_grayscale_float(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        g = img
    else:
        # BGR to gray
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32)
    if g.max() > 1.0:
        g /= 255.0
    return g

def read_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        # if saved as RGB, take first channel
        m = m[:,:,0]
    m = m.astype(np.int32)
    # sanitize to 0..3 if needed
    uniq = set(np.unique(m).tolist())
    if not uniq.issubset(VALID_MASK_VALUES):
        # Case A: 0, 85, 170, 255
        if m.max() > 3:
             m = (m / 85).astype(np.int32)
             m = np.clip(m, 0, 3)
        else:
             m = np.clip(m, 0, 3)
    return m

def write_mask(path: str, mask: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Map 0,1,2,3 -> 0,85,170,255 for visibility/compatibility
    m = (mask * 85).astype(np.uint8)
    cv2.imwrite(path, m)

def list_frames(folder: str) -> List[Dict[str,str]]:
    # return list of dicts with img, pred, gt if exists
    files = os.listdir(folder)
    imgs = [f for f in files if f.endswith("_img.png")]
    frames = []
    for imgf in sorted(imgs):
        stem = imgf[:-8]  # remove _img.png
        pred = stem + "_pred.png"
        gt = stem + "_gt.png"
        rec = {
            "stem": stem,
            "img": os.path.join(folder, imgf),
            "pred": os.path.join(folder, pred),
        }
        if gt in files:
            rec["gt"] = os.path.join(folder, gt)
        frames.append(rec)
    return frames
