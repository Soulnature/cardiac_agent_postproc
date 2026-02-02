from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import cv2
from scipy import ndimage as ndi

from .io_utils import to_grayscale_float
from .geom import boundary_4n, label_cc, keep_largest_cc, remove_small_cc, fill_holes, bbox, compactness
from .atlas import match_atlas, align_mask

@dataclass
class RQSResult:
    total: float
    terms: Dict[str, float]
    aux: Dict[str, float]

def clamp(x: float, lo: float=0.0, hi: float=100.0) -> float:
    return float(max(lo, min(hi, x)))

def sobel_edges(img: np.ndarray, percentile: float=85) -> np.ndarray:
    g = to_grayscale_float(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    thr = np.percentile(mag, percentile)
    return (mag > thr)

def valve_plane_y(mask: np.ndarray) -> int|None:
    U = (mask>0)
    if U.sum()==0:
        return None
    x0,y0,x1,y1 = bbox(U)
    h = max(1, y1-y0+1)
    return int(round(y0 + 0.08*h))

def _area(mask: np.ndarray, c: int) -> int:
    return int(np.count_nonzero(mask==c))

def _nonbg(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))

def _touch_pixels(mask: np.ndarray) -> int:
    M1 = (mask==1)
    M3 = (mask==3)
    if M1.sum()==0 or M3.sum()==0:
        return 0
    dil = ndi.binary_dilation(M1, iterations=1)
    return int(np.count_nonzero(dil & M3))

def _components_penalty(mask: np.ndarray, view_type: str, rv_optional_thr: float) -> float:
    P = 0.0
    nonbg = _nonbg(mask)
    rv_ratio = _area(mask,1)/max(1, nonbg)
    req = {2:1, 3:1}
    if view_type=="2ch":
        req[1] = 0
    else:
        req[1] = 1
    for c in [1,2,3]:
        Mc = (mask==c)
        _, n = label_cc(Mc)
        if req[c]==1:
            if c==1 and rv_ratio < rv_optional_thr:
                # allow empty RV
                if Mc.sum()==0:
                    continue
            P += 12.0 * max(0, n-1)
            if Mc.sum()==0:
                P += 12.0
        else:
            if Mc.sum()>0:
                P += 40.0
    return float(P)

def _islands_penalty(mask: np.ndarray, small_thr: int, cap: float=30.0) -> float:
    nonbg = _nonbg(mask)
    SMALL = max(small_thr, int(0.002*nonbg))
    P = 0.0
    for c in [1,2,3]:
        lab, n = label_cc(mask==c)
        sum_small = 0
        for i in range(1, n+1):
            comp = (lab==i)
            if comp.sum() < SMALL:
                sum_small += int(comp.sum())
        P += 20.0 * min(1.0, sum_small / (0.02*nonbg + 1.0))
    return float(min(cap, P))

def _valve_leak_penalty(mask: np.ndarray) -> float:
    nonbg = _nonbg(mask)
    vy = valve_plane_y(mask)
    if vy is None:
        return 0.0
    leak = int(np.count_nonzero((mask>0) & (np.arange(mask.shape[0])[:,None] < vy)))
    return float(25.0 * min(1.0, leak / (0.02*nonbg + 1.0)))

def _double_cavity_penalty(mask: np.ndarray) -> float:
    lab, n = label_cc(mask==3)
    return 30.0 if n >= 2 else 0.0

def _holes_penalty(mask: np.ndarray) -> float:
    M3 = (mask==3)
    M2 = (mask==2)
    if M3.sum()>0:
        fill3 = ndi.binary_fill_holes(M3)
        holes3 = int(np.count_nonzero(fill3 & (~M3)) > 0)
    else:
        holes3 = 0
    if M2.sum()>0:
        fill2 = ndi.binary_fill_holes(M2)
        holes2 = int(np.count_nonzero(fill2 & (~M2)) > 0)
    else:
        holes2 = 0
    return float(10.0 * min(3.0, holes3 + 0.5*holes2))

def _size_drift_penalty(mask: np.ndarray, pred0: np.ndarray, atlas_entry: dict|None, atlas_params: dict|None) -> tuple[float,float]:
    nonbg0 = max(1, int(np.count_nonzero(pred0)))
    drift = float(np.count_nonzero(mask!=pred0) / (nonbg0 + 1.0))
    atlas_direction_score = 0.0
    if atlas_entry is not None and atlas_params is not None:
        aligned_old, _ = align_mask(pred0, target_diag=float(atlas_entry["align_params"]["target_diag"]))
        aligned_new, _ = align_mask(mask, target_diag=float(atlas_entry["align_params"]["target_diag"]))
        # direction: changed pixels move to higher atlas prob for their new label vs old label
        changed = (aligned_new != aligned_old)
        if changed.sum() > 0:
            P = atlas_entry["prob_maps"]
            old_lbl = aligned_old[changed]
            new_lbl = aligned_new[changed]
            yy, xx = np.nonzero(changed)
            better = 0
            for i in range(len(xx)):
                o = int(old_lbl[i]); n = int(new_lbl[i])
                if n in [1,2,3] and o in [1,2,3]:
                    if P[n][yy[i],xx[i]] > P[o][yy[i],xx[i]]:
                        better += 1
                elif n in [1,2,3] and o==0:
                    # compare to background prob approx
                    bgp = 1.0 - (P[1][yy[i],xx[i]] + P[2][yy[i],xx[i]] + P[3][yy[i],xx[i]])
                    if P[n][yy[i],xx[i]] > bgp:
                        better += 1
            atlas_direction_score = float(better / (len(xx) + 1.0))
        eff = drift * (1.0 - 0.6*atlas_direction_score)
    else:
        eff = drift
    P = 25.0 * min(1.0, eff / 0.35)
    return float(P), float(atlas_direction_score)

def _edge_misalignment_penalty(mask: np.ndarray, E: np.ndarray, overlap_target: float=0.55) -> tuple[float, dict]:
    Psum = 0.0
    overlaps = {}
    for c in [1,2,3]:
        B = boundary_4n(mask==c)
        denom = B.sum() + 1.0
        ov = float(np.count_nonzero(B & E) / denom)
        overlaps[c] = ov
        pc = 8.0 * max(0.0, overlap_target - ov) / overlap_target
        Psum += pc
    return float(min(25.0, Psum)), overlaps

def _enclosure_reward(mask: np.ndarray) -> float:
    M3 = (mask==3)
    M2 = (mask==2)
    if M3.sum()==0 or M2.sum()==0:
        return 0.0
    ring = ndi.binary_dilation(M3, iterations=2) & (~M3)
    coverage = float(np.count_nonzero(ring & M2) / (ring.sum() + 1.0))
    val = 20.0 * min(1.0, max(0.0, (coverage - 0.35) / 0.35))
    return float(val)

def _shape_prior_penalty_and_match(mask: np.ndarray, atlas_entry: dict|None) -> tuple[float,float,float]:
    if atlas_entry is None:
        return 0.0, 0.0, 0.0
    aligned, _ = align_mask(mask, target_diag=float(atlas_entry["align_params"]["target_diag"]))
    # area deviation
    area_pen = 0.0
    for c in [1,2,3]:
        mu, sd = atlas_entry["area_stats"][c]
        z = abs(np.count_nonzero(mask==c) - mu) / (sd + 1e-8)
        area_pen += min(1.0, max(0.0, z - 1.5) / 2.0)
    P_area = 8.0 * area_pen
    # spatial overlap and shape match
    low_pen = 0.0
    confs = []
    for c in [1,2,3]:
        Mc = (aligned==c).astype(np.float32)
        denom = Mc.sum() + 1e-8
        conf = float((atlas_entry["prob_maps"][c] * Mc).sum() / denom)
        confs.append(conf)
        low_pen += max(0.0, 0.4 - conf) / 0.4
    P_spatial = 8.0 * min(1.0, low_pen / 2.0)
    P_shape = float(min(20.0, P_area + P_spatial))
    mean_conf = float(np.mean(confs))
    R_shape = 15.0 * min(1.0, max(0.0, (mean_conf - 0.3) / 0.4))
    return P_shape, R_shape, mean_conf

def _boundary_roughness_penalty(mask: np.ndarray, contour_epsilon: float=1.5) -> float:
    # Implemented as in spec (approx): sharp turns + solidity defect proxies.
    Psum = 0.0
    for c in [1,2,3]:
        Mc = (mask==c).astype(np.uint8)
        B = boundary_4n(Mc>0)
        if B.sum() < 10:
            continue
        cnts, _ = cv2.findContours(Mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sharp_ratio = 0.0
        total_samples = 0
        sharp = 0
        for cnt in cnts:
            if len(cnt) < 10:
                continue
            pts = cnt[:,0,:]  # Nx2
            # sample every 3rd point
            idx = np.arange(0, len(pts), 3)
            pts_s = pts[idx]
            if len(pts_s) < 10:
                continue
            # compute angles using i-2, i, i+2
            for i in range(2, len(pts_s)-2):
                p0 = pts_s[i-2].astype(np.float32)
                p1 = pts_s[i].astype(np.float32)
                p2 = pts_s[i+2].astype(np.float32)
                v1 = p0 - p1
                v2 = p2 - p1
                n1 = np.linalg.norm(v1) + 1e-6
                n2 = np.linalg.norm(v2) + 1e-6
                cosang = float(np.clip(np.dot(v1,v2)/(n1*n2), -1.0, 1.0))
                ang = np.degrees(np.arccos(cosang))  # 0..180
                # sharp turn if deviation from 180 exceeds 60 => angle < 120
                if ang < 120.0:
                    sharp += 1
                total_samples += 1
        sharp_ratio = float(sharp / (total_samples + 1.0))

        # solidity
        area = float(Mc.sum())
        if area <= 0:
            continue
        cnts, _ = cv2.findContours(Mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull)) + 1.0
        solidity = float(area / hull_area)

        if c==3:
            defect = max(0.0, 0.8 - solidity) / 0.3
        elif c==1:
            defect = max(0.0, 0.5 - solidity) / 0.3
        else:
            defect = 0.0

        pc = 5.0 * min(1.0, sharp_ratio / 0.15) + 3.0 * min(1.0, defect)
        Psum += pc
    return float(min(20.0, Psum))

def compute_rqs(mask: np.ndarray,
                img: np.ndarray,
                view_type: str,
                pred0: np.ndarray,
                atlas: dict|None,
                cfg: dict) -> RQSResult:
    weights = cfg["reward_weights"]
    nonbg = _nonbg(mask)
    rv_ratio = _area(mask,1)/max(1, nonbg)
    touch = _touch_pixels(mask)

    E = sobel_edges(img, percentile=float(cfg["edge"]["sobel_percentile"]))

    atlas_entry, atlas_cid, atlas_match_score, atlas_params = (None, None, 0.0, None)
    if atlas is not None:
        atlas_entry, atlas_cid, atlas_match_score, atlas_params = match_atlas(mask, view_type, atlas)

    P_touch = 0.0
    if view_type in ("3ch","4ch"):
        if touch > 0:
            P_touch = 50.0 + 10.0 * min(5.0, touch/50.0)
    # 2ch: 0
    P_rv_2ch = 0.0
    if view_type=="2ch":
        P_rv_2ch = 80.0 * min(1.0, rv_ratio / float(cfg["view_inference"]["rv_ratio_2ch_threshold"]))

    P_components = _components_penalty(mask, view_type, rv_optional_thr=float(cfg["view_inference"]["rv_ratio_rv_optional_threshold"]))
    small_thr = int(cfg["ops"]["small_component_floor"])
    P_islands = _islands_penalty(mask, small_thr=small_thr)
    P_valve_leak = _valve_leak_penalty(mask)
    P_double_cavity = _double_cavity_penalty(mask)
    P_holes = _holes_penalty(mask)
    P_size_drift, atlas_direction_score = _size_drift_penalty(mask, pred0, atlas_entry, atlas_params)
    P_edge_misalignment, overlaps = _edge_misalignment_penalty(mask, E, overlap_target=float(cfg["edge"]["overlap_target"]))
    R_enclosure = _enclosure_reward(mask)
    P_shape_prior, R_shape_match, mean_conf = _shape_prior_penalty_and_match(mask, atlas_entry)
    P_boundary_roughness = _boundary_roughness_penalty(mask, contour_epsilon=float(cfg["ops"]["smooth"]["contour_epsilon"]))

    total = 100.0         - weights["P_touch"]*P_touch         - weights["P_rv_2ch"]*P_rv_2ch         - weights["P_components"]*P_components         - weights["P_islands"]*P_islands         - weights["P_valve_leak"]*P_valve_leak         - weights["P_double_cavity"]*P_double_cavity         - weights["P_holes"]*P_holes         - weights["P_size_drift"]*P_size_drift         - weights["P_edge_misalignment"]*P_edge_misalignment         - weights["P_shape_prior"]*P_shape_prior         - weights["P_boundary_roughness"]*P_boundary_roughness         + weights["R_enclosure"]*R_enclosure         + weights["R_shape_match"]*R_shape_match

    terms = {
        "P_touch": float(P_touch),
        "P_rv_2ch": float(P_rv_2ch),
        "P_components": float(P_components),
        "P_islands": float(P_islands),
        "P_valve_leak": float(P_valve_leak),
        "P_double_cavity": float(P_double_cavity),
        "P_holes": float(P_holes),
        "P_size_drift": float(P_size_drift),
        "P_edge_misalignment": float(P_edge_misalignment),
        "P_shape_prior": float(P_shape_prior),
        "P_boundary_roughness": float(P_boundary_roughness),
        "R_enclosure": float(R_enclosure),
        "R_shape_match": float(R_shape_match),
    }
    aux = {
        "rv_ratio": float(rv_ratio),
        "touch_pixel_count": float(touch),
        "atlas_cluster_id": float(atlas_cid) if atlas_cid is not None else -1.0,
        "atlas_match_score": float(atlas_match_score),
        "atlas_direction_score": float(atlas_direction_score),
        "mean_conf": float(mean_conf),
        "boundary_edge_overlap_c1": float(overlaps.get(1,0.0)),
        "boundary_edge_overlap_c2": float(overlaps.get(2,0.0)),
        "boundary_edge_overlap_c3": float(overlaps.get(3,0.0)),
    }
    return RQSResult(total=clamp(total), terms=terms, aux=aux)
