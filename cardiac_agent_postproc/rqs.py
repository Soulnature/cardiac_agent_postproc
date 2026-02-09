from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import numpy as np
import cv2
from scipy import ndimage as ndi
import joblib
import os

from .io_utils import to_grayscale_float
from .geom import boundary_4n, label_cc, keep_largest_cc, remove_small_cc, fill_holes, bbox, compactness
from .atlas import match_atlas, align_mask
from .quality_model import QualityScorer, QUALITY_FEATURE_NAMES

# Phase 3: Surrogate Model
_SURROGATE_MODEL = None
_QUALITY_SCORERS: dict[str, QualityScorer | None] = {}
_FEATURE_COLS = ['P_touch', 'P_rv_2ch', 'P_components', 'P_islands', 'P_valve_leak', 
                 'P_double_cavity', 'P_holes', 'P_size_drift', 'P_shape_prior', 
                 'P_atlas_mismatch', 'P_atlas_lv_mismatch', 'P_lv_shape', 
                 'P_slice_consistency', 'P_boundary_complexity', 'P_myo_thickness_var', 
                 'R_enclosure', 'R_shape_match', 'atlas_match_score', 'mean_conf']

def _load_surrogate(path: str):
    global _SURROGATE_MODEL
    if _SURROGATE_MODEL is None and os.path.exists(path):
        try:
            _SURROGATE_MODEL = joblib.load(path)
            print(f"[RQS] Loaded Surrogate Model from {path}")
        except Exception as e:
            print(f"[RQS] Failed to load Surrogate: {e}")


def _get_quality_scorer(cfg: dict) -> QualityScorer | None:
    q_cfg = cfg.get("quality_model", {})
    if not q_cfg.get("enabled", False):
        return None
    path = q_cfg.get("path", "quality_model.pkl")
    if path in _QUALITY_SCORERS:
        return _QUALITY_SCORERS[path]
    scorer = QualityScorer.load(path)
    if scorer is None:
        print(f"[RQS] Warning: quality model not found at {path}")
    _QUALITY_SCORERS[path] = scorer
    return scorer

@dataclass
class RQSResult:
    total: float
    terms: Dict[str, float]
    aux: Dict[str, float]

def _calc_mean_conf(mask, view_type):
    # Dummy placeholder if confidence map not available
    # In optimizer.py we have the map, but here we might not?
    # Actually, train_surrogate used 'mean_conf' from LOG, which came from optimizer.
    # We should calculate it if possible, or pass it in.
    # For now, return 0.5 (Average) or modify signature.
    return 0.5

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

def _components_penalty(mask: np.ndarray, view_type: str, rv_optional_thr: float, weight: float) -> float:
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
                # allow empty RV if small predicted ratio
                if Mc.sum()==0:
                    continue
            P += weight * max(0, n-1)
            if Mc.sum()==0:
                P += weight
        else:
            if Mc.sum()>0:
                # 40.0 seems to be a fixed constant in user code ("unexpected presence")
                # User code: if view=="2ch" and area>0: p+=40.
                # keeping 40.0 as hardcoded for now or make it a param? 
                # User config doesn't have P_unexpected.
                P += 40.0
    return float(P)

def _islands_penalty(mask: np.ndarray, small_thr: int, weight: float, cap: float=30.0) -> float:
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
        P += weight * min(1.0, sum_small / (0.02*nonbg + 1.0))
    return float(min(cap, P))

def _valve_leak_penalty(mask: np.ndarray, weight: float, threshold: float=0.05) -> float:
    nonbg = _nonbg(mask)
    vy = valve_plane_y(mask)
    if vy is None:
        return 0.0
    leak = int(np.count_nonzero((mask>0) & (np.arange(mask.shape[0])[:,None] < vy)))
    return float(weight * min(1.0, max(0.0, leak/(nonbg+1.0) - threshold)/threshold))

def _double_cavity_penalty(mask: np.ndarray, weight: float) -> float:
    lab, n = label_cc(mask==3)
    return weight if n >= 2 else 0.0

def _holes_penalty(mask: np.ndarray, weight: float) -> float:
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
    return float(weight * min(3.0, holes3 + 0.5*holes2))

def _size_drift_penalty(mask: np.ndarray, pred0: np.ndarray, atlas_entry: dict|None, weight: float, threshold: float=0.35) -> tuple[float,float]:
    nonbg0 = max(1, int(np.count_nonzero(pred0)))
    drift = float(np.count_nonzero(mask!=pred0) / (nonbg0 + 1.0))
    atlas_direction_score = 0.0
    
    if atlas_entry is not None:
        aligned_old, _ = align_mask(pred0, target_diag=float(atlas_entry["align_params"]["target_diag"]))
        aligned_new, _ = align_mask(mask, target_diag=float(atlas_entry["align_params"]["target_diag"]))
        
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
                    bgp = 1.0 - (P[1][yy[i],xx[i]] + P[2][yy[i],xx[i]] + P[3][yy[i],xx[i]])
                    if P[n][yy[i],xx[i]] > bgp:
                        better += 1
            atlas_direction_score = float(better / (len(xx) + 1.0))
        eff = drift * (1.0 - 0.6*atlas_direction_score)
    else:
        eff = drift
    P = weight * min(1.0, eff / threshold)
    return float(P), float(atlas_direction_score)

def _enclosure_reward(mask: np.ndarray, weight: float) -> float:
    M3 = (mask==3)
    M2 = (mask==2)
    if M3.sum()==0 or M2.sum()==0:
        return 0.0
    ring = ndi.binary_dilation(M3, iterations=2) & (~M3)
    coverage = float(np.count_nonzero(ring & M2) / (ring.sum() + 1.0))
    # V5: w=5, reduced
    val = weight * min(1.0, max(0.0, (coverage - 0.35) / 0.35))
    return float(val)

def _shape_prior_and_match(mask: np.ndarray, atlas_entry: dict|None, valid_keys: dict) -> tuple[float, float, float, float, float, float]:
    # Returns: P_shape_prior, P_atlas_mismatch, P_atlas_lv_mismatch, R_shape_match, mean_conf, P_lv_shape
    if atlas_entry is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    aligned, _ = align_mask(mask, target_diag=float(atlas_entry["align_params"]["target_diag"]))
    
    # 1. P_shape_prior (Area + Spatial) - V5: cap 4
    area_pen = 0.0
    for c in [1,2,3]:
        mu, sd = atlas_entry["area_stats"][c]
        z = abs(np.count_nonzero(mask==c) - mu) / (sd + 1e-8)
        area_pen += min(1.0, max(0.0, z - 1.5) / 2.0)
    
    low_pen = 0.0
    confs = []
    overlaps = {}
    for c in [1,2,3]:
        Mc = (aligned==c).astype(np.float32)
        denom = Mc.sum() + 1e-8
        conf = float((atlas_entry["prob_maps"][c] * Mc).sum() / denom)
        confs.append(conf)
        overlaps[c] = conf
        low_pen += max(0.0, 0.4 - conf) / 0.4
    
    # weights keys: "P_shape_prior", "P_atlas_mismatch", "P_atlas_lv_mismatch", "R_shape_match", "P_lv_shape"
    # Assuming caller passes dict with these keys and also special "P_shape_prior_area" etc if needed? 
    # For now assume split is fixed ratio or allow detail tuning?
    # User code: P_shape_prior_area=1.0, P_shape_prior_spatial=1.0. Total P_shape_prior = sum clipped at cap.
    
    # Let's trust just P_shape_prior total weight for now.
    # Actually wait, User Code: 
    # P_shape_prior_area=1.0, P_shape_prior_spatial=1.0, P_shape_prior_cap=4.0
    # In my config: P_shape_prior=1.0 (total weight? Or area weight?)
    # My previous implementation was: P_area (weight 1 in formula) + P_spatial (weight 1) -> sum -> clip(4.0) -> Multiplied by P_shape_prior(1.0)
    # So effectively same.
    
    # Correct refactor: Return RAW terms? No, User Reference puts weights inside helper.
    # So I apply weights HERE.
    
    w_shape = valid_keys.get("P_shape_prior", 1.0)
    P_area = 1.0 * area_pen
    P_spatial = 1.0 * min(1.0, low_pen / 2.0)
    # The 'weight' in config P_shape_prior is likely a multiplier for the whole term?
    # User Reference: terms["P_shape_prior"] = min(cap, P_area_weighted + P_spatial_weighted)
    # where P_area used weight 1.0.
    
    P_shape_prior = w_shape * float(min(4.0, P_area + P_spatial))
    
    # 5. P_atlas_mismatch
    mean_overlap = float(np.mean(confs))
    P_atlas_mismatch = valid_keys.get("P_atlas_mismatch", 40.0) * min(1.0, max(0.0, 0.35 - mean_overlap) / 0.35)
    
    # 6. P_atlas_lv_mismatch (class 3 only)
    overlap_lv = overlaps.get(3, 0.0)
    P_atlas_lv_mismatch = valid_keys.get("P_atlas_lv_mismatch", 8.0) * min(1.0, max(0.0, 0.44 - overlap_lv) / 0.12)
    
    # 7. P_lv_shape (Ellipse fit IoU for LV)
    M3 = (mask==3).astype(np.uint8)
    if M3.sum() > 10:
        cnts, _ = cv2.findContours(M3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            if len(cnt) >= 5:
                # fitEllipse needs >= 5 points
                ellipse = cv2.fitEllipse(cnt)
                canvas = np.zeros_like(M3)
                cv2.ellipse(canvas, ellipse, 1, -1)
                union = np.count_nonzero((M3>0)|(canvas>0))
                inter = np.count_nonzero((M3>0)&(canvas>0))
                iou = inter / (union + 1e-8)
                P_lv_shape = valid_keys.get("P_lv_shape", 8.0) * min(1.0, max(0.0, 0.80 - iou) / 0.12)
            else:
                P_lv_shape = valid_keys.get("P_lv_shape", 8.0)
        else:
             P_lv_shape = valid_keys.get("P_lv_shape", 8.0)
    else:
         P_lv_shape = 0.0 # too small to judge

    # 20. R_shape_match
    R_shape_match = valid_keys.get("R_shape_match", 15.0) * min(1.0, max(0.0, (mean_overlap - 0.2) / 0.4))

    return P_shape_prior, P_atlas_mismatch, P_atlas_lv_mismatch, R_shape_match, mean_overlap, P_lv_shape

def _slice_consistency_penalty(mask: np.ndarray, siblings: List[np.ndarray], view_type: str, weight: float) -> float:
    # V5 Tier 2
    if not siblings:
        return 0.0
    
    # a) Centroid deviation
    centroid_pen_sum = 0.0
    count_c = 0
    H, W = mask.shape
    norm_dim = max(H,W)
    
    for c in [1,2,3]:
        Mc = (mask==c)
        if Mc.sum()==0: continue
        
        cy, cx = ndi.center_of_mass(Mc)
        max_dist = 0.0
        valid_sib = False
        for sib in siblings:
            Sc = (sib==c)
            if Sc.sum()==0: continue
            sy, sx = ndi.center_of_mass(Sc)
            dist = np.sqrt((cy-sy)**2 + (cx-sx)**2)
            max_dist = max(max_dist, dist)
            valid_sib = True
        
        if valid_sib:
            ratio = max_dist / norm_dim
            pen = max(0.0, ratio - 0.15) / 0.10
            centroid_pen_sum += pen
            count_c += 1
            
    P_centroid = 6.0 * min(1.0, centroid_pen_sum / max(1, count_c))

    # b) Area ratio consistency
    nonbg = _nonbg(mask)
    area_pen_sum = 0.0
    count_a = 0
    
    for c in [1,2,3]:
        Mc = (mask==c)
        ratio = Mc.sum() / (nonbg + 1.0)
        max_diff = 0.0
        valid_sib = False
        for sib in siblings:
            Sc = (sib==c)
            if Sc.sum()==0 and Mc.sum()==0: continue # both empty is fine
            
            s_nonbg = _nonbg(sib)
            s_ratio = Sc.sum() / (s_nonbg + 1.0)
            diff = abs(ratio - s_ratio)
            max_diff = max(max_diff, diff)
            valid_sib = True
            
        if valid_sib:
            pen = max(0.0, max_diff - 0.25) / 0.15
            area_pen_sum += pen
            count_a += 1
            
    P_area = 4.0 * min(1.0, area_pen_sum / max(1, count_a))

    # c) Orientation
    # simple PCA angle
    def get_angle(m):
        pts = np.argwhere(m > 0)
        if len(pts) < 10: return None
        # y,x -> need to swap for consistent angle or just PCA
        pts = pts.astype(np.float32)
        # PCA
        mean = np.mean(pts, axis=0)
        centered = pts - mean
        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eig(cov)
        idx = np.argsort(evals)[::-1]
        evec = evecs[:, idx[0]] # major axis
        angle = np.degrees(np.arctan2(evec[0], evec[1])) # y,x
        return angle
        
    ang = get_angle(mask)
    P_orient = 0.0
    if ang is not None:
        max_ang_diff = 0.0
        valid_sib = False
        for sib in siblings:
            s_ang = get_angle(sib)
            if s_ang is not None:
                # angle difference modulo 180 (axis symmetry) or 360? 
                # Heart axis usually directed, but mask shape is slightly ambiguous. 
                # Let's assume directed range -180..180.
                d = abs(ang - s_ang)
                if d > 180: d = 360 - d
                # Also axis has 180 ambiguity? usually LAX is directed up-down.
                # If ambiguity, check min(d, 180-d). 
                # V5 spec says "block rotation > 20 deg".
                # Let's assume 180 ambiguity exists for ellipses but heart has distinct top/down.
                # We'll use direct diff.
                max_ang_diff = max(max_ang_diff, d)
                valid_sib = True
        
        if valid_sib:
             P_orient = 5.0 * min(1.0, max(0.0, max_ang_diff - 20.0) / 25.0)

    val = min(15.0, P_centroid + P_area + P_orient)
    # Apply global weight multiplier (User code doesn't explicitly show this helper but likely 15.0 was cap, weight is scalar)
    # Let's assume weight is multiplier for the capped term.
    # Note: P_centroid (6) + P_area (4) + P_orient (5) = 15. 
    # If config weight is different from 15, we scale.
    # Scale factor: weight/15.0
    return float(val * (weight/15.0))

def _boundary_complexity_penalty(mask: np.ndarray, weight: float) -> float:
    # P_boundary_complexity
    B = boundary_4n(mask>0)
    nonbg = _nonbg(mask)
    ratio = B.sum() / (nonbg + 1.0)
    return float(weight * min(1.0, max(0.0, ratio - 0.13) / 0.05))

def _myo_thickness_var_penalty(mask: np.ndarray, weight: float) -> float:
    M2 = (mask==2)
    M3 = (mask==3)
    if M2.sum()==0 or M3.sum()==0:
        return 0.0
    
    # thickness of M2 approx: distance from inner boundary (adj to M3)
    inner_b = boundary_4n(M3)
    if inner_b.sum()==0: return 0.0
    
    dist_map = ndi.distance_transform_edt(~inner_b)
    # sample at M2 pixels that are on the OUTER boundary of M2
    outer_b = boundary_4n(M2) & (~inner_b) # roughly
    
    if outer_b.sum() == 0: return 0.0
    
    thicknesses = dist_map[outer_b]
    mu = thicknesses.mean()
    sd = thicknesses.std()
    cv = sd / (mu + 1e-8)
    
    return float(weight * min(1.0, max(0.0, cv - 0.36) / 0.15))

def compute_rqs(mask: np.ndarray,
                img: np.ndarray,
                view_type: str,
                pred0: np.ndarray,
                atlas: dict|None,
                siblings: List[np.ndarray],
                cfg: dict,
                z_norm: float|None = None) -> RQSResult:
    weights = cfg["reward_weights"]
    nonbg = _nonbg(mask)
    rv_ratio = _area(mask,1)/max(1, nonbg)
    touch = _touch_pixels(mask)

    atlas_entry, atlas_cid, atlas_match_score, atlas_params = (None, None, 0.0, None)
    if atlas is not None:
        atlas_entry, atlas_cid, atlas_match_score, atlas_params = match_atlas(mask, view_type, atlas, z_norm=z_norm)

    # 1) P_touch (User code: 50 + extra).
    P_touch = 0.0
    w_touch = weights.get("P_touch", 50.0)
    if view_type in ("3ch","4ch") and w_touch > 0:
        if touch > 0:
             # Logic from user code: weights["P_touch"] + 10 * min(5, touch/50)
             P_touch = w_touch + 10.0 * min(5.0, touch/50.0)

    # 2) P_rv_2ch
    P_rv_2ch = 0.0
    w_rv = weights.get("P_rv_2ch", 80.0)
    if view_type=="2ch" and w_rv > 0:
        P_rv_2ch = w_rv * min(1.0, rv_ratio / float(cfg["view_inference"]["rv_ratio_2ch_threshold"]))

    # 3) P_components
    P_components = _components_penalty(
        mask, 
        view_type, 
        rv_optional_thr=float(cfg["view_inference"]["rv_ratio_rv_optional_threshold"]),
        weight=weights.get("P_components", 18.0)
    )

    # 4) P_islands
    small_thr = int(cfg["ops"]["small_component_floor"])
    P_islands = _islands_penalty(
        mask, 
        small_thr=small_thr, 
        weight=weights.get("P_islands", 20.0),
        cap=weights.get("P_islands_cap", 30.0)
    )

    # 5) P_valve_leak
    P_valve_leak = _valve_leak_penalty(
        mask, 
        weight=weights.get("P_valve_leak", 3.0),
        threshold=weights.get("P_valve_leak_threshold", 0.05)
    )

    # 6) P_double_cavity
    P_double_cavity = _double_cavity_penalty(
        mask, 
        weight=weights.get("P_double_cavity", 30.0)
    )

    # 7) P_holes
    P_holes = _holes_penalty(
        mask, 
        weight=weights.get("P_holes", 10.0)
    )

    # 8) P_size_drift
    P_size_drift, atlas_direction_score = _size_drift_penalty(
        mask, 
        pred0, 
        atlas_entry,
        weight=weights.get("P_size_drift", 25.0),
        threshold=weights.get("P_size_drift_threshold", 0.35)
    )
    
    # 9) P_edge_misalignment
    # Use Sobel edges from image (Data Fidelity)
    # Distance transform of edge map
    # We penalized pixels in Mask boundary that are far from Image edges
    
    # helper:
    def _calc_edge_pen(mask, img_edges):
        if mask.sum() == 0: return 0.0
        B = boundary_4n(mask)
        if B.sum() == 0: return 0.0
        
        # Distance from nearest edge
        # We pre-calculated edge_map (binary)?
        # rqs.py doesn't receive edge_map directly in compute_rqs signature currently?
        # Wait, compute_rqs signature has `img`. We can compute edges on fly or pass them.
        # Computing on fly is slow for inner loop.
        # But let's check optimization loop... it computes E once per outer loop? 
        # Actually ops.py generate_candidates uses E. 
        # compute_rqs needs E.
        # For now, compute on fly to verify logic, optimize later or pass in.
        
        # Optimization: To avoid recomputing sobel every time, we typically pass it.
        # BUT current signature is `compute_rqs(..., img, ...)`
        # Let's compute it. It's 128x128, fast enough.
        
        # Actually, let's look at `sobel_edges` helper at top of file.
        pass

    # Implementation:
    # We need to compute edges if not passed.
    # To be clean, we'll assume we compute it here.
    edges = sobel_edges(img, percentile=float(cfg["edge"]["sobel_percentile"]))
    
    # Distance transform of inverted edges (0 on edge, growing away)
    # Using scipy.ndimage.distance_transform_edt
    # Invert edges: True where there is NO edge.
    dt = ndi.distance_transform_edt(~edges)
    
    # Mask boundary
    B = boundary_4n(mask)
    if B.sum() > 0:
        dists = dt[B]
        # Average distance of boundary pixels to nearest image edge
        # We want to minimize this.
        # Penalty: 0 if avg_dist < 1.0 px
        # Linear scale up
        avg_dist = dists.mean()
        # threshold: 1.5 px tolerance
        P_edge_misalignment = weights.get("P_edge_misalignment", 20.0) * min(1.0, max(0.0, avg_dist - 1.5) / 3.0)
    else:
        P_edge_misalignment = 0.0

    
    # 10) R_enclosure
    R_enclosure = _enclosure_reward(
        mask, 
        weight=weights.get("R_enclosure", 5.0)
    )
    
    # 11) Shape Prior & Atlas Mismatch Group
    P_shape_prior, P_atlas_mismatch, P_atlas_lv_mismatch, R_shape_match, mean_conf, P_lv_shape = _shape_prior_and_match(
        mask, 
        atlas_entry,
        valid_keys=weights # Pass all weights for extraction
    )
    
    # 12) P_slice_consistency
    P_slice_consistency = _slice_consistency_penalty(
        mask, 
        siblings, 
        view_type,
        weight=weights.get("P_slice_consistency", 15.0)
    )

    # 14) P_boundary_complexity
    P_boundary_complexity = _boundary_complexity_penalty(
        mask, 
        weight=weights.get("P_boundary_complexity", 20.0)
    )

    # 15) P_myo_thickness_var
    P_myo_thickness_var = _myo_thickness_var_penalty(
        mask, 
        weight=weights.get("P_myo_thickness_var", 10.0)
    )

    # Total Summation - Single Weighting
    # Terms already contain the weight. Subtracted directly.
    total = 100.0 \
            - P_touch \
            - P_rv_2ch \
            - P_components \
            - P_islands \
            - P_valve_leak \
            - P_double_cavity \
            - P_holes \
            - P_size_drift \
            - P_shape_prior \
            - P_atlas_mismatch \
            - P_atlas_lv_mismatch \
            - P_lv_shape \
            - P_slice_consistency \
            - P_boundary_complexity \
            - P_myo_thickness_var \
            - P_edge_misalignment \
            + R_enclosure \
            + R_shape_match
            
    # Skipped: P_boundary_roughness, P_intensity_uniformity

    terms = {
        "P_touch": float(P_touch),
        "P_rv_2ch": float(P_rv_2ch),
        "P_components": float(P_components),
        "P_islands": float(P_islands),
        "P_valve_leak": float(P_valve_leak),
        "P_double_cavity": float(P_double_cavity),
        "P_holes": float(P_holes),
        "P_size_drift": float(P_size_drift),
        "P_shape_prior": float(P_shape_prior),
        "P_atlas_mismatch": float(P_atlas_mismatch),
        "P_atlas_lv_mismatch": float(P_atlas_lv_mismatch),
        "P_lv_shape": float(P_lv_shape),
        "P_slice_consistency": float(P_slice_consistency),
        "P_boundary_complexity": float(P_boundary_complexity),
        "P_myo_thickness_var": float(P_myo_thickness_var),
        "P_edge_misalignment": float(P_edge_misalignment),
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
    }

    q_cfg = cfg.get("quality_model", {})
    if q_cfg.get("enabled", False):
        scorer = _get_quality_scorer(cfg)
        if scorer is not None:
            quality_pred, quality_feats = scorer.score(mask, img, view_type)
            aux["quality_score"] = float(quality_pred)
            aux["quality_is_bad"] = bool(quality_pred < float(q_cfg.get("bad_threshold", 0.8)))
            if q_cfg.get("log_features", False):
                for name in QUALITY_FEATURE_NAMES:
                    aux[f"feat_{name}"] = float(quality_feats.get(name, 0.0))
            q_scaled = quality_pred * 100.0
            blend_mode = q_cfg.get("blend_mode", "none")
            if blend_mode == "replace":
                total = q_scaled
            elif blend_mode == "hybrid":
                alpha = float(q_cfg.get("blend_alpha", 0.5))
                total = alpha * total + (1.0 - alpha) * q_scaled

    # Phase 3: Surrogate Prediction
    if weights.get("use_surrogate", False):
        path = weights.get("surrogate_path", "surrogate_model.pkl")
        _load_surrogate(path)
        if _SURROGATE_MODEL is not None:
            # Must match _FEATURE_COLS order exactly
            # ['P_touch', 'P_rv_2ch', 'P_components', 'P_islands', 'P_valve_leak', 
            #  'P_double_cavity', 'P_holes', 'P_size_drift', 'P_shape_prior', 
            #  'P_atlas_mismatch', 'P_atlas_lv_mismatch', 'P_lv_shape', 
            #  'P_slice_consistency', 'P_boundary_complexity', 'P_myo_thickness_var', 
            #  'R_enclosure', 'R_shape_match', 'atlas_match_score', 'mean_conf']
            
            # Ensure variables are defined (P_slice_consistency might be missing if section skipped?)
            # Actually, compute_rqs flow runs all sections sequentially.
            
            feats = [
                P_touch, P_rv_2ch, P_components, P_islands, P_valve_leak,
                P_double_cavity, P_holes, P_size_drift, P_shape_prior,
                P_atlas_mismatch, P_atlas_lv_mismatch, P_lv_shape,
                locals().get('P_slice_consistency', 0.0), # Safer access
                locals().get('P_boundary_complexity', 0.0),
                locals().get('P_myo_thickness_var', 0.0),
                R_enclosure, R_shape_match, atlas_match_score, mean_conf
            ]
            
            # Predict Dice (0.0 - 1.0)
            pred_dice = _SURROGATE_MODEL.predict([feats])[0]
            
            # Convert to RQS scale (0-100)
            # RQS typically targets > 80. Dice 0.9 -> 90.
            total = pred_dice * 100.0

    return RQSResult(total=clamp(total), terms=terms, aux=aux)
