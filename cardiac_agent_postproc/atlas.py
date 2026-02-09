from __future__ import annotations
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, Any, List
import numpy as np
from sklearn.cluster import KMeans
from scipy import ndimage as ndi

from .geom import compactness, eccentricity_largest_cc, centroid, bbox

AtlasKey = Tuple[str, int]

def _compute_features(mask: np.ndarray, z_norm: float = 0.5) -> np.ndarray:
    nonbg = np.count_nonzero(mask)
    if nonbg == 0:
        return np.zeros(6, dtype=np.float32)
    a1 = np.count_nonzero(mask==1)/nonbg
    a2 = np.count_nonzero(mask==2)/nonbg
    a3 = np.count_nonzero(mask==3)/nonbg
    c3 = compactness(mask==3)
    e3 = eccentricity_largest_cc(mask==3)
    # z_norm is critical for slice consistency
    return np.array([a1,a2,a3,c3,e3, z_norm], dtype=np.float32)

def _foreground_union(mask: np.ndarray) -> np.ndarray:
    return (mask>0)

def _center_of_mass(binary: np.ndarray) -> tuple[float,float]:
    ys, xs = np.nonzero(binary)
    if len(xs)==0:
        return (0.0,0.0)
    return (float(xs.mean()), float(ys.mean()))

def _scale_to_diag(binary: np.ndarray, target_diag: float) -> float:
    x0,y0,x1,y1 = bbox(binary)
    w = max(1, x1-x0+1)
    h = max(1, y1-y0+1)
    diag = (w*w + h*h) ** 0.5
    if diag <= 1e-6:
        return 1.0
    return float(target_diag/diag)

def _warp_mask(mask: np.ndarray, tx: float, ty: float, scale: float) -> np.ndarray:
    # affine: scale then translate
    h,w = mask.shape
    M = np.array([[scale, 0, tx],
                  [0, scale, ty]], dtype=np.float32)
    warped = ndi.affine_transform(mask, 
                                 matrix=np.array([[1/scale,0],[0,1/scale]], dtype=np.float32),
                                 offset=np.array([-ty/scale, -tx/scale], dtype=np.float32),
                                 order=0, mode="constant", cval=0)
    # NOTE: ndi affine uses inverse mapping; we used offset above. Keep consistent with inverse mapping below.
    return warped.astype(np.int32)

def align_mask(mask: np.ndarray, target_diag: float=128.0) -> tuple[np.ndarray, dict]:
    h,w = mask.shape
    U = _foreground_union(mask)
    cx,cy = _center_of_mass(U)
    # translate so CoM at center
    tx0 = (w/2.0 - cx)
    ty0 = (h/2.0 - cy)
    # scale so bbox diag matches target
    scale = _scale_to_diag(U, target_diag)
    # because ndi.affine_transform handles scale about origin, we approximate:
    # We first scale around center by composing translations. Here we do a practical approach:
    # Apply translation in original coords then scaling around image center.
    # Implement via cv2 for clarity.
    import cv2
    M = np.array([[scale, 0, (1-scale)*(w/2.0) + tx0],
                  [0, scale, (1-scale)*(h/2.0) + ty0]], dtype=np.float32)
    aligned = cv2.warpAffine(mask.astype(np.uint8), M, (w,h), flags=cv2.INTER_NEAREST, borderValue=0).astype(np.int32)
    params = {"M": M, "scale": scale, "tx0": tx0, "ty0": ty0, "target_diag": float(target_diag)}
    return aligned, params

def apply_inverse_affine_to_prob(prob: np.ndarray, M: np.ndarray) -> np.ndarray:
    # prob in atlas space -> image space; use inverse warp
    import cv2
    h,w = prob.shape
    Minv = cv2.invertAffineTransform(M)
    out = cv2.warpAffine(prob.astype(np.float32), Minv, (w,h), flags=cv2.INTER_LINEAR, borderValue=0.0)
    return out

def build_atlas(gt_masks: List[np.ndarray], view_types: List[str], z_values: List[float], cfg: dict) -> Dict[AtlasKey, dict]:
    target_diag = float(cfg["shape_atlas"]["target_diag"])
    kmeans_max = int(cfg["shape_atlas"]["kmeans_max"])
    kmeans_min = int(cfg["shape_atlas"]["kmeans_min"])
    per_view_div = int(cfg["shape_atlas"]["per_view_divisor"])
    min_samples_for_cluster = int(cfg["shape_atlas"]["min_samples_for_clustering"])

    atlas: Dict[AtlasKey, dict] = {}
    for view in sorted(set(view_types)):
        idx = [i for i,v in enumerate(view_types) if v==view]
        masks = [gt_masks[i] for i in idx]
        zs = [z_values[i] for i in idx]
        N = len(masks)
        if N == 0:
            continue
        feats = np.stack([_compute_features(m, z) for m,z in zip(masks, zs)], axis=0)
        if N < min_samples_for_cluster:
            labels = np.zeros(N, dtype=int)
            K = 1
        else:
            K = min(kmeans_max, max(kmeans_min, N // per_view_div))
            if K < 1:
                K = 1
            if K == 1:
                labels = np.zeros(N, dtype=int)
            else:
                km = KMeans(n_clusters=K, random_state=0, n_init="auto")
                labels = km.fit_predict(feats)
        for k in range(int(labels.max())+1):
            cluster_masks = [masks[i] for i in range(N) if labels[i]==k]
            if not cluster_masks:
                continue
            aligned_masks = []
            align_params_any = None
            for m in cluster_masks:
                a, params = align_mask(m, target_diag=target_diag)
                aligned_masks.append(a)
                align_params_any = params
            H,W = aligned_masks[0].shape
            prob_maps = {}
            for c in [1,2,3]:
                stack = np.stack([(am==c).astype(np.float32) for am in aligned_masks], axis=0)
                prob_maps[c] = stack.mean(axis=0)
            # stats in original scale
            area_stats = {}
            centroid_stats = {}
            for c in [1,2,3]:
                areas = np.array([np.count_nonzero(m==c) for m in cluster_masks], dtype=np.float32)
                mu = float(areas.mean())
                sd = float(areas.std(ddof=0) + 1e-6)
                area_stats[c] = (mu, sd)
                # centroid relative to foreground CoM
                rel = []
                for m in cluster_masks:
                    U = _foreground_union(m)
                    cx,cy = _center_of_mass(U)
                    ccx, ccy = centroid(m==c)
                    rel.append([ccx-cx, ccy-cy])
                rel = np.array(rel, dtype=np.float32)
                centroid_stats[c] = (rel.mean(axis=0).tolist(), (rel.std(axis=0)+1e-6).tolist())
            atlas[(view, int(k))] = {
                "prob_maps": prob_maps,
                "area_stats": area_stats,
                "centroid_stats": centroid_stats,
                "align_params": {"target_diag": target_diag},
                "n_samples": len(cluster_masks),
            }
    return atlas

def save_atlas(path: str, atlas: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(atlas, f)

def load_atlas(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

def match_atlas(mask: np.ndarray, view_type: str, atlas: dict, z_norm: float|None = None) -> tuple[dict|None, int|None, float, dict|None]:
    # returns best_entry, cluster_id, score, align_params
    keys = [k for k in atlas.keys() if k[0]==view_type]
    if not keys:
        return None, None, 0.0, None
    best_score = -999.0
    best_key = None
    aligned, params = align_mask(mask, target_diag=float(atlas[keys[0]]["align_params"]["target_diag"]))
    
    for (v,cid) in keys:
        entry = atlas[(v,cid)]
        
        # 1. Image Overlap Score
        score_cs = []
        for c in [1,2,3]:
            Mc = (aligned==c).astype(np.float32)
            denom = Mc.sum() + 1e-8
            score = float((entry["prob_maps"][c] * Mc).sum() / denom)
            score_cs.append(score)
        overlap_score = float(np.mean(score_cs))
        
        # 2. Slice Consistency Penalty (User Req)
        # If we know the slice position, we should prefer the cluster that represents that position.
        z_penalty = 0.0
        if z_norm is not None and "feat_centroid" in entry:
            # feat_centroid has 6 dims [a1,a2,a3,c3,e3, z]
            # check if 6 dims
            feats = entry["feat_centroid"]
            if len(feats) >= 6:
                cluster_z = feats[5]
                # penalize distance. weight? 
                # Diff 0.5 (half heart) should range from 0 to 1. 0.5 is huge.
                # Let's say weight 2.0 -> if diff is 0.5, penalty 1.0 (kills overlap score).
                z_penalty = 2.0 * abs(z_norm - cluster_z)
        
        final_score = overlap_score - z_penalty

        if final_score > best_score:
            best_score = final_score
            best_key = (v,cid)

    if best_key is None:
        return None, None, 0.0, params
    # Return overlap score as the reported match score, but selection used the penalty
    # Wait, RQS uses this score to reward shape match. If we return the penalized score, 
    # a correct shape at wrong Z gets low score. That's good.
    # But RQS uses it for `R_shape_match`.
    # Let's return the *penalized* score so strictly consistent matches are rewarded.
    return atlas[best_key], int(best_key[1]), float(best_score), params
