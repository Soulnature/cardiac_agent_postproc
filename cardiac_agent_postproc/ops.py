from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
from scipy import ndimage as ndi

from .geom import keep_largest_cc, remove_small_cc, fill_holes, boundary_4n, label_cc
from .atlas import apply_inverse_affine_to_prob, match_atlas, align_mask

def _merge(M1: np.ndarray, M2: np.ndarray, M3: np.ndarray) -> np.ndarray:
    # priority: M2 > M3 > M1
    out = np.zeros_like(M1, dtype=np.int32)
    out[M1.astype(bool)] = 1
    out[M3.astype(bool)] = 3
    out[M2.astype(bool)] = 2
    return out

def _small_thr(mask: np.ndarray, cfg: dict) -> int:
    nonbg = max(1, int(np.count_nonzero(mask)))
    return max(int(cfg["ops"]["small_component_floor"]), int(cfg["ops"]["small_component_frac"] * nonbg))

def topology_cleanup(mask: np.ndarray, view_type: str, cfg: dict) -> np.ndarray:
    thr = _small_thr(mask, cfg)
    M1 = (mask==1)
    M2 = (mask==2)
    M3 = (mask==3)
    if view_type=="2ch":
        M1 = np.zeros_like(M1, dtype=bool)
    else:
        M1 = remove_small_cc(M1, thr)
        M1 = keep_largest_cc(M1) if M1.sum()>0 else M1
    M2 = remove_small_cc(M2, thr)
    M3 = remove_small_cc(M3, thr)
    if M2.sum()>0: M2 = keep_largest_cc(M2)
    if M3.sum()>0: M3 = keep_largest_cc(M3)
    return _merge(M1,M2,M3)

def fill_holes_ops(mask: np.ndarray, cfg: dict) -> np.ndarray:
    M1 = (mask==1)
    M2 = (mask==2)
    M3 = (mask==3)
    M3f = fill_holes(M3)
    M2f = M2
    # fill holes in M2 only if it doesn't reduce enclosure proxy too much; leave decision to RQS selection.
    # Here we provide a candidate with M2 filled as well.
    return _merge(M1, M2f, M3f)

def fill_holes_ops_m2(mask: np.ndarray, cfg: dict) -> np.ndarray:
    M1 = (mask==1)
    M2 = (mask==2)
    M3 = (mask==3)
    return _merge(M1, fill_holes(M2), fill_holes(M3))

def morph_close_open(mask: np.ndarray, cfg: dict, close2_open3: bool=True) -> np.ndarray:
    k = int(cfg["ops"]["morph_kernel3"])
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    M1 = (mask==1).astype(np.uint8)
    M2 = (mask==2).astype(np.uint8)
    M3 = (mask==3).astype(np.uint8)
    if close2_open3:
        M2n = cv2.morphologyEx(M2, cv2.MORPH_CLOSE, ker)
        M3n = cv2.morphologyEx(M3, cv2.MORPH_OPEN, ker)
    else:
        M2n = cv2.morphologyEx(M2, cv2.MORPH_OPEN, ker)
        M3n = cv2.morphologyEx(M3, cv2.MORPH_CLOSE, ker)
    return _merge(M1>0, M2n>0, M3n>0)

def dilate_or_erode(mask: np.ndarray, cfg: dict, cls: int, op: str) -> np.ndarray:
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    Mc = (mask==cls).astype(np.uint8)
    if op=="dilate":
        Mcn = cv2.dilate(Mc, ker, iterations=1)>0
    else:
        Mcn = cv2.erode(Mc, ker, iterations=1)>0
    M1 = (mask==1)
    M2 = (mask==2)
    M3 = (mask==3)
    if cls==1: M1 = Mcn
    if cls==2: M2 = Mcn
    if cls==3: M3 = Mcn
    return _merge(M1,M2,M3)

def valve_suppress(mask: np.ndarray, valve_y: int|None, cfg: dict) -> np.ndarray:
    if valve_y is None:
        return mask.copy()
    out = mask.copy()
    out[:valve_y,:] = 0
    return out

def rv_lv_barrier(mask: np.ndarray, cfg: dict) -> np.ndarray:
    # implements spec F
    M1 = (mask==1)
    M2 = (mask==2)
    M3 = (mask==3)
    if M1.sum()==0 or M3.sum()==0:
        return mask.copy()
    touch_zone = ndi.binary_dilation(M1, iterations=1) & M3
    if touch_zone.sum()==0:
        return mask.copy()
    new_M2 = M2 | ndi.binary_dilation(touch_zone, iterations=2)
    new_M1 = M1 & (~new_M2)
    new_M3 = M3 & (~new_M2)
    return _merge(new_M1, new_M2, new_M3)

def edge_snapping_proxy(mask: np.ndarray, edge_map: np.ndarray, cfg: dict) -> np.ndarray:
    # simpler proxy: erode then dilate myocardium constrained by edges proximity
    # (fast and robust; exact pixel-shift snapping is complex)
    M1 = (mask==1)
    M2 = (mask==2).astype(np.uint8)
    M3 = (mask==3)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    M2e = cv2.erode(M2, ker, iterations=1)
    M2d = cv2.dilate(M2e, ker, iterations=1)
    # constrain by edges: keep only pixels within 2px to edge_map OR originally in M2
    dist = ndi.distance_transform_edt(~edge_map.astype(bool))
    keep = (dist <= 2.0) | (M2.astype(bool))
    M2n = (M2d>0) & keep
    return _merge(M1, M2n, M3)

def atlas_guided_grow_prune(mask: np.ndarray, view_type: str, atlas: dict, cfg: dict) -> List[np.ndarray]:
    # Spec H (1) & (2) (simplified but faithful)
    entry, cid, score, params = match_atlas(mask, view_type, atlas)
    if entry is None:
        return []
    aligned, ap = align_mask(mask, target_diag=float(entry["align_params"]["target_diag"]))
    M = ap["M"]
    cand = []
    # bring prob maps to image space
    Pimg = {c: apply_inverse_affine_to_prob(entry["prob_maps"][c], M) for c in [1,2,3]}
    # background prob
    Pbg = 1.0 - (Pimg[1] + Pimg[2] + Pimg[3])
    # (1) confident grow
    out = mask.copy()
    for c in [2,3,1]:  # priority M2 > M3 > M1
        Rc = (Pimg[c] > 0.7)
        yy, xx = np.nonzero(Rc)
        for y,x in zip(yy,xx):
            cur = int(out[y,x])
            if cur == c:
                continue
            curp = Pbg[y,x] if cur==0 else Pimg[cur][y,x]
            if Pimg[c][y,x] > curp:
                out[y,x] = c
    cand.append(out)

    # (2) prune impossible
    out2 = mask.copy()
    for c in [1,2,3]:
        Ic = (Pimg[c] < 0.05) & (out2==c)
        yy,xx = np.nonzero(Ic)
        for y,x in zip(yy,xx):
            # choose best among bg and 1/2/3
            probs = [Pbg[y,x], Pimg[1][y,x], Pimg[2][y,x], Pimg[3][y,x]]
            out2[y,x] = int(np.argmax(probs))
    cand.append(out2)

    return cand

def atlas_constrained_morph(mask: np.ndarray, view_type: str, atlas: dict, cfg: dict) -> List[np.ndarray]:
    entry, cid, score, params = match_atlas(mask, view_type, atlas)
    if entry is None:
        return []
    aligned, ap = align_mask(mask, target_diag=float(entry["align_params"]["target_diag"]))
    M = ap["M"]
    Pimg = {c: apply_inverse_affine_to_prob(entry["prob_maps"][c], M) for c in [1,2,3]}
    Pbg = 1.0 - (Pimg[1] + Pimg[2] + Pimg[3])

    out_list = []
    for c in [2,3]:
        Mc = (mask==c)
        expand_zone = ndi.binary_dilation(Mc, iterations=2) & (~Mc) & (Pimg[c] > 0.3)
        M1 = (mask==1); M2=(mask==2); M3=(mask==3)
        if c==2:
            M2n = M2 | expand_zone
            out_list.append(_merge(M1,M2n,M3))
        else:
            M3n = M3 | expand_zone
            out_list.append(_merge(M1,M2,M3n))

        shrink_zone = Mc & (Pimg[c] < 0.1)
        out = mask.copy()
        yy,xx = np.nonzero(shrink_zone)
        for y,x in zip(yy,xx):
            probs = [Pbg[y,x], Pimg[1][y,x], Pimg[2][y,x], Pimg[3][y,x]]
            out[y,x] = int(np.argmax(probs))
        out_list.append(out)
    return out_list

def boundary_smoothing(mask: np.ndarray, cfg: dict, method: str="contour") -> np.ndarray:
    M1 = (mask==1).astype(np.uint8)
    M2 = (mask==2).astype(np.uint8)
    M3 = (mask==3).astype(np.uint8)

    if method=="contour":
        eps = float(cfg["ops"]["smooth"]["contour_epsilon"])
        def smooth_bin(B):
            cnts, _ = cv2.findContours(B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            out = np.zeros_like(B)
            for c in cnts:
                if len(c) < 20:
                    cv2.drawContours(out, [c], -1, 255, thickness=-1)
                    continue
                approx = cv2.approxPolyDP(c, epsilon=eps, closed=True)
                cv2.drawContours(out, [approx], -1, 255, thickness=-1)
            return out
        M2s = smooth_bin(M2)
        M3s = smooth_bin(M3)
        return _merge(M1>0, M2s>0, M3s>0)

    if method=="gaussian":
        sigma = float(cfg["ops"]["smooth"]["gaussian_sigma"])
        M2f = ndi.gaussian_filter(M2.astype(np.float32), sigma=sigma) > 0.5
        M3f = ndi.gaussian_filter(M3.astype(np.float32), sigma=sigma) > 0.5
        return _merge(M1>0, M2f, M3f)

    if method=="morph":
        r = int(cfg["ops"]["smooth"]["morph_r"])
        k = 2*r+1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        M2n = cv2.morphologyEx(M2, cv2.MORPH_CLOSE, ker)
        M2n = cv2.morphologyEx(M2n, cv2.MORPH_OPEN, ker)
        M3n = cv2.morphologyEx(M3, cv2.MORPH_CLOSE, ker)
        M3n = cv2.morphologyEx(M3n, cv2.MORPH_OPEN, ker)
        return _merge(M1>0, M2n>0, M3n>0)

    return mask.copy()

def generate_candidates(mask: np.ndarray,
                        pred0: np.ndarray,
                        img: np.ndarray,
                        view_type: str,
                        edge_map: np.ndarray,
                        valve_y: int|None,
                        atlas: dict|None,
                        cfg: dict) -> List[Tuple[str,np.ndarray]]:
    cands: List[Tuple[str,np.ndarray]] = []
    # candidate 0: current
    cands.append(("current", mask.copy()))

    # A topology cleanup
    cands.append(("topology_cleanup", topology_cleanup(mask, view_type, cfg)))

    # B fill holes
    cands.append(("fill_holes", fill_holes_ops(mask, cfg)))
    cands.append(("fill_holes_m2", fill_holes_ops_m2(mask, cfg)))

    # C morph close/open
    cands.append(("close2_open3", morph_close_open(mask, cfg, close2_open3=True)))
    cands.append(("open2_close3", morph_close_open(mask, cfg, close2_open3=False)))

    # D greedy dilate/erode
    for cls in [2,3]:
        cands.append((f"{cls}_dilate", dilate_or_erode(mask, cfg, cls, "dilate")))
        cands.append((f"{cls}_erode", dilate_or_erode(mask, cfg, cls, "erode")))

    # RV: erode if touching (optimizer decides via reward)
    if view_type in ("3ch","4ch"):
        cands.append(("rv_erode", dilate_or_erode(mask, cfg, 1, "erode")))

    # E valve suppression
    cands.append(("valve_suppress", topology_cleanup(valve_suppress(mask, valve_y, cfg), view_type, cfg)))

    # F barrier
    if view_type in ("3ch","4ch"):
        cands.append(("rv_lv_barrier", rv_lv_barrier(mask, cfg)))

    # G edge snapping proxy
    cands.append(("edge_snap_proxy", topology_cleanup(edge_snapping_proxy(mask, edge_map, cfg), view_type, cfg)))

    # H shape prior guided
    if atlas is not None:
        for i,cm in enumerate(atlas_guided_grow_prune(mask, view_type, atlas, cfg)):
            cands.append((f"atlas_growprune_{i}", topology_cleanup(cm, view_type, cfg)))
        for i,cm in enumerate(atlas_constrained_morph(mask, view_type, atlas, cfg)):
            cands.append((f"atlas_morph_{i}", topology_cleanup(cm, view_type, cfg)))

    # I boundary smoothing
    cands.append(("smooth_contour", topology_cleanup(boundary_smoothing(mask, cfg, "contour"), view_type, cfg)))
    cands.append(("smooth_gaussian", topology_cleanup(boundary_smoothing(mask, cfg, "gaussian"), view_type, cfg)))
    cands.append(("smooth_morph", topology_cleanup(boundary_smoothing(mask, cfg, "morph"), view_type, cfg)))

    return cands

def is_valid_candidate(mask: np.ndarray) -> bool:
    if mask.min() < 0 or mask.max() > 3:
        return False
    if np.count_nonzero(mask==2) == 0:
        return False
    if np.count_nonzero(mask==3) == 0:
        return False
    return True
