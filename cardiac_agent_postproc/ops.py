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

def safe_topology_fix(mask: np.ndarray, view_type: str, cfg: dict) -> np.ndarray:
    """
    Gen 2 Mutation: Class-Aware Topology Repair.
    Ensures no class is lost. Fills holes and removes small islands.
    """
    out = np.zeros_like(mask)
    labels = [1, 2, 3] # RV, Myo, LV
    
    for c in labels:
        Mc = (mask == c)
        if np.count_nonzero(Mc) == 0:
            continue
            
        # 1. Fill Holes
        Mc_filled = ndi.binary_fill_holes(Mc)
        
        # 2. Open (remove small connections/noise)
        # Mc_opened = ndi.binary_opening(Mc_filled, structure=np.ones((3,3)))
        
        # 3. Largest Component (Keep only the main body)
        # Identify components
        labeled, num_features = ndi.label(Mc_filled)
        if num_features > 1:
            sizes = ndi.sum(Mc_filled, labeled, range(num_features + 1))
            largest_label = np.argmax(sizes[1:]) + 1
            Mc_final = (labeled == largest_label)
        else:
            Mc_final = Mc_filled
            
        # SURVIVAL CHECK
        if np.count_nonzero(Mc_final) == 0:
             # Lethal mutation avoided: Revert to input
             Mc_final = Mc
             
        out[Mc_final] = c
        
    return out

def remove_islands(mask: np.ndarray) -> np.ndarray:
    """Standalone remove small islands for all classes."""
    out = np.zeros_like(mask)
    labels = [1, 2, 3] 
    for c in labels:
        Mc = (mask == c)
        if np.count_nonzero(Mc) == 0: continue
        
        # Keep largest component only
        labeled, num = ndi.label(Mc)
        if num > 1:
            sizes = ndi.sum(Mc, labeled, range(num + 1))
            largest_idx = np.argmax(sizes[1:]) + 1
            Mc_final = (labeled == largest_idx)
        else:
            Mc_final = Mc
        out[Mc_final] = c
    return out

def morph_close_large(mask: np.ndarray, cfg: dict) -> np.ndarray:
    """Stronger closing for larger holes/gaps using 9x9 kernel."""
    k = 9
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    M1 = (mask==1).astype(np.uint8)
    M2 = (mask==2).astype(np.uint8)
    M3 = (mask==3).astype(np.uint8)
    
    # Close all classes
    M1n = cv2.morphologyEx(M1, cv2.MORPH_CLOSE, ker)
    M2n = cv2.morphologyEx(M2, cv2.MORPH_CLOSE, ker)
    M3n = cv2.morphologyEx(M3, cv2.MORPH_CLOSE, ker)
    
    return _merge(M1n>0, M2n>0, M3n>0)

def morphological_gap_close(mask: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Specifically targets Myocardium (2) gaps by aggressive closing (15px kernel).
    Prioritizes Myo over LV/RV to fix intrusions.
    """
    k = 15 # Very large kernel to bridge significant gaps
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    M2 = (mask==2).astype(np.uint8)
    
    # Closing = Dilate -> Erode. Bridges gaps < kernel size.
    M2_closed = cv2.morphologyEx(M2, cv2.MORPH_CLOSE, ker)
    
    M1 = (mask==1)
    M3 = (mask==3)
    
    # _merge prioritizes M2, so this will overwrite LV/RV at the gap
    return _merge(M1, M2_closed>0, M3)

def close_holes(mask: np.ndarray) -> np.ndarray:
    """Standalone binary fill holes for all classes."""
    out = np.zeros_like(mask)
    labels = [1, 2, 3] 
    for c in labels:
        Mc = (mask == c)
        if np.count_nonzero(Mc) == 0: continue
        Mc_filled = ndi.binary_fill_holes(Mc)
        out[Mc_filled] = c
    return out

def smooth_morph(mask: np.ndarray, kernel: int=3) -> np.ndarray:
    """Standalone morphological smoothing."""
    # Already exists as 'boundary_smoothing(..., method="morph")' but creating simple alias
    labels = [1, 2, 3]
    out = np.zeros_like(mask)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel,kernel))
    for c in labels:
        Mc = (mask == c).astype(np.uint8)
        if np.count_nonzero(Mc) == 0: continue
        # Open-Close to smooth
        closed = cv2.morphologyEx(Mc, cv2.MORPH_CLOSE, ker)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, ker)
        out[opened > 0] = c
    return out

def _topology_cleanup_v5_legacy(mask: np.ndarray, view_type: str, cfg: dict) -> np.ndarray:
    """
    Revised V5: Safe Topology Cleanup with Survival Check.
    Wraps all operations to ensure no class is completely lost.
    """
    out = np.zeros_like(mask)
    labels = [1, 2, 3] # RV, Myo, LV
    if view_type == "2ch":
        labels = [2, 3] # No RV in 2ch

    for c in labels:
        Mc = (mask == c)
        if np.count_nonzero(Mc) == 0:
            continue
            
        # 1. Fill Holes
        Mc_filled = ndi.binary_fill_holes(Mc)
        
        # 2. Open (remove small connections/noise) - Optional, skip for safety
        
        # 3. Largest Component
        labeled, num_features = ndi.label(Mc_filled)
        if num_features > 1:
            thr = _small_thr(mask, cfg)
            # Filter small
            sizes = ndi.sum(Mc_filled, labeled, range(num_features + 1))
            # If largest is too small? No, keep largest anyway ensuring existence.
            largest_label = np.argmax(sizes[1:]) + 1
            Mc_final = (labeled == largest_label)
        else:
            Mc_final = Mc_filled
            
        # SURVIVAL CHECK
        if np.count_nonzero(Mc_final) == 0:
             # Lethal Op detected: Revert to input for this class
             Mc_final = Mc
             
        out[Mc_final] = c
        
    return out

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
    # Kernel size logic: 
    # "dilate" / "erode" -> 3x3 (Standard)
    # "dilate_large" / "erode_large" -> 5x5
    k = 3
    if "large" in op:
        k = 5
    
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    Mc = (mask==cls).astype(np.uint8)
    
    # Get iterations from config (default 1)
    iters = int(cfg["ops"].get("erosion_iterations", 1))

    if "dilate" in op:
        iters = int(cfg["ops"].get("dilation_iterations", 1)) # Separated for flexibility
        Mcn = cv2.dilate(Mc, ker, iterations=iters)>0
    else:
        # Safety Check: Prevent suicide erosion
        pre_count = np.count_nonzero(Mc)
        Mcn = cv2.erode(Mc, ker, iterations=iters)>0
        post_count = np.count_nonzero(Mcn)
        
        loss_ratio = (pre_count - post_count) / max(1, pre_count)
        max_loss = float(cfg["ops"].get("max_erosion_loss", 0.3))
        
        if loss_ratio > max_loss:
            # Too aggressive! Try reducing iterations or kernel
            if iters > 1:
                # Retry with 1 iteration
                Mcn = cv2.erode(Mc, ker, iterations=1)>0
                post_count = np.count_nonzero(Mcn)
                loss_ratio = (pre_count - post_count) / max(1, pre_count)
            
            # If still too much loss, REVERT (no-op)
            if loss_ratio > max_loss:
                # print(f"DEBUG: Erosion halted for class {cls}. Loss {loss_ratio:.2f} > {max_loss}")
                return mask.copy() # Safe exit
        
    M1 = (mask==1)
    M2 = (mask==2)
    M3 = (mask==3)
    if cls==1: M1 = Mcn
    if cls==2: M2 = Mcn
    if cls==3: M3 = Mcn
    return _merge(M1,M2,M3)

def erode_lv_expand_myo(mask: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Erodes LV (class 3) and expands Myocardium (class 2) into the vacated space.
    Ensures continuity so that LV shrinkage doesn't leave gaps.
    """
    # 1. Identify current LV
    M3 = (mask == 3).astype(np.uint8)
    if np.count_nonzero(M3) == 0:
        return mask.copy()

    # 2. Erode LV (standard 3x3 kernel)
    k = 3
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # Coupled erosion needs to be stronger to be effective. Default to 2 if not specified.
    # If global erosion_iterations is set to something higher, use that.
    global_iters = int(cfg["ops"].get("erosion_iterations", 1))
    iters = max(2, global_iters)
    
    # Safety check (same as dilate_or_erode)
    pre_count = np.count_nonzero(M3)
    M3n = cv2.erode(M3, ker, iterations=iters) > 0
    post_count = np.count_nonzero(M3n)
    
    loss_ratio = (pre_count - post_count) / max(1, pre_count)
    max_loss = float(cfg["ops"].get("max_erosion_loss", 0.3))
    
    if loss_ratio > max_loss:
        if iters > 1:
            M3n = cv2.erode(M3, ker, iterations=1) > 0 # Retry with 1 iter
            post_count = np.count_nonzero(M3n)
            loss_ratio = (pre_count - post_count) / max(1, pre_count)
        
        if loss_ratio > max_loss:
            return mask.copy() # Safe exit

    # 3. Identify vacated pixels (Old LV - New LV)
    vacated = (M3 > 0) & (~M3n)
    
    # 4. Reassign to Myo (class 2), but respect RV (class 1)
    M1 = (mask == 1)
    M2 = (mask == 2)
    
    # New Myo includes old Myo + Vacated LV (excluding any RV overlap, though LV/RV shouldn't overlap)
    M2n = M2 | vacated
    
    # Reassemble: RV (1), Myo (2), LV (3)
    # _merge prioritizes M2 over M1/M3 usually, but here M3n is strictly smaller.
    # We want: 1 stays 1, 2 gains vacated, 3 shrinks.
    return _merge(M1, M2n, M3n)

def erode_rv_expand_myo_conditional(mask: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Erodes RV (class 1). Vacated pixels become Myocardium (class 2) ONLY if
    they are adjacent to existing Myocardium (Septal Wall).
    Otherwise they become Background (Free Wall).
    """
    # 1. Identify current RV
    M1 = (mask == 1).astype(np.uint8)
    if np.count_nonzero(M1) == 0:
        return mask.copy()

    # 2. Erode RV (standard 3x3 kernel)
    k = 3
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # Coupled erosion needs to be stronger. Default to 2.
    global_iters = int(cfg["ops"].get("erosion_iterations", 1))
    iters = max(2, global_iters)
    
    # Safety Check
    pre_count = np.count_nonzero(M1)
    M1n = cv2.erode(M1, ker, iterations=iters) > 0
    post_count = np.count_nonzero(M1n)
    
    loss_ratio = (pre_count - post_count) / max(1, pre_count)
    max_loss = float(cfg["ops"].get("max_erosion_loss", 0.3))
    
    if loss_ratio > max_loss:
        if iters > 1:
            M1n = cv2.erode(M1, ker, iterations=1) > 0
            post_count = np.count_nonzero(M1n)
            loss_ratio = (pre_count - post_count) / max(1, pre_count)
        
        if loss_ratio > max_loss:
            return mask.copy()

    # 3. Identify vacated pixels
    vacated = (M1 > 0) & (~M1n)
    
    # 4. Conditional Expansion: Only expand Myo if vacated pixel touches existing Myo
    M2 = (mask == 2).astype(np.uint8)
    # Dilate Myo to find adjacency
    M2_neighbor = cv2.dilate(M2, ker, iterations=1) > 0
    
    # Vacated pixels that touch Myo become new Myo (Septal shift)
    new_myo_pixels = vacated & M2_neighbor
    
    # Updates
    M2n = (M2 > 0) | new_myo_pixels
    
    # Reassemble: RV (shrunk), Myo (conditionally expanded), LV (unchanged)
    M3 = (mask == 3)
    return _merge(M1n, M2n, M3)

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
    
    # AGGRESSIVE: Dilate 5 times (approx 5px radius) to find near-misses
    dilated_M1 = ndi.binary_dilation(M1, iterations=5)
    touch_zone = dilated_M1 & M3
    
    if touch_zone.sum()==0:
        # Extreme case: Try dilating M3 as well if M1 dilation alone didn't reach
        dilated_M3 = ndi.binary_dilation(M3, iterations=5)
        touch_zone = dilated_M1 & dilated_M3
    
    if touch_zone.sum()==0:
        return mask.copy()
    
    # Create barrier: Dialte touch zone significantly to form a thick wall
    # Using 3 iterations (~3px radius) -> 6px thick wall
    barrier = ndi.binary_dilation(touch_zone, iterations=3)
    
    # Enforce Myo (2) on barrier
    new_M2 = M2 | barrier
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

def atlas_guided_grow_prune(mask: np.ndarray, view_type: str, atlas: dict, cfg: dict, z_norm: float|None=None) -> List[np.ndarray]:
    # Spec H (1) & (2) (simplified but faithful)
    entry, cid, score, params = match_atlas(mask, view_type, atlas, z_norm=z_norm)
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


def atlas_class_repair(mask: np.ndarray, view_type: str, atlas: dict,
                       cfg: dict, target_class: int,
                       z_norm: float | None = None) -> np.ndarray | None:
    """Selectively repair a SINGLE class using Atlas probability maps.

    Unlike atlas_guided_grow_prune which modifies all classes, this function
    ONLY grows the target_class into regions where the Atlas says it should
    exist but the current mask has background or a conflicting label.
    Other classes are left completely untouched unless they overlap with
    high-confidence target_class regions.

    Args:
        mask:         current segmentation mask
        view_type:    '4ch', '2ch', etc.
        atlas:        shape atlas dict
        cfg:          config dict
        target_class: integer label to repair (1=RV, 2=Myo, 3=LV)
        z_norm:       optional z-normalization for atlas matching

    Returns:
        repaired mask or None if atlas matching fails
    """
    entry, cid, score, params = match_atlas(mask, view_type, atlas, z_norm=z_norm)
    if entry is None:
        return None

    aligned, ap = align_mask(mask, target_diag=float(entry["align_params"]["target_diag"]))
    M = ap["M"]

    # Bring probability maps to image space
    Pimg = {c: apply_inverse_affine_to_prob(entry["prob_maps"][c], M)
            for c in [1, 2, 3]}

    Ptarget = Pimg[target_class]
    out = mask.copy()

    # --- Phase 1: Grow target_class into background or low-confidence regions ---
    # Where Atlas says target_class should be with high confidence
    grow_region = Ptarget > 0.5
    # Only modify pixels that are currently background (0) or a class with
    # LOWER Atlas probability than target_class
    yy, xx = np.nonzero(grow_region)
    for y, x in zip(yy, xx):
        cur = int(out[y, x])
        if cur == target_class:
            continue  # already correct
        # Current pixel's atlas probability
        if cur == 0:
            cur_prob = 0.0
        else:
            cur_prob = float(Pimg[cur][y, x])
        # Only replace if target has significantly higher probability
        if float(Ptarget[y, x]) > cur_prob + 0.15:
            out[y, x] = target_class

    # --- Phase 2: Prune target_class from impossible regions ---
    # Where current mask has target_class but Atlas says it shouldn't be there
    prune_region = (out == target_class) & (Ptarget < 0.05)
    if np.any(prune_region):
        Pbg = 1.0 - (Pimg[1] + Pimg[2] + Pimg[3])
        yy, xx = np.nonzero(prune_region)
        for y, x in zip(yy, xx):
            probs = [float(Pbg[y, x]), float(Pimg[1][y, x]),
                     float(Pimg[2][y, x]), float(Pimg[3][y, x])]
            out[y, x] = int(np.argmax(probs))

    changed = int(np.sum(out != mask))
    if changed == 0:
        return None
    return out


def neighbor_class_repair(mask: np.ndarray, neighbor_mask: np.ndarray,
                          target_class: int) -> np.ndarray | None:
    """Repair a specific class using a neighbor slice's mask as a shape donor.

    Aggressiveness is automatically determined:
    - If target_class is MISSING (area < 50% of neighbor): AGGRESSIVE mode
      → replace ALL conflicting pixels (bg + any other class) in donor region
    - Otherwise: CONSERVATIVE mode
      → only replace bg + classes that are oversized vs neighbor

    Args:
        mask:          current segmentation mask to repair
        neighbor_mask: mask from an adjacent slice (same patient, same view)
        target_class:  integer label to repair (1=RV, 2=Myo, 3=LV)

    Returns:
        repaired mask or None if no changes made
    """
    if mask.shape != neighbor_mask.shape:
        return None

    donor_region = (neighbor_mask == target_class)
    if not np.any(donor_region):
        return None

    # Detect if target class is "missing" — severely undersized vs neighbor
    cur_target_area = np.count_nonzero(mask == target_class)
    nb_target_area = np.count_nonzero(donor_region)
    missing = cur_target_area < nb_target_area * 0.5  # < 50% → missing

    # Build replaceability per class
    replaceable = np.zeros(4, dtype=bool)
    replaceable[0] = True  # background is always replaceable

    if missing:
        # AGGRESSIVE: target class is missing, replace ALL other classes
        for c in [1, 2, 3]:
            if c != target_class:
                replaceable[c] = True
    else:
        # CONSERVATIVE: only replace classes that are oversized vs neighbor
        for c in [1, 2, 3]:
            if c == target_class:
                continue
            cur_area = np.count_nonzero(mask == c)
            nb_area = np.count_nonzero(neighbor_mask == c)
            if nb_area > 0 and cur_area > nb_area * 1.2:
                replaceable[c] = True

    out = mask.copy()
    yy, xx = np.nonzero(donor_region)
    changed = 0
    for y, x in zip(yy, xx):
        cur = int(out[y, x])
        if cur == target_class:
            continue
        if replaceable[cur]:
            out[y, x] = target_class
            changed += 1

    if changed == 0:
        return None
    return out

def atlas_replace_op(mask: np.ndarray, view_type: str, atlas: dict, cfg: dict, z_norm: float|None=None) -> List[np.ndarray]:
    """
    Aggressively replace the current mask with the best-matching Atlas entry.
    Useful when the current mask is structurally broken beyond repair.
    """
    # Note: match_atlas now includes z_norm penalty if passed via RQS, 
    # but here we call it without z_norm? 
    # Ops doesn't have z_norm. This is a limitation. 
    # However, match_atlas is robust enough with overlap score usually.
    # Ideally we should pass z_norm to generate_candidates too... 
    # For now, let's rely on Image Overlap to find the best shape.
    entry, cid, score, params = match_atlas(mask, view_type, atlas, z_norm=z_norm)
    if entry is None:
        return []
        
    aligned, ap = align_mask(mask, target_diag=float(entry["align_params"]["target_diag"]))
    M = ap["M"]
    
    # We want to warp the ATLAS PROB MAPS back to Image Space and threshold them directly.
    # This creates a "pure" atlas shape in the image frame.
    Pimg = {c: apply_inverse_affine_to_prob(entry["prob_maps"][c], M) for c in [1,2,3]}
    Pbg = 1.0 - (Pimg[1] + Pimg[2] + Pimg[3])
    
    out = np.zeros_like(mask)
    h, w = mask.shape
    
    # Argmax reconstruction
    # To be safe, avoiding explicit loops for speed
    stack = np.stack([Pbg, Pimg[1], Pimg[2], Pimg[3]], axis=0) # 4, H, W
    out = np.argmax(stack, axis=0).astype(np.int32)
    
    return [out]

def atlas_constrained_morph(mask: np.ndarray, view_type: str, atlas: dict, cfg: dict, z_norm: float|None=None) -> List[np.ndarray]:
    entry, cid, score, params = match_atlas(mask, view_type, atlas, z_norm=z_norm)
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

def myo_bridge(mask: np.ndarray) -> np.ndarray:
    """
    Targeted fix for disconnected myocardium: find the gap between the two
    largest myo components and draw a thin (3 px) bridge of class 2 between
    the closest pair of boundary pixels.

    Returns the original mask unchanged if myo has <= 1 component.
    """
    M2 = (mask == 2).astype(np.uint8)
    labeled, num = ndi.label(M2)
    if num <= 1:
        return mask.copy()

    # Sizes of each component (index 0 = background of label map)
    sizes = ndi.sum(M2, labeled, range(num + 1))
    # Two largest component labels (skip index 0)
    order = np.argsort(sizes[1:])[::-1] + 1  # descending, 1-based
    lab_a, lab_b = int(order[0]), int(order[1])

    comp_a = (labeled == lab_a)
    comp_b = (labeled == lab_b)

    # Distance transform from comp_a → find closest pixel in comp_b
    dist_a = ndi.distance_transform_edt(~comp_a)
    # Mask dist_a to only comp_b boundary pixels for efficiency
    border_b = comp_b & (ndi.binary_dilation(comp_b) != comp_b)
    if border_b.sum() == 0:
        border_b = comp_b

    # Find the pixel in comp_b closest to comp_a
    dists_at_b = np.where(border_b, dist_a, np.inf)
    idx_b = np.unravel_index(np.argmin(dists_at_b), dists_at_b.shape)

    # Now find closest pixel in comp_a to idx_b
    dist_b = ndi.distance_transform_edt(~comp_b)
    border_a = comp_a & (ndi.binary_dilation(comp_a) != comp_a)
    if border_a.sum() == 0:
        border_a = comp_a
    dists_at_a = np.where(border_a, dist_b, np.inf)
    idx_a = np.unravel_index(np.argmin(dists_at_a), dists_at_a.shape)

    # Draw bridge line (3 px thick) between idx_a and idx_b
    out = mask.copy()
    bridge = np.zeros_like(M2)
    cv2.line(bridge, (int(idx_a[1]), int(idx_a[0])), (int(idx_b[1]), int(idx_b[0])),
             1, thickness=3)
    out[bridge > 0] = 2
    return out


def lv_bridge(mask: np.ndarray) -> np.ndarray:
    """
    Targeted fix for fragmented LV: bridge the two largest LV components
    with a thin line of class 3.
    """
    M3 = (mask == 3).astype(np.uint8)
    labeled, num = ndi.label(M3)
    if num <= 1:
        return mask.copy()

    sizes = ndi.sum(M3, labeled, range(num + 1))
    order = np.argsort(sizes[1:])[::-1] + 1
    lab_a, lab_b = int(order[0]), int(order[1])

    comp_a = (labeled == lab_a)
    comp_b = (labeled == lab_b)

    dist_a = ndi.distance_transform_edt(~comp_a)
    border_b = comp_b & (ndi.binary_dilation(comp_b) != comp_b)
    if border_b.sum() == 0:
        border_b = comp_b
    dists_at_b = np.where(border_b, dist_a, np.inf)
    idx_b = np.unravel_index(np.argmin(dists_at_b), dists_at_b.shape)

    dist_b = ndi.distance_transform_edt(~comp_b)
    border_a = comp_a & (ndi.binary_dilation(comp_a) != comp_a)
    if border_a.sum() == 0:
        border_a = comp_a
    dists_at_a = np.where(border_a, dist_b, np.inf)
    idx_a = np.unravel_index(np.argmin(dists_at_a), dists_at_a.shape)

    out = mask.copy()
    bridge = np.zeros_like(M3)
    cv2.line(bridge, (int(idx_a[1]), int(idx_a[0])), (int(idx_b[1]), int(idx_b[0])),
             1, thickness=3)
    out[bridge > 0] = 3
    return out


def generate_candidates(mask: np.ndarray,
                        pred0: np.ndarray,
                        img: np.ndarray,
                        view_type: str,
                        edge_map: np.ndarray,
                        valve_y: int|None,
                        atlas: dict|None,
                        cfg: dict,
                        z_norm: float|None = None) -> List[Tuple[str,np.ndarray]]:
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
        cands.append((f"{cls}_dilate_large", dilate_or_erode(mask, cfg, cls, "dilate_large")))
        cands.append((f"{cls}_erode_large", dilate_or_erode(mask, cfg, cls, "erode_large")))

    # RV: erode if touching (optimizer decides via reward)
    if view_type in ("3ch","4ch"):
        cands.append(("rv_erode", dilate_or_erode(mask, cfg, 1, "erode")))
        cands.append(("rv_erode_large", dilate_or_erode(mask, cfg, 1, "erode_large")))

    # E valve suppression
    cands.append(("valve_suppress", topology_cleanup(valve_suppress(mask, valve_y, cfg), view_type, cfg)))

    # F barrier
    if view_type in ("3ch","4ch"):
        cands.append(("rv_lv_barrier", rv_lv_barrier(mask, cfg)))

    # G edge snapping proxy
    cands.append(("edge_snap_proxy", topology_cleanup(edge_snapping_proxy(mask, edge_map, cfg), view_type, cfg)))

    # H shape prior guided
    if atlas is not None:
        for i,cm in enumerate(atlas_guided_grow_prune(mask, view_type, atlas, cfg, z_norm=z_norm)):
            cands.append((f"atlas_growprune_{i}", topology_cleanup(cm, view_type, cfg)))
        # Atlas Morph (DISABLED: Also causes Class Dropout if Atlas incomplete)
        # for i,cm in enumerate(atlas_constrained_morph(mask, view_type, atlas, cfg, z_norm=z_norm)):
        #     cands.append((f"atlas_morph_{i}", topology_cleanup(cm, view_type, cfg)))
        
        # Aggressive Replace (DISABLED: Causes Class Dropout/Regression)
        # for i,cm in enumerate(atlas_replace_op(mask, view_type, atlas, cfg, z_norm=z_norm)):
        #     cands.append((f"atlas_replace_{i}", topology_cleanup(cm, view_type, cfg)))

    # Gen 2 Mutation:    # F/G/H Morphological Ops (Risky in Multiclass - Disabled for Gen 4 Stability)
    # cands.append(("2_dilate", topology_cleanup(morph_op(mask, 2, "dilate", cfg), view_type, cfg)))
    # cands.append(("2_erode", topology_cleanup(morph_op(mask, 2, "erode", cfg), view_type, cfg)))
    # cands.append(("3_dilate", topology_cleanup(morph_op(mask, 3, "dilate", cfg), view_type, cfg)))
    # cands.append(("3_erode", topology_cleanup(morph_op(mask, 3, "erode", cfg), view_type, cfg)))
    
    # Specific anatomical fixes
    cands.append(("safe_topology_fix", safe_topology_fix(mask, view_type, cfg)))
    cands.append(("smooth_contour", topology_cleanup(boundary_smoothing(mask, cfg, "contour"), view_type, cfg)))
    cands.append(("smooth_gaussian", topology_cleanup(boundary_smoothing(mask, cfg, "gaussian"), view_type, cfg)))
    cands.append(("smooth_morph", topology_cleanup(boundary_smoothing(mask, cfg, "morph"), view_type, cfg)))
    
    # New Gap Closing Ops
    cands.append(("morphological_gap_close", morphological_gap_close(mask, cfg)))
    cands.append(("morph_close_large", morph_close_large(mask, cfg)))

    # Intensity-Guided Ops (formerly Polish)
    cands.append(("expand_myo_intensity", expand_myo_intensity_guided(mask, img)))
    cands.append(("expand_lv_intensity", expand_blood_pool_intensity(mask, img, 3, cfg)))
    if view_type in ("3ch", "4ch"):
        cands.append(("expand_rv_intensity", expand_blood_pool_intensity(mask, img, 1, cfg)))

    # Targeted bridge operations for disconnected structures
    cands.append(("myo_bridge", myo_bridge(mask)))
    cands.append(("lv_bridge", lv_bridge(mask)))
    cands.append(("convex_hull_fill", convex_hull_fill(mask)))

    return cands


# ---------------------------------------------------------------------------
# Intensity-Guided Operations (Ported from Reference)
# ---------------------------------------------------------------------------

def expand_blood_pool_intensity(mask: np.ndarray, img: np.ndarray, target_class: int, cfg: dict) -> np.ndarray:
    """
    Expand a blood pool class (1=RV, 3=LV) into bright background using intensity guidance.
    Ported from comprehensive_fix_v2.py
    """
    if img is None: return mask.copy()
    
    # Ensure grayscale for intensity calculations
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    target_mask = (mask == target_class)
    myo_mask = (mask == 2)
    bg_mask = (mask == 0)

    if not np.any(target_mask):
        return mask.copy()

    # Get intensity threshold
    blood_mean = img[target_mask].mean()
    blood_std = img[target_mask].std()
    myo_mean = img[myo_mask].mean() if np.any(myo_mask) else 0

    # Blood pool is brighter than myo
    intensity_min = (blood_mean + myo_mean) / 2 - blood_std

    # Find bright background near the target structure
    bright_bg = bg_mask & (img > intensity_min)

    # Must be near heart structures
    heart = (mask > 0)
    heart_dilated = ndi.binary_dilation(heart, iterations=15)
    expandable = bright_bg & heart_dilated
    
    # Don't expand into other structures (buffer)
    other_structures = (mask > 0) & (mask != target_class)
    other_buffer = ndi.binary_dilation(other_structures, iterations=2)
    expandable = expandable & ~other_buffer

    fixed_mask = mask.copy()
    current = target_mask.copy()
    max_expansion = 500
    total_added = 0
    ref_size = np.sum(target_mask)

    for _ in range(100):
        if total_added >= max_expansion: break
        
        dilated = ndi.binary_dilation(current, iterations=1)
        new_pixels = dilated & expandable & ~current
        
        if not np.any(new_pixels): break
        
        current = current | new_pixels
        total_added += np.sum(new_pixels)
        
    fixed_mask[current & ~target_mask] = target_class
    return fixed_mask

def expand_myo_intensity_guided(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Expand myocardium to close gaps where LV/RV touches background,
    BUT only if the gap pixels are dark (like myocardium).
    Ported from comprehensive_fix_v2.py (expand_myo_to_close_gaps)
    """
    if img is None: return mask.copy()

    # Ensure grayscale for intensity calculations
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    myo_mask = (mask == 2)
    blood_mask = (mask == 1) | (mask == 3)
    bg_mask = (mask == 0)

    if not np.any(myo_mask): return mask.copy()

    # Find where blood touches background (gaps)
    blood_dilated = ndi.binary_dilation(blood_mask, iterations=1)
    gaps = blood_dilated & bg_mask & ~blood_mask

    if not np.any(gaps): return mask.copy()
    
    # Gaps that are near myo
    myo_dilated = ndi.binary_dilation(myo_mask, iterations=8)
    fillable_gaps = gaps & myo_dilated
    
    if not np.any(fillable_gaps): return mask.copy()

    # Check intensity - myo should be dark
    myo_mean = img[myo_mask].mean()
    myo_std = img[myo_mask].std()
    # Relaxed threshold: Allow brighter pixels (up to 2.5 std devs) to be included
    # This helps close gaps where partial volume effect makes the gap slightly brighter
    intensity_max = myo_mean + 2.5 * myo_std

    # Only fill dark gaps
    dark_gaps = fillable_gaps & (img < intensity_max)
    
    # Fallback: if very small gap, fill anyway
    if not np.any(dark_gaps) and np.sum(fillable_gaps) < 100: # Increased from 50
        dark_gaps = fillable_gaps
        
    fixed_mask = mask.copy()
    current_myo = myo_mask.copy()
    total_added = 0
    max_expansion = 500 # Increased from 150 to allow larger gap closure
    
    for _ in range(200): # Increased iterations
        if total_added >= max_expansion: break

        dilated = ndi.binary_dilation(current_myo, iterations=1)
        new_pixels = dilated & dark_gaps & ~current_myo & (fixed_mask == 0)

        if not np.any(new_pixels): break

        fixed_mask[new_pixels] = 2
        current_myo = (fixed_mask == 2)
        total_added += np.sum(new_pixels)
        
    return fixed_mask

def topology_cleanup(mask: np.ndarray, view_type: str, cfg: dict) -> np.ndarray:
    """
    Revised V6: Safe Topology Cleanup with Component Ratio Check.
    Preserves disconnected components if they are significant (e.g. > 10% of total area).
    """
    out = np.zeros_like(mask)
    labels = [1, 2, 3] # RV, Myo, LV
    if view_type == "2ch":
        labels = [2, 3] # No RV in 2ch

    for c in labels:
        Mc = (mask == c)
        if np.count_nonzero(Mc) == 0:
            continue
            
        # 1. Fill Holes
        Mc_filled = ndi.binary_fill_holes(Mc)
        
        # 2. Filter Small Components (but keep fractured large parts)
        labeled, num_features = ndi.label(Mc_filled)
        if num_features > 1:
            # Calculate areas
            sizes = ndi.sum(Mc_filled, labeled, range(num_features + 1))
            total_area = np.sum(sizes[1:])
            
            # Keep any component that is > 10% of total area
            # Or always keep largest.
            # AND keep components > absolute size (e.g. 50px)
            
            largest_idx = np.argmax(sizes[1:]) + 1
            
            keep_mask = np.zeros_like(labeled, dtype=bool)
            keep_mask[labeled == largest_idx] = True
            
            # Check others
            for i in range(1, num_features + 1):
                if i == largest_idx: continue
                ratio = sizes[i] / (total_area + 1e-6)
                if ratio > 0.10: # Keep if >10% of total
                    keep_mask[labeled == i] = True
            
            Mc_final = keep_mask
        else:
            Mc_final = Mc_filled
            
        # SURVIVAL CHECK
        if np.count_nonzero(Mc_final) == 0:
             Mc_final = Mc
             
        out[Mc_final] = c
        
    return out


def convex_hull_fill(mask: np.ndarray) -> np.ndarray:
    """
    Advanced gap filling: Find the largest Myocardium component and fill its 
    Convex Hull, but restrain it to the Myo ring shape by masking out the 
    known LV and RV.
    """
    M2 = (mask == 2).astype(np.uint8)
    # Find largest component
    labeled, num = ndi.label(M2)
    if num == 0:
        return mask.copy()
        
    sizes = ndi.sum(M2, labeled, range(num + 1))
    largest_idx = np.argmax(sizes[1:]) + 1
    largest_comp = (labeled == largest_idx).astype(np.uint8)
    
    # Get contours
    contours, _ = cv2.findContours(largest_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask.copy()
        
    # Simplify contour slightly to avoid noise
    c = contours[0]
    hull = cv2.convexHull(c)
    
    # Draw hull
    hull_mask = np.zeros_like(M2)
    cv2.drawContours(hull_mask, [hull], -1, 1, thickness=-1)
    
    # Restrain hull:
    # 1. Remove LV (3) and RV (1) - don't overwrite them
    # 2. Keep original Myo (2)
    # 3. Only add pixels that are in the Hull but NOT in LV/RV
    
    M1 = (mask == 1)
    M3 = (mask == 3)
    
    # New Myo = (Hull AND NOT (LV OR RV))
    # But wait, Hull covers the CENTER (LV) too!
    # So we must subtract the LV *hole*... but if LV is missing?
    # Better: Hull includes the central hole. We need to subtract the HOLE.
    # If M3 exists, we subtract M3. 
    # If M3 is small or missing, we might accidentally fill the ventricle.
    # SAFETY: Only apply if LV exists and is reasonably large.
    
    if np.sum(M3) < 100:
        return mask.copy() # Unsafe to use hull if LV is missing (would fill center)
        
    # Subtract LV (dilated slightly to be safe)
    M3_dil = ndi.binary_dilation(M3, iterations=2)
    M1_dil = ndi.binary_dilation(M1, iterations=1)
    
    new_M2 = (hull_mask > 0) & (~M3_dil) & (~M1_dil)
    
    # Final merge
    result = _merge(M1, new_M2, M3)
    
    if np.array_equal(result, mask):
        # DEBUG: Save visualization of why it failed
        debug_img = np.zeros((H, W, 3), dtype=np.uint8)
        debug_img[mask == 2] = [0, 255, 0] # Green Myo
        debug_img[mask == 3] = [0, 0, 255] # Blue LV
        debug_img[hull_mask > 0] = [100, 100, 100] # Gray Hull (background)
        debug_img[M3_dil > 0] = [0, 0, 100] # Dark Blue Dilated LV
        # import cv2  <-- REMOVED to fix shadowing
        cv2.imwrite("/data484_5/xzhao14/cardiac_agent_postproc/convex_hull_fail.png", debug_img)
        
    return result

def is_valid_candidate(mask: np.ndarray) -> bool:
    if mask.min() < 0 or mask.max() > 3:
        return False
    # Relaxed: Allow missing classes (soft penalty in RQS P_components instead)
    if np.count_nonzero(mask > 0) == 0:
        return False
    return True
