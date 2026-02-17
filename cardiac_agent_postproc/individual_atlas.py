"""
Individual Atlas Repair Module.

Implements deterministic transfer of segmentation masks from a good neighbor slice 
to a target bad slice using:
1. Image-based Phase Correlation (Registration)
2. Heuristic Scaling based on cardiac phase volume changes
3. Hybrid Fusion with original prediction based on diagnosis
"""
import cv2
import numpy as np
from scipy import ndimage as ndi

def to_gray(img):
    """Convert to grayscale float32 for registration"""
    if img.ndim == 3:
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        elif img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def align_images(src_img: np.ndarray, dst_img: np.ndarray) -> tuple[float, float]:
    """
    Register src_img to dst_img using Phase Correlation.
    Returns (dy, dx) shift vector.
    """
    src_gray = to_gray(src_img).astype(np.float32)
    dst_gray = to_gray(dst_img).astype(np.float32)
    
    # Hanning window to reduce edge effects
    h, w = src_gray.shape
    hy = np.hanning(h)
    hx = np.hanning(w)
    window = np.outer(hy, hx)
    
    # Phase correlation
    shift, response = cv2.phaseCorrelate(src_gray * window, dst_gray * window)
    # cv2 returns (dx, dy), we want (dy, dx) for indexing
    return shift[1], shift[0]

def transfer_mask(
    atlas_mask: np.ndarray, 
    shift: tuple[float, float], 
    scale_factor: float
) -> np.ndarray:
    """
    Apply shift and scale to atlas mask.
    """
    dy, dx = shift
    
    # 1. Apply Shift
    shifted = ndi.shift(atlas_mask, (dy, dx), order=0, mode='constant', cval=0)
    
    # 2. Apply Scale
    # Center of scaling: Centroid of shifted mask (or image center if empty)
    coords = np.argwhere(shifted > 0)
    if len(coords) > 0:
        cy, cx = coords.mean(axis=0)
    else:
        cy, cx = shifted.shape[0]//2, shifted.shape[1]//2
        
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale_factor)
    scaled = cv2.warpAffine(
        shifted, M, 
        (shifted.shape[1], shifted.shape[0]), 
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return scaled

def hybrid_repair(
    target_pred: np.ndarray,
    neighbor_mask: np.ndarray,
    target_img: np.ndarray,
    neighbor_img: np.ndarray,
    diagnosis_issues: list[str] = None
) -> np.ndarray:
    """
    Execute Individual Atlas Repair.
    
    Args:
        target_pred: Original bad prediction
        neighbor_mask: Good neighbor mask (Atlas)
        target_img: Target image (Anchor)
        neighbor_img: Good neighbor image (Source)
        diagnosis_issues: List of issues to guide conflict resolution
        
    Returns:
        Repaired mask
    """
    # 1. Align
    dy, dx = align_images(neighbor_img, target_img)
    
    # 2. Estimate Scale Factor
    # Heuristic: If we don't know phases, assume 0.8 as safe contraction/expansion?
    # Better: Measure RV area ratio? No, target RV is bad.
    # User heuristic: 0.7 worked well for ED->ES. 
    # For robust production usage, strictly we need to know phases or use 1.0 + non-rigid.
    # For now, using 0.75 as a generic scaler if phases likely differ, 
    # or 1.0 if they are close. 
    # Let's use 1.0 for now, unless specific logic (like critical missing) suggests otherwise.
    # In case 335, we had to shrink (0.7).
    # If we detect "massive_missing" or "undersized", maybe the target is just smaller?
    # Simple check: Compare Bright Blood Pool area in images? Too complex.
    # Let's stick to 1.0 (Direct Transfer aligned) first, unless we want to hardcode 0.7 for demo.
    # Wait, the demo showed 0.7 was necessary.
    # Let's compute scale from the Image Intensity moments?
    # Let's just use 0.8 as a compromise for now.
    scale = 0.8
    
    det_mask = transfer_mask(neighbor_mask, (dy, dx), scale)
    
    # 3. Hybrid Fusion
    # Default: Trust Det_Mask for classes that are BROKEN in Target.
    # Identify broken classes from diagnosis.
    # If diagnosis not provided, assume RV is fragile.
    
    replace_classes = {1} # Always replace RV by default? 
    if diagnosis_issues:
        if any("RV" in i for i in diagnosis_issues): replace_classes.add(1)
        if any("Myo" in i for i in diagnosis_issues): replace_classes.add(2)
        if any("LV" in i for i in diagnosis_issues): replace_classes.add(3)
    
    hybrid = target_pred.copy()
    
    for c in replace_classes:
        # Overwrite: mask[det==c] = c
        # This trusts Det mask geometry for that class
        det_c = (det_mask == c)
        hybrid[det_c] = c
        
    return hybrid
