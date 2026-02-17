"""
Inter-slice consistency repair operations.

Uses masks from neighboring slices (same patient, same view) to detect and
correct misaligned or shifted segmentations. These operations treat the
neighbor slices as a **local, patient-specific atlas** — far more accurate
than the global atlas because the anatomy is identical.

Neighbor masks are expected as Dict[int, np.ndarray] mapping slice_index →
mask. The current slice's index is implicit (not in the dict).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

from .geom import keep_largest_cc, centroid, bbox

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLASS_LABELS = {1: "RV", 2: "Myo", 3: "LV"}


def _structure_centroid(mask: np.ndarray, cls: int) -> Optional[Tuple[float, float]]:
    """Return (cx, cy) of the largest connected component of class `cls`."""
    binary = (mask == cls).astype(np.uint8)
    if binary.sum() == 0:
        return None
    lcc = keep_largest_cc(binary)
    ys, xs = np.nonzero(lcc)
    return (float(xs.mean()), float(ys.mean()))


def _structure_area(mask: np.ndarray, cls: int) -> int:
    return int(np.count_nonzero(mask == cls))


def _neighbor_centroids(
    neighbor_masks: Dict[int, np.ndarray], cls: int
) -> List[Tuple[float, float]]:
    """Collect centroids of `cls` across all neighbors (skipping missing)."""
    centroids = []
    for _idx, nmask in neighbor_masks.items():
        c = _structure_centroid(nmask, cls)
        if c is not None:
            centroids.append(c)
    return centroids


def _neighbor_areas(
    neighbor_masks: Dict[int, np.ndarray], cls: int
) -> List[int]:
    return [_structure_area(m, cls) for m in neighbor_masks.values()]


# ---------------------------------------------------------------------------
# Op 1: Neighbor Centroid Alignment
# ---------------------------------------------------------------------------

def neighbor_centroid_align(
    mask: np.ndarray,
    neighbor_masks: Dict[int, np.ndarray],
    sigma_threshold: float = 2.0,
) -> np.ndarray:
    """
    Detect and correct shifted structures by comparing per-class centroids
    with neighbor consensus.

    For each class (RV, Myo, LV):
    - Compute centroid in current slice and all neighbors
    - If current centroid deviates > sigma_threshold * std from neighbor mean,
      shift the entire class region toward the neighbor mean

    Returns corrected mask (or original if no shift needed).
    """
    if not neighbor_masks:
        return mask.copy()

    out = mask.copy()
    h, w = mask.shape
    shifted_any = False

    for cls in [1, 2, 3]:
        cur_c = _structure_centroid(mask, cls)
        if cur_c is None:
            continue

        nb_centroids = _neighbor_centroids(neighbor_masks, cls)
        if len(nb_centroids) < 1:
            continue

        # Compute neighbor mean and std
        nb_arr = np.array(nb_centroids)
        mean_cx, mean_cy = nb_arr[:, 0].mean(), nb_arr[:, 1].mean()

        if len(nb_centroids) >= 2:
            std_cx = max(nb_arr[:, 0].std(), 3.0)  # min 3px std
            std_cy = max(nb_arr[:, 1].std(), 3.0)
        else:
            # With only 1 neighbor, use fixed tolerance
            std_cx = std_cy = 8.0

        # Check if current centroid is an outlier
        dx = cur_c[0] - mean_cx
        dy = cur_c[1] - mean_cy

        is_outlier_x = abs(dx) > sigma_threshold * std_cx
        is_outlier_y = abs(dy) > sigma_threshold * std_cy

        if not (is_outlier_x or is_outlier_y):
            continue

        # Compute shift to move toward neighbor mean (not fully, just 70%)
        shift_x = -int(round(dx * 0.7)) if is_outlier_x else 0
        shift_y = -int(round(dy * 0.7)) if is_outlier_y else 0

        if shift_x == 0 and shift_y == 0:
            continue

        # Shift this class's pixels
        cls_mask = (mask == cls).astype(np.uint8)
        M_affine = np.array(
            [[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32
        )
        shifted_cls = cv2.warpAffine(
            cls_mask, M_affine, (w, h),
            flags=cv2.INTER_NEAREST, borderValue=0
        )

        # Clear old class pixels and place shifted ones
        # Only place where destination is background or same class
        old_region = out == cls
        out[old_region] = 0  # clear old position

        new_region = shifted_cls > 0
        # Don't overwrite other classes — only fill background
        can_place = (out == 0) & new_region
        out[can_place] = cls
        shifted_any = True

    return out if shifted_any else mask.copy()


# ---------------------------------------------------------------------------
# Op 2: Neighbor Shape Prior (Local Mini-Atlas)
# ---------------------------------------------------------------------------

def neighbor_shape_prior(
    mask: np.ndarray,
    neighbor_masks: Dict[int, np.ndarray],
    grow_threshold: float = 0.6,
    prune_threshold: float = 0.1,
) -> np.ndarray:
    """
    Build a local probability map from neighbor slices and use it to
    grow/prune the current mask — same idea as atlas_guided_grow_prune but
    using actual patient neighbors instead of global atlas clusters.

    For each pixel:
    - If neighbor consensus says class C is likely (>grow_threshold) but
      current mask has background → grow
    - If neighbor consensus says class C is unlikely (<prune_threshold) but
      current mask has C → prune (replace with best neighbor vote)

    Returns corrected mask.
    """
    if not neighbor_masks:
        return mask.copy()

    h, w = mask.shape

    # Build probability maps from neighbors (aligned by centroid)
    # Step 1: Compute global foreground centroid offset for each neighbor
    #         and apply translation before building prob map
    cur_fg = (mask > 0)
    if cur_fg.sum() == 0:
        return mask.copy()

    cur_cy, cur_cx = ndi.center_of_mass(cur_fg)

    prob = {c: np.zeros((h, w), dtype=np.float32) for c in [0, 1, 2, 3]}
    n_valid = 0

    for _idx, nmask in neighbor_masks.items():
        nb_fg = (nmask > 0)
        if nb_fg.sum() == 0:
            continue

        nb_cy, nb_cx = ndi.center_of_mass(nb_fg)
        # Translation to align neighbor to current slice
        tx = int(round(cur_cx - nb_cx))
        ty = int(round(cur_cy - nb_cy))

        M_affine = np.array(
            [[1, 0, tx], [0, 1, ty]], dtype=np.float32
        )

        # Warp each class channel
        for c in [1, 2, 3]:
            cls_bin = (nmask == c).astype(np.uint8)
            warped = cv2.warpAffine(
                cls_bin, M_affine, (w, h),
                flags=cv2.INTER_NEAREST, borderValue=0
            )
            prob[c] += warped.astype(np.float32)

        n_valid += 1

    if n_valid == 0:
        return mask.copy()

    # Normalize to probabilities
    for c in [1, 2, 3]:
        prob[c] /= n_valid
    prob[0] = 1.0 - (prob[1] + prob[2] + prob[3])
    prob[0] = np.clip(prob[0], 0, 1)

    out = mask.copy()

    # Grow: where neighbor consensus is strong but current mask is empty
    for c in [2, 3, 1]:  # priority: Myo > LV > RV
        grow_zone = (prob[c] > grow_threshold) & (out == 0)
        out[grow_zone] = c

    # Prune: where current mask has a class but neighbors strongly disagree
    for c in [1, 2, 3]:
        prune_zone = (prob[c] < prune_threshold) & (out == c)
        if prune_zone.sum() == 0:
            continue
        # Replace with best neighbor vote
        ys, xs = np.nonzero(prune_zone)
        for y, x in zip(ys, xs):
            probs = [prob[0][y, x], prob[1][y, x], prob[2][y, x], prob[3][y, x]]
            out[y, x] = int(np.argmax(probs))

    return out


# ---------------------------------------------------------------------------
# Op 3: Neighbor Area Consistency
# ---------------------------------------------------------------------------

def neighbor_area_consistency(
    mask: np.ndarray,
    neighbor_masks: Dict[int, np.ndarray],
    sigma_threshold: float = 2.0,
) -> np.ndarray:
    """
    Check if area of each structure is consistent with neighbors.
    If current slice has abnormally large/small structure relative to
    neighbors, erode or dilate to bring it closer to the neighbor mean.

    Returns corrected mask.
    """
    if not neighbor_masks:
        return mask.copy()

    out = mask.copy()
    changed = False

    for cls in [1, 2, 3]:
        cur_area = _structure_area(mask, cls)
        if cur_area == 0:
            continue

        nb_areas = _neighbor_areas(neighbor_masks, cls)
        nb_areas = [a for a in nb_areas if a > 0]
        if len(nb_areas) < 1:
            continue

        mean_area = np.mean(nb_areas)
        if len(nb_areas) >= 2:
            std_area = max(np.std(nb_areas), mean_area * 0.05)  # min 5% of mean
        else:
            std_area = mean_area * 0.15  # 15% tolerance with 1 neighbor

        deviation = (cur_area - mean_area) / std_area

        if abs(deviation) < sigma_threshold:
            continue

        # Structure is too large → erode; too small → dilate
        cls_bin = (mask == cls).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Scale iterations by how far off we are
        iters = min(3, max(1, int(abs(deviation) - sigma_threshold + 1)))

        if deviation > 0:
            # Too large → erode
            corrected = cv2.erode(cls_bin, kernel, iterations=iters)
        else:
            # Too small → dilate
            corrected = cv2.dilate(cls_bin, kernel, iterations=iters)

        # Apply: clear old, set new (respecting other classes)
        old_region = out == cls
        out[old_region] = 0

        if deviation > 0:
            # Erode: just place the smaller region
            new_region = corrected > 0
            out[new_region & (out == 0)] = cls
        else:
            # Dilate: only fill background, don't overwrite other classes
            new_region = corrected > 0
            out[new_region & (out == 0)] = cls

        changed = True

    return out if changed else mask.copy()


# ---------------------------------------------------------------------------
# Convenience: detect misalignment
# ---------------------------------------------------------------------------

def detect_misalignment(
    mask: np.ndarray,
    neighbor_masks: Dict[int, np.ndarray],
    sigma_threshold: float = 2.0,
) -> List[Dict]:
    """
    Return a list of detected misalignment issues (empty = no issues).
    Each issue is a dict with keys: class, type, deviation.
    """
    issues = []
    if not neighbor_masks:
        return issues

    for cls in [1, 2, 3]:
        label = CLASS_LABELS[cls]

        # Centroid check
        cur_c = _structure_centroid(mask, cls)
        if cur_c is None:
            continue

        nb_centroids = _neighbor_centroids(neighbor_masks, cls)
        if len(nb_centroids) >= 1:
            nb_arr = np.array(nb_centroids)
            mean_cx, mean_cy = nb_arr[:, 0].mean(), nb_arr[:, 1].mean()
            std_cx = max(nb_arr[:, 0].std(), 3.0) if len(nb_centroids) >= 2 else 8.0
            std_cy = max(nb_arr[:, 1].std(), 3.0) if len(nb_centroids) >= 2 else 8.0

            dx = abs(cur_c[0] - mean_cx) / std_cx
            dy = abs(cur_c[1] - mean_cy) / std_cy

            if dx > sigma_threshold or dy > sigma_threshold:
                issues.append({
                    "class": label,
                    "type": "centroid_shift",
                    "deviation_x": round(dx, 2),
                    "deviation_y": round(dy, 2),
                })

        # Area check
        cur_area = _structure_area(mask, cls)
        nb_areas = [a for a in _neighbor_areas(neighbor_masks, cls) if a > 0]
        if nb_areas:
            mean_a = np.mean(nb_areas)
            std_a = max(np.std(nb_areas), mean_a * 0.05) if len(nb_areas) >= 2 else mean_a * 0.15
            dev = abs(cur_area - mean_a) / std_a
            if dev > sigma_threshold:
                issues.append({
                    "class": label,
                    "type": "area_inconsistency",
                    "deviation": round(dev, 2),
                    "current_area": cur_area,
                    "neighbor_mean": round(float(mean_a)),
                })

    return issues
