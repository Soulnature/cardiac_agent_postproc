"""
anatomy_score.py
================
No-GT anatomy quality score based on source-learned feature distributions
with view-specific anatomical constraints.

Pipeline:
  1. fit_source_stats(pred_dir, view_type)  -> AnatomyStats  (pickle-able)
  2. compute_anatomy_score(mask, img, view_type, stats)
       -> (score: float [0,100], detail: dict)
  3. diagnose_from_score(detail, view_type)
       -> List[str]  rule-based diagnoses, replaces LLM diagnosis

View-specific feature sets
--------------------------
ALL views:
  myo_enclosure_ratio    fraction of LV-ring covered by Myo (R_enclosure proxy)
  myo_lv_ratio           myo_area / lv_area
  lv_compactness         4π·lv_area / lv_perim²
  intensity_contrast     between-class var / within-class var
  boundary_gradient_rv   mean Sobel mag at RV  boundary (0 if no RV)
  boundary_gradient_lv   mean Sobel mag at LV  boundary
  boundary_gradient_myo  mean Sobel mag at Myo boundary

2CH-specific  (RV should be absent):
  log_lv_area_frac       log(lv_area / fg_area)
  lv_eccentricity        elongation of LV (should be high ~0.85)
  rv_contamination       rv_area / fg_area  (should be ~0)

3CH-specific  (small crescent RV):
  log_lv_area_frac
  log_rv_lv_ratio        log(rv / lv)  — key discriminator
  rv_eccentricity        RV shape elongation (crescent → high)
  lv_eccentricity        LV shape (moderately elongated)
  rv_lv_centroid_lateral normalised |cx_rv - cx_lv| / img_width  (side-by-side)
  rv_solidity            rv_area / convex_hull_rv_area  (fragmented → low)
  myo_solidity           myo_area / convex_hull_myo_area

4CH-specific  (full RV, side-by-side):
  log_lv_area_frac
  log_rv_lv_ratio        — most discriminative
  lv_eccentricity        less elongated than 2CH
  rv_lv_centroid_dy      normalised |cy_rv - cy_lv| / img_height  (small → side-by-side)
  rv_lv_centroid_lateral normalised |cx_rv - cx_lv| / img_width
  rv_solidity            rv_area / convex_hull_rv_area  (fragmented → low)
  myo_solidity           myo_area / convex_hull_myo_area

Scoring  (v2: directional one-sided Mahalanobis)
-------
  z_i = clip( (f_i - median_i) / (1.4826·MAD_i) , -5, +5 )

  Directional penalty per feature:
    direction[i] = +1 → high value is bad  → penalty_i = max(0,  z_i)²
    direction[i] = -1 → low  value is bad  → penalty_i = max(0, -z_i)²
    direction[i] =  0 → symmetric          → penalty_i = z_i²

  weighted_mahal = sqrt( Σ w_i·penalty_i / Σ w_i )
  base_score     = 100·exp(-0.5·weighted_mahal)
  hard_penalty   = sum of hard-violation deductions (0–60 pts)
  final_score    = max(0, base_score - hard_penalty)

Diagnosis
---------
  diagnose_from_score(detail, view_type) maps extreme z-scores to
  actionable repair tags compatible with the planner/executor:
    rv_too_small, rv_too_large, rv_present_in_2ch,
    myo_insufficient_enclosure, myo_too_thin, myo_too_thick,
    lv_too_large, lv_too_small, poor_boundary_rv, poor_boundary_lv,
    rv_not_lateral (3ch/4ch), rv_lv_vertical_misalign (4ch)
"""
from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .geom import (
    boundary_4n,
    centroid as geom_centroid,
    compactness as geom_compactness,
    eccentricity_largest_cc,
)
from .io_utils import read_mask
from .rqs import sobel_edges


# ─── per-class boundary gradient ──────────────────────────────────────────────

def _boundary_gradient(binary: np.ndarray, sobel_mag: np.ndarray) -> float:
    """Mean Sobel magnitude sampled at boundary pixels of `binary`."""
    if binary.sum() < 4:
        return 0.0
    B = boundary_4n(binary.astype(bool))
    if B.sum() == 0:
        return 0.0
    return float(sobel_mag[B].mean())


def _sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """Continuous Sobel magnitude map (float32, same shape as img)."""
    from scipy.ndimage import sobel as scipy_sobel
    img_f = img.astype(np.float32)
    sx = scipy_sobel(img_f, axis=1)
    sy = scipy_sobel(img_f, axis=0)
    return np.sqrt(sx ** 2 + sy ** 2)


# ─── myo enclosure ratio ──────────────────────────────────────────────────────

def _myo_enclosure_ratio(mask: np.ndarray) -> float:
    """
    Fraction of the LV-ring (LV dilated 2px) that is covered by Myo.
    1.0 → Myo perfectly wraps LV; 0.0 → no enclosure.
    """
    lv = (mask == 3).astype(np.uint8)
    myo = (mask == 2).astype(np.uint8)
    if lv.sum() < 4:
        return 0.0
    lv_ring = ndi.binary_dilation(lv.astype(bool), iterations=2) & ~lv.astype(bool)
    ring_pixels = lv_ring.sum()
    if ring_pixels == 0:
        return 0.0
    covered = (lv_ring & myo.astype(bool)).sum()
    return float(covered) / float(ring_pixels)


# ─── intensity contrast ───────────────────────────────────────────────────────

def _intensity_contrast(mask: np.ndarray, img: np.ndarray) -> float:
    """Between-class / within-class intensity variance (Fisher criterion)."""
    means, vars_ = [], []
    for cls in [1, 2, 3]:
        pixels = img[mask == cls].astype(float)
        if len(pixels) < 10:
            continue
        means.append(pixels.mean())
        vars_.append(pixels.var())
    if len(means) < 2:
        return 0.0
    between = float(np.var(means))
    within = float(np.mean(vars_)) if vars_ else 1.0
    return between / max(within, 1.0)


# ─── solidity (area / convex-hull area) ───────────────────────────────────────

def _solidity(binary: np.ndarray) -> float:
    """
    Area / convex-hull area.
    1.0 = perfectly convex, <1.0 = concave / fragmented.
    Note: myocardial ring is naturally concave (solidity ~0.6–0.8 when correct).
    """
    b = binary.astype(bool)
    if b.sum() < 10:
        return float("nan")
    try:
        from skimage.morphology import convex_hull_image
        hull = convex_hull_image(b)
        return float(b.sum()) / float(max(hull.sum(), 1))
    except Exception:
        return float("nan")


# ─── safe helpers ─────────────────────────────────────────────────────────────

def _safe_log(x: float, eps: float = 1e-3) -> float:
    return float(np.log(max(x, eps)))


def _area(mask: np.ndarray, cls: int) -> float:
    return float((mask == cls).sum())


# ─── main feature extractor ───────────────────────────────────────────────────

def compute_anatomy_features(
    mask: np.ndarray,
    img: np.ndarray,
    view_type: str,
) -> Dict[str, float]:
    """
    Compute anatomy features from prediction mask + raw image (no GT).
    Returns dict keyed by feature name; NaN if a feature cannot be computed.
    """
    view = _normalize_view(view_type)
    is_2c = view == "2c"
    is_3c = view == "3c"
    is_4c = view == "4c"
    has_rv = is_3c or is_4c

    rv_a = _area(mask, 1)
    myo_a = _area(mask, 2)
    lv_a = _area(mask, 3)
    fg_a = rv_a + myo_a + lv_a
    h, w = mask.shape

    feats: Dict[str, float] = {}

    if lv_a < 10 or fg_a < 20:
        return {k: float("nan") for k in _feature_names(view)}

    # ── shared ──────────────────────────────────────────────────────────────
    feats["myo_enclosure_ratio"] = _myo_enclosure_ratio(mask)
    feats["myo_lv_ratio"] = myo_a / lv_a
    feats["lv_compactness"] = geom_compactness(mask == 3)
    feats["intensity_contrast"] = _intensity_contrast(mask, img)

    # per-class boundary gradients
    sobel_mag = _sobel_magnitude(img.astype(np.float32))
    feats["boundary_gradient_lv"] = _boundary_gradient(mask == 3, sobel_mag)
    feats["boundary_gradient_myo"] = _boundary_gradient(mask == 2, sobel_mag)
    feats["boundary_gradient_rv"] = _boundary_gradient(mask == 1, sobel_mag) if has_rv else 0.0

    # LV eccentricity (all views, different expected range)
    feats["lv_eccentricity"] = eccentricity_largest_cc(mask == 3)

    # ── 2CH-specific ─────────────────────────────────────────────────────────
    if is_2c:
        feats["log_lv_area_frac"] = _safe_log(lv_a / fg_a)
        feats["rv_contamination"] = rv_a / max(fg_a, 1.0)  # should be ~0

    # ── 3CH-specific ─────────────────────────────────────────────────────────
    elif is_3c:
        feats["log_lv_area_frac"] = _safe_log(lv_a / fg_a)
        feats["log_rv_lv_ratio"] = _safe_log(rv_a / lv_a)
        feats["rv_eccentricity"] = eccentricity_largest_cc(mask == 1) if rv_a > 10 else float("nan")
        # lateral distance: RV centroid should be to the side of LV
        cx_lv, _ = geom_centroid(mask == 3)
        if rv_a > 10:
            cx_rv, _ = geom_centroid(mask == 1)
            feats["rv_lv_centroid_lateral"] = abs(cx_rv - cx_lv) / max(w, 1)
        else:
            feats["rv_lv_centroid_lateral"] = float("nan")
        # solidity: RV in 3C is crescent → naturally low solidity; fragmentation → very low
        feats["rv_solidity"] = _solidity(mask == 1) if rv_a > 10 else float("nan")
        feats["myo_solidity"] = _solidity(mask == 2) if myo_a > 10 else float("nan")

    # ── 4CH-specific ─────────────────────────────────────────────────────────
    elif is_4c:
        feats["log_lv_area_frac"] = _safe_log(lv_a / fg_a)
        feats["log_rv_lv_ratio"] = _safe_log(rv_a / lv_a)
        cx_lv, cy_lv = geom_centroid(mask == 3)
        if rv_a > 10:
            cx_rv, cy_rv = geom_centroid(mask == 1)
            feats["rv_lv_centroid_dy"] = abs(cy_rv - cy_lv) / max(h, 1)
            feats["rv_lv_centroid_lateral"] = abs(cx_rv - cx_lv) / max(w, 1)
        else:
            feats["rv_lv_centroid_dy"] = float("nan")
            feats["rv_lv_centroid_lateral"] = float("nan")
        # solidity features — RV in 4C should be compact; fragmented RV → low
        feats["rv_solidity"] = _solidity(mask == 1) if rv_a > 10 else float("nan")
        # Myo in 4C is a thick ring → naturally low solidity; blob myo → high (bad)
        feats["myo_solidity"] = _solidity(mask == 2) if myo_a > 10 else float("nan")

    return feats


# ─── per-view feature name lists ─────────────────────────────────────────────

_SHARED_FEATURES = [
    "myo_enclosure_ratio",
    "myo_lv_ratio",
    "lv_compactness",
    "intensity_contrast",
    "boundary_gradient_lv",
    "boundary_gradient_myo",
    "boundary_gradient_rv",
    "lv_eccentricity",
]

_VIEW_FEATURES: Dict[str, List[str]] = {
    "2c": _SHARED_FEATURES + ["log_lv_area_frac", "rv_contamination"],
    "3c": _SHARED_FEATURES + [
        "log_lv_area_frac", "log_rv_lv_ratio",
        "rv_eccentricity", "rv_lv_centroid_lateral",
        "rv_solidity", "myo_solidity",
    ],
    "4c": _SHARED_FEATURES + [
        "log_lv_area_frac", "log_rv_lv_ratio",
        "rv_lv_centroid_dy", "rv_lv_centroid_lateral",
        "rv_solidity", "myo_solidity",
    ],
}

# Per-feature importance weights — default (2C)
_FEATURE_WEIGHTS: Dict[str, float] = {
    "log_rv_lv_ratio":          1.5,
    "myo_enclosure_ratio":      3.2,
    "rv_contamination":         3.0,   # 2c: any RV is a big problem
    "myo_lv_ratio":             2.2,
    "log_lv_area_frac":         1.0,
    "boundary_gradient_rv":     2.0,
    "boundary_gradient_lv":     2.0,
    "boundary_gradient_myo":    1.8,
    "lv_eccentricity":          1.5,
    "rv_eccentricity":          2.0,
    "rv_lv_centroid_dy":        2.5,
    "rv_lv_centroid_lateral":   2.5,
    "lv_compactness":           0.8,
    "intensity_contrast":       1.0,
    "rv_solidity":              2.5,
    "myo_solidity":             2.0,
}

# 4C-specific weights — calibrated to UKB empirical |Spearman rho| ordering.
# NOTE: myo_lv_ratio intentionally low (rho=-0.26 but generates false negatives
#   for patients with naturally thick myocardium). myo_solidity similarly reduced
#   because thickness varies legitimately between patients.
_FEATURE_WEIGHTS_4C: Dict[str, float] = {
    "rv_lv_centroid_dy":        3.5,   # strongest 4C discriminator (rho=+0.47)
    "rv_solidity":              3.0,   # rho=+0.33
    "rv_lv_centroid_lateral":   2.5,   # rho=-0.27 (low lateral→bad)
    "boundary_gradient_rv":     2.0,   # rho=+0.21
    "intensity_contrast":       1.5,   # rho=+0.20
    "myo_enclosure_ratio":      3.0,   # user policy: prioritize Myo integrity
    "myo_lv_ratio":             2.2,   # user policy: tighten Myo thickness consistency
    "boundary_gradient_lv":     1.0,   # rho=+0.13
    "boundary_gradient_myo":    1.3,   # emphasize Myo contour support
    "myo_solidity":             1.6,   # raise Myo structural penalty weight
    "lv_eccentricity":          0.5,   # rho=+0.10 (low signal)
    "lv_compactness":           0.3,   # rho=-0.09 (noise)
    "log_lv_area_frac":         0.3,   # rho=+0.08 (noise)
    "log_rv_lv_ratio":          0.3,   # rho=-0.02 (noise! reduce heavily)
}

# 3C-specific weights — calibrated to UKB empirical |Spearman rho| ordering:
#   boundary_gradient_lv (-0.50), rv_lv_centroid_lateral (+0.48),
#   rv_eccentricity (-0.46), boundary_gradient_rv (-0.42), rv_solidity (+0.42),
#   lv_eccentricity (+0.31), myo_enclosure_ratio (+0.30)
#   NOISE: log_rv_lv_ratio (-0.04), log_lv_area_frac (-0.07), myo_lv_ratio (+0.08)
_FEATURE_WEIGHTS_3C: Dict[str, float] = {
    "boundary_gradient_lv":     4.0,   # rho=-0.50 (high gradient in 3C → bad position)
    "rv_lv_centroid_lateral":   4.0,   # rho=+0.48 (low lateral → stacked → bad)
    "rv_eccentricity":          3.5,   # rho=-0.46 (over-elongated RV → bad)
    "boundary_gradient_rv":     3.5,   # rho=-0.42 (high gradient → wrong position)
    "rv_solidity":              3.0,   # rho=+0.42 (fragmented RV → bad)
    "lv_eccentricity":          2.5,   # rho=+0.31
    "myo_enclosure_ratio":      3.2,   # user policy: prioritize Myo integrity
    "lv_compactness":           1.0,   # rho=-0.21 (penalize overly compact in 3C)
    "boundary_gradient_myo":    1.2,   # increase Myo contour weight
    "intensity_contrast":       0.8,   # rho=-0.14 (weak)
    "myo_lv_ratio":             1.2,   # tighten Myo/LV thickness ratio control
    "myo_solidity":             1.0,   # increase ring-structure consistency pressure
    "log_lv_area_frac":         0.3,   # rho=-0.06 (noise)
    "log_rv_lv_ratio":          0.3,   # rho=-0.04 (noise)
}

# Per-feature penalization direction:
#   -1 → penalize when z < 0 (low value is bad, high is fine)
#   +1 → penalize when z > 0 (high value is bad, low is fine)
#    0 → symmetric (both directions penalized equally)
#
# Determined from:
#   - Empirical Spearman correlation with Dice on UKB dataset
#   - Anatomical priors (which direction is physiologically meaningful)
_FEATURE_DIRECTIONS: Dict[str, int] = {
    # Low value is bad (-1): penalize only when below-median
    "myo_enclosure_ratio":      -1,   # low enclosure → broken ring
    "myo_lv_ratio":              0,   # too thin OR too thick both bad → symmetric
    "intensity_contrast":       -1,   # low contrast → poorly defined boundaries
    "boundary_gradient_rv":     -1,   # low gradient → boundary not on edge
    "boundary_gradient_lv":     -1,   # low gradient → boundary not on edge (4C)
    "boundary_gradient_myo":    -1,   # low gradient → boundary not on edge
    "lv_eccentricity":          -1,   # low eccentricity → LV too round
    "rv_eccentricity":           0,   # crescent in 3C; view-specific override below
    "rv_lv_centroid_lateral":   -1,   # low lateral → RV/LV stacked (wrong in 3C/4C)
    "rv_lv_centroid_dy":        -1,   # low dy → both at same height (4C: data shows rho=+0.47)
    "rv_solidity":              -1,   # low solidity → fragmented RV
    "myo_solidity":             +1,   # high solidity → blob myo (ring should be concave)
                                      # (for 4C, overridden to symmetric in _FEATURE_DIRECTIONS_4C)
    "lv_compactness":            0,   # symmetric: too round OR too elongated
    "log_lv_area_frac":          0,   # LV size: can be too small OR too large
    "log_rv_lv_ratio":           0,   # RV/LV ratio: can be too small OR too large
    "rv_contamination":         +1,   # high RV in 2C is bad
}

# 3C-specific direction overrides:
#   boundary gradients are NEGATIVELY correlated with Dice in 3C
#   (high gradient → boundary placed at wrong tissue, despite image having an edge there)
_FEATURE_DIRECTIONS_3C: Dict[str, int] = {
    "boundary_gradient_lv":     +1,   # 3C rho=-0.502: high gradient → lower Dice
    "boundary_gradient_rv":     +1,   # 3C rho=-0.424: high gradient → lower Dice
    "rv_eccentricity":          +1,   # 3C rho=-0.457: over-elongated RV → worse
    "rv_lv_centroid_lateral":   -1,   # 3C rho=+0.481: more lateral → better
    "rv_solidity":              -1,   # 3C rho=+0.417: more solid → better
    "lv_compactness":           +1,   # 3C data: high compactness (neg_z) → better (rho=-0.21)
}

# 4C-specific direction overrides:
#   myo_solidity in 4C: symmetric is safer (thick-myo patients have high solidity but good Dice)
_FEATURE_DIRECTIONS_4C: Dict[str, int] = {
    "myo_solidity":              0,   # symmetric for 4C: thick-myo patients get less penalized
}

# 2C-specific direction overrides:
#   In 2C, lv_compactness rho=-0.322: high compactness = too-round LV = bad segmentation
#   (2C LV should be elongated, not circular)
_FEATURE_DIRECTIONS_2C: Dict[str, int] = {
    "lv_compactness":           +1,   # high compactness → too round → bad in 2C
    "myo_lv_ratio":              0,   # symmetric: both too-thin and too-thick penalized
}

# 2C-specific weights — calibrated to UKB empirical |Spearman rho|:
#   lv_compactness (-0.322), intensity_contrast (+0.277), lv_eccentricity (+0.213),
#   myo_lv_ratio (+0.212), log_lv_area_frac (-0.212)
#   NOISE: myo_enclosure_ratio (-0.052), boundary_gradient_lv (-0.064)
#   CONSTANT IN UKB 2C (useless): rv_contamination=0, boundary_gradient_rv=0
_FEATURE_WEIGHTS_2C: Dict[str, float] = {
    "lv_compactness":           2.5,   # rho=-0.322 (strongest 2C signal)
    "intensity_contrast":       2.0,   # rho=+0.277
    "lv_eccentricity":          1.5,   # rho=+0.213
    "myo_lv_ratio":             2.0,   # user policy: prioritize Myo thickness plausibility
    "log_lv_area_frac":         1.5,   # rho=-0.212
    "boundary_gradient_myo":    1.2,   # strengthen Myo contour quality
    "myo_enclosure_ratio":      1.5,   # user policy: prioritize Myo enclosure
    "boundary_gradient_lv":     0.5,   # rho=-0.064 (low signal)
    "rv_contamination":         1.0,   # class-presence priority (2C should keep RV near-zero)
    "boundary_gradient_rv":     0.0,   # constant=0 for 2C (no RV) → exclude
}


def _normalize_view(view_type: str) -> str:
    """Normalize view string to canonical 2-char form: '2ch'→'2c', '4CH'→'4c', etc."""
    v = re.sub(r"[^a-z0-9]", "", str(view_type).lower())
    _MAP = {"2ch": "2c", "3ch": "3c", "4ch": "4c"}
    return _MAP.get(v, v)


def _feature_names(view: str) -> List[str]:
    return _VIEW_FEATURES.get(view, _SHARED_FEATURES)


def _get_direction(feat_name: str, view: str) -> int:
    """Return penalization direction for a feature, with view-specific overrides."""
    if view == "3c" and feat_name in _FEATURE_DIRECTIONS_3C:
        return _FEATURE_DIRECTIONS_3C[feat_name]
    if view == "4c" and feat_name in _FEATURE_DIRECTIONS_4C:
        return _FEATURE_DIRECTIONS_4C[feat_name]
    if view == "2c" and feat_name in _FEATURE_DIRECTIONS_2C:
        return _FEATURE_DIRECTIONS_2C[feat_name]
    return _FEATURE_DIRECTIONS.get(feat_name, 0)


def _get_weight(feat_name: str, view: str) -> float:
    """Return importance weight for a feature, with view-specific overrides."""
    if view == "4c" and feat_name in _FEATURE_WEIGHTS_4C:
        return _FEATURE_WEIGHTS_4C[feat_name]
    if view == "3c" and feat_name in _FEATURE_WEIGHTS_3C:
        return _FEATURE_WEIGHTS_3C[feat_name]
    if view == "2c" and feat_name in _FEATURE_WEIGHTS_2C:
        return _FEATURE_WEIGHTS_2C[feat_name]
    return _FEATURE_WEIGHTS.get(feat_name, 1.0)


# ─── Distribution fitting ─────────────────────────────────────────────────────

@dataclass
class AnatomyStats:
    """Fitted source distribution for one view type."""
    view_type: str
    feature_names: List[str]
    medians: np.ndarray          # (n_features,)
    mads: np.ndarray             # (n_features,) scaled MAD
    weights: np.ndarray          # (n_features,) importance weights
    directions: np.ndarray       # (n_features,) -1/0/+1 penalization direction
    n_samples: int
    raw_features: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "view_type": self.view_type,
            "feature_names": self.feature_names,
            "medians": self.medians.tolist(),
            "mads": self.mads.tolist(),
            "weights": self.weights.tolist(),
            "directions": self.directions.tolist(),
            "n_samples": self.n_samples,
        }


def fit_source_stats(
    pred_dir: str,
    view_type: str,
    img_dir: Optional[str] = None,
    min_samples: int = 5,
) -> AnatomyStats:
    """
    Scan *_pred.png in pred_dir for the given view_type,
    compute anatomy features, and fit a robust distribution.
    img_dir defaults to pred_dir.
    """
    img_dir = img_dir or pred_dir
    view = _normalize_view(view_type)
    feat_names = _feature_names(view)

    pred_files = sorted(
        p for p in os.listdir(pred_dir)
        if p.endswith("_pred.png") and f"lax_{view}" in p
    )

    rows: List[List[float]] = []
    for fname in pred_files:
        stem = fname.replace("_pred.png", "")
        img_path = os.path.join(img_dir, f"{stem}_img.png")
        pred_path = os.path.join(pred_dir, fname)
        try:
            mask = read_mask(pred_path)
            img = np.array(Image.open(img_path).convert("L"))
            feats = compute_anatomy_features(mask, img, view)
            row = [feats.get(k, float("nan")) for k in feat_names]
            if any(np.isnan(v) for v in row):
                continue
            rows.append(row)
        except Exception:
            continue

    if len(rows) < min_samples:
        raise ValueError(
            f"Only {len(rows)} valid samples for view={view} in {pred_dir}."
        )

    arr = np.array(rows, dtype=float)
    medians = np.median(arr, axis=0)
    mads = np.median(np.abs(arr - medians), axis=0) * 1.4826
    mads = np.maximum(mads, 1e-6)

    weights = np.array(
        [_get_weight(k, view) for k in feat_names], dtype=float
    )
    directions = np.array(
        [_get_direction(k, view) for k in feat_names], dtype=float
    )

    return AnatomyStats(
        view_type=view,
        feature_names=feat_names,
        medians=medians,
        mads=mads,
        weights=weights,
        directions=directions,
        n_samples=len(rows),
        raw_features=arr,
    )


# ─── Hard anatomical constraint violations ───────────────────────────────────

def _hard_violations(
    mask: np.ndarray,
    view_type: str,
    feats: Dict[str, float],
) -> Tuple[float, List[str]]:
    """View-specific hard anatomical rules. Returns (penalty 0-60, violated names)."""
    view = _normalize_view(view_type)
    penalty = 0.0
    violated: List[str] = []

    rv_a = _area(mask, 1)
    lv_a = _area(mask, 3)
    myo_a = _area(mask, 2)

    # ── rules for all views ──────────────────────────────────────────────────
    # LV absent
    if lv_a < 30:
        penalty += 35.0
        violated.append("lv_absent")
    # Myo absent
    if myo_a < 20:
        penalty += 35.0
        violated.append("myo_absent")

    # Myo enclosure integrity priority:
    # treat severe LV enclosure loss as near-hard failure even when Myo exists.
    myo_enc = feats.get("myo_enclosure_ratio", float("nan"))
    if myo_a >= 20 and lv_a >= 30 and np.isfinite(myo_enc):
        severe_th = 0.35 if view in {"3c", "4c"} else 0.20
        low_th = 0.50 if view in {"3c", "4c"} else 0.30
        if myo_enc < severe_th:
            penalty += 20.0
            violated.append("myo_enclosure_critical")
        elif myo_enc < low_th:
            penalty += 10.0
            violated.append("myo_enclosure_low")

    # ── 2CH rules ────────────────────────────────────────────────────────────
    if view == "2c":
        # RV should not appear in 2CH
        fg_a = rv_a + myo_a + lv_a
        rv_frac = rv_a / max(fg_a, 1)
        if rv_frac > 0.10:
            penalty += min(25.0, rv_frac * 100)
            violated.append("rv_in_2ch")
        # LV should be elongated
        lv_ecc = feats.get("lv_eccentricity", float("nan"))
        if not np.isnan(lv_ecc) and lv_ecc < 0.40:
            penalty += 10.0
            violated.append("lv_not_elongated_2ch")

    # ── 3CH rules ────────────────────────────────────────────────────────────
    elif view == "3c":
        # RV should be present but small
        if rv_a < 20:
            penalty += 22.0
            violated.append("rv_absent_3ch")
        rv_lv = rv_a / max(lv_a, 1)
        if rv_lv > 2.0:
            penalty += 20.0
            violated.append("rv_dominant_3ch")

    # ── 4CH rules ────────────────────────────────────────────────────────────
    elif view == "4c":
        # RV should be present
        if rv_a < 30:
            penalty += 28.0
            violated.append("rv_absent_4ch")
        # RV/LV ratio sanity
        rv_lv = rv_a / max(lv_a, 1)
        if rv_lv < 0.05:
            penalty += 25.0
            violated.append("rv_critically_small_4ch")
        elif rv_lv > 4.0:
            penalty += 20.0
            violated.append("rv_critically_large_4ch")
        # RV and LV should be side-by-side (dy small)
        dy = feats.get("rv_lv_centroid_dy", float("nan"))
        if not np.isnan(dy) and dy > 0.35:
            penalty += 15.0
            violated.append("rv_lv_not_lateral_4ch")

    return float(min(penalty, 60.0)), violated


# ─── Scoring ──────────────────────────────────────────────────────────────────

def compute_anatomy_score(
    mask: np.ndarray,
    img: np.ndarray,
    view_type: str,
    stats: AnatomyStats,
) -> Tuple[float, Dict[str, float]]:
    """
    Score a single prediction against the fitted source statistics.

    Uses directional one-sided Mahalanobis: each feature is only penalized
    when it deviates in the anatomically "bad" direction (per _FEATURE_DIRECTIONS).

    Returns
    -------
    score  : float in [0, 100], higher = more anatomically plausible
    detail : dict with per-feature z-scores, raw values, mahal, violations
    """
    view = _normalize_view(view_type)
    feats = compute_anatomy_features(mask, img, view_type)

    feat_vals = np.array(
        [feats.get(k, float("nan")) for k in stats.feature_names], dtype=float
    )
    nan_mask = np.isnan(feat_vals)

    # Robust z-scores, capped at ±5σ
    z = np.where(
        nan_mask,
        0.0,
        np.clip((feat_vals - stats.medians) / stats.mads, -5.0, 5.0),
    )

    # Directional penalty per feature
    # directions: -1 → penalize z<0, +1 → penalize z>0, 0 → penalize both
    dirs = stats.directions if hasattr(stats, "directions") else np.zeros_like(z)
    penalty_terms = np.where(
        dirs == -1, np.maximum(0.0, -z) ** 2,      # low value is bad
        np.where(
            dirs == +1, np.maximum(0.0, z) ** 2,   # high value is bad
            z ** 2,                                  # symmetric
        ),
    )

    # Cap each feature's penalty at z=3 equivalent (=9.0) to prevent any single
    # outlier feature from dominating the score (handles legitimate patient-anatomy variation)
    penalty_terms = np.minimum(penalty_terms, 9.0)

    # Weighted directional Mahalanobis
    w = stats.weights.copy()
    w[nan_mask] = 0.0
    w_sum = max(w.sum(), 1e-9)
    weighted_mahal = float(np.sqrt((w * penalty_terms).sum() / w_sum))

    base_score = float(100.0 * np.exp(-0.5 * weighted_mahal))

    # Hard violations
    hard_penalty, violated = _hard_violations(mask, view, feats)
    score = float(max(0.0, base_score - hard_penalty))

    # Build detail dict
    detail: Dict[str, float] = {}
    for i, k in enumerate(stats.feature_names):
        detail[f"z_{k}"] = float(z[i])
        detail[f"val_{k}"] = float(feat_vals[i])
    detail["mahal"] = weighted_mahal
    detail["base_score"] = base_score
    detail["hard_penalty"] = hard_penalty
    detail["hard_violations"] = violated          # type: ignore[assignment]
    detail["n_valid_features"] = float((~nan_mask).sum())

    return score, detail


# ─── Rule-based diagnosis ─────────────────────────────────────────────────────

# z-score threshold for a feature to be considered anomalous
_DIAG_Z_THRESH = 2.0

_DIAG_RULES: Dict[str, List[Tuple[str, float, str]]] = {
    # (feature, sign*threshold, diagnosis_tag)
    # sign > 0: flag if z > threshold; sign < 0: flag if z < -threshold
    "2c": [
        ("rv_contamination",        +2.0, "rv_present_in_2ch"),
        ("myo_enclosure_ratio",     -2.0, "myo_insufficient_enclosure"),
        ("myo_lv_ratio",            -2.0, "myo_too_thin"),
        ("myo_lv_ratio",            +2.5, "myo_too_thick"),
        ("lv_eccentricity",         -2.0, "lv_not_elongated"),
        ("boundary_gradient_lv",    -2.0, "poor_boundary_lv"),
        ("boundary_gradient_myo",   -2.0, "poor_boundary_myo"),
        ("intensity_contrast",      -2.0, "poor_intensity_separation"),
    ],
    "3c": [
        ("log_rv_lv_ratio",         -2.5, "rv_too_small"),
        ("log_rv_lv_ratio",         +2.5, "rv_too_large"),
        ("myo_enclosure_ratio",     -2.0, "myo_insufficient_enclosure"),
        ("myo_lv_ratio",            -2.0, "myo_too_thin"),
        ("myo_lv_ratio",            +2.5, "myo_too_thick"),
        ("rv_lv_centroid_lateral",  -2.0, "rv_not_lateral"),
        ("rv_eccentricity",         +2.0, "rv_not_crescent"),  # high ecc → bad in 3C
        ("rv_solidity",             -2.0, "rv_fragmented"),
        ("boundary_gradient_lv",    +2.0, "poor_boundary_lv"),  # high gradient in 3C → wrong position
        ("boundary_gradient_rv",    +2.0, "poor_boundary_rv"),
        ("intensity_contrast",      -2.0, "poor_intensity_separation"),
    ],
    "4c": [
        ("log_rv_lv_ratio",         -2.5, "rv_too_small"),
        ("log_rv_lv_ratio",         +2.5, "rv_too_large"),
        ("myo_enclosure_ratio",     -2.0, "myo_insufficient_enclosure"),
        ("myo_lv_ratio",            -2.0, "myo_too_thin"),
        ("myo_lv_ratio",            +2.5, "myo_too_thick"),
        ("rv_lv_centroid_dy",       -2.0, "rv_lv_vertical_misalign"),  # low dy = bad (rho+0.47)
        ("rv_lv_centroid_lateral",  -2.0, "rv_not_lateral"),
        ("rv_solidity",             -2.0, "rv_fragmented"),
        ("boundary_gradient_rv",    -2.0, "poor_boundary_rv"),
        ("boundary_gradient_lv",    -2.0, "poor_boundary_lv"),
        ("intensity_contrast",      -2.0, "poor_intensity_separation"),
        ("log_lv_area_frac",        -2.0, "lv_too_small"),
        ("log_lv_area_frac",        +2.0, "lv_too_large"),
    ],
}


def diagnose_from_score(
    detail: Dict,
    view_type: str,
    top_k: int = 4,
) -> List[str]:
    """
    Map extreme feature z-scores → actionable repair tags.
    Returns up to `top_k` diagnoses sorted by severity (|z|).
    Hard violations are prepended.

    Compatible with the existing planner/executor repair tag vocabulary.
    """
    view = _normalize_view(view_type)
    diagnoses: List[Tuple[float, str]] = []

    # Hard violations first (highest priority)
    for v in (detail.get("hard_violations") or []):
        diagnoses.append((99.0, v))

    # Z-score driven
    rules = _DIAG_RULES.get(view, _DIAG_RULES.get("4c", []))
    for feat, signed_thr, tag in rules:
        z = detail.get(f"z_{feat}", 0.0)
        if isinstance(z, float) and not np.isnan(z):
            if signed_thr > 0 and z > signed_thr:
                diagnoses.append((abs(z), tag))
            elif signed_thr < 0 and z < signed_thr:
                diagnoses.append((abs(z), tag))

    # Deduplicate + sort by severity
    seen: set = set()
    sorted_diag: List[str] = []
    for _, tag in sorted(diagnoses, key=lambda x: -x[0]):
        if tag not in seen:
            seen.add(tag)
            sorted_diag.append(tag)
        if len(sorted_diag) >= top_k:
            break

    return sorted_diag


def _direction_label(direction: int) -> str:
    if direction < 0:
        return "penalize_low"
    if direction > 0:
        return "penalize_high"
    return "symmetric"


def _directional_anomaly_strength(z: float, direction: int) -> float:
    if not np.isfinite(z):
        return 0.0
    if direction < 0:
        return max(0.0, -float(z))
    if direction > 0:
        return max(0.0, float(z))
    return abs(float(z))


def build_anatomy_llm_summary(
    score: float,
    detail: Dict[str, Any],
    view_type: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Build a compact, structured summary for LLM consumption.

    The summary intentionally avoids dumping all raw feature values. It exposes:
    - global quality (`score`, `risk_level`)
    - hard anatomical violations
    - top directional anomalies from per-feature z-scores
    - rule-based action tags from `diagnose_from_score`
    """
    view = _normalize_view(view_type)
    feat_names = _feature_names(view)
    hard_violations_raw = detail.get("hard_violations", [])
    if isinstance(hard_violations_raw, list):
        hard_violations = [str(x).strip() for x in hard_violations_raw if str(x).strip()]
    else:
        hard_violations = []

    try:
        score_val = float(score)
    except Exception:
        score_val = 0.0
    score_val = max(0.0, min(100.0, score_val))

    hard_count = len(hard_violations)
    if score_val < 35.0 or hard_count >= 2:
        risk_level = "high"
    elif score_val < 60.0 or hard_count >= 1:
        risk_level = "medium"
    else:
        risk_level = "low"

    anomalies: List[Dict[str, Any]] = []
    for feat in feat_names:
        z_key = f"z_{feat}"
        if z_key not in detail:
            continue
        try:
            z_val = float(detail.get(z_key))
        except Exception:
            continue
        if not np.isfinite(z_val):
            continue
        direction = int(_get_direction(feat, view))
        strength = _directional_anomaly_strength(z_val, direction)
        if strength <= 0.0:
            continue
        anomalies.append(
            {
                "feature": feat,
                "z": float(z_val),
                "direction": _direction_label(direction),
                "is_bad_direction": bool(strength > 0.0),
                "strength": float(strength),
            }
        )

    anomalies.sort(key=lambda x: (float(x["strength"]), abs(float(x["z"]))), reverse=True)
    top_anomalies = anomalies[: max(1, int(top_k))]
    for item in top_anomalies:
        # Keep prompt payload compact; sorting already consumed this field.
        item.pop("strength", None)

    action_tags = diagnose_from_score(detail, view, top_k=max(1, int(top_k)))

    return {
        "view": view,
        "score": float(score_val),
        "risk_level": risk_level,
        "hard_violations": hard_violations,
        "top_anomalies": top_anomalies,
        "action_tags": list(action_tags),
    }


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_stats(stats_dict: Dict[str, AnatomyStats], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(stats_dict, f)


def load_stats(path: str) -> Dict[str, AnatomyStats]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── CLI: fit and save ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--views", default="2c,3c,4c")
    ap.add_argument("--output", default="anatomy_stats.pkl")
    ap.add_argument("--extra-pred-dir", default=None,
                    help="Fallback dir for views not found in primary (e.g. UKB for 2c/3c)")
    args = ap.parse_args()

    stats_dict: Dict[str, AnatomyStats] = {}
    for view in [v.strip() for v in args.views.split(",")]:
        for d in filter(None, [args.pred_dir, args.extra_pred_dir]):
            try:
                s = fit_source_stats(d, view_type=view)
                stats_dict[view] = s
                print(f"  {view}: n={s.n_samples}")
                for k, med, mad, dr in zip(s.feature_names, s.medians, s.mads, s.directions):
                    dir_str = {-1: "penalize_low", 0: "symmetric", 1: "penalize_high"}.get(int(dr), "?")
                    print(f"    {k:35s} median={med:.4f}  mad={mad:.4f}  dir={dir_str}")
                break
            except ValueError as e:
                print(f"  {view} in {d}: {e}")

    save_stats(stats_dict, args.output)
    print(f"\nSaved → {args.output}")
