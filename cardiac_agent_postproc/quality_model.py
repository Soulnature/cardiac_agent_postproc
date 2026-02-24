from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import os
import warnings
import joblib
import numpy as np
import cv2
from scipy import ndimage as ndi

from .geom import boundary_4n, label_cc, compactness, eccentricity_largest_cc
from .io_utils import to_grayscale_float

# ==========================================
# Constants Definition (Avoiding Magic Numbers)
# ==========================================
CLS_BG = 0
CLS_RV = 1
CLS_MYO = 2
CLS_LV = 3
EPSILON = 1e-6  # Small value to prevent division by zero

QUALITY_FEATURE_NAMES: List[str] = [
    "nonbg_ratio",
    "area_frac_rv",
    "area_frac_myo",
    "area_frac_lv",
    "cc_count_rv",
    "cc_count_myo",
    "cc_count_lv",
    "hole_frac_lv",
    "hole_frac_myo",
    "touch_ratio",
    "valve_leak_ratio",
    "boundary_over_area",
    "thickness_mean",
    "thickness_std",
    "thickness_cv",
    "lv_compactness",
    "lv_eccentricity",
    "intensity_lv_mean",
    "intensity_myo_mean",
    "intensity_bg_mean",
    "intensity_lv_bg_diff",
    "intensity_myo_bg_diff",
    "edge_distance_mean",
    "edge_distance_std",
    "rv_missing",
    "myo_missing",
    "lv_missing",
    "view_is_2ch",
    "view_is_3ch",
    "view_is_4ch",
    # --- Newly Added Features ---
    "rv_false_positive",     # RV should not appear in 2CH view
    "myo_coverage_lv",       # Coverage ratio of myocardium on LV boundary
    "lv_length_width_ratio", # LV length-to-width ratio
    "rv_lv_area_ratio"       # Area ratio between RV and LV
]


def _area(mask: np.ndarray, cls: int) -> int:
    return int(np.count_nonzero(mask == cls))


def _component_count(mask: np.ndarray, cls: int) -> int:
    lab, n = label_cc(mask == cls)
    return int(n)


def _hole_fraction(mask: np.ndarray, cls: int) -> float:
    region = (mask == cls)
    if not region.any():
        return 0.0
    filled = ndi.binary_fill_holes(region)
    holes = np.count_nonzero(filled & (~region))
    return float(holes / max(1, region.sum()))


def _touch_ratio(mask: np.ndarray) -> float:
    rv = (mask == CLS_RV)
    lv = (mask == CLS_LV)
    if rv.sum() == 0 or lv.sum() == 0:
        return 0.0
    dilated = ndi.binary_dilation(rv, iterations=1)
    touch = np.count_nonzero(dilated & lv)
    return float(touch / max(1, lv.sum()))


def _myo_coverage_ratio(mask: np.ndarray) -> float:
    """Calculate the coverage ratio of myocardium on the LV boundary to detect myocardial gaps."""
    lv = (mask == CLS_LV)
    myo = (mask == CLS_MYO)
    if not lv.any() or not myo.any():
        return 0.0

    lv_boundary = boundary_4n(lv)
    if lv_boundary.sum() == 0:
        return 0.0

    # Dilate the myocardium by 1 pixel and check how many pixels overlap with the LV boundary
    dilated_myo = ndi.binary_dilation(myo, iterations=1)
    covered = np.count_nonzero(lv_boundary & dilated_myo)
    return float(covered / max(1, lv_boundary.sum()))


def _lv_length_width_ratio(mask: np.ndarray) -> float:
    """Calculate the length-to-width ratio of the LV."""
    lv = (mask == CLS_LV).astype(np.uint8)
    if not lv.any():
        return 0.0
    contours, _ = cv2.findContours(lv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 5:  # minAreaRect requires at least 5 points
        return 0.0
    
    _, (w, h), _ = cv2.minAreaRect(largest_contour)
    major = max(w, h)
    minor = min(w, h)
    return float(major / max(EPSILON, minor))


def _valve_leak_ratio(mask: np.ndarray) -> float:
    """Dynamically calculate the valve leak ratio using PCA (Principal Component Analysis)."""
    lv = (mask == CLS_LV)
    if not lv.any():
        return 0.0
    
    ys, xs = np.nonzero(lv)
    pts = np.column_stack((xs, ys)).astype(np.float32)
    if len(pts) < 5:
        return 0.0

    # 1. Use PCA to calculate the major axis direction of the LV
    mean_pt, eigenvectors = cv2.PCACompute(pts, mean=None)
    main_axis = eigenvectors[0]  # [dx, dy]

    # 2. Ensure the major axis vector points to the apex 
    # (Assuming the image base is at the top with smaller Y, and apex at the bottom with larger Y)
    if main_axis[1] < 0:
        main_axis = -main_axis

    # 3. Project all LV points onto the major axis
    projections = np.dot(pts - mean_pt.flatten(), main_axis)
    min_proj, max_proj = projections.min(), projections.max()

    # 4. Define the valve plane: 8% below the basal end (min_proj)
    length = max_proj - min_proj
    valve_threshold = min_proj + 0.08 * length

    # 5. Calculate the proportion of pixels in the entire mask that extend beyond this plane (leakage)
    fg = mask > 0
    f_ys, f_xs = np.nonzero(fg)
    f_pts = np.column_stack((f_xs, f_ys)).astype(np.float32)
    
    f_projections = np.dot(f_pts - mean_pt.flatten(), main_axis)
    leak_count = np.count_nonzero(f_projections < valve_threshold)

    return float(leak_count / max(1, len(f_pts)))


def _edge_distance_stats(mask: np.ndarray, gray: np.ndarray, threshold_pct: float = 85.0) -> Tuple[float, float]:
    """Calculate boundary distance statistics. Optimized to directly receive a pre-computed grayscale image."""
    boundary = boundary_4n(mask > 0)
    if boundary.sum() == 0:
        return 0.0, 0.0
    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag, threshold_pct)
    edges = mag > thr
    
    dt = ndi.distance_transform_edt(~edges)
    dists = dt[boundary]
    return float(np.mean(dists)), float(np.std(dists))


def _thickness_stats(mask: np.ndarray) -> Tuple[float, float, float]:
    myo = mask == CLS_MYO
    lv = mask == CLS_LV
    if myo.sum() == 0 or lv.sum() == 0:
        return 0.0, 0.0, 0.0
    
    lv_boundary = boundary_4n(lv)
    if lv_boundary.sum() == 0:
        return 0.0, 0.0, 0.0
        
    dist = ndi.distance_transform_edt(~lv_boundary)
    outer = boundary_4n(myo) & (~lv_boundary)
    if outer.sum() == 0:
        return 0.0, 0.0, 0.0
        
    samples = dist[outer]
    mean = float(samples.mean())
    std = float(samples.std())
    cv = float(std / max(EPSILON, mean)) # Unified defensive programming
    return mean, std, cv


def _intensity_stats(mask: np.ndarray, gray: np.ndarray, cls: int) -> float:
    """Calculate the average intensity of a region. Optimized to directly receive a pre-computed grayscale image."""
    region = gray[mask == cls]
    if region.size == 0:
        return 0.0
    return float(region.mean())


def extract_quality_features(mask: np.ndarray, img: np.ndarray, view_type: str) -> Dict[str, float]:
    h, w = mask.shape
    total = h * w
    nonbg = np.count_nonzero(mask)
    features: Dict[str, float] = {}
    
    # Pre-compute the grayscale image once to avoid performance bottlenecks from repeated calculations
    gray_img = to_grayscale_float(img)

    features["nonbg_ratio"] = float(nonbg / max(1, total))
    features["area_frac_rv"] = float(_area(mask, CLS_RV) / max(1, total))
    features["area_frac_myo"] = float(_area(mask, CLS_MYO) / max(1, total))
    features["area_frac_lv"] = float(_area(mask, CLS_LV) / max(1, total))

    features["cc_count_rv"] = float(_component_count(mask, CLS_RV))
    features["cc_count_myo"] = float(_component_count(mask, CLS_MYO))
    features["cc_count_lv"] = float(_component_count(mask, CLS_LV))

    features["hole_frac_lv"] = float(_hole_fraction(mask, CLS_LV))
    features["hole_frac_myo"] = float(_hole_fraction(mask, CLS_MYO))

    features["touch_ratio"] = _touch_ratio(mask)
    features["valve_leak_ratio"] = _valve_leak_ratio(mask)

    boundary_pixels = boundary_4n(mask > 0)
    features["boundary_over_area"] = float(boundary_pixels.sum() / max(1, nonbg))

    thickness_mean, thickness_std, thickness_cv = _thickness_stats(mask)
    features["thickness_mean"] = thickness_mean
    features["thickness_std"] = thickness_std
    features["thickness_cv"] = thickness_cv

    lv_region = mask == CLS_LV
    features["lv_compactness"] = float(compactness(lv_region))
    features["lv_eccentricity"] = float(eccentricity_largest_cc(lv_region))

    # Pass the pre-computed gray_img
    lv_mean = _intensity_stats(mask, gray_img, CLS_LV)
    myo_mean = _intensity_stats(mask, gray_img, CLS_MYO)
    bg_mean = _intensity_stats(mask, gray_img, CLS_BG)
    features["intensity_lv_mean"] = lv_mean
    features["intensity_myo_mean"] = myo_mean
    features["intensity_bg_mean"] = bg_mean
    features["intensity_lv_bg_diff"] = lv_mean - bg_mean
    features["intensity_myo_bg_diff"] = myo_mean - bg_mean

    # Pass the pre-computed gray_img
    edge_mean, edge_std = _edge_distance_stats(mask, gray_img)
    features["edge_distance_mean"] = edge_mean
    features["edge_distance_std"] = edge_std

    features["rv_missing"] = 1.0 if _area(mask, CLS_RV) == 0 else 0.0
    features["myo_missing"] = 1.0 if _area(mask, CLS_MYO) == 0 else 0.0
    features["lv_missing"] = 1.0 if _area(mask, CLS_LV) == 0 else 0.0

    features["view_is_2ch"] = 1.0 if view_type == "2ch" else 0.0
    features["view_is_3ch"] = 1.0 if view_type == "3ch" else 0.0
    features["view_is_4ch"] = 1.0 if view_type == "4ch" else 0.0

    # --- Newly Added Features ---
    features["rv_false_positive"] = 1.0 if (view_type == "2ch" and features["rv_missing"] == 0.0) else 0.0
    features["myo_coverage_lv"] = _myo_coverage_ratio(mask)
    features["lv_length_width_ratio"] = _lv_length_width_ratio(mask)
    
    area_rv = _area(mask, CLS_RV)
    area_lv = _area(mask, CLS_LV)
    features["rv_lv_area_ratio"] = float(area_rv / max(1, area_lv))

    # Ensure all expected features are returned
    for name in QUALITY_FEATURE_NAMES:
        if name not in features:
            features[name] = 0.0
    return features


@dataclass
class QualityScorer:
    model_path: str
    model: Any
    feature_names: List[str]
    classifier: Any = None
    threshold: float = 0.5
    use_classifier: bool = False

    @classmethod
    def load(cls, path: str) -> "QualityScorer" | None:
        if not path or not os.path.exists(path):
            return None
        try:
            from sklearn.exceptions import InconsistentVersionWarning
        except Exception:
            InconsistentVersionWarning = Warning  # type: ignore[assignment]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InconsistentVersionWarning)
                payload = joblib.load(path)
        except Exception as e:
            print(f"[QualityScorer] Failed to load model '{path}': {e}")
            return None
            
        model = payload.get("model") if isinstance(payload, dict) else payload
        feature_names = payload.get("feature_names", QUALITY_FEATURE_NAMES) if isinstance(payload, dict) else QUALITY_FEATURE_NAMES
        if model is None:
            return None
            
        classifier = payload.get("classifier") if isinstance(payload, dict) else None
        threshold = float(payload.get("threshold", 0.5)) if isinstance(payload, dict) else 0.5
        use_classifier = bool(payload.get("use_classifier", False)) if isinstance(payload, dict) else False
        
        return cls(
            model_path=path,
            model=model,
            feature_names=feature_names,
            classifier=classifier,
            threshold=threshold,
            use_classifier=use_classifier,
        )

    @property
    def has_classifier(self) -> bool:
        """Returns True if any classify() strategy is available."""
        return self.classifier is not None or self.model is not None

    def score(self, mask: np.ndarray, img: np.ndarray, view_type: str) -> Tuple[float, Dict[str, float]]:
        feats = extract_quality_features(mask, img, view_type)
        vec = np.array([feats[name] for name in self.feature_names], dtype=np.float32).reshape(1, -1)
        pred = float(self.model.predict(vec)[0])
        pred = float(max(0.0, min(1.0, pred)))
        return pred, feats

    def classify(self, mask: np.ndarray, img: np.ndarray, view_type: str) -> Tuple[bool, float, Dict[str, float]]:
        """Classify a sample as bad or good."""
        feats = extract_quality_features(mask, img, view_type)
        vec = np.array([feats[name] for name in self.feature_names], dtype=np.float32).reshape(1, -1)

        if self.use_classifier and self.classifier is not None:
            prob = float(self.classifier.predict_proba(vec)[0, 1])
            is_bad = prob >= self.threshold
            return is_bad, prob, feats

        # In regressor mode, directly use the predicted Dice to calculate Bad Probability
        pred_dice = float(self.model.predict(vec)[0])
        pred_dice = float(max(0.0, min(1.0, pred_dice)))
        is_bad = pred_dice < self.threshold
        
        # Optimized pseudo-probability mapping: lower Dice results in higher bad_prob; bad_prob is 0 when Dice is 1
        bad_prob = 1.0 - pred_dice
        return is_bad, bad_prob, feats