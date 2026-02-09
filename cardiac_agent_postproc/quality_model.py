from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import os
import joblib
import numpy as np
import cv2
from scipy import ndimage as ndi

from .geom import boundary_4n, label_cc, compactness, eccentricity_largest_cc
from .io_utils import to_grayscale_float

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
    rv = (mask == 1)
    lv = (mask == 3)
    if rv.sum() == 0 or lv.sum() == 0:
        return 0.0
    dilated = ndi.binary_dilation(rv, iterations=1)
    touch = np.count_nonzero(dilated & lv)
    return float(touch / max(1, lv.sum()))


def _valve_plane(mask: np.ndarray) -> int | None:
    fg = mask > 0
    if not fg.any():
        return None
    ys, xs = np.nonzero(fg)
    y0 = ys.min()
    y1 = ys.max()
    height = max(1, y1 - y0 + 1)
    return int(round(y0 + 0.08 * height))


def _valve_leak_ratio(mask: np.ndarray) -> float:
    vp = _valve_plane(mask)
    if vp is None:
        return 0.0
    nonbg = np.count_nonzero(mask)
    if nonbg == 0:
        return 0.0
    leak = np.count_nonzero((mask > 0) & (np.arange(mask.shape[0])[:, None] < vp))
    return float(leak / nonbg)


def _edge_distance_stats(mask: np.ndarray, img: np.ndarray) -> Tuple[float, float]:
    boundary = boundary_4n(mask > 0)
    if boundary.sum() == 0:
        return 0.0, 0.0
    gray = to_grayscale_float(img)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag, 85)
    edges = mag > thr
    dt = ndi.distance_transform_edt(~edges)
    dists = dt[boundary]
    return float(np.mean(dists)), float(np.std(dists))


def _thickness_stats(mask: np.ndarray) -> Tuple[float, float, float]:
    myo = mask == 2
    lv = mask == 3
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
    cv = float(std / (mean + 1e-6))
    return mean, std, cv


def _intensity_stats(mask: np.ndarray, img: np.ndarray, cls: int) -> float:
    gray = to_grayscale_float(img)
    region = gray[mask == cls]
    if region.size == 0:
        return 0.0
    return float(region.mean())


def extract_quality_features(mask: np.ndarray, img: np.ndarray, view_type: str) -> Dict[str, float]:
    h, w = mask.shape
    total = h * w
    nonbg = np.count_nonzero(mask)
    features: Dict[str, float] = {}

    features["nonbg_ratio"] = float(nonbg / max(1, total))
    features["area_frac_rv"] = float(_area(mask, 1) / max(1, total))
    features["area_frac_myo"] = float(_area(mask, 2) / max(1, total))
    features["area_frac_lv"] = float(_area(mask, 3) / max(1, total))

    features["cc_count_rv"] = float(_component_count(mask, 1))
    features["cc_count_myo"] = float(_component_count(mask, 2))
    features["cc_count_lv"] = float(_component_count(mask, 3))

    features["hole_frac_lv"] = float(_hole_fraction(mask, 3))
    features["hole_frac_myo"] = float(_hole_fraction(mask, 2))

    features["touch_ratio"] = _touch_ratio(mask)
    features["valve_leak_ratio"] = _valve_leak_ratio(mask)

    boundary_pixels = boundary_4n(mask > 0)
    features["boundary_over_area"] = float(boundary_pixels.sum() / max(1, nonbg))

    thickness_mean, thickness_std, thickness_cv = _thickness_stats(mask)
    features["thickness_mean"] = thickness_mean
    features["thickness_std"] = thickness_std
    features["thickness_cv"] = thickness_cv

    lv_region = mask == 3
    features["lv_compactness"] = float(compactness(lv_region))
    features["lv_eccentricity"] = float(eccentricity_largest_cc(lv_region))

    lv_mean = _intensity_stats(mask, img, 3)
    myo_mean = _intensity_stats(mask, img, 2)
    bg_mean = _intensity_stats(mask, img, 0)
    features["intensity_lv_mean"] = lv_mean
    features["intensity_myo_mean"] = myo_mean
    features["intensity_bg_mean"] = bg_mean
    features["intensity_lv_bg_diff"] = lv_mean - bg_mean
    features["intensity_myo_bg_diff"] = myo_mean - bg_mean

    edge_mean, edge_std = _edge_distance_stats(mask, img)
    features["edge_distance_mean"] = edge_mean
    features["edge_distance_std"] = edge_std

    features["rv_missing"] = 1.0 if _area(mask, 1) == 0 else 0.0
    features["myo_missing"] = 1.0 if _area(mask, 2) == 0 else 0.0
    features["lv_missing"] = 1.0 if _area(mask, 3) == 0 else 0.0

    features["view_is_2ch"] = 1.0 if view_type == "2ch" else 0.0
    features["view_is_3ch"] = 1.0 if view_type == "3ch" else 0.0
    features["view_is_4ch"] = 1.0 if view_type == "4ch" else 0.0

    # ensure all expected features exist
    for name in QUALITY_FEATURE_NAMES:
        if name not in features:
            features[name] = 0.0
    return features


@dataclass
class QualityScorer:
    model_path: str
    model: Any
    feature_names: List[str]

    @classmethod
    def load(cls, path: str) -> "QualityScorer" | None:
        if not path or not os.path.exists(path):
            return None
        payload = joblib.load(path)
        model = payload.get("model") if isinstance(payload, dict) else payload
        feature_names = payload.get("feature_names", QUALITY_FEATURE_NAMES) if isinstance(payload, dict) else QUALITY_FEATURE_NAMES
        if model is None:
            return None
        return cls(model_path=path, model=model, feature_names=feature_names)

    def score(self, mask: np.ndarray, img: np.ndarray, view_type: str) -> Tuple[float, Dict[str, float]]:
        feats = extract_quality_features(mask, img, view_type)
        vec = np.array([feats[name] for name in self.feature_names], dtype=np.float32).reshape(1, -1)
        pred = float(self.model.predict(vec)[0])
        pred = float(max(0.0, min(1.0, pred)))
        return pred, feats
