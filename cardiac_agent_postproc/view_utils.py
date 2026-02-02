from __future__ import annotations
import os
import numpy as np

def infer_view_type(filename: str, mask: np.ndarray, rv_ratio_threshold: float=0.02) -> tuple[str, str]:
    # returns (view_type, reason)
    low = filename.lower()
    if "2ch" in low:
        return "2ch", "filename"
    if "3ch" in low:
        return "3ch", "filename"
    if "4ch" in low:
        return "4ch", "filename"

    nonbg = np.count_nonzero(mask)
    rv = np.count_nonzero(mask == 1)
    rv_ratio = rv / max(1, nonbg)
    if rv_ratio < rv_ratio_threshold:
        return "2ch", f"inferred(rv_ratio={rv_ratio:.4f}<{rv_ratio_threshold})"
    return "4ch", f"inferred(rv_ratio={rv_ratio:.4f}>={rv_ratio_threshold})"
