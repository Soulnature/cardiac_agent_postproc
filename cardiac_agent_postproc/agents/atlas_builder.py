from __future__ import annotations
import os
import time
from typing import Dict, Any, List
import numpy as np
from ..io_utils import list_frames, read_mask
from ..view_utils import infer_view_type
from ..atlas import build_atlas, save_atlas, load_atlas

class AtlasBuilderAgent:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self, source_dir: str, atlas_path: str) -> dict:
        if os.path.exists(atlas_path):
            # load cache
            return load_atlas(atlas_path)

        frames = list_frames(source_dir)
        gt_masks = []
        views = []
        for fr in frames:
            if "gt" not in fr:
                continue
            m = read_mask(fr["gt"])
            v, _ = infer_view_type(os.path.basename(fr["stem"]), m, rv_ratio_threshold=float(self.cfg["view_inference"]["rv_ratio_2ch_threshold"]))
            gt_masks.append(m)
            views.append(v)

        if len(gt_masks) == 0:
            raise RuntimeError(f"No *_gt.png found in SOURCE_DIR: {source_dir}")

        atlas = build_atlas(gt_masks, views, self.cfg)
        save_atlas(atlas_path, atlas)
        return atlas
