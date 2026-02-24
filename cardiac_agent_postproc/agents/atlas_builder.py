from __future__ import annotations
import os
import time
from typing import Dict, Any, List
import numpy as np
from ..io_utils import list_frames, read_mask, read_image
from ..view_utils import infer_view_type
from ..atlas import build_atlas, save_atlas, load_atlas
class AtlasBuilderAgent:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self, source_dir: str, atlas_path: str) -> dict:
        
        # Always build fresh if "Self-Imitation" is desired (implied by user req)
        # But for speed, we can check basic existence. 
        # User wants to "refer to good labels as atlas". This implies dynamic rebuild.
        # Let's force rebuild or at least check properly.
        
        frames = list_frames(source_dir)
        good_masks = []
        good_zs = []
        views = []
        
        print("[AtlasBuilder] Building Self-Imitation Atlas from Predicted Labels (RQS no longer evaluated).")

        count_total = 0
        count_accepted = 0

        for fr in frames:
            count_total += 1
            # Use PREDICTION, not GT (User Req: "predict good label -> atlas")
            m = read_mask(fr["pred"])
            img = read_image(fr["img"])
            v, _ = infer_view_type(os.path.basename(fr["stem"]), m, rv_ratio_threshold=float(self.cfg["view_inference"]["rv_ratio_2ch_threshold"]))
            
            good_masks.append(m)
            views.append(v)

            # Parse slice index (User Req: filename "..._019")
            try:
                # stem: "205_original_lax_4c_019" -> 19
                z_str = fr["stem"].split("_")[-1]
                z_val = float(int(z_str))
            except:
                z_val = 0.0

            # Normalize: assume max 64 slices for normalization stability (0..1)
            z_norm = min(1.0, z_val / 64.0)
            good_zs.append(z_norm)

            count_accepted += 1
            
        print(f"[AtlasBuilder] Accepted {count_accepted}/{count_total} masks for Atlas.")

        if len(good_masks) == 0:
            print("[AtlasBuilder] WARNING: No good masks found! Falling back to ALL masks to avoid crash.")
            for fr in frames:
                m = read_mask(fr["pred"])
                v, _ = infer_view_type(os.path.basename(fr["stem"]), m, rv_ratio_threshold=float(self.cfg["view_inference"]["rv_ratio_2ch_threshold"]))
                good_masks.append(m)
                views.append(v)
                try:
                    z_str = fr["stem"].split("_")[-1]
                    z_val = float(int(z_str))
                except:
                    z_val = 0.0
                good_zs.append(min(1.0, z_val/64.0))

        atlas = build_atlas(good_masks, views, good_zs, self.cfg)
        save_atlas(atlas_path, atlas)
        return atlas
