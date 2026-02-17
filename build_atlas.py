import os
import re
import cv2
import yaml
import numpy as np
from tqdm import tqdm
from cardiac_agent_postproc.atlas import build_atlas, save_atlas
from cardiac_agent_postproc.io_utils import read_mask

def main():
    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)

    source_dir = cfg["paths"]["source_dir"]
    print(f"Scanning {source_dir}...")

    gt_masks = []
    view_types = []
    z_values = [] # normalized Z (0..1)
    
    # Regex to capture view and slice index
    # validation/210_original_lax_4c_029.png -> view=4ch
    # We need to parse view type from filename.
    # Pattern: {id}_original_{view_code}_{slice}.png
    # view_code: lax_4c -> 4ch, lax_3c -> 3ch, lax_2c -> 2ch, sax -> sax
    
    # We only care about GT files: {stem}_gt.png
    # But files in source_dir are likely predictions or original inputs?
    # run_batch_repair says: gt_path = os.path.join(source_dir, f"{ctx.stem}_gt.png")
    
    files = os.listdir(source_dir)
    gt_files = [f for f in files if f.endswith("_gt.png")]
    print(f"Found {len(gt_files)} GT masks.")
    
    # Pre-parse to group by patient for Z-normalization?
    # For now, just assume Z=0.5 if we can't determine it easily, or parse from filename?
    # Filename: 210_original_lax_4c_029_gt.png
    # The last number is slice index relative to temporal frame? No, it's 3D slice index?
    # Actually MnM2 filenames are messy.
    # Let's try to extract view type at least.

    for f in tqdm(gt_files):
        path = os.path.join(source_dir, f)
        mask = read_mask(path)
        if mask is None: continue
        
        # Parse view
        # 210_original_lax_4c_029_gt.png
        if "_lax_4c_" in f:
            view = "4ch"
        elif "_lax_3c_" in f:
            view = "3ch"
        elif "_lax_2c_" in f:
            view = "2ch"
        elif "_sax_" in f:
            view = "sax"
        else:
            continue
            
        gt_masks.append(mask)
        view_types.append(view)
        z_values.append(0.5) # Placeholder
        
    print("Building atlas...")
    atlas = build_atlas(gt_masks, view_types, z_values, cfg)
    
    out_path = "shape_atlas.pkl"
    save_atlas(out_path, atlas)
    print(f"Saved atlas to {out_path} with {len(atlas)} entries.")
    print(f"Keys: {list(atlas.keys())}")

if __name__ == "__main__":
    main()
