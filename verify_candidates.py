import sys
import os
import yaml
import numpy as np
import cv2
sys.path.append(os.getcwd())

from cardiac_agent_postproc.io_utils import read_mask, read_image
from cardiac_agent_postproc.ops import generate_candidates
from cardiac_agent_postproc.rqs import sobel_edges, valve_plane_y
from cardiac_agent_postproc.view_utils import infer_view_type
from cardiac_agent_postproc.atlas import load_atlas

def test():
    # Load config
    with open("config/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Load Atlas
    atlas_path = "results/inputs/shape_atlas.pkl"
    print(f"Loading atlas from {atlas_path}...")
    atlas = load_atlas(atlas_path)
    print(f"Atlas Keys: {list(atlas.keys())}")

    # Pick the worse file from logs (307)
    stem = "307_original_lax_4c_024"
    pred_path = f"results/inputs/{stem}_pred.png"
    img_path = f"results/inputs/{stem}_img.png"
    
    print(f"Processing {stem}...")
    mask = read_mask(pred_path)
    pred0 = mask.copy()
    img = read_image(img_path)
    
    view, reason = infer_view_type(stem, mask)
    print(f"Inferred View: {view} ({reason})")

    E = sobel_edges(img, percentile=float(cfg["edge"]["sobel_percentile"]))
    vy = valve_plane_y(mask)

    from cardiac_agent_postproc.rqs import compute_rqs
    
    # Need siblings for RQS? Just use empty list for test
    sibs = []

    print("Generating candidates...")
    cands = generate_candidates(mask, pred0, img, view, E, vy, atlas, cfg)
    
    print("Scoring candidates...")
    
    # Score current first
    res_cur = compute_rqs(mask, img, view, pred0, atlas, sibs, cfg)
    print(f"CURRENT RQS: {res_cur.total:.2f}")
    print(f"  Breakdown: {res_cur.terms}")

    for name, cm in cands:
        if np.count_nonzero(cm)==0:
            print(f"  - {name}: INVALID (empty)")
            continue
            
        res = compute_rqs(cm, img, view, pred0, atlas, sibs, cfg)
        delta = res.total - res_cur.total
        print(f"  - {name}: RQS={res.total:.2f} (delta={delta:.2f})")
        # Print breakdown if delta is negative (worse) or improving
        if "atlas" in name:
            print(f"    Terms: {res.terms}")

if __name__ == "__main__":
    test()
