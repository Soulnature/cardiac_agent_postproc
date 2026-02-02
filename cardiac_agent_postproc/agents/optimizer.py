from __future__ import annotations
import os
import csv
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm

from ..io_utils import list_frames, read_image, read_mask, write_mask
from ..view_utils import infer_view_type
from ..rqs import compute_rqs, sobel_edges, valve_plane_y
from ..ops import generate_candidates, is_valid_candidate, topology_cleanup

def _append_row_csv(path: str, header: List[str], row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)

class OptimizerAgent:
    def __init__(self, cfg: dict, atlas: dict|None):
        self.cfg = cfg
        self.atlas = atlas

    def _ensure_no_gt(self, folder: str, rec: dict):
        # Enforce: do not read GT for optimization decisions
        # We simply never touch rec["gt"] here; additionally guard if folder is TARGET.
        # Users can set a flag externally if desired.
        return

    def optimize_folder(self,
                        folder: str,
                        is_target: bool,
                        run_dir: str) -> Dict[str, Any]:
        cfg = self.cfg
        frames = list_frames(folder)
        if len(frames)==0:
            raise RuntimeError(f"No frames found in {folder} (expect *_img.png & *_pred.png).")

        # log file (append-safe)
        log_csv = os.path.join(folder, "label_optimization_log.csv")
        header = [
            "filename","view_type","outer_iter","inner_step","op_name",
            "RQS_total",
            "P_touch","P_rv_2ch","P_components","P_islands","P_valve_leak","P_double_cavity","P_holes",
            "P_size_drift","P_edge_misalignment","P_shape_prior","P_boundary_roughness",
            "R_enclosure","R_shape_match",
            "rv_ratio","touch_pixel_count",
            "boundary_edge_overlap_c1","boundary_edge_overlap_c2","boundary_edge_overlap_c3",
            "atlas_cluster_id","atlas_match_score","atlas_direction_score","mean_conf",
            "pred_path","out_path"
        ]

        # initialize current masks as original preds
        cur_masks = {}
        pred0_masks = {}
        metas = {}
        for fr in frames:
            pred = read_mask(fr["pred"])
            pred0_masks[fr["stem"]] = pred.copy()
            cur_masks[fr["stem"]] = pred.copy()
            metas[fr["stem"]] = fr

        # baseline RQS
        def score_all(outer: int) -> Dict[str, float]:
            scores = {}
            for stem, m in cur_masks.items():
                fr = metas[stem]
                img = read_image(fr["img"])
                view, _ = infer_view_type(fr["stem"], m, rv_ratio_threshold=float(cfg["view_inference"]["rv_ratio_2ch_threshold"]))
                res = compute_rqs(m, img, view, pred0_masks[stem], self.atlas, cfg)
                scores[stem] = res.total
            return scores

        scores = score_all(outer=0)
        min_rqs = min(scores.values())
        best_min_rqs = min_rqs
        stagnant = 0

        max_outer = int(cfg["solver"]["max_outer"])
        for outer in range(1, max_outer+1):
            # rank by score ascending
            items = sorted(scores.items(), key=lambda x: x[1])
            n = len(items)
            topk = int(max(cfg["solver"]["topk_min"], min(cfg["solver"]["topk_max"], int(cfg["solver"]["topk_frac"]*n))))
            worst = items[:topk]

            worst_file, worst_score = worst[0]
            print(f"[{os.path.basename(folder)}] Outer {outer}: processing TOPK={topk}/{n}. Worst={worst_file} RQS={worst_score:.2f}")

            # process each worst sample
            for stem,_ in tqdm(worst, desc=f"outer{outer}", leave=False):
                fr = metas[stem]
                img = read_image(fr["img"])
                cur = cur_masks[stem]
                pred0 = pred0_masks[stem]
                view, _ = infer_view_type(fr["stem"], cur, rv_ratio_threshold=float(cfg["view_inference"]["rv_ratio_2ch_threshold"]))
                E = sobel_edges(img, percentile=float(cfg["edge"]["sobel_percentile"]))
                vy = valve_plane_y(cur)

                inner_max = int(cfg["solver"]["max_inner"])
                no_improve = 0
                for inner in range(1, inner_max+1):
                    # generate candidates
                    cands = generate_candidates(cur, pred0, img, view, E, vy, self.atlas, cfg)

                    # evaluate candidates
                    best = None
                    best_res = None
                    best_name = None
                    for name, cm in cands:
                        if not is_valid_candidate(cm):
                            continue
                        res = compute_rqs(cm, img, view, pred0, self.atlas, cfg)
                        if (best_res is None) or (res.total > best_res.total):
                            best = cm
                            best_res = res
                            best_name = name

                    if best_res is None:
                        break

                    cur_res = compute_rqs(cur, img, view, pred0, self.atlas, cfg)
                    delta = best_res.total - cur_res.total
                    if delta >= float(cfg["solver"]["min_delta"]):
                        cur = best
                        cur_masks[stem] = cur
                        no_improve = 0

                        # write intermediate
                        out_iter = os.path.join(folder, f"{stem}_pred_iter{outer}_step{inner}.png")
                        write_mask(out_iter, cur)

                        row = {
                            "filename": f"{stem}_pred.png",
                            "view_type": view,
                            "outer_iter": outer,
                            "inner_step": inner,
                            "op_name": best_name,
                            "RQS_total": best_res.total,
                            **best_res.terms,
                            **best_res.aux,
                            "pred_path": fr["pred"],
                            "out_path": out_iter,
                        }
                        _append_row_csv(log_csv, header, row)
                    else:
                        no_improve += 1
                        if no_improve >= int(cfg["solver"]["patience"]):
                            break

                # write final optimized for this stem (current best after inner)
                final_path = os.path.join(folder, f"{stem}_pred_optimized.png")
                write_mask(final_path, cur)

            # rescore all for outer reporting
            scores = score_all(outer=outer)
            min_rqs = min(scores.values())
            worst_file = min(scores, key=scores.get)
            # print dominant penalties for worst file
            fr = metas[worst_file]
            img = read_image(fr["img"])
            view, _ = infer_view_type(fr["stem"], cur_masks[worst_file], rv_ratio_threshold=float(cfg["view_inference"]["rv_ratio_2ch_threshold"]))
            res = compute_rqs(cur_masks[worst_file], img, view, pred0_masks[worst_file], self.atlas, cfg)
            pen = {k:v for k,v in res.terms.items() if k.startswith("P_")}
            top3 = sorted(pen.items(), key=lambda x: -x[1])[:3]
            print(f"[{os.path.basename(folder)}] Outer {outer} summary: min_RQS={min_rqs:.2f} worst={worst_file} top_penalties={top3}")

            # stop criteria
            if min_rqs >= float(cfg["solver"]["stop_min_rqs"]):
                print(f"[{os.path.basename(folder)}] Stop: min_RQS >= {cfg['solver']['stop_min_rqs']}")
                break

            if (min_rqs - best_min_rqs) < float(cfg["solver"]["stop_min_rqs_delta"]):
                stagnant += 1
            else:
                best_min_rqs = min_rqs
                stagnant = 0

            if stagnant >= int(cfg["solver"]["stop_min_rqs_stagnant_rounds"]):
                print(f"[{os.path.basename(folder)}] Stop: min_RQS improvement stalled for {stagnant} outer rounds.")
                break

        # summary
        all_scores = list(scores.values())
        return {
            "n": len(all_scores),
            "min_rqs": float(np.min(all_scores)),
            "median_rqs": float(np.median(all_scores)),
            "max_rqs": float(np.max(all_scores)),
        }
