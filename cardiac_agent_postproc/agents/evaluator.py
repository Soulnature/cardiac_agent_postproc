from __future__ import annotations
import os, csv
from typing import Dict, Any, List
import numpy as np

from ..io_utils import list_frames, read_mask
from ..view_utils import infer_view_type
from ..eval_metrics import dice_macro, hd95_macro

def _append_row_csv(path: str, header: List[str], row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)

def _write_csv(path: str, header: List[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

class EvaluatorAgent:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def evaluate_target(self, target_dir: str, outer_iter: int) -> Dict[str, Any]:
        frames = list_frames(target_dir)
        report_path = os.path.join(target_dir, "target_dice_report.csv")
        summary_path = os.path.join(target_dir, "target_dice_summary.csv")

        rep_header = [
            "filename","view_type","outer_iter",
            "dice_macro","dice_c1","dice_c2","dice_c3",
            "hd95_macro","hd95_c1","hd95_c2","hd95_c3",
            "optimized_pred_path"
        ]
        rows = []
        dice_list = []
        hd_list = []
        for fr in frames:
            if "gt" not in fr:
                continue
            stem = fr["stem"]
            gt = read_mask(fr["gt"])
            # prefer optimized if exists
            opt_path = os.path.join(target_dir, f"{stem}_pred_optimized.png")
            pred_path = opt_path if os.path.exists(opt_path) else fr["pred"]
            pred = read_mask(pred_path)
            view, _ = infer_view_type(stem, pred, rv_ratio_threshold=float(self.cfg["view_inference"]["rv_ratio_2ch_threshold"]))

            dmac,d1,d2,d3 = dice_macro(pred, gt)
            hmac,h1,h2,h3 = hd95_macro(pred, gt)

            dice_list.append(dmac)
            hd_list.append(hmac)
            row = {
                "filename": f"{stem}_pred.png",
                "view_type": view,
                "outer_iter": outer_iter,
                "dice_macro": dmac,
                "dice_c1": d1,
                "dice_c2": d2,
                "dice_c3": d3,
                "hd95_macro": hmac,
                "hd95_c1": h1,
                "hd95_c2": h2,
                "hd95_c3": h3,
                "optimized_pred_path": pred_path
            }
            rows.append(row)
            _append_row_csv(report_path, rep_header, row)

        if len(rows)==0:
            raise RuntimeError("No TARGET frames with *_gt.png found for evaluation.")

        dice_arr = np.array(dice_list, dtype=np.float32)
        hd_arr = np.array(hd_list, dtype=np.float32)

        worst_dice_idx = int(np.argmin(dice_arr))
        worst_hd_idx = int(np.argmax(hd_arr))

        summ = {
            "outer_iter": outer_iter,
            "min_dice": float(np.min(dice_arr)),
            "median_dice": float(np.median(dice_arr)),
            "mean_dice": float(np.mean(dice_arr)),
            "worst_file_by_dice": rows[worst_dice_idx]["filename"],
            "max_hd95": float(np.max(hd_arr)),
            "median_hd95": float(np.median(hd_arr)),
            "mean_hd95": float(np.mean(hd_arr)),
            "worst_file_by_hd95": rows[worst_hd_idx]["filename"],
        }

        # overwrite summary each run for cleanliness
        _write_csv(summary_path, list(summ.keys()), [summ])

        print(f"[TARGET Eval] outer={outer_iter} minDice={summ['min_dice']:.4f} medDice={summ['median_dice']:.4f} meanDice={summ['mean_dice']:.4f} worstDice={summ['worst_file_by_dice']}")
        print(f"[TARGET Eval] outer={outer_iter} maxHD95={summ['max_hd95']:.4f} medHD95={summ['median_hd95']:.4f} meanHD95={summ['mean_hd95']:.4f} worstHD95={summ['worst_file_by_hd95']}")
        return summ
