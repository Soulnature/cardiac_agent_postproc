from __future__ import annotations
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from cardiac_agent_postproc.io_utils import read_image, read_mask
from cardiac_agent_postproc.view_utils import infer_view_type
from cardiac_agent_postproc.quality_model import QUALITY_FEATURE_NAMES, extract_quality_features


def build_dataset(data_dir: str, metrics_csv: str, limit: int | None = None):
    df = pd.read_csv(metrics_csv)
    feats = []
    labels = []
    rows = df.to_dict("records")
    if limit:
        rows = rows[:limit]
    for row in rows:
        stem = row["id"].replace(".png", "") if row["id"].endswith(".png") else row["id"]
        img_path = os.path.join(data_dir, f"{stem}_img.png")
        pred_path = os.path.join(data_dir, f"{stem}_pred.png")
        if not (os.path.exists(img_path) and os.path.exists(pred_path)):
            continue
        img = read_image(img_path)
        mask = read_mask(pred_path)
        view = row.get("view")
        if isinstance(view, str):
            v_lower = view.lower()
            if "2" in v_lower:
                view_type = "2ch"
            elif "3" in v_lower:
                view_type = "3ch"
            elif "4" in v_lower:
                view_type = "4ch"
            else:
                view_type, _ = infer_view_type(stem, mask)
        else:
            view_type, _ = infer_view_type(stem, mask)
        feat_dict = extract_quality_features(mask, img, view_type)
        feats.append([feat_dict[name] for name in QUALITY_FEATURE_NAMES])
        labels.append(float(row.get("mean_dice", 0.0)))
    return np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train a quality scoring model from weak labels.")
    parser.add_argument("--data-dir", default="results/Input_MnM2/all_frames_export", help="Folder containing *_img.png and *_pred.png files")
    parser.add_argument("--metrics-csv", default="results/Input_MnM2/metrics_report.csv", help="CSV with columns id, mean_dice, view")
    parser.add_argument("--output", default="quality_model.pkl", help="Output path for the trained model")
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out fraction for evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of samples for quick experiments")
    args = parser.parse_args()

    X, y = build_dataset(args.data_dir, args.metrics_csv, limit=args.limit)
    if len(X) == 0:
        raise RuntimeError("No training samples found. Check data paths and metrics CSV.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(random_state=42, n_estimators=400, max_depth=3)),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    payload = {
        "model": pipeline,
        "feature_names": QUALITY_FEATURE_NAMES,
        "train_meta": {
            "mae": mae,
            "r2": r2,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
    }
    joblib.dump(payload, args.output)
    print(f"Saved quality model to {args.output}")
    print(f"Metrics -> MAE: {mae:.4f}, R2: {r2:.4f}")


if __name__ == "__main__":
    main()
