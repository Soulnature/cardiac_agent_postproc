
import cv2
import numpy as np
import collections

path = "/data484_5/xzhao14/cardiac_agent_postproc/results/Input_MnM2/all_frames_export/335_original_lax_4c_009_pred_iter1_step1.png"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
if np.max(img) > 3:
    img = np.round(img / 85).astype(np.uint8)

counts = collections.Counter(img.flatten())
print(f"File: {path}")
print(f"Pixel Counts: {dict(counts)}")
print(f"Has Class 3 (LV)? {3 in counts and counts[3] > 0}")
