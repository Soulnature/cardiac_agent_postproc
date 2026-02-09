
import numpy as np
import cv2
from cardiac_agent_postproc.io_utils import read_mask

def test_reader():
    # Only need one file
    path = "/data484_2/mferrari2/results_segmentation/MnM2/all_frames_export/201_original_lax_4c_000_pred.png"
    
    # Raw check
    raw = cv2.imread(path, -1)
    print(f"RAW File Values: {np.unique(raw)}")
    
    # Function check
    mask = read_mask(path)
    print(f"read_mask result: {np.unique(mask)}")
    
    if 1 in mask and 2 in mask:
        print("SUCCESS: Classes 1 and 2 preserved.")
    else:
        print("FAILURE: Classes lost!")

if __name__ == "__main__":
    test_reader()
