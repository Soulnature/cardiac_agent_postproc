
import numpy as np
import cv2
import os
from cardiac_agent_postproc.io_utils import write_mask, read_mask

def test_writer():
    path = "results/test_write.png"
    # Create a dummy mask [0, 1, 2, 3]
    mask = np.array([[0, 1], [2, 3]], dtype=np.int32)
    
    # Write it
    write_mask(path, mask)
    print(f"Wrote mask to {path}")
    
    # Read back RAW
    raw = cv2.imread(path, -1)
    print(f"Read back raw unique values: {np.unique(raw)}")
    
    if np.array_equal(np.unique(raw), np.array([0, 85, 170, 255])):
        print("SUCCESS: Scaled correctly.")
    else:
        print("FAILURE: Incorrect scaling.")
    
    # Cleanup
    if os.path.exists(path):
        os.remove(path)

if __name__ == "__main__":
    test_writer()
