import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

def convex_hull_fill_debug(mask: np.ndarray):
    M2 = (mask == 2).astype(np.uint8)
    M3 = (mask == 3).astype(np.uint8)
    
    # Simulate the logic
    labeled, num = ndi.label(M2)
    sizes = ndi.sum(M2, labeled, range(num + 1))
    largest_idx = np.argmax(sizes[1:]) + 1
    largest_comp = (labeled == largest_idx).astype(np.uint8)
    
    contours, _ = cv2.findContours(largest_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    hull = cv2.convexHull(cnt)
    
    hull_mask = np.zeros_like(M2)
    cv2.drawContours(hull_mask, [hull], -1, 1, thickness=-1)
    
    # The original logic subtraction
    M3_dil = ndi.binary_dilation(M3, iterations=2)
    
    # Result
    new_M2 = (hull_mask > 0) & (~M3_dil)
    
    return hull_mask, M3_dil, new_M2

# Create synthetic C-shape Myo with protruding LV
H, W = 100, 100
mask = np.zeros((H, W), dtype=np.uint8)

# Myo: C-shape
cv2.ellipse(mask, (50, 50), (30, 30), 0, 45, 315, 2, 5) # Draw Myo (val 2)

# LV: Filling the center AND protruding through the gap (angle 0)
cv2.circle(mask, (50, 50), 25, 3, -1) # Center LV (val 3)
cv2.line(mask, (50, 50), (90, 50), 3, 5) # Protrusion through gap

# Visualization
hull_mask, M3_dil, new_M2 = convex_hull_fill_debug(mask)

print(f"Hull Area: {np.sum(hull_mask)}")
print(f"LV Dilated Area: {np.sum(M3_dil)}")
print(f"New Myo Area: {np.sum(new_M2)}")
print(f"Original Myo Area: {np.sum(mask == 2)}")

if np.sum(new_M2) <= np.sum(mask == 2):
    print("FAILURE: Operation added 0 pixels (or removed pixels). Gap preserved due to LV protrusion.")
else:
    print("SUCCESS: Operation added pixels.")
