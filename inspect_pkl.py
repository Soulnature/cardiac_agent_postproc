
import joblib
import os

path = "shape_atlas.pkl"
if not os.path.exists(path):
    print("File not found.")
    exit()

data = joblib.load(path)
print(f"Type: {type(data)}")
if isinstance(data, dict):
    print("Keys:", data.keys())
    for k in data:
        print(f"Key '{k}': Type {type(data[k])}")
        if isinstance(data[k], dict):
             print(f"  Subkeys: {list(data[k].keys())[:5]}...")
             if 'prob_maps' in data[k]:
                 pm = data[k]['prob_maps']
                 print(f"  prob_maps Type: {type(pm)}")
                 print(f"  prob_maps Keys: {list(pm.keys())}")
                 first_val = list(pm.values())[0]
                 import numpy as np
                 if isinstance(first_val, np.ndarray):
                      print(f"  Value Shape: {first_val.shape}")
