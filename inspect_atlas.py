import pickle
import sys
import os

sys.path.append(os.getcwd())
from cardiac_agent_postproc.atlas import load_atlas

try:
    atlas = load_atlas("results/inputs/shape_atlas.pkl")
    print("Atlas loaded. Keys:")
    for k in atlas.keys():
        print(f"  {k}")
except Exception as e:
    print(f"Error: {e}")
