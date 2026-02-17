import pickle
import sys

def inspect(path):
    try:
        with open(path, "rb") as f:
            atlas = pickle.load(f)
        print(f"Atlas loaded from {path}")
        print(f"Keys: {list(atlas.keys())}")
    except Exception as e:
        print(f"Error loading atlas: {e}")

if __name__ == "__main__":
    inspect("shape_atlas.pkl")
