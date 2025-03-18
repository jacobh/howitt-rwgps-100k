import numpy as np
from ..lib.osm import build_spatial_index

def test_build_spatial_index(npz_path: str = "../data/highways_coords.npz", indices=(0, 1)):
    print(f"Loading highway coordinates from '{npz_path}'...")
    data = np.load(npz_path, allow_pickle=True)
    coords = data["coords"]
    print(f"Total features loaded: {len(coords)}")
    
    tree = build_spatial_index(coords)

    print(f"STRTree built with {len(tree)} geometries.")
    
if __name__ == "__main__":
    test_build_spatial_index()