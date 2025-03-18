import numpy as np

def inspect_highway_coords(npz_path: str = "../data/highways_coords.npz", indices=(0, 1)):
    print(f"Loading highway coordinates from '{npz_path}'...")
    data = np.load(npz_path, allow_pickle=True)
    coords = data["coords"]
    print(f"Total features loaded: {len(coords)}")
    
    for i in indices:
        if i < len(coords):
            arr = coords[i]
            print(f"\nCoords[{i}]: shape = {arr.shape}, dtype = {arr.dtype}")
            print(arr)
        else:
            print(f"\nIndex {i} is out of range. Total features: {len(coords)}")
    
if __name__ == "__main__":
    inspect_highway_coords()