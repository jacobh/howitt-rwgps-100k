import numpy as np
from ..lib.osm import load_highway_data, convert_all_highway_features_to_numpy

def load_and_convert_all_highway_features():
    """Load and convert all highway features to numpy arrays"""
    print("Loading highway data...")
    highway_collection = load_highway_data("../data/vic_and_tas_highways.json")
    
    print("Converting highway data to numpy arrays...")
    highway_coords_numpy_list = convert_all_highway_features_to_numpy(highway_collection['features'])

     # Save the list of numpy arrays to an NPZ file.
    print("Dumping highway coordinates to '../data/highways_coords.npz'...")
    highway_coords_obj_array = np.array(highway_coords_numpy_list, dtype=object)
    np.savez("../data/highways_coords.npz", coords=highway_coords_obj_array)

    print("Highway coordinates successfully dumped to '../data/highways_coords.npz'")
 

    print("Done!")

if __name__ == "__main__":
    load_and_convert_all_highway_features()
