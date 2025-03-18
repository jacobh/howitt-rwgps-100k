import json
import geojson
import numpy as np
from typing import List

def load_highway_data(file_path) -> geojson.FeatureCollection:
    """Load highway data from GeoJSON file using async I/O"""
    with open(file_path, "r") as f:
        content = f.read()
        data = json.loads(content)
        return data


def convert_highway_geometry_to_numpy(highway_feature: geojson.Feature) -> np.ndarray:
    """Convert highway geometry to numpy arrays"""
    
    return np.array(list(geojson.coords(highway_feature)))


def convert_all_highway_features_to_numpy(highway_features: List[geojson.Feature]) -> List[np.ndarray]:
    """Convert highway geometry to numpy arrays"""

    return [convert_highway_geometry_to_numpy(highway_feature) for highway_feature in highway_features]
