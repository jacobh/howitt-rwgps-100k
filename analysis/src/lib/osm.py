import json
import geojson
import numpy as np
from typing import List, Union
import shapely

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


def build_spatial_index(highway_coords: List[np.ndarray]) -> shapely.STRtree:
    """Build spatial index for highway coordinates"""

    geometries: List[Union[shapely.geometry.LineString, shapely.geometry.Point]] = []
    for coords in highway_coords:
        if len(coords) > 1:
            geometries.append(shapely.geometry.LineString(coords))
        else:
            geometries.append(shapely.geometry.Point(coords[0]))
    
    tree = shapely.STRtree(geometries)
    return tree
