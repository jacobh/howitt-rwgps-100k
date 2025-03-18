import json
import geojson
from prefect import task
import numpy as np
from typing import List

# from prefect.cache_policies import TASK_SOURCE, INPUTS

@task
def load_highway_data(file_path) -> geojson.FeatureCollection:
    """Load highway data from GeoJSON file using async I/O"""
    with open(file_path, "r") as f:
        content = f.read()
        data = json.loads(content)
        return data

@task
def convert_highway_geometry_to_numpy(highway_feature: geojson.Feature) -> np.ndarray:
    """Convert highway geometry to numpy arrays"""
    
    return np.ndarray(list(*geojson.coords(highway_feature)))

@task
def convert_highway_features_to_numpy(highway_features: List[geojson.Feature]):
    """Convert highway geometry to numpy arrays"""

    return [convert_highway_geometry_to_numpy(highway_feature) for highway_feature in highway_features]