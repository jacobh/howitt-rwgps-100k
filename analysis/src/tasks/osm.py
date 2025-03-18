import json
import geojson
from prefect import task
import numpy as np
from typing import List
import itertools
from prefect.utilities.annotations import quote

# from prefect.cache_policies import TASK_SOURCE, INPUTS

@task
def load_highway_data(file_path) -> geojson.FeatureCollection:
    """Load highway data from GeoJSON file using async I/O"""
    with open(file_path, "r") as f:
        content = f.read()
        data = json.loads(content)
        return data


def convert_highway_geometry_to_numpy(highway_feature: geojson.Feature) -> np.ndarray:
    """Convert highway geometry to numpy arrays"""
    
    return np.array(list(geojson.coords(highway_feature)))

@task
def convert_batch_highway_features_to_numpy(highway_features: List[geojson.Feature]) -> List[np.ndarray]:
    return [convert_highway_geometry_to_numpy(highway_feature) for highway_feature in highway_features]

@task
def convert_all_highway_features_to_numpy(highway_features: List[geojson.Feature]) -> List[np.ndarray]:
    """Convert highway geometry to numpy arrays"""

    batch_size = 1000
    
    # Split into batches
    batches = [highway_features[i:i+batch_size] for i in range(0, len(highway_features), batch_size)]

    print(f"batches: {len(batches)}")

    futures = convert_batch_highway_features_to_numpy.map(quote(batches))
    

    results = [future.result() for future in futures]
    return list(itertools.chain(*results))