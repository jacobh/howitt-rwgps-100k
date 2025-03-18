from ..models.rwgps import Trip
from typing import List, Any, Dict
import numpy as np
import msgpack
# import shapely

def load_batch_data(batch_file_path: str) -> List[Trip]:
    """Load batch data from MessagePack file using async I/O"""
    with open(batch_file_path, "rb") as f:
        content = f.read()
        parsed: List[Dict[str, Any]] = msgpack.unpackb(content)
        return [Trip.parse_obj(t['trip']) for t in parsed]

def build_trip_coords(trip: Trip) -> np.ndarray:
    """Build a numpy array of coordinates from a trip"""
    track_points = trip.track_points
    coords = np.array([[point.x, point.y] for point in track_points])

    coords = filter_nil_coords(coords)

    return coords

# def build_trip_linestring(trip: Trip) -> shapely.LineString:
#     """Build a shapely LineString from a trip"""
#     track_points = trip.track_points
#     return shapely.LineString([[point.x, point.y] for point in track_points])

def filter_nil_coords(coords: np.ndarray) -> np.ndarray:
    """
    Remove rows with None values from a 2D numpy array.

    Args:
        coords: A numpy array of shape (n, 2) with x, y pairs.

    Returns:
        A numpy array containing only rows where both x and y are not None.
    """
    mask = ~np.equal(coords, np.array(None, dtype=object))
    valid_rows = np.all(mask, axis=1)
    return coords[valid_rows]
