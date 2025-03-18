from ..models.rwgps import Trip
from ..models.trip_dimensions import TripDimensions
from typing import List, Any, Dict
import numpy as np
import msgpack
from pydantic import ValidationError

# import shapely

def load_batch_data(batch_file_path: str) -> List[Trip]:
    """Load batch data from a MessagePack file and skip rides missing the distance field."""
    with open(batch_file_path, "rb") as f:
        content = f.read()
        parsed: List[Dict[str, Any]] = msgpack.unpackb(content)
        trips = []
        for t in parsed:
            try:
                trips.append(Trip.parse_obj(t["trip"]))
            except ValidationError as e:
                # Skips the ride if validation fails, e.g., when the distance field is missing.
                print(f"Skipping ride in batch {batch_file_path} due to validation error: {e}")
                continue
        return trips

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

def build_trip_dimensions(trip: Trip) -> TripDimensions:
    """
    Build a TripDimensions instance from a Trip.

    Args:
        trip: A Trip instance containing trip details and track points.

    Returns:
        A TripDimensions instance with the trip's summary and dimensions extracted.
    """
    track_points = trip.track_points

    # Build coordinates array with (longitude, latitude) format.
    coords = [
        (pt.x, pt.y) if pt.x is not None and pt.y is not None else None
        for pt in track_points
    ]

    elevation = [pt.e for pt in track_points]
    distance = [pt.d for pt in track_points]
    speed = [pt.s for pt in track_points]
    time = [pt.t for pt in track_points]
    heart_rate = [pt.h for pt in track_points]
    temperature = [pt.T for pt in track_points]
    cadence = [pt.c for pt in track_points]
    power = [pt.p for pt in track_points]

    dimensions = TripDimensions(
        id=trip.id,
        user_id=trip.user_id,
        distance_m=trip.distance,
        created_at=trip.created_at,
        departed_at=trip.departed_at,
        coords=coords,
        elevation=elevation,
        distance=distance,
        speed=speed,
        time=time,
        heart_rate=heart_rate,
        temperature=temperature,
        cadence=cadence,
        power=power,
    )

    return dimensions