import numpy as np
import shapely
from ..lib.osm import build_spatial_index
from ..lib.trip import load_batch_data, build_trip_coords
from ..lib.geo import generate_bbox, pad_bbox, numpy_bbox_to_shapely
from ..models.rwgps import Trip
import math
from typing import List

def get_spatial_index() -> shapely.STRtree:
    npz_path: str = "../data/highways_coords.npz"

    print(f"Loading highway coordinates from '{npz_path}'...")
    data = np.load(npz_path, allow_pickle=True)
    coords = data["coords"]
    print(f"Total features loaded: {len(coords)}")
    
    tree = build_spatial_index(coords)

    print(f"STRTree built with {len(tree)} geometries.")

    return tree

def get_batch_trips() -> List[Trip]:
    print("Loading trip batch data...")
    return load_batch_data("../data/trip_batches/trip_batch_10.msgpack")


def process_trip_highways() -> None:
    tree = get_spatial_index()
    trips = get_batch_trips()

    trip_highway_idxs = []
    for trip in trips:
        print(f"Processing trip {trip.id}...")

        trip_coords = build_trip_coords(trip)

        trip_distance_m = trip.distance

        segment_count = max(1, math.ceil(trip_distance_m / 500))
        print(f"Trip distance: {trip_distance_m} m, dividing into {segment_count} segments.")

        # Divide trip_coords into roughly equal buckets
        trip_segments = np.array_split(trip_coords, segment_count)
        print(f"Created {len(trip_segments)} segments for trip {trip.id}.")

        bboxes = []
        for segment in trip_segments:
            if len(segment) > 0:
                bbox2d = generate_bbox(segment)
                bbox2d = pad_bbox(bbox2d, 250)
                bbox_shapely = numpy_bbox_to_shapely(bbox2d)
                bboxes.append(bbox_shapely)

        highway_idxs: List[int] = []
        for bbox in bboxes:
            highway_idxs.extend(tree.query(bbox))

        print(f"Found {len(highway_idxs)} highway segments for trip {trip.id}.")
        trip_highway_idxs.append(highway_idxs)

    import pprint
    pprint.pprint(trip_highway_idxs)


if __name__ == "__main__":
    process_trip_highways()
