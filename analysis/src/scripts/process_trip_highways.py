import numpy as np
import shapely
from ..lib.osm import build_spatial_index
from ..lib.geo import generate_bbox, pad_bbox, numpy_bbox_to_shapely
from ..models.trip_dimensions import TripDimensions
from ..models.trip_segments import TripSegmentIndex, TripSegmentIndexes
import math
from typing import List
import msgpack


def get_spatial_index() -> shapely.STRtree:
    npz_path: str = "../data/highways_coords.npz"

    print(f"Loading highway coordinates from '{npz_path}'...")
    data = np.load(npz_path, allow_pickle=True)
    coords = data["coords"]
    print(f"Total features loaded: {len(coords)}")

    tree = build_spatial_index(coords)

    print(f"STRTree built with {len(tree)} geometries.")

    return tree


def load_trip_dimensions() -> List[TripDimensions]:
    """
    Reload trip dimensions from the existing MessagePack file.
    """
    print("Reloading trip dimensions batch data...")
    with open("../data/trip_dimensions/trip_dimensions_10.msgpack", "rb") as f:
        raw = msgpack.unpackb(f.read(), raw=False)
        return [TripDimensions.model_validate(item) for item in raw]


def process_trip_highways() -> None:
    tree = get_spatial_index()
    trips = load_trip_dimensions()

    trip_segments_list: List[TripSegmentIndexes] = []
    for trip in trips:
        print(f"Processing trip {trip.id}...")

        # Exclude None values and convert to numpy array.
        trip_coords = np.array(
            [p if p is not None else [np.nan, np.nan] for p in trip.coords]
        )
        trip_distance_m = trip.distance_m

        segment_count = max(1, math.ceil(trip_distance_m / 200))
        print(
            f"Trip distance: {trip_distance_m} m, dividing into {segment_count} segments."
        )

        segments: List[TripSegmentIndex] = []
        # Split the indices instead of the data directly to retain start indexes.
        indices = np.arange(len(trip_coords))
        indices_splits = np.array_split(indices, segment_count)

        for split in indices_splits:
            if len(split) == 0:
                continue
            start_idx = int(split[0])
            segment_coords = trip_coords[split]
            segment_coords = segment_coords[~np.isnan(segment_coords).any(axis=1)]

            if len(segment_coords) == 0:
                continue

            # Generate the bounding box for the segment and pad it.
            bbox2d = generate_bbox(segment_coords)
            bbox2d = pad_bbox(bbox2d, 250)
            bbox_shapely = numpy_bbox_to_shapely(bbox2d)

            # Obtain candidate highway indexes for this segment.
            highway_idxs: List[int] = list(tree.query(bbox_shapely))
            # print(f"Segment starting at index {start_idx} for trip {trip.id} has {len(highway_idxs)} highway segments.")

            segment_obj = TripSegmentIndex(
                start_idx=start_idx, candidate_highway_indexes=highway_idxs
            )
            segments.append(segment_obj)

        trip_seg_obj = TripSegmentIndexes(trip_id=trip.id, segments=segments)
        trip_segments_list.append(trip_seg_obj)

    # For example, print out the JSON representations of the TripSegments instances.
    print("Finished processing trips. Generated TripSegments instances:")
    # for ts in trip_segments_list:
    #     # Using the pydantic's model json() method for pretty printing
    #     print(ts.json())


if __name__ == "__main__":
    process_trip_highways()
