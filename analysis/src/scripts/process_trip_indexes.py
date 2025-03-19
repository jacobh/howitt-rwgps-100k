import numpy as np
import jax.numpy as jnp
import shapely
from ..lib.osm import build_spatial_index
from ..lib.geo import (
    generate_bbox,
    pad_bbox,
    numpy_bbox_to_shapely,
    haversine_distances,
)
from ..models.trip_dimensions import TripDimensions
from ..models.trip_segments import TripSegmentIndex, TripSegmentIndexes
import math
from typing import List
import msgpack


def get_highway_data() -> np.ndarray:
    npz_path: str = "../data/highways_coords.npz"

    print(f"Loading highway coordinates from '{npz_path}'...")
    data = np.load(npz_path, allow_pickle=True)
    coords = data["coords"]
    print(f"Total features loaded: {len(coords)}")

    return np.array(coords)


def get_spatial_index(coords: np.ndarray) -> shapely.STRtree:
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


def process_trip_indexes() -> None:
    highway_coords = get_highway_data()

    tree = get_spatial_index(highway_coords)
    trips = load_trip_dimensions()

    trip_segments_list: List[TripSegmentIndexes] = []
    for trip in trips:
        print(f"Processing trip {trip.id}...")

        trip_coords = np.array(
            [p if p is not None else [np.nan, np.nan] for p in trip.coords]
        )
        trip_distance_m = trip.distance_m

        # Calculate segment count based on distance, but ensure segments don't exceed 128 coords
        min_segments_by_distance = max(1, math.ceil(trip_distance_m / 200))
        min_segments_by_coords = max(1, math.ceil(len(trip_coords) / 128))
        segment_count = max(min_segments_by_distance, min_segments_by_coords)

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
            bbox2d = pad_bbox(bbox2d, 200)
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

    for i, trip_segments in enumerate(trip_segments_list):
        trip_dimensions = trips[i]

        for j, segment in enumerate(trip_segments.segments):
            # Get start index for this segment
            start_idx = segment.start_idx

            # Determine end index by looking at next segment or using the end of trip
            if j < len(trip_segments.segments) - 1:
                end_idx = trip_segments.segments[j + 1].start_idx
            else:
                end_idx = len(trip_dimensions.coords)

            # Extract coordinates for this segment
            segment_coords = np.array(
                [p for p in trip_dimensions.coords[start_idx:end_idx] if p is not None]
            )

            if len(segment_coords) == 0:
                continue

            def pad_linestring(coords, target_length=1024):
                current_length = len(coords)

                # If already at or exceeding target length, truncate
                if current_length >= target_length:
                    return coords[:target_length]

                # Calculate padding needed
                padding_needed = target_length - current_length

                # Use np.pad with 'edge' mode to repeat the last coordinate
                padded_coords = np.pad(
                    coords, pad_width=((0, padding_needed), (0, 0)), mode="edge"
                )

                return padded_coords

            candidate_highways_coords = np.array(
                [
                    pad_linestring(highway_coords[idx])
                    for idx in segment.candidate_highway_indexes
                ]
            )

            if len(candidate_highways_coords) == 0:
                continue

            distances = haversine_distances(
                ref_linestring=jnp.array(pad_linestring(segment_coords, 128)),
                target_linestrings=jnp.array(candidate_highways_coords),
            )

            # distances = distances.block_until_ready()

            # Now you can use segment_coords for further processing
            # For example, print the number of coordinates in each segment
            print(
                f"Segment {j} of trip {trip_dimensions.id}: {len(segment_coords)} coordinates, {len(segment.candidate_highway_indexes)} candidate highways"
            )


if __name__ == "__main__":
    process_trip_indexes()
