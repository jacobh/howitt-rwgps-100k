import numpy as np
import jax.numpy as jnp
import shapely
from ..lib.osm import build_spatial_index
from ..lib.geo import (
    generate_bbox,
    pad_bbox,
    numpy_bbox_to_shapely,
    mean_min_distances,
)
from ..models.trip_dimensions import TripDimensions
from ..models.trip_segments import TripSegmentIndex, TripSegmentIndexes
import math
from typing import List
import msgpack

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


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


def pad_linestring(
    coords: np.ndarray, target_length=1024
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad a linestring to a target length by repeating the last coordinate.

    Parameters:
    -----------
    coords : np.ndarray
        Array of coordinates with shape (N, 2) where each point is [lon, lat]
    target_length : int, default=1024
        Target length to pad to

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        - Padded coordinates with shape (target_length, 2)
        - Boolean mask with shape (target_length,) where True indicates original points
          and False indicates padding
    """
    current_length = len(coords)

    # Create the mask array - True for original points, False for padding
    mask = np.ones(target_length, dtype=bool)

    # If already at or exceeding target length, truncate
    if current_length >= target_length:
        return coords[:target_length], mask

    # Update mask to mark padded positions as False
    mask[current_length:] = False

    # Calculate padding needed
    padding_needed = target_length - current_length

    # Use np.pad with 'edge' mode to repeat the last coordinate
    padded_coords = np.pad(coords, pad_width=((0, padding_needed), (0, 0)), mode="edge")

    return padded_coords, mask


def pad_highways(
    highways_coords: np.ndarray, highways_mask: np.ndarray, target_length=512
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad the highways_coords array and corresponding mask to have target_length highways.
    If there are fewer than target_length highways, the last highway is repeated and
    marked as padding in the mask.
    If there are more, the arrays are truncated.

    Args:
        highways_coords: Array of shape (n_highways, n_points, 2) containing highway coordinates
        highways_mask: Array of shape (n_highways, n_points) containing boolean masks for each
                      highway point (True for valid points, False for padding)
        target_length: Target number of highways

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Padded array of shape (target_length, n_points, 2)
            - Updated mask of shape (target_length, n_points) with padded highways marked as all False
    """
    if len(highways_coords) == 0:
        # Return empty arrays for both coordinates and mask
        return np.array([]), np.array([])

    current_length = len(highways_coords)

    # If already at or exceeding target length, truncate both arrays
    if current_length >= target_length:
        return highways_coords[:target_length], highways_mask[:target_length]

    # Calculate padding needed
    padding_needed = target_length - current_length

    # Pad the coordinates with 'edge' mode to repeat the last highway
    padded_highways = np.pad(
        highways_coords, pad_width=((0, padding_needed), (0, 0), (0, 0)), mode="edge"
    )

    # For the mask, we pad with zeros (False) to indicate the padded highways are all padding
    # This creates a mask where all points in the padded highways are marked as padding
    padded_mask = np.pad(
        highways_mask,
        pad_width=((0, padding_needed), (0, 0)),
        mode="constant",
        constant_values=False,
    )

    return padded_highways, padded_mask


def process_trip_indexes() -> None:
    highway_coords = get_highway_data()

    tree = get_spatial_index(highway_coords)
    trips = load_trip_dimensions()

    trip_segments_list: List[TripSegmentIndexes] = []
    for trip in trips[:5]:
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

        # distance_tasks = []
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

            candidate_highways_coords_and_padding_masks = [
                pad_linestring(highway_coords[idx])
                for idx in segment.candidate_highway_indexes
            ]

            if len(candidate_highways_coords_and_padding_masks) == 0:
                continue

            candidate_highways_coords = np.array(
                [x[0] for x in candidate_highways_coords_and_padding_masks]
            )
            candidate_highways_masks = np.array(
                [x[1] for x in candidate_highways_coords_and_padding_masks]
            )

            candidate_highways_coords, candidate_highways_masks = pad_highways(
                candidate_highways_coords, candidate_highways_masks, 256
            )

            ref_coords, ref_mask = pad_linestring(segment_coords, 128)

            # Calculate mean minimum distances
            mean_min_distances_result = mean_min_distances(
                ref_linestring=jnp.array(ref_coords),
                target_linestrings=jnp.array(candidate_highways_coords),
                ref_mask=jnp.array(ref_mask),
                target_masks=jnp.array(candidate_highways_masks),
            )

            # Convert to numpy array
            mean_min_distances_result = np.array(mean_min_distances_result)

            # Get only the distances for the actual highways (not padding)
            # We know there are len(segment.candidate_highway_indexes) real highways
            valid_distances = mean_min_distances_result[
                : len(segment.candidate_highway_indexes)
            ]

            # Find the index of the highway with minimum mean distance
            if len(valid_distances) > 0:
                min_distance_idx = np.argmin(valid_distances)
                best_matched_highway_idx = segment.candidate_highway_indexes[
                    min_distance_idx
                ]
                min_distance = valid_distances[min_distance_idx]

                print(
                    f"  - Best match: Highway idx {best_matched_highway_idx} with mean distance {min_distance:.2f} meters"
                )
            else:
                best_matched_highway_idx = None
                print("  - No valid highways found for this segment")

            # Now you can use segment_coords for further processing
            # For example, print the number of coordinates in each segment
            print(
                f"Segment {j} of trip {trip_dimensions.id}: {len(segment_coords)} coordinates, {len(segment.candidate_highway_indexes)} candidate highways"
            )


if __name__ == "__main__":
    process_trip_indexes()
