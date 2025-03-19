import numpy as np
import jax
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

def pad_highways(highways_coords, target_length=512):
    """
    Pad the highways_coords array to have target_length highways.
    If there are fewer than target_length highways, the last highway is repeated.
    If there are more, the array is truncated.
    
    Args:
        highways_coords: Array of shape (n_highways, n_points, 2) containing highway coordinates
        target_length: Target number of highways
        
    Returns:
        Padded array of shape (target_length, n_points, 2)
    """
    if len(highways_coords) == 0:
        return np.array([])
        
    current_length = len(highways_coords)
    
    # If already at or exceeding target length, truncate
    if current_length >= target_length:
        return highways_coords[:target_length]
    
    # Calculate padding needed
    padding_needed = target_length - current_length
    
    # Use np.pad with 'edge' mode to repeat the last highway
    # For a 3D array, pad_width needs three tuples, one for each dimension
    # We only want to pad the first dimension (highways)
    padded_highways = np.pad(
        highways_coords, 
        pad_width=((0, padding_needed), (0, 0), (0, 0)), 
        mode="edge"
    )
    
    return padded_highways


batch_haversine = jax.jit(jax.vmap(haversine_distances, in_axes=(0, 0)))


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

            candidate_highways_coords = np.array(
                [
                    pad_linestring(highway_coords[idx])
                    for idx in segment.candidate_highway_indexes
                ]
            )

            if len(candidate_highways_coords) == 0:
                continue

            candidate_highways_coords = pad_highways(candidate_highways_coords, 256)

            # distance_tasks.append(
            #     (jnp.array(pad_linestring(segment_coords, 128)), jnp.array(candidate_highways_coords))
            # )
            distances = haversine_distances(jnp.array(pad_linestring(segment_coords, 128)), jnp.array(candidate_highways_coords))

            # distances = distances.block_until_ready()

            # Now you can use segment_coords for further processing
            # For example, print the number of coordinates in each segment
            print(
                f"Segment {j} of trip {trip_dimensions.id}: {len(segment_coords)} coordinates, {len(segment.candidate_highway_indexes)} candidate highways"
            )

        # # Extract arrays from the list of tuples
        # # Process in minibatches of 64 segments
        # batch_size = 32
        # all_distances = []
        
        # for i in range(0, len(distance_tasks), batch_size):
        #     batch_tasks = distance_tasks[i:i+batch_size]
            
        #     # Skip if batch is empty
        #     if not batch_tasks:
        #         continue
                
        #     print(f"Processing minibatch {i//batch_size + 1} with {len(batch_tasks)} segments")
            
        #     # Extract arrays from the current batch
        #     batch_ref_linestrings = jnp.array([task[0] for task in batch_tasks])
        #     batch_target_linestrings = jnp.array([task[1] for task in batch_tasks])

        #     # print(batch_ref_linestrings.shape)
        #     # print(batch_target_linestrings.shape)

        #     batch_ref_linestrings = jnp.pad(
        #         batch_ref_linestrings,
        #         pad_width=[(0, batch_size - len(batch_tasks)), (0, 0), (0, 0)],
        #         mode="empty"
        #     )
        #     batch_target_linestrings = jnp.pad(
        #         batch_target_linestrings,
        #         pad_width=[(0, batch_size - len(batch_tasks)), (0, 0), (0, 0), (0, 0)],
        #         mode="empty"
        #     )
            
        #     # Use vmap to process this batch
        #     batch_distances = batch_haversine(batch_ref_linestrings, batch_target_linestrings)
            
        #     # # Force computation to complete for this batch
        #     # batch_distances = batch_distances.block_until_ready()
            
        #     # Store the results
        #     # all_distances.append(batch_distances)
            
        # # Combine results from all batches if needed
        # if all_distances:
        #     distances = jnp.concatenate(all_distances, axis=0)
        # else:
        #     distances = jnp.array([])


if __name__ == "__main__":
    process_trip_indexes()
