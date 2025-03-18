import jax
import jax.numpy as jnp

def create_trip_segments(arrays, deltas, min_distance_m=1000):
    """
    Split a trip into segments by dividing it into roughly equal parts based on total distance.
    
    Args:
        arrays: Dictionary of JAX arrays containing track data
        deltas: Dictionary of JAX arrays containing point-to-point deltas
        min_distance_m: Target distance in meters for each segment (default: 1000m)
        
    Returns:
        List of tuples (segment_arrays, segment_deltas), where each tuple represents a segment
    """
    n_points = arrays["latlng"].shape[0]
    
    if n_points <= 2:
        return [(arrays, deltas)]
    
    # Calculate total distance of the trip
    if "distance_m" in deltas and deltas["distance_m"].size > 0:
        total_distance = jnp.sum(deltas["distance_m"])
    else:
        # If no distance deltas available, return a single segment
        return [(arrays, deltas)]
    
    # Calculate number of segments based on total distance and min_distance_m
    num_segments = max(1, int(total_distance / min_distance_m))
    
    # Calculate points per segment, ensuring at least 2 points per segment
    points_per_segment = max(2, n_points // num_segments)
    
    # Create even segment boundaries
    boundaries = []
    for i in range(1, num_segments):
        # Calculate boundary index, ensuring we don't exceed array length
        boundary_idx = min(i * points_per_segment, n_points - 1)
        boundaries.append(boundary_idx)
    
    # Create segments based on boundaries
    segments = []
    
    # First segment (if we have boundaries)
    if boundaries:
        first_boundary = boundaries[0]
        segment_arrays = {k: v[:first_boundary+1] for k, v in arrays.items()}
        segment_deltas = {k: v[:first_boundary] for k, v in deltas.items()}
        segments.append((segment_arrays, segment_deltas))
        
        # Middle segments
        for i in range(1, len(boundaries)):
            start = boundaries[i-1]
            end = boundaries[i]
            segment_arrays = {k: v[start:end+1] for k, v in arrays.items()}
            segment_deltas = {k: v[start:end] for k, v in deltas.items()}
            segments.append((segment_arrays, segment_deltas))
        
        # Last segment
        last_boundary = boundaries[-1]
        segment_arrays = {k: v[last_boundary:] for k, v in arrays.items()}
        segment_deltas = {k: v[last_boundary:] for k, v in deltas.items()}
        segments.append((segment_arrays, segment_deltas))
    else:
        # No boundaries found, just return the original data
        segments.append((arrays, deltas))
    
    return segments

@jax.jit
def create_all_segment_bboxes(segments, scale_factor=0.2):
    """
    Generate bounding boxes for all trip segments using JAX's vmap.
    Pads segments with their first point to ensure uniform length for efficient vectorization.
    Scales the bounding boxes by a specified factor for margin.
    
    Args:
        segments: List of tuples (segment_arrays, segment_deltas) as returned by create_trip_segments
        scale_factor: Factor by which to scale up the bounding box (default: 0.2 or 20%)
    
    Returns:
        JAX array: Array of bounding boxes, each in format (min_lng, min_lat, max_lng, max_lat)
    """
    if not segments:
        return jnp.zeros((0, 4))  # Return empty array with shape (0, 4)
    
    # Find the maximum segment length
    max_length = max(segment[0]["latlng"].shape[0] for segment in segments)
    
    # Pad each segment to the maximum length by repeating the first point
    def pad_segment_latlng(latlng, max_len):
        """Pad a latlng array to max_len by repeating the first point"""
        current_len = latlng.shape[0]
        
        if current_len == max_len:
            return latlng
        
        # Create padding using the first point
        first_point = latlng[0:1]  # Shape: (1, 2)
        padding = jnp.tile(first_point, (max_len - current_len, 1))  # Shape: (max_len - current_len, 2)
        
        # Concatenate the original array with padding
        return jnp.concatenate([latlng, padding], axis=0)
    
    # Pad all segments and stack into a batch
    padded_latlng_arrays = [pad_segment_latlng(segment[0]["latlng"], max_length) for segment in segments]
    latlng_batch = jnp.stack(padded_latlng_arrays)  # Shape: (num_segments, max_length, 2)
    
    # Define our bounding box calculation function with scaling
    @jax.jit
    def calculate_scaled_bbox(latlng):
        """Calculate bounding box from a latlng array and scale it up"""
        min_lat = jnp.min(latlng[:, 0])
        min_lng = jnp.min(latlng[:, 1])
        max_lat = jnp.max(latlng[:, 0])
        max_lng = jnp.max(latlng[:, 1])
        
        # Calculate the center of the bounding box
        center_lat = (min_lat + max_lat) / 2.0
        center_lng = (min_lng + max_lng) / 2.0
        
        # Calculate the dimensions of the bounding box
        lat_range = max_lat - min_lat
        lng_range = max_lng - min_lng
        
        # Scale up the dimensions by the scale factor
        scaled_lat_range = lat_range * (1.0 + scale_factor)
        scaled_lng_range = lng_range * (1.0 + scale_factor)
        
        # Calculate the new min/max coordinates based on the scaled dimensions
        scaled_min_lat = center_lat - (scaled_lat_range / 2.0)
        scaled_min_lng = center_lng - (scaled_lng_range / 2.0)
        scaled_max_lat = center_lat + (scaled_lat_range / 2.0)
        scaled_max_lng = center_lng + (scaled_lng_range / 2.0)
        
        return jnp.array([scaled_min_lng, scaled_min_lat, scaled_max_lng, scaled_max_lat])
    
    # Vectorize across the batch dimension
    batched_calculate_bbox = jax.vmap(calculate_scaled_bbox)
    
    # Calculate all scaled bounding boxes in one vectorized operation
    return batched_calculate_bbox(latlng_batch)
