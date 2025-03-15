import numpy as np
from .geo import haversine_distance

def create_trip_segments(arrays, deltas, min_distance_m=1000):
    """
    Split a trip into segments, each with a minimum distance traveled.
    
    Args:
        arrays: Dictionary of JAX arrays containing track data
        deltas: Dictionary of JAX arrays containing point-to-point deltas
        min_distance_m: Minimum distance in meters for each segment (default: 1000m)
        
    Returns:
        List of tuples (segment_arrays, segment_deltas), where each tuple represents a segment
    """
    n_points = arrays["latlng"].shape[0]
    
    if n_points <= 2:
        return [(arrays, deltas)]
    
    # Find segment boundaries
    boundaries = []
    segment_start = 0
    
    # Convert to numpy for easier processing
    latlng = np.array(arrays["latlng"])
    
    for i in range(1, n_points):
        first_point = latlng[segment_start]
        current_point = latlng[i]
        
        # Calculate haversine distance
        lat1, lng1 = first_point
        lat2, lng2 = current_point
        
        # We'll use the haversine_distance function from your existing code
        distance = haversine_distance(lat1, lng1, lat2, lng2)
        
        if distance >= min_distance_m:
            boundaries.append(i)
            segment_start = i
    
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