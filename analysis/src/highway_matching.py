# Add to src/geo.py or create a new file src/highway_matching.py

import jax
import jax.numpy as jnp
from .geo import vectorized_haversine_distances


def extract_highway_coordinates(highway_feature):
    """
    Extract coordinates from a highway GeoJSON feature as JAX array.

    Args:
        highway_feature: GeoJSON feature representing a highway

    Returns:
        JAX array of shape (n_points, 2) with [lat, lng] coordinates
    """
    coords = []
    geometry = highway_feature["geometry"]

    # Handle different geometry types
    if geometry["type"] == "LineString":
        # LineString has a single list of coordinates
        # GeoJSON coordinates are in [lng, lat] order, convert to [lat, lng]
        coords = [[point[1], point[0]] for point in geometry["coordinates"]]
    elif geometry["type"] == "MultiLineString":
        # MultiLineString has a list of lists of coordinates
        for line in geometry["coordinates"]:
            coords.extend([[point[1], point[0]] for point in line])

    # Convert to JAX array
    return jnp.array(coords)

def make_highway_jax(highway, max_length=256):
    """
    Convert a highway GeoJSON feature to a padded JAX array.
    
    Args:
        highway: GeoJSON feature representing a highway
        max_length: Maximum length for the padded array
        
    Returns:
        Tuple of (padded_coords, valid) where:
        - padded_coords: JAX array of shape (max_length, 2) with [lat, lng] coordinates
        - valid: Boolean indicating if the highway had valid coordinates
    """
    try:
        # Extract coordinates from highway
        coords = extract_highway_coordinates(highway)
        
        # Ensure we have at least one coordinate
        if coords.shape[0] > 0:
            # Create padded array of the first point repeated
            first_point = coords[0]
            padded_coords = jnp.tile(first_point, (max_length, 1))
            
            # Copy the actual coordinates (limited to max_length)
            actual_length = min(coords.shape[0], max_length)
            padded_coords = padded_coords.at[:actual_length].set(coords[:actual_length])
            
            return padded_coords, True
        else:
            # Return a dummy array filled with zeros and False flag
            return jnp.zeros((max_length, 2)), False
            
    except Exception as e:
        print(f"Error extracting coordinates from highway: {e}")
        return jnp.zeros((max_length, 2)), False

def make_highway_jaxs(highways, max_length=256):
    """
    Convert multiple highway GeoJSON features to padded JAX arrays.
    
    Args:
        highways: List of GeoJSON features representing highways
        max_length: Maximum length for each padded array
        
    Returns:
        Tuple of (highway_batch, valid_highways) where:
        - highway_batch: JAX array of shape (n_valid_highways, max_length, 2)
        - valid_highways: List of valid highway objects
    """
    padded_highways = []
    valid_highways = []
    
    for highway in highways:
        padded_coords, is_valid = make_highway_jax(highway, max_length)
        if is_valid:
            padded_highways.append(padded_coords)
            valid_highways.append(highway)
    
    if not padded_highways:
        return jnp.zeros((0, max_length, 2)), []
    
    # Stack highways into a single batch array
    highway_batch = jnp.stack(padded_highways)
    
    return highway_batch, valid_highways


@jax.jit
def min_distance_to_highway(point, highway_coords):
    """
    Find the minimum distance from a point to any point on a highway.

    Args:
        point: Single point as [lat, lng]
        highway_coords: JAX array of highway coordinates of shape (m_points, 2)

    Returns:
        Minimum distance in meters
    """
    # Calculate distances from point to all highway points
    distances = vectorized_haversine_distances(point, highway_coords)

    # Return the minimum distance
    return jnp.min(distances)


def segment_nearest_highway(segment_latlng, highways):
    """
    Find the highway that best matches a segment based on minimum distance.
    
    Args:
        segment_latlng: JAX array of shape (n_points, 2) with lat/lng coordinates
        highways: List of highways, each with a geometry to compare against
        
    Returns:
        Tuple of (best_highway_index, min_total_distance, highway_feature)
    """
    if not highways:
        return (None, float('inf'), None)
    
    # Get padded highway coordinates and valid highways
    highway_batch, valid_highways = make_highway_jaxs(highways)
    
    if not valid_highways:
        return (None, float('inf'), None)
    
    # Define a function to calculate distance from a single point to a highway
    def point_to_highway_distance(point, highway_coords):
        # Calculate distances from point to all highway points
        distances = vectorized_haversine_distances(point, highway_coords)
        # Return minimum distance
        return jnp.min(distances)
    
    # Define a function to calculate total distance from all segment points to a highway
    @jax.jit
    def segment_to_highway_distance(highway_coords):
        # Vectorize across segment points
        point_distances = jax.vmap(
            lambda point: point_to_highway_distance(point, highway_coords)
        )(segment_latlng)
        # Sum distances
        return jnp.sum(point_distances)
    
    # Vectorize across all highways in the batch
    highway_distances = jax.vmap(segment_to_highway_distance)(highway_batch)
    
    # Find the highway with minimum total distance
    best_index = jnp.argmin(highway_distances)
    
    return (int(best_index), float(highway_distances[best_index]), valid_highways[int(best_index)])


def all_segments_nearest_highways(segments, all_nearby_highways):
    """
    Find the nearest highway for each segment.

    Args:
        segments: List of segment data (each containing latlng arrays)
        all_nearby_highways: List where each element is a list of nearby highways for that segment

    Returns:
        List of (segment_index, best_highway, min_distance) tuples
    """
    results = []

    for i, (segment_arrays, _) in enumerate(segments):
        nearby_highways = all_nearby_highways[i]

        # Skip if no nearby highways
        if not nearby_highways:
            results.append((i, None, float("inf")))
            continue

        # Get the segment's lat/lng coordinates
        segment_latlng = segment_arrays["latlng"]

        # Find the best matching highway
        best_highway_index, min_distance, best_highway = segment_nearest_highway(
            segment_latlng, nearby_highways
        )

        # Store the result
        results.append((i, best_highway, min_distance))

    return results
