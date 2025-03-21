import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import shapely


def haversine_distances(
    ref_linestring: jnp.ndarray, 
    target_linestrings: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate haversine distances between reference linestring points and multiple target linestrings.
    
    Parameters:
    -----------
    ref_linestring : np.ndarray
        Array of reference linestring points with shape (N, 2) where each point is [lon, lat]
    target_linestrings : np.ndarray
        Array of target linestrings with shape (L, M, 2) where:
        - L is the number of linestrings
        - M is the number of points per linestring
        - 2 represents [lon, lat] coordinates
    radius : float, optional
        Earth radius in meters, default is 6371000.0
    
    Returns:
    --------
    np.ndarray
        Array of shape (L, N, M) containing haversine distances in meters
        where distances[j, i, k] is the distance between ref_linestring[i] and 
        target_linestrings[j, k]
        - First dimension (L): index of target linestring
        - Second dimension (N): index of reference point
        - Third dimension (M): index of target point
    
    Notes:
    ------
    This function can handle large arrays using vectorized operations for efficiency.
    """
    radius: float = 6371000.0  # Earth radius in meters

    # Get dimensions
    n_refs = ref_linestring.shape[0]
    
    # Reshape reference points to allow broadcasting
    # Shape: (N, 1, 1, 2) for broadcasting with target linestrings
    ref_reshaped = ref_linestring.reshape(n_refs, 1, 1, 2)
    
    # Convert to radians - we'll do this after reshaping for proper broadcasting
    ref_lon = jnp.radians(ref_reshaped[..., 0])  # Shape (N, 1, 1)
    ref_lat = jnp.radians(ref_reshaped[..., 1])  # Shape (N, 1, 1)
    target_lon = jnp.radians(target_linestrings[..., 0])  # Shape (L, M)
    target_lat = jnp.radians(target_linestrings[..., 1])  # Shape (L, M)
    
    # Differences with broadcasting
    dlon = target_lon - ref_lon  # Broadcasting to shape (N, L, M)
    dlat = target_lat - ref_lat  # Broadcasting to shape (N, L, M)
    
    # Haversine formula
    a = jnp.sin(dlat/2)**2 + jnp.cos(ref_lat) * jnp.cos(target_lat) * jnp.sin(dlon/2)**2
    c = 2 * jnp.arcsin(jnp.sqrt(a))
    distance = radius * c  # Distance in meters
    
    # Transpose the dimensions to get shape (L, N, M)
    distance = jnp.transpose(distance, (1, 0, 2))
    
    return distance

@jax.jit
@eqx.debug.assert_max_traces(max_traces=1)
def mean_min_distances(
    ref_linestring: jnp.ndarray, 
    target_linestrings: jnp.ndarray,
    ref_mask: jnp.ndarray,
    target_masks: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate the mean of minimum distances between a reference linestring and multiple target linestrings,
    accounting for padding in both the reference linestring and target linestrings.
    
    Parameters:
    -----------
    ref_linestring : jnp.ndarray
        Array of reference linestring points with shape (N, 2) where each point is [lon, lat]
    target_linestrings : jnp.ndarray
        Array of target linestrings with shape (L, M, 2), where some linestrings may be
        entirely padding (all points masked out)
    ref_mask : jnp.ndarray
        Boolean mask of shape (N,) where True indicates valid points and False indicates padding
        in the reference linestring.
    target_masks : jnp.ndarray
        Boolean mask of shape (L, M) where True indicates valid points and False indicates padding
        in the target linestrings. For entirely padded targets, all values in that row will be False.
    
    Returns:
    --------
    jnp.ndarray
        Array of shape (L,) containing the mean minimum distance from each valid reference
        point to each target linestring. Returns NaN for:
        - Entirely padded target linestrings (all target mask values are False)
        - Cases where all reference points are masked out
    """
    # 1. Compute distances between reference points and target linestrings
    distances = haversine_distances(ref_linestring, target_linestrings)
    
    # 2. Apply target masks to set distances from/to padded target points to infinity
    # This ensures they won't be selected as minimums
    # Reshape target_masks to (L, 1, M) for proper broadcasting with distances (L, N, M)
    n_refs = ref_linestring.shape[0]
    expanded_target_masks = jnp.expand_dims(target_masks, 1)  # Shape (L, 1, M)
    # Repeat the mask for each reference point
    expanded_target_masks = jnp.repeat(expanded_target_masks, n_refs, axis=1)  # Shape (L, N, M)
    
    # Set distances to padded target points to infinity
    masked_distances = jnp.where(expanded_target_masks, distances, jnp.inf)
    
    # 3. For each reference point, find the minimum distance to each target linestring
    # For entirely padded target linestrings, this will be infinity for all reference points
    min_distances = jnp.min(masked_distances, axis=2)  # Shape (L, N)
    
    # 4. Apply reference mask to exclude padded reference points from the mean calculation
    expanded_ref_mask = jnp.expand_dims(ref_mask, 0)  # Shape (1, N)
    
    # Calculate mean only over valid reference points
    valid_distances = jnp.where(expanded_ref_mask, min_distances, 0.0)
    valid_ref_counts = jnp.sum(expanded_ref_mask, axis=1)  # Shape (L,)
    
    # Calculate sum of valid distances
    sum_valid_distances = jnp.sum(valid_distances, axis=1)
    
    # 5. Identify completely padded target linestrings
    valid_target_counts = jnp.sum(target_masks, axis=1)  # Shape (L,)
    has_valid_targets = valid_target_counts > 0  # Targets with at least one valid point
    
    # 6. Calculate mean distances, handling edge cases
    mean_distances = sum_valid_distances / valid_ref_counts
    
    # Return NaN for:
    # - Completely padded target linestrings (all mask values are False)
    # - Cases where all reference points are masked out
    valid_computation = (valid_ref_counts > 0) & has_valid_targets
    mean_distances = jnp.where(valid_computation, mean_distances, jnp.nan)
    
    return mean_distances


def generate_bbox(coords: np.ndarray) -> np.ndarray:
    """
    Generate a bounding box for a trip as a 2x2 matrix.

    The bounding box is represented as:
        [[min_x, min_y],
         [max_x, max_y]]
    
    Args:
        trip_coords: A numpy array of shape (N, 2) containing the trip's (x, y) coordinates.
    
    Returns:
        A 2x2 numpy array representing the bounding box.
        
    Raises:
        ValueError: If the input array is empty.
    """
    if coords.size == 0:
        raise ValueError("Input trip_coords is empty, cannot generate bounding box.")
    
    # Compute the minimum and maximum coordinates along each axis
    min_vals = np.min(coords, axis=0)
    max_vals = np.max(coords, axis=0)
    
    # Stack the min and max values into a 2x2 matrix
    return np.vstack((min_vals, max_vals))

def pad_bbox(bbox: np.ndarray, padding_m: float) -> np.ndarray:
    """
    Pad a geographic bounding box by a specified distance in meters on all sides.

    The bounding box is expected to be in lon, lat pairs in the following format:
        [[min_lon, min_lat],
         [max_lon, max_lat]]
    Since the padding is provided in meters, this function converts the
    meter padding to degrees using an approximate conversion:
      - For latitude: 1 degree is approximately 111.32 km.
      - For longitude: 1 degree varies with latitude, approximately 
        111.32 km * cos(latitude), where latitude is taken as the center latitude.
    
    Args:
        bbox: A 2x2 numpy array representing the bounding box.
        padding_m: A float representing the padding distance in meters.
    
    Returns:
        A 2x2 numpy array representing the padded bounding box.
        
    Raises:
        ValueError: If the input bbox does not have shape (2, 2).
    """
    if bbox.shape != (2, 2):
        raise ValueError("Input bbox must be a 2x2 numpy array.")
    
    min_lon, min_lat = bbox[0]
    max_lon, max_lat = bbox[1]
    
    # Compute the center latitude for adjusting the longitude conversion factor
    center_lat = (min_lat + max_lat) / 2.0

    # Approximate conversion factors:
    # 1 degree latitude is ~111320 meters.
    lat_deg_per_meter = 1.0 / 111320.0
    # 1 degree longitude (~ at center latitude) in meters
    long_deg_per_meter = 1.0 / (111320.0 * np.cos(np.deg2rad(center_lat)))
    
    # Convert the padding from meters to degrees
    lat_padding = padding_m * lat_deg_per_meter
    long_padding = padding_m * long_deg_per_meter
    
    padded_min = np.array([min_lon - long_padding, min_lat - lat_padding])
    padded_max = np.array([max_lon + long_padding, max_lat + lat_padding])
    
    return np.vstack((padded_min, padded_max))


def numpy_bbox_to_shapely(bbox: np.ndarray) -> shapely.geometry.Polygon:
    """
    Convert a 2x2 numpy array bounding box to a Shapely Polygon.
    
    The input bbox must be a 2x2 numpy array:
        [[min_x, min_y],
         [max_x, max_y]]
    
    Returns:
        A Shapely Polygon representing the bounding box.
        
    Raises:
        ValueError: If the input array does not have shape (2, 2).
    """
    if bbox.shape != (2, 2):
        raise ValueError("Input bbox must be a 2x2 numpy array.")
    
    min_x, min_y = bbox[0]
    max_x, max_y = bbox[1]
    
    return shapely.geometry.box(min_x, min_y, max_x, max_y)