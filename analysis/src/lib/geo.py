import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import shapely


def haversine_distances(
    ref_linestring: jnp.ndarray, target_linestrings: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate haversine distances between reference linestring points and multiple target linestrings.

    Parameters:
    -----------
    ref_linestring : jnp.ndarray
        Array of reference linestring points with shape (N, 2) where each point is [lon, lat]
    target_linestrings : jnp.ndarray
        Array of target linestrings with shape (L, M, 2) where:
        - L is the number of linestrings
        - M is the number of points per linestring
        - 2 represents [lon, lat] coordinates

    Returns:
    --------
    jnp.ndarray
        Array of shape (L, M, N) containing haversine distances in meters
        where distances[j, k, i] is the distance between ref_linestring[i] and
        target_linestrings[j, k]
        - First dimension (L): index of target linestring
        - Second dimension (M): index of target point
        - Third dimension (N): index of reference point
    """
    radius: float = 6371000.0  # Earth radius in meters

    # Get dimensions
    n_refs = ref_linestring.shape[0]

    # Reshape reference points to allow broadcasting
    # Shape: (1, 1, N, 2) for broadcasting with target linestrings
    ref_reshaped = ref_linestring.reshape(1, 1, n_refs, 2)

    # Convert to radians - we'll do this after reshaping for proper broadcasting
    ref_lon = jnp.radians(ref_reshaped[..., 0])  # Shape (1, 1, N)
    ref_lat = jnp.radians(ref_reshaped[..., 1])  # Shape (1, 1, N)
    target_lon = jnp.radians(target_linestrings[..., 0])  # Shape (L, M)
    target_lat = jnp.radians(target_linestrings[..., 1])  # Shape (L, M)

    # Reshape target coordinates to (L, M, 1) for broadcasting
    target_lon = jnp.expand_dims(target_lon, -1)  # Shape (L, M, 1)
    target_lat = jnp.expand_dims(target_lat, -1)  # Shape (L, M, 1)

    # Differences with broadcasting
    dlon = target_lon - ref_lon  # Broadcasting to shape (L, M, N)
    dlat = target_lat - ref_lat  # Broadcasting to shape (L, M, N)

    # Haversine formula
    a = (
        jnp.sin(dlat / 2) ** 2
        + jnp.cos(ref_lat) * jnp.cos(target_lat) * jnp.sin(dlon / 2) ** 2
    )
    c = 2 * jnp.arcsin(jnp.sqrt(a))
    distance = radius * c  # Distance in meters

    return distance  # Shape (L, M, N)


@jax.jit
@eqx.debug.assert_max_traces(max_traces=24)
def mean_min_distances(
    ref_linestring: jnp.ndarray,
    target_linestrings: jnp.ndarray,
    ref_mask: jnp.ndarray,
    target_masks: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate the mean of minimum distances between a reference linestring and multiple target linestrings,
    accounting for padding in both the reference linestring and target linestrings.

    Parameters:
    -----------
    ref_linestring : jnp.ndarray
        Array of reference linestring points with shape (N, 2) where each point is [lon, lat]
        and N is the number of points in the reference linestring
    target_linestrings : jnp.ndarray
        Array of target linestrings with shape (L, M, 2) where:
        - L is the number of target linestrings
        - M is the number of points per target linestring
        - 2 represents [lon, lat] coordinates
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
        point to each target linestring. Each value represents:
        - For each target linestring: the average of the minimum distances from each valid
          reference point to any valid point in that target linestring

        Returns NaN for:
        - Entirely padded target linestrings (all target mask values are False)
        - Cases where all reference points are masked out
    """
    # 1. Compute distances between reference points and target linestrings
    distances = haversine_distances(
        ref_linestring, target_linestrings
    )  # Shape (L, M, N)

    # 2. Apply target masks to set distances from/to padded target points to infinity
    # Reshape target_masks to (L, M, 1) for proper broadcasting with distances (L, M, N)
    expanded_target_masks = jnp.expand_dims(target_masks, -1)  # Shape (L, M, 1)

    # Set distances to padded target points to infinity
    masked_distances = jnp.where(expanded_target_masks, distances, jnp.inf)

    # 3. For each reference point, find the minimum distance to each target linestring
    # We need to find min over dimension M
    min_distances = jnp.min(masked_distances, axis=1)  # Shape (L, N)

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


def linestring_distance(linestring: np.ndarray) -> float:
    """
    Calculate the total distance of a linestring in meters using haversine formula.

    Parameters:
    -----------
    linestring : np.ndarray
        Array of linestring points with shape (N, 2) where each point is [lon, lat]

    Returns:
    --------
    float
        Total distance of the linestring in meters

    Notes:
    ------
    Returns 0.0 if the linestring has fewer than 2 points.
    """
    if len(linestring) < 2:
        return 0.0

    radius = 6371000.0  # Earth radius in meters

    # Convert to radians
    points_rad = np.radians(linestring)

    # Get consecutive points
    points1 = points_rad[:-1]  # All points except the last one
    points2 = points_rad[1:]  # All points except the first one

    # Unpack coordinates
    lon1, lat1 = points1[:, 0], points1[:, 1]
    lon2, lat2 = points2[:, 0], points2[:, 1]

    # Differences
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = radius * c  # Distances in meters

    # Sum all segment distances
    total_distance = np.sum(distances)

    return float(total_distance)

def linestring_offset(linestring: np.ndarray) -> np.ndarray:
    """
    Calculate the offset in meters between the first and last point of a linestring.
    The first point is treated as (0,0), and the result will show x/y offsets in meters.
    
    Parameters:
    -----------
    linestring : np.ndarray
        Array of linestring points with shape (N, 2) where each point is [lon, lat]
        
    Returns:
    --------
    np.ndarray
        Array with shape (2,) containing [x_offset, y_offset] in meters
        where x_offset is the east-west offset and y_offset is the north-south offset
    
    Notes:
    ------
    Returns [0.0, 0.0] if the linestring has fewer than 2 points.
    Uses spherical approximation to calculate offsets.
    """
    if len(linestring) < 2:
        return np.array([0.0, 0.0])

    endpoints = np.vstack([linestring[0], linestring[-1]])
    
    # Convert all points to radians for vectorized operations
    points_rad = np.radians(endpoints)
    
    # Extract first and last points (now in radians)
    first_point = points_rad[0]
    last_point = points_rad[-1]
    
    # Unpack coordinates
    lon1, lat1 = first_point
    lon2, lat2 = last_point
    
    # Earth radius in meters
    radius = 6371000.0
    
    # Calculate x offset (east-west)
    # We use the latitude of the first point for the longitude scaling
    x_offset = radius * np.cos(lat1) * (lon2 - lon1)
    
    # Calculate y offset (north-south)
    y_offset = radius * (lat2 - lat1)
    
    return np.array([float(x_offset), float(y_offset)])