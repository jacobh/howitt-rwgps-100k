import jax
import jax.numpy as jnp
import numpy as np
import shapely

def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians using JAX"""
    return degrees * jnp.pi / 180.0


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees using JAX"""
    return radians * 180.0 / jnp.pi

def degrees_to_radians_jax(degrees: jnp.ndarray) -> jnp.ndarray:
    """Convert degrees to radians using JAX"""
    return degrees * jnp.pi / 180.0

def radians_to_degrees_jax(radians: jnp.ndarray) -> jnp.ndarray:
    """Convert radians to degrees using JAX"""
    return radians * 180.0 / jnp.pi

@jax.jit
def haversine_distance_jax(p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the great-circle distance between two points
    using the haversine formula with JAX.

    Args:
        lat1, lng1: Latitude and longitude of point 1 in degrees
        lat2, lng2: Latitude and longitude of point 2 in degrees

    Returns:
        Distance between points in meters
    """
    lng1, lat1 = p1
    lng2, lat2 = p2

    # Convert lat/lng from degrees to radians
    lat1_rad = degrees_to_radians(lat1)
    lng1_rad = degrees_to_radians(lng1)
    lat2_rad = degrees_to_radians(lat2)
    lng2_rad = degrees_to_radians(lng2)

    # Haversine formula
    dlng = lng2_rad - lng1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        jnp.sin(dlat / 2.0) ** 2
        + jnp.cos(lat1_rad) * jnp.cos(lat2_rad) * jnp.sin(dlng / 2.0) ** 2
    )
    c = 2.0 * jnp.arctan2(jnp.sqrt(a), jnp.sqrt(1.0 - a))

    # Earth radius in meters
    R = 6371000.0
    return R * c


def haversine_distance_cpu(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate the great-circle distance between two points
    using the haversine formula with NumPy.

    Args:
        p1: A numpy array [longitude, latitude] of point 1 in degrees
        p2: A numpy array [longitude, latitude] of point 2 in degrees

    Returns:
        Distance between points in meters
    """
    # Extract coordinates
    lng1, lat1 = p1
    lng2, lat2 = p2

    # Convert lat/lng from degrees to radians
    lat1_rad = degrees_to_radians(lat1)
    lng1_rad = degrees_to_radians(lng1)
    lat2_rad = degrees_to_radians(lat2)
    lng2_rad = degrees_to_radians(lng2)

    # Haversine formula
    dlng = lng2_rad - lng1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlng / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    # Earth radius in meters
    R = 6371000.0
    return R * c

@jax.jit
def calculate_bearing_jax(p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the initial bearing between two points using JAX.

    Args:
        p1: A JAX array [longitude, latitude] of point 1 in degrees
        p2: A JAX array [longitude, latitude] of point 2 in degrees

    Returns:
        Bearing in degrees (0-360)
    """
    # Extract coordinates
    lng1, lat1 = p1
    lng2, lat2 = p2
    
    # Convert lat/lng from degrees to radians
    lat1_rad = degrees_to_radians(lat1)
    lng1_rad = degrees_to_radians(lng1)
    lat2_rad = degrees_to_radians(lat2)
    lng2_rad = degrees_to_radians(lng2)

    dlng = lng2_rad - lng1_rad

    x = jnp.sin(dlng) * jnp.cos(lat2_rad)
    y = jnp.cos(lat1_rad) * jnp.sin(lat2_rad) - jnp.sin(lat1_rad) * jnp.cos(
        lat2_rad
    ) * jnp.cos(dlng)

    bearing_rad = jnp.arctan2(x, y)
    # Convert to degrees and normalize to 0-360
    return (radians_to_degrees_jax(bearing_rad) + 360.0) % 360.0

def vectorized_haversine_distances(reference_point: jnp.ndarray, points: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate haversine distances from a reference point to multiple points.
    
    Args:
        reference_point: A single point [lat, lng] in degrees
        points: Array of points with shape (n, 2) where each row is [lat, lng] in degrees
        
    Returns:
        Array of distances in meters from reference point to each point in points
    """
    # Extract coordinates
    ref_lng, ref_lat,  = reference_point
    
    # Convert to radians
    ref_lat_rad = degrees_to_radians(ref_lat)
    ref_lng_rad = degrees_to_radians(ref_lng)
    points_lng_rad = degrees_to_radians_jax(points[:, 0])
    points_lat_rad = degrees_to_radians_jax(points[:, 1])
    
    # Calculate differences
    dlng = points_lng_rad - ref_lng_rad
    dlat = points_lat_rad - ref_lat_rad
    
    # Haversine formula
    a = (
        jnp.sin(dlat / 2.0) ** 2
        + jnp.cos(ref_lat_rad) * jnp.cos(points_lat_rad) * jnp.sin(dlng / 2.0) ** 2
    )
    c = 2.0 * jnp.arctan2(jnp.sqrt(a), jnp.sqrt(1.0 - a))
    
    # Earth radius in meters
    R = 6371000.0
    return R * c

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