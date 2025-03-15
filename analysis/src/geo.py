import jax
import jax.numpy as jnp

def degrees_to_radians(degrees):
    """Convert degrees to radians using JAX"""
    return degrees * jnp.pi / 180.0


def radians_to_degrees(radians):
    """Convert radians to degrees using JAX"""
    return radians * 180.0 / jnp.pi

@jax.jit
def haversine_distance(lat1, lng1, lat2, lng2):
    """
    Calculate the great-circle distance between two points
    using the haversine formula with JAX.

    Args:
        lat1, lng1: Latitude and longitude of point 1 in degrees
        lat2, lng2: Latitude and longitude of point 2 in degrees

    Returns:
        Distance between points in meters
    """
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

def calculate_bearing(lat1, lng1, lat2, lng2):
    """
    Calculate the initial bearing between two points using JAX.

    Args:
        lat1, lng1: Latitude and longitude of point 1 in degrees
        lat2, lng2: Latitude and longitude of point 2 in degrees

    Returns:
        Bearing in degrees (0-360)
    """
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
    return (radians_to_degrees(bearing_rad) + 360.0) % 360.0

def vectorized_haversine_distances(reference_point, points):
    """
    Calculate haversine distances from a reference point to multiple points.
    
    Args:
        reference_point: A single point [lat, lng] in degrees
        points: Array of points with shape (n, 2) where each row is [lat, lng] in degrees
        
    Returns:
        Array of distances in meters from reference point to each point in points
    """
    # Extract coordinates
    ref_lat, ref_lng = reference_point
    
    # Convert to radians
    ref_lat_rad = degrees_to_radians(ref_lat)
    ref_lng_rad = degrees_to_radians(ref_lng)
    points_lat_rad = degrees_to_radians(points[:, 0])
    points_lng_rad = degrees_to_radians(points[:, 1])
    
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