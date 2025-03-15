import jax
import jax.numpy as jnp


def degrees_to_radians(degrees):
    """Convert degrees to radians using JAX"""
    return degrees * jnp.pi / 180.0


def radians_to_degrees(radians):
    """Convert radians to degrees using JAX"""
    return radians * 180.0 / jnp.pi


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


def calculate_elevation_difference(elev1, elev2):
    """
    Calculate elevation difference between two points.

    Args:
        elev1: Elevation of point 1 in meters
        elev2: Elevation of point 2 in meters

    Returns:
        Elevation difference in meters (positive if climbing, negative if descending)
    """
    return elev2 - elev1


def calculate_time_difference(time1, time2):
    """
    Calculate time difference between two points.

    Args:
        time1: Time value of point 1 (in seconds)
        time2: Time value of point 2 (in seconds)

    Returns:
        Time difference in seconds
    """
    return time2 - time1


def point_delta(arrays, i):
    """
    Calculate delta between the current point (i) and the next point (i+1)

    Args:
        arrays: Dictionary of JAX arrays containing track data
        i: Index of the current point

    Returns:
        tuple containing:
            elevation_m: Elevation difference in meters (can be negative)
            distance_m: Distance traveled in meters
            bearing: Bearing in degrees (0-360)
            duration_secs: Time duration in seconds
    """
    # Get current and next points
    point1 = arrays["latlng"][i]  # [lat, lng]
    point2 = arrays["latlng"][i + 1]  # [lat, lng]

    # Calculate elevation difference
    elevation_m = calculate_elevation_difference(
        arrays["elevation"][i], arrays["elevation"][i + 1]
    )

    # Calculate distance
    distance_m = haversine_distance(
        point1[0],
        point1[1],  # lat1, lng1
        point2[0],
        point2[1],  # lat2, lng2
    )

    # Calculate bearing
    bearing = calculate_bearing(
        point1[0],
        point1[1],  # lat1, lng1
        point2[0],
        point2[1],  # lat2, lng2
    )

    # Calculate time duration in seconds
    duration_secs = calculate_time_difference(arrays["time"][i], arrays["time"][i + 1])

    # Return all values
    return elevation_m, distance_m, bearing, duration_secs


@jax.jit
def all_point_deltas(arrays):
    """
    Calculate deltas between all consecutive points in the track at once using JAX vectorization.

    Args:
        arrays: Dictionary of JAX arrays containing track data

    Returns:
        Dictionary containing arrays of:
            elevation_m: Elevation differences in meters (can be negative)
            distance_m: Distances traveled in meters
            bearing: Bearings in degrees (0-360)
            duration_secs: Time durations in seconds
    """
    # Get total number of points
    n_points = len(arrays["latlng"])

    # Handle empty or single-point tracks using regular Python control flow
    if n_points <= 1:
        return {
            "elevation_m": jnp.array([]),
            "distance_m": jnp.array([]),
            "bearing": jnp.array([]),
            "duration_secs": jnp.array([]),
        }

    # Create indices for all points except the last one
    indices = jnp.arange(n_points - 1)

    # Vectorize the point_delta function
    vectorized_point_delta = jax.vmap(lambda i: point_delta(arrays, i))

    # Calculate all deltas at once
    elev_diffs, distances, bearings, durations = vectorized_point_delta(indices)

    return {
        "elevation_m": elev_diffs,
        "distance_m": distances,
        "bearing": bearings,
        "duration_secs": durations,
    }
