from typing import Tuple, Optional
import numpy as np

def agg_elevations(elevations: np.ndarray) -> Tuple[float, float]:
    """
    Aggregate elevation data to compute gain and loss.
    
    Args:
        elevations (np.ndarray): Array of elevation values.
        
    Returns:
        Tuple[float, float]: A tuple containing (elevation_gain_m, elevation_loss_m).
        Both values are positive numbers, with loss being the absolute value
        of the sum of negative elevation changes.
    """
    if len(elevations) < 2:
        return 0.0, 0.0
    
    # Calculate differences between consecutive elevation points
    diffs = np.diff(elevations)
    
    # Sum all positive differences for elevation gain
    gain_m = float(np.sum(diffs[diffs > 0]))
    
    # Sum absolute value of negative differences for elevation loss
    loss_m = float(np.abs(np.sum(diffs[diffs < 0])))
    
    return gain_m, loss_m

def agg_moving_time_secs(coords: np.ndarray, times: np.ndarray) -> Optional[float]:
    """
    Aggregate moving time in seconds based on coordinate and time data.
    Calculates total time where movement speed is greater than 1 km/h.
    
    Args:
        coords: Array of coordinate points with shape (N, 2) where each point is [lon, lat]
        times: Array of timestamps in seconds with shape (N,)
        
    Returns:
        Total moving time in seconds, or None if inputs are invalid
    """
    if len(coords) < 2 or len(coords) != len(times):
        return None
    
    # Calculate distance between consecutive points using haversine formula
    radius = 6371000.0  # Earth radius in meters
    coords_rad = np.radians(coords)
    
    # Calculate differences between consecutive coordinates
    points1 = coords_rad[:-1]  # All points except the last one
    points2 = coords_rad[1:]   # All points except the first one
    
    # Unpack coordinates
    lon1, lat1 = points1[:, 0], points1[:, 1]
    lon2, lat2 = points2[:, 0], points2[:, 1]
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = radius * c  # Distances in meters
    
    # Calculate time differences between consecutive points in seconds
    time_diffs = np.diff(times)
    
    # Calculate speeds in km/h
    # speed = distance / time (convert m/s to km/h by multiplying by 3.6)
    speeds = (distances / time_diffs) * 3.6  # km/h
    
    # Sum time intervals where speed is greater than 1 km/h
    moving_time = np.sum(time_diffs[speeds > 1.0])
    
    return float(moving_time)
