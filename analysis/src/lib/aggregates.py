from typing import Tuple
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