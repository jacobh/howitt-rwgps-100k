from typing import List, Optional, Tuple
from pydantic import BaseModel

class TripDimensions(BaseModel):
    id: int
    user_id: int
    distance_m: float
    created_at: int
    departed_at: int

    # dimensions
    coords: List[Optional[Tuple[float, float]]]
    elevation: List[Optional[float]]
    distance: List[Optional[float]]
    speed: List[Optional[float]]
    time: List[Optional[float]]
    heart_rate: List[Optional[float]]
    temperature: List[Optional[float]]
    cadence: List[Optional[float]]
    power: List[Optional[float]]
