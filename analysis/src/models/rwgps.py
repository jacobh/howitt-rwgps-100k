from typing import List, Optional, Any, Literal
from pydantic import BaseModel

# Literal types corresponding to the various string unions
ActivityType = Literal[
    "cycling",
    "cycling:mountain",
    "cycling:road",
    "cycling:gravel",
    "running",
    "walking",
    "hiking",
]

TrackType = Literal["loop", "out_and_back", "point_to_point"]

Terrain = Literal["flat", "rolling", "climbing", "mountainous"]

Difficulty = Literal["easy", "moderate", "challenging", "difficult", "extreme"]

Visibility = Literal[0, 1, 2, 3, 4]


# Trip Details API Response Types


class TrackPoint(BaseModel):
    x: Optional[float] = None  # longitude
    y: Optional[float] = None  # latitude
    e: Optional[float] = None  # elevation (m)
    d: Optional[float] = None  # distance (m)
    s: Optional[float] = None  # speed (m/s)
    t: Optional[float] = None  # timestamp
    T: Optional[float] = None  # temperature (C)
    h: Optional[float] = None  # heart rate
    c: Optional[float] = None  # cadence
    p: Optional[float] = None  # power


class Trip(BaseModel):
    id: int
    user_id: int
    name: str
    # administrative_area: str
    # country_code: str
    # locality: Optional[str] = None
    created_at: int
    departed_at: int
    # description: Optional[str]
    distance: float
    # duration: Optional[float] = None
    # moving_time: Optional[float] = None
    # elevation_gain: float
    # elevation_loss: float
    # avg_speed: Optional[float] = None
    # max_speed: Optional[float] = None
    # min_hr: Optional[float] = None
    # max_hr: Optional[float] = None
    # activity_type: Optional[ActivityType] = None
    # visibility: Visibility
    # likes_count: Optional[int] = None
    # views: Optional[int] = None
    # first_lat: float
    # first_lng: float
    # last_lat: float
    # last_lng: float
    # ne_lat: Optional[float] = None
    # ne_lng: Optional[float] = None
    # sw_lat: Optional[float] = None
    # sw_lng: Optional[float] = None
    track_points: List[TrackPoint]
    # metrics: Any
    # highlighted_photo_id: Optional[int]


class TripDetailsResponse(BaseModel):
    trip: Trip
    # permissions: Any
    # user: Optional[Any]
