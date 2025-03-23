from typing import List
from pydantic import BaseModel
from .rwgps import ActivityType


class TripSegmentIndex(BaseModel):
    idx: int
    start_idx: int
    end_idx: int
    candidate_highway_indexes: List[int]


class TripSegmentIndexes(BaseModel):
    trip_id: int
    segments: List[TripSegmentIndex]


class TripSegmentDimensions(BaseModel):
    user_id: int
    trip_id: int
    activity_type: ActivityType
    elevation_gain_m: List[float]
    elevation_loss_m: List[float]
    offset_x: List[float]
    offset_y: List[float]
    distance_m: List[float]
    elapsed_time_secs: List[float]
    moving_time_secs: List[float]
    matched_highway_idx: List[int]
    matched_boundary_idxs: List[List[int]]
    mean_heart_rate_bpm: List[float]
    mean_temperature_c: List[float]
    mean_cadence_rpm: List[float]
    mean_power_w: List[float]
