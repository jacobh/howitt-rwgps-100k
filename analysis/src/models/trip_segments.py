from typing import List, Optional
from pydantic import BaseModel
from .rwgps import ActivityType


class TripSegmentIndex(BaseModel):
    idx: int
    start_idx: int
    end_idx: int
    candidate_highway_indexes: List[int]
    boundary_indexes: List[int]


class TripSegmentIndexes(BaseModel):
    trip_id: int
    segments: List[TripSegmentIndex]


class TripSegmentData(BaseModel):
    elevation_gain_m: float
    elevation_loss_m: float
    offset_x: float
    offset_y: float
    distance_m: float
    elapsed_time_secs: float
    moving_time_secs: float
    matched_highway_idx: Optional[int]
    matched_boundary_idxs: List[int]
    mean_heart_rate_bpm: Optional[float]
    mean_temperature_c: Optional[float]
    mean_cadence_rpm: Optional[float]
    mean_power_w: Optional[float]


class TripSegmentDimensions(BaseModel):
    user_id: int
    trip_id: int
    activity_type: Optional[ActivityType]
    elevation_gain_m: List[float]
    elevation_loss_m: List[float]
    offset_x: List[float]
    offset_y: List[float]
    distance_m: List[float]
    elapsed_time_secs: List[float]
    moving_time_secs: List[float]
    matched_highway_idx: List[Optional[int]]
    matched_boundary_idxs: List[List[int]]
    mean_heart_rate_bpm: List[Optional[float]]
    mean_temperature_c: List[Optional[float]]
    mean_cadence_rpm: List[Optional[float]]
    mean_power_w: List[Optional[float]]


def collect_trip_segment_dimensions(
    user_id: int,
    trip_id: int,
    activity_type: Optional[ActivityType],
    segments: List[TripSegmentData],
) -> TripSegmentDimensions:
    """
    Collects segment data from a list of TripSegmentData objects and
    organizes them into a TripSegmentDimensions object.
    
    Args:
        user_id: The user identifier
        trip_id: The trip identifier
        activity_type: The type of activity (cycling, running, etc.)
        segments: List of segment data objects
        
    Returns:
        A TripSegmentDimensions object with collected data
    """
    return TripSegmentDimensions(
        user_id=user_id,
        trip_id=trip_id,
        activity_type=activity_type,
        elevation_gain_m=[segment.elevation_gain_m for segment in segments],
        elevation_loss_m=[segment.elevation_loss_m for segment in segments],
        offset_x=[segment.offset_x for segment in segments],
        offset_y=[segment.offset_y for segment in segments],
        distance_m=[segment.distance_m for segment in segments],
        elapsed_time_secs=[segment.elapsed_time_secs for segment in segments],
        moving_time_secs=[segment.moving_time_secs for segment in segments],
        matched_highway_idx=[segment.matched_highway_idx for segment in segments],
        matched_boundary_idxs=[segment.matched_boundary_idxs for segment in segments],
        mean_heart_rate_bpm=[segment.mean_heart_rate_bpm for segment in segments],
        mean_temperature_c=[segment.mean_temperature_c for segment in segments],
        mean_cadence_rpm=[segment.mean_cadence_rpm for segment in segments],
        mean_power_w=[segment.mean_power_w for segment in segments],
    )