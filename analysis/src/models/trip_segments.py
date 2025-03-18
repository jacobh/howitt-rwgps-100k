from typing import List
from pydantic import BaseModel


class TripSegment(BaseModel):
    start_idx: int
    candidate_highway_indexes: List[int]


class TripSegments(BaseModel):
    trip_id: int
    segments: List[TripSegment]
    
