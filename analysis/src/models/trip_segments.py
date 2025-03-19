from typing import List
from pydantic import BaseModel


class TripSegmentIndex(BaseModel):
    start_idx: int
    candidate_highway_indexes: List[int]


class TripSegmentIndexes(BaseModel):
    trip_id: int
    segments: List[TripSegmentIndex]
    
