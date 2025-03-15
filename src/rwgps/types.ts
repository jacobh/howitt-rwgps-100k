/**
 * RWGPS API Types
 */

// Common types
export type Coordinate = {
  lat: number;
  lng: number;
};

export type BoundingBox = [number, number, number, number]; // [south, west, north, east]

export type ActivityType =
  | "cycling"
  | "cycling:mountain"
  | "cycling:road"
  | "cycling:gravel"
  | "running"
  | "walking"
  | "hiking";

export type TrackType = "loop" | "out_and_back" | "point_to_point";

export type Terrain = "flat" | "rolling" | "climbing" | "mountainous";

export type Difficulty =
  | "easy"
  | "moderate"
  | "challenging"
  | "difficult"
  | "extreme";

export type Visibility = 0 | 1 | 2 | 3 | 4; // 0 = public, 4 = private, etc.

// Explore API Request Parameters
export interface ExploreTripsParams {
  boundingBox: BoundingBox;
  models?: ("trips" | "routes")[];
  sortBy?:
    | "relevance desc"
    | "date desc"
    | "date asc"
    | "distance desc"
    | "distance asc";
  limit?: number;
  offset?: number;
  nextAssets?: string;
}

// Explore API Response Types
export interface TripSummary {
  id: number;
  type: string;
  url: string;
  administrative_area: string;
  avg_speed: number;
  distance: number;
  duration: number;
  elevation_gain: number;
  elevation_loss: number;
  first_lat: number;
  first_lng: number;
  last_lat: number;
  last_lng: number;
  locality: string;
  name: string;
  activity_type: ActivityType;
  byline_name: string;
  simplified_polyline: string;
  track_type: TrackType;
  terrain: Terrain;
  difficulty: Difficulty;
}

export interface UserSummary {
  type: string;
  id: number;
  user: {
    id: number;
    name: string;
    profile_photo_path: string;
    locality: string;
    country_code: string;
  };
}

export interface Permissions {
  navigate: boolean;
  customize_export_options: boolean;
  export_as_history: boolean;
  can_be_copied: boolean;
  can_be_modified: boolean;
}

export interface ExploreMetadata {
  api_version: number;
  explore: {
    bounding_box: Coordinate[];
    models: string[];
    total_count: number;
    next_page_url: string | null;
  };
  type: string;
}

export interface ExploreTripsResponse {
  results: TripSummary[];
  total_count: number;
  extras: UserSummary[];
  permissions: Record<string, Permissions>;
  meta: ExploreMetadata;
}

// Trip Details API Response Types
export interface TrackPoint {
  x: number; // longitude
  y: number; // latitude
  e: number; // elevation (m)
  d: number; // distance (m)
  s: number; // speed (m/s)
  t: number; // timestamp
  T?: number; // temperature (C)
  h?: number; // heart rate
  c?: number; // cadence
  p?: number; // power
}

export interface MetricSummary {
  max: number;
  min: number;
  avg: number;
}

export interface HillSegment {
  first_i: number;
  last_i: number;
  ele_gain: number;
  ele_loss: number;
  distance: number;
  avg_grade: number;
  is_climb: boolean;
}

export interface TripMetrics {
  ele: MetricSummary;
  hr?: MetricSummary;
  speed: MetricSummary;
  grade: MetricSummary;
  temp?: MetricSummary;
  stationary: boolean;
  pace: number;
  movingPace: number;
  vam: number;
  hills: HillSegment[];
}

export interface User {
  id: number;
  name: string;
  locality: string;
  administrative_area: string;
  country_code: string;
  created_at: string;
  account_level: number;
}

export interface Trip {
  id: number;
  name: string;
  administrative_area: string;
  country_code: string;
  locality: string;
  created_at: string;
  departed_at: string;
  description: string | null;
  distance: number;
  duration: number;
  moving_time: number;
  elevation_gain: number;
  elevation_loss: number;
  avg_speed: number;
  max_speed: number;
  min_hr?: number;
  max_hr?: number;
  activity_type: ActivityType;
  visibility: Visibility;
  likes_count: number;
  views: number;

  // Bounding box coordinates
  first_lat: number;
  first_lng: number;
  last_lat: number;
  last_lng: number;
  ne_lat: number;
  ne_lng: number;
  sw_lat: number;
  sw_lng: number;

  // Track points and metrics
  track_points: TrackPoint[];
  metrics: TripMetrics;
}

export interface TripDetailsResponse {
  trip: Trip;
  permissions: Permissions;
  user?: User;
}
