#!/usr/bin/env python
# analysis/src/scripts/analyze_trip_segments.py

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path


def load_trip_segment_dimensions() -> List[Dict[str, Any]]:
    """Load trip segment dimensions from JSON file."""
    file_path = Path("../data/trip_segment_dimensions.json")
    print(f"Loading trip segment dimensions from {file_path}...")
    with open(file_path, "r") as f:
        return json.load(f)


def calculate_stats(
    values: List[float], use_absolute: bool = False
) -> Dict[str, float]:
    """Calculate median, mean, and stddev for a list of values, handling None values."""
    # Filter out None values
    clean_values = [v for v in values if v is not None]

    if not clean_values:
        return {"median": None, "mean": None, "stddev": None}

    # Use absolute values if specified
    if use_absolute:
        clean_values = [abs(v) for v in clean_values]

    return {
        "median": float(np.median(clean_values)),
        "mean": float(np.mean(clean_values)),
        "stddev": float(np.std(clean_values)) if len(clean_values) > 1 else 0.0,
    }


def analyze_trip_segments(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze trip segment dimensions and return statistics per trip and overall."""
    metrics = [
        "elevation_gain_m", "elevation_loss_m", "offset_x", "offset_y", 
        "distance_m", "elapsed_time_secs", "moving_time_secs", 
        "mean_heart_rate_bpm", "mean_temperature_c", "mean_cadence_rpm", "mean_power_w"
    ]
    
    # Metrics that should use absolute values
    absolute_metrics = ["offset_x", "offset_y"]
    
    # Create a dictionary to store stats per trip
    trip_stats = {}
    
    # Create lists to aggregate all values for overall stats
    all_values = {metric: [] for metric in metrics}
    
    # Track segment counts for all trips
    all_segment_counts = []
    
    # Process each trip
    for trip in data:
        trip_id = trip["trip_id"]
        segment_count = len(trip["distance_m"])
        all_segment_counts.append(segment_count)
        
        trip_stats[trip_id] = {
            "user_id": trip["user_id"],
            "activity_type": trip["activity_type"],
            "segment_count": segment_count,
            "stats": {}
        }
        
        # Calculate stats for each metric for this trip
        for metric in metrics:
            use_absolute = metric in absolute_metrics
            trip_stats[trip_id]["stats"][metric] = calculate_stats(trip[metric], use_absolute)
            
            # Add values to overall aggregation (filtering out None values)
            values_to_add = [v for v in trip[metric] if v is not None]
            if use_absolute:
                values_to_add = [abs(v) for v in values_to_add]
            all_values[metric].extend(values_to_add)
    
    # Calculate overall stats
    overall_stats = {
        metric: calculate_stats(all_values[metric]) for metric in metrics
    }
    
    # Calculate segment count statistics
    segment_count_stats = {
        "mean": float(np.mean(all_segment_counts)),
        "median": float(np.median(all_segment_counts)),
        "stddev": float(np.std(all_segment_counts)) if len(all_segment_counts) > 1 else 0.0,
        "min": int(min(all_segment_counts)),
        "max": int(max(all_segment_counts))
    }
    
    # Count unique trips and users
    unique_users = len(set(trip["user_id"] for trip in data))
    
    return {
        "trip_stats": trip_stats,
        "overall_stats": overall_stats,
        "segment_count_stats": segment_count_stats,
        "summary": {
            "total_trips": len(data),
            "unique_users": unique_users,
            "total_segments": sum(all_segment_counts)
        }
    }

def format_output(analysis: Dict[str, Any]) -> None:
    """Format and print the analysis results."""
    # Print summary
    summary = analysis["summary"]
    segment_stats = analysis["segment_count_stats"]
    
    print(f"\n===== SUMMARY =====")
    print(f"Total Trips: {summary['total_trips']}")
    print(f"Unique Users: {summary['unique_users']}")
    print(f"Total Segments: {summary['total_segments']}")
    
    # Print segment count statistics
    print(f"\n===== SEGMENT COUNT STATISTICS =====")
    print(f"Mean segments per trip: {segment_stats['mean']:.2f}")
    print(f"Median segments per trip: {segment_stats['median']:.2f}")
    print(f"StdDev of segments per trip: {segment_stats['stddev']:.2f}")
    print(f"Min segments in a trip: {segment_stats['min']}")
    print(f"Max segments in a trip: {segment_stats['max']}")
    
    # Print overall stats
    print(f"\n===== OVERALL STATISTICS =====")
    overall = analysis["overall_stats"]
    
    # Convert to DataFrame for nicer display
    stats_df = pd.DataFrame({
        "Metric": list(overall.keys()),
        "Mean": [overall[m]["mean"] for m in overall],
        "Median": [overall[m]["median"] for m in overall],
        "StdDev": [overall[m]["stddev"] for m in overall]
    })
    
    print(stats_df.to_string(index=False))
    
    # Print per-trip summary (limited to first 5 trips for brevity)
    print(f"\n===== PER-TRIP STATISTICS (showing first 5) =====")
    trips = list(analysis["trip_stats"].items())[:5]
    
    for trip_id, trip_data in trips:
        print(f"\nTrip ID: {trip_id}")
        print(f"Activity Type: {trip_data['activity_type']}")
        print(f"Segments: {trip_data['segment_count']}")
        
        trip_stats_df = pd.DataFrame({
            "Metric": list(trip_data["stats"].keys()),
            "Mean": [trip_data["stats"][m]["mean"] for m in trip_data["stats"]],
            "Median": [trip_data["stats"][m]["median"] for m in trip_data["stats"]],
            "StdDev": [trip_data["stats"][m]["stddev"] for m in trip_data["stats"]]
        })
        
        print(trip_stats_df.to_string(index=False))


def main():
    data = load_trip_segment_dimensions()
    analysis = analyze_trip_segments(data)
    format_output(analysis)

    # Save the full analysis to a JSON file (optional)
    output_path = Path("../data/trip_segment_analysis.json")
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nFull analysis saved to {output_path}")


if __name__ == "__main__":
    main()

"""
Loading trip segment dimensions from ../data/trip_segment_dimensions.json...

===== SUMMARY =====
Total Trips: 999
Unique Users: 666
Total Segments: 125551

===== SEGMENT COUNT STATISTICS =====
Mean segments per trip: 125.68
Median segments per trip: 91.00
StdDev of segments per trip: 124.96
Min segments in a trip: 1
Max segments in a trip: 1147

===== OVERALL STATISTICS =====
             Metric       Mean     Median      StdDev
   elevation_gain_m   4.905912   1.700000   41.781417
   elevation_loss_m   4.920113   1.700000   25.687116
           offset_x 146.330517 158.867784   85.780899
           offset_y 151.317525 167.570754   86.673763
         distance_m 263.202959 249.082901  102.361157
  elapsed_time_secs  87.120023  39.000000 2879.572842
   moving_time_secs  51.007527  37.000000   44.605678
mean_heart_rate_bpm 126.066081 126.785714   23.274773
 mean_temperature_c  18.232804  18.000000    7.499277
   mean_cadence_rpm  63.978164  72.257143   26.310948
       mean_power_w 145.062574 144.522774  120.660840
"""