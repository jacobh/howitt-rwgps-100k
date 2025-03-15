# analysis/src/main.py
import json
import jax.numpy as jnp
from pathlib import Path
import aiofiles
import asyncio
import atexit

from .point_delta import all_point_deltas
from .segments import create_trip_segments, create_all_segment_bboxes
from .spatial_service import AsyncSpatialService


async def load_trip_data(trip_file_path):
    """Load trip data from JSON file using async I/O"""
    async with aiofiles.open(trip_file_path, "r") as f:
        content = await f.read()
        return json.loads(content)


async def load_highway_data(file_path):
    """Load highway data from GeoJSON file using async I/O"""
    async with aiofiles.open(file_path, "r") as f:
        content = await f.read()
        data = json.loads(content)
        return data


async def analyze_trip_file(trip_file_path, spatial_service):
    """Analyze a single trip file and return results"""
    try:
        # Load trip data
        trip_data = await load_trip_data(trip_file_path)

        # Get track points and summary
        track_points = get_track_points(trip_data)

        if not track_points:
            print(f"No track points found in {trip_file_path}")
            return None

        # Get data arrays for further analysis
        arrays = extract_jax_arrays(track_points)

        deltas = all_point_deltas(arrays)

        segments = create_trip_segments(arrays, deltas, min_distance_m=200)

        bboxes = create_all_segment_bboxes(segments)

        # Use the spatial service to find nearby highways
        nearby_highways = await spatial_service.find_nearby_highways(bboxes)

        print(
            f"Found {sum(len(highways) for highways in nearby_highways)} nearby highways across {len(nearby_highways)} segments"
        )

        return {
            "arrays": arrays,
            "segments": segments,
            "nearby_highways": nearby_highways,
        }

    except Exception as e:
        print(f"Error analyzing trip {trip_file_path}: {e}")
        return None


def get_track_points(trip_data):
    """Extract track points from trip data"""
    if "trip" in trip_data and "track_points" in trip_data["trip"]:
        return trip_data["trip"]["track_points"]
    return []


def extract_jax_arrays(track_points):
    """
    Extract track points into JAX arrays for analysis.
    Pre-initializes arrays with zeros and populates them only if data exists.
    Returns a dictionary of JAX arrays.
    """
    if not track_points:
        return {}

    # Get the length of track points
    n_points = len(track_points)

    # Initialize all possible arrays with zeros
    arrays = {
        "latlng": jnp.zeros((n_points, 2)),  # [lat, lng]
        "elevation": jnp.zeros(n_points),
        "distance": jnp.zeros(n_points),
        "speed": jnp.zeros(n_points),
        "time": jnp.zeros(n_points),
        "heart_rate": jnp.zeros(n_points),
        "temperature": jnp.zeros(n_points),
        "cadence": jnp.zeros(n_points),
        "power": jnp.zeros(n_points),
    }

    # Check first point to determine which fields exist and populate corresponding arrays
    if "y" in track_points[0] and "x" in track_points[0]:
        arrays["latlng"] = jnp.array(
            [[point["y"], point["x"]] for point in track_points]
        )

    if "e" in track_points[0]:
        arrays["elevation"] = jnp.array([point.get("e", 0) for point in track_points])

    if "d" in track_points[0]:
        arrays["distance"] = jnp.array([point.get("d", 0) for point in track_points])

    if "s" in track_points[0]:
        arrays["speed"] = jnp.array([point.get("s", 0) for point in track_points])

    if "t" in track_points[0]:
        arrays["time"] = jnp.array([point.get("t", 0) for point in track_points])

    if "h" in track_points[0]:
        arrays["heart_rate"] = jnp.array([point.get("h", 0) for point in track_points])

    if "T" in track_points[0]:
        arrays["temperature"] = jnp.array([point.get("T", 0) for point in track_points])

    if "c" in track_points[0]:
        arrays["cadence"] = jnp.array([point.get("c", 0) for point in track_points])

    if "p" in track_points[0]:
        arrays["power"] = jnp.array([point.get("p", 0) for point in track_points])

    return arrays


async def main():
    # Load highway data first
    highway_file = Path("../data/vic_and_tas_highways.json")
    try:
        highway_data = await load_highway_data(highway_file)
        if "features" in highway_data:
            print(
                f"Loaded highway data: {len(highway_data['features'])} features found"
            )

            # Initialize the spatial service with the highway data
            spatial_service = AsyncSpatialService(highway_data)

            # Register cleanup function to ensure the service is properly shut down
            atexit.register(spatial_service.shutdown)
        else:
            print("No features found in highway data")
            return
    except Exception as e:
        print(f"Error loading highway data: {e}")
        return

    # Find trip files
    data_dir = Path("../data/trips")
    trip_files = list(data_dir.glob("trip_*.json"))

    if not trip_files:
        print(f"No trip files found in {data_dir}")
        return

    # Limit to 1000 files or use all available if less than 1000
    max_files = min(10, len(trip_files))
    trip_files = trip_files[:max_files]
    print(f"Analyzing {max_files} trip files...")

    # Create tasks for all files
    tasks = [analyze_trip_file(trip_file, spatial_service) for trip_file in trip_files]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Filter out None results (failed analyses)
    valid_results = [r for r in results if r is not None]

    print(
        f"\nAnalysis complete: {len(valid_results)} successful out of {max_files} files"
    )

    # Report some statistics
    if valid_results:
        # Count how many trips have each type of data
        data_counts = {}
        for result in valid_results:
            for key in result["arrays"].keys():
                data_counts[key] = data_counts.get(key, 0) + 1

        print("\nData availability across trips:")
        for key, count in data_counts.items():
            print(f"  {key}: {count} trips ({count / len(valid_results) * 100:.1f}%)")

    # Explicitly shut down the spatial service
    spatial_service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
