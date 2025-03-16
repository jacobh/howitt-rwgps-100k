# analysis/src/main.py
import json
import jax.numpy as jnp
from pathlib import Path
import aiofiles
import asyncio
import atexit
import msgpack  # Standard msgpack library

from .point_delta import all_point_deltas
from .segments import create_trip_segments, create_all_segment_bboxes
from .spatial_service import AsyncSpatialService


async def load_batch_data(batch_file_path):
    """Load batch data from MessagePack file using async I/O"""
    async with aiofiles.open(batch_file_path, "rb") as f:
        content = await f.read()
        return msgpack.unpackb(content)


async def load_highway_data(file_path):
    """Load highway data from GeoJSON file using async I/O"""
    async with aiofiles.open(file_path, "r") as f:
        content = await f.read()
        data = json.loads(content)
        return data


async def analyze_trip(trip_data, spatial_service, trip_id=None):
    """Analyze a single trip and return results"""
    try:
        # Get track points and summary
        track_points = get_track_points(trip_data)

        if not track_points:
            if trip_id:
                print(f"No track points found in trip {trip_id}")
            return None

        # Get data arrays for further analysis
        arrays = extract_jax_arrays(track_points)

        deltas = all_point_deltas(arrays)

        segments = create_trip_segments(arrays, deltas, min_distance_m=200)

        bboxes = create_all_segment_bboxes(segments)

        # Use the spatial service to find nearby highways
        nearby_highways = await spatial_service.find_nearby_highways(bboxes)

        if trip_id:
            print(
                f"Trip {trip_id}: Found {sum(len(highways) for highways in nearby_highways)} nearby highways across {len(nearby_highways)} segments"
            )

        return {
            "trip_id": trip_id,
            "arrays": arrays,
            "segments": segments,
            "nearby_highways": nearby_highways,
        }

    except Exception as e:
        if trip_id:
            print(f"Error analyzing trip {trip_id}: {e}")
        else:
            print(f"Error analyzing trip: {e}")
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


async def process_batch(batch_file, spatial_service, max_concurrent=100, max_trips=None):
    """Process a batch file containing multiple trips"""
    print(f"Loading batch file: {batch_file.name}")
    try:
        # Load all trips from the batch file
        batch_data = await load_batch_data(batch_file)
        
        print(f"Loaded {len(batch_data)} trips from {batch_file.name}")
        
        # Limit the number of trips to process if specified
        if max_trips and max_trips < len(batch_data):
            trips_to_process = batch_data[:max_trips]
            print(f"Processing {max_trips} out of {len(batch_data)} trips")
        else:
            trips_to_process = batch_data
        
        # Process trips concurrently with a limit on max concurrent tasks
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(trip_idx, trip):
            async with semaphore:
                # Extract trip ID from the trip data if available, otherwise use index
                trip_id = trip.get("id", f"{batch_file.stem}_{trip_idx}")
                return await analyze_trip(trip, spatial_service, trip_id)
        
        # Create tasks for all trips in the batch
        tasks = [
            analyze_with_semaphore(i, trip) 
            for i, trip in enumerate(trips_to_process)
        ]
        
        # Run all tasks and collect results
        results = await asyncio.gather(*tasks)
        
        # Filter out None results (failed analyses)
        valid_results = [r for r in results if r is not None]
        
        print(f"Batch {batch_file.name}: {len(valid_results)} successful out of {len(trips_to_process)} trips")
        
        return valid_results
        
    except Exception as e:
        print(f"Error processing batch {batch_file}: {e}")
        return []


async def main():
    # Set up parameters
    max_batches = 1  # Number of batch files to process
    max_trips_per_batch = 100  # Max trips to process from each batch
    max_concurrent_trips = 50  # Max concurrent trip analyses
    
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

    # Find batch files
    batches_dir = Path("../data/trip_batches")
    batch_files = list(batches_dir.glob("trip_batch_*.msgpack"))

    if not batch_files:
        print(f"No batch files found in {batches_dir}")
        return

    # Limit to specified number of batch files
    batch_files = batch_files[:max_batches]
    print(f"Found {len(batch_files)} batch files, processing up to {max_batches}")

    # Process each batch file
    all_results = []
    for batch_file in batch_files:
        batch_results = await process_batch(
            batch_file, 
            spatial_service, 
            max_concurrent=max_concurrent_trips,
            max_trips=max_trips_per_batch
        )
        all_results.extend(batch_results)

    print(f"\nProcessing complete: {len(all_results)} trips successfully analyzed")

    # Report some statistics
    if all_results:
        # Count how many trips have each type of data
        data_counts = {}
        for result in all_results:
            for key in result["arrays"].keys():
                data_counts[key] = data_counts.get(key, 0) + 1

        print("\nData availability across trips:")
        for key, count in sorted(data_counts.items()):
            print(f"  {key}: {count} trips ({count / len(all_results) * 100:.1f}%)")

    # Explicitly shut down the spatial service
    spatial_service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())