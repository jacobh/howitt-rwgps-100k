import glob
import os
import msgpack
from typing import List

from ..lib.trip import load_batch_data, build_trip_dimensions
from ..models.rwgps import Trip
from ..models.trip_dimensions import TripDimensions


def get_batch_filenames() -> List[str]:
    # Glob all msgpack files in the trip_batches directory.
    return glob.glob("../data/trip_batches/*.msgpack")


def write_trip_dimensions(trip_dimensions: List[TripDimensions], output_filename: str) -> None:
    print(f"Writing {len(trip_dimensions)} trip dimensions to {output_filename}...")
    with open(output_filename, "wb") as f:
        # Each TripDimensions object is serialized by its model_dump() method.
        f.write(msgpack.packb([trip.model_dump() for trip in trip_dimensions]))


def process_batch(batch_filename: str) -> None:
    print(f"\nProcessing batch file: {batch_filename}")
    trips: List[Trip] = load_batch_data(batch_filename)
    print(f"Loaded {len(trips)} trips from {batch_filename}...")
    trip_dimensions = [build_trip_dimensions(trip) for trip in trips]
    
    # Create output filename by replacing 'trip_batch' with 'trip_dimensions' in the batch filename.
    base_name = os.path.basename(batch_filename)
    name_without_ext, _ = os.path.splitext(base_name)
    out_base_name = name_without_ext.replace("trip_batch", "trip_dimensions")
    output_filename = os.path.join("../data/trip_dimensions", f"{out_base_name}.msgpack")
    
    write_trip_dimensions(trip_dimensions, output_filename)
    print(f"Finished processing batch file: {batch_filename}")


def process_all_batches() -> None:
    batch_filenames = get_batch_filenames()
    
    if not batch_filenames:
        print("No batch files found. Exiting.")
        return

    print(f"Found {len(batch_filenames)} batch files to process...")
    for batch_filename in batch_filenames:
        process_batch(batch_filename)
    
    print("\nAll batches processed!")


if __name__ == "__main__":
    process_all_batches()