from typing import List
import msgpack
from ..lib.trip import load_batch_data, build_trip_dimensions
from ..models.rwgps import Trip
from ..models.trip_dimensions import TripDimensions


def get_batch_trips() -> List[Trip]:
    print("Loading trip batch data...")
    return load_batch_data("../data/trip_batches/trip_batch_10.msgpack")


def write_trip_dimensions(trip_dimensions: List[TripDimensions]) -> None:
    print("Writing trip dimensions to file...")
    with open("../data/trip_dimensions/trip_dimensions_10.msgpack", "wb") as f:
        f.write(msgpack.packb([trip.model_dump() for trip in trip_dimensions]))


def process_trip_highways() -> None:
    trips = get_batch_trips()

    print(f"Processing {len(trips)} trips...")
    trip_dimensions = [build_trip_dimensions(trip) for trip in trips]

    print(f"Writing {len(trip_dimensions)} trip dimensions to file...")
    write_trip_dimensions(trip_dimensions)

    print("Done!")


if __name__ == "__main__":
    process_trip_highways()
