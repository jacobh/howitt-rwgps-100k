from prefect import flow
from prefect_ray import RayTaskRunner
from pathlib import Path
from .tasks.osm import load_highway_data, convert_all_highway_features_to_numpy
from prefect.utilities.annotations import quote
import time


@flow(
    log_prints=True,
    task_runner=RayTaskRunner(),
)
def process_rides(data_dir: Path):
    highway_file = data_dir / "vic_and_tas_highways.json"

    highways = load_highway_data.submit(highway_file).result()
    highway_coords_list = convert_all_highway_features_to_numpy(quote(highways["features"])).result()

    print(f"generate coords: {len(highway_coords_list)}")


if __name__ == "__main__":
    data_dir = Path("../data")  # Change this path as needed

    process_rides(data_dir)
    
    try:
        print("Press CTRL+C to exit...")
        while True:
            time.sleep(1)  # Sleep for 1 second between checks, much more CPU-friendly
    except KeyboardInterrupt:
        print("\nExiting gracefully...")