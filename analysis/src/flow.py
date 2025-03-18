from prefect import flow
from pathlib import Path
from .tasks.osm import load_highway_data, convert_highway_features_to_numpy


@flow(log_prints=True)
def process_rides(data_dir: Path):
    highway_file = data_dir / "vic_and_tas_highways.json"

    highways = load_highway_data(highway_file)
    highway_coords_list = convert_highway_features_to_numpy(highways['features'])

    print(f'generate coords: {len(highway_coords_list)}')


if __name__ == "__main__":
    data_dir = Path("../data")  # Change this path as needed

    process_rides(data_dir)
