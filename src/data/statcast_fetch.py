from pybaseball import statcast, cache
import datetime as dt
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data"


def refresh_statcast_data(min_year=2023, override_refresh=False):
    parquet_files = os.listdir(DATA_DIRECTORY)
    if (not parquet_files) or override_refresh:
        max_year = dt.datetime.now().year
        for i in range(min_year, max_year+1):
            # Each year is about 770k records
            print(f'Fetching full year records for {i}')
            current = statcast(start_dt=f'{i}-03-01',
                            end_dt=f'{i}-11-15',
                            verbose=False)
            export_path = os.path.join(DATA_DIRECTORY, f'statcast_{i}.parquet')
            current.to_parquet(export_path, index=False)