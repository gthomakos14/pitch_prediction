from pybaseball import statcast
import datetime as dt
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIRECTORY = PROJECT_ROOT / "data/raw"


def refresh_statcast_data(min_year=2023, override_refresh=False):
    parquet_files = os.listdir(RAW_DIRECTORY)
    if (not parquet_files) or override_refresh:
        if not override_refresh:
            print(f'No statcast data found. Refreshing going back to {min_year}...')
        else:
            print('Override received. Refreshing statcast parquets now...')
        max_year = dt.datetime.now().year
        for i in range(min_year, max_year+1):
            # Each year is about 770k records
            print(f'Fetching full year records for {i}')
            current = statcast(start_dt=f'{i}-03-01',
                            end_dt=f'{i}-11-15',
                            verbose=False)
            export_path = os.path.join(RAW_DIRECTORY, f'statcast_{i}.parquet')
            current.to_parquet(export_path, index=False)