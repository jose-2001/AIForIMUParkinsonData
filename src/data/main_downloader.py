import pandas as pd
from dotenv import load_dotenv
from unidecode import unidecode
from imu_helper import ImuData
from firebase_json_downloader import get_data_summary, get_measurement
from json_formater import imu_data2dataframe, measurement_has_valid_keys
from src.utils.basic_imu_data_attributes import IMU_DATA_ATTRIBUTES


"""
This is the central file of the data module, it uses all the required
scripts to download the core data (from IMUs) of the project. If this file
is executed it will not download  by itself the data, we strongly recommend using
it as part of a notebook and store the downloaded data in a csv, pickle or excel 
file (as convenient for storage and analytic purposes).
"""


def run_data_download() -> pd.DataFrame:
    """
        Downloads all the data from the IMUs stored in the Database
        and stores them in the raw data directory. returns a dataframe
        with all the timeseries. Every row is a different timestamp with
        the corresponding measures.
    """
    imu_df = pd.DataFrame(columns=IMU_DATA_ATTRIBUTES)
    record_count = 0
    imu_count = 0
    saved_dates = []

    print('Starting download from firebase...')
    summary = get_data_summary()
    print('Summary download completed.')
    print()

    for record_id in summary:
        record_id = record_id.replace(' ', '')
        measurements_dict: dict = summary.get(record_id, {})
        for date in measurements_dict.values():
            record_count += 1
            measurement = get_measurement(unidecode(record_id), date)
            if (measurement is not None) and measurement_has_valid_keys(measurement) and (date not in saved_dates):
                imu_count += 1
                saved_dates.append(date)
                new_imu_data = ImuData(record_id, date, measurement)
                new_imu_df = imu_data2dataframe(new_imu_data)
                imu_df = pd.concat([imu_df, new_imu_df])
            print(f'Num. Records: {record_count}\t|\tCurrent record: {record_id}\t|\tAmount IMU records: {imu_count}\t')

    return imu_df


if __name__ == "__main__":
    load_dotenv()
    run_data_download()
    # test()
