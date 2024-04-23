import json
from unidecode import unidecode
from dotenv import load_dotenv

import pandas as pd

from src.settings import ROOT_DIR
from src.data.imu_helper import ImuData
from src.data.firebase_json_downloader import get_measurement, get_data_summary
from src.data.json_formater import imu_data2dataframe, measurement_has_valid_keys
from src.data.characteristics_extraction import generar_nombres_columnas, extraer_caracteristicas


def execute_prediction(patient_id: str):
    dates = _get_patient_measurement_dates(patient_id)
    date = _get_desired_date_by_user(dates)

    if date == str(len(dates) + 1):
        # Get all dates
        # Transform all to DF
        # Execute predictions for all dates
        pass
    else:
        date = dates[date]
        measure = get_measurement(unidecode(patient_id), date)
        patient_df = _from_measure2dataframe(patient_id, date, measure)

        df_columns = generar_nombres_columnas(preprocessing=False)
        patient_df_features = pd.DataFrame(columns=df_columns)
        patient_df_features = extraer_caracteristicas(patient_df, patient_df_features, preprocessing=False)

        print(patient_df_features)


def _from_measure2dataframe(patient_id: str, date: str, measure: dict) -> pd.DataFrame:
    if (measure is not None) and measurement_has_valid_keys(measure):
        imu_data = ImuData(patient_id, date, measure)
        patient_df = imu_data2dataframe(imu_data)
    else:
        raise Exception("Selected measure is not valid. Check if that measure was made with IMUs or accelerometers.")
    return patient_df


def _get_desired_date_by_user(dates: list) -> int:
    print("Here is a list of dates of the patient's measures:")
    for i in range(len(dates)):
        print(f'{i + 1}. {dates[i]}')

    print(f'{len(dates) + 1}. Use all')
    date = None
    while date is None:
        date = int(input('Please select one of the dates (Enter the number): ')) - 1

        if date not in range(0, len(dates) + 1):
            print('Invalid date. Please try again.')
            date = None

    return date


def _get_patient_measurement_dates(patient_id: str) -> list:
    summary = get_data_summary()
    patient_measurements: dict = summary[patient_id]
    dates = list(patient_measurements.values())

    if len(dates) == 0:
        raise Exception("No date was found for the entered patient id.")

    return dates


if __name__ == '__main__':
    load_dotenv()
    patient_cc = str(input("Please enter the patient's cc: "))
    execute_prediction(patient_cc)
