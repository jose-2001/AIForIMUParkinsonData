import json
from unidecode import unidecode
from dotenv import load_dotenv
import warnings

import pandas as pd
import numpy as np

from src.settings import ROOT_DIR
from src.data.imu_helper import ImuData
from src.data.firebase_json_downloader import get_measurement, get_data_summary
from src.data.json_formater import imu_data2dataframe, measurement_has_valid_keys
from src.data.characteristics_extraction import generar_nombres_columnas, extraer_caracteristicas
from src.preprocessing.summarizer_preprocessing import predict_symptom, BEST_MODEL_BY_SYMPTOM
from src.pipeline.summarizer_usage import predict_parkinson


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

        symptoms = ['tremor', 'posture', 'laterality', 'asa', 'dysk']
        results = {}

        for symptom in symptoms:
            symptom: str
            if 'laterality' in symptom:
                results[f'{symptom}_left'] = list()
                results[f'{symptom}_right'] = list()
            else:
                results[symptom] = list()

            for index, value in patient_df_features.iterrows():
                model = BEST_MODEL_BY_SYMPTOM[symptom]
                prediction = predict_symptom(symptom, model, value)
                if 'laterality' in symptom:
                    prediction = prediction[0]
                    results[f'{symptom}_right'].append(prediction[1])
                    results[f'{symptom}_left'].append(prediction[2])
                else:
                    prediction = prediction[0][1]
                    results[symptom].append(prediction)

            if 'laterality' in symptom:
                results[f'{symptom}_right'] = np.mean(results[f'{symptom}_right'])
                results[f'{symptom}_left'] = np.mean(results[f'{symptom}_left'])
            else:
                results[symptom] = np.mean(results[symptom])

        predict_parkinson(patient_df_features)
        results['PD'] = np.mean(patient_df_features['PD'].to_numpy())
        return results


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
    warnings.filterwarnings('ignore')
    patient_cc = str(input("Please enter the patient's cc: "))
    results = execute_prediction(patient_cc)
    print(results)
