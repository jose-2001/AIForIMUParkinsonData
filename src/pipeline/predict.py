import os
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
from src.data.characteristics_extraction import generate_column_names, extract_features
from src.preprocessing.summarizer_preprocessing import predict_symptom, BEST_MODEL_BY_SYMPTOM
from src.pipeline.summarizer_usage import predict_parkinson

RESULTS_PATH = str(ROOT_DIR) + '/results/predictions'


def execute_prediction(patient_id: str, date: str = None):
    results = {}
    measure = get_measurement(unidecode(patient_id), date)
    patient_df = _from_measure2dataframe(patient_id, date, measure)

    df_columns = generate_column_names(preprocessing=False)
    patient_df_features = pd.DataFrame(columns=df_columns)
    patient_df_features = extract_features(patient_df, patient_df_features, preprocessing=False)

    symptoms = ['tremor', 'posture', 'laterality', 'asa', 'dysk']

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


def _save_prediction(results_dict: dict, patient_id: str) -> None:
    patient_path = RESULTS_PATH + f'/{patient_id}/'
    os.makedirs(patient_path, exist_ok=True)
    print(len(results.keys()))
    if len(results.keys()) > 1:
        date_measure = 'all_measures'
    else:
        date_measure = list(results.keys())[0]

    file_name = patient_path + f'prediction_{date_measure}.json'
    with open(file_name, 'w') as file:
        json.dump(results_dict, file)
    file.close()
    print(f'Prediction saved in {file_name}')


if __name__ == '__main__':
    load_dotenv()
    warnings.filterwarnings('ignore')
    patient_cc = str(input("Please enter the patient's cc: "))

    dates = list(set(_get_patient_measurement_dates(patient_cc)))
    date = _get_desired_date_by_user(dates)

    results = {}
    if date == str(len(dates) + 1):
        for date in dates:
            results[date] = execute_prediction(patient_cc, date)
    else:
        date = dates[date]
        results[date] = execute_prediction(patient_cc, date)

    _save_prediction(results, patient_cc)
