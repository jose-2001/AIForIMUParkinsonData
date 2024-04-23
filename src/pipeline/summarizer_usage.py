import pickle

import pandas as pd

from src.settings import ROOT_DIR
from src.preprocessing.summarizer_preprocessing import _load_model, predict_symptom

MODEL_FEATURES = {
    'PD': ['imu_gyroZ_spine_std', 'imu_gyroZ_spine_range',
           'imu_gyroZ_spine_median_abs_dev', 'imu_gyroZ_spine_interquartile_range',
           'imu_angleZ_left_std', 'imu_angleZ_left_mean_abs_dev',
           'imu_angleZ_left_max', 'tremor_knn', 'tremor_gboost', 'posture_knn',
           'posture_gboost', 'laterality_svm', 'laterality_knn',
           'laterality_gboost', 'asa_svm', 'asa_knn', 'asa_gboost',
           'imu_angleZ_right_std', 'imu_angleZ_right_mean_abs_dev',
           'imu_angleZ_right_max']
}

USED_SYMPTOMS_MODELS = {
    'tremor': ['knn', 'gboost'],
    'posture': ['knn', 'gboost'],
    'laterality': ['knn', 'gboost', 'svm'],
    'asa': ['knn', 'gboost', 'svm']
}


def predict_parkinson(patient_df: pd.DataFrame):
    _add_prediction_columns(patient_df)
    features = MODEL_FEATURES['PD']
    model = _load_model('summarizer', 'rf')
    patient_df_features = patient_df[features].copy()
    parkinson_predictions = []
    for index, row in patient_df_features.iterrows():
        row = row.to_numpy().reshape(1, -1)
        prediction = model.predict(row)
        parkinson_predictions.append(prediction[0])

    patient_df['PD'] = parkinson_predictions


def _add_prediction_columns(patient_df: pd.DataFrame):
    predictions = {}
    for symptom in USED_SYMPTOMS_MODELS.keys():
        for model in USED_SYMPTOMS_MODELS[symptom]:
            column_name = f'{symptom}_{model}'
            predictions[column_name] = []
            for index, row in patient_df.iterrows():
                predictions[column_name].append(predict_symptom(symptom, model, row))

    for key in predictions.keys():
        values = predictions[key]
        for index in range(len(values)):
            if 'laterality' in key:
                temp_list: list = list(values[index][0])
                max_value = max(temp_list)
                values[index] = temp_list.index(max_value)

            else:
                values[index] = values[index][0][1]

        predictions[key] = values

    for key in predictions.keys():
        patient_df[key] = predictions[key]
