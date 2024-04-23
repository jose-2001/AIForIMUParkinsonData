import pickle

import pandas as pd

from src.settings import ROOT_DIR

RESULTS_DIR = ROOT_DIR / 'results' / 'models'

# Features used by each model
MODELS_FEATURES = {
    'tremor': ['imu_gyroX_right_std', 'imu_gyroX_right_min', 'imu_gyroX_right_max',
                'imu_accY_right_std', 'imu_accZ_right_std', 'imu_accZ_right_max',
                'imu_angularZ_right_std', 'imu_gyroX_left_std', 'imu_gyroX_left_min',
                'imu_gyroX_left_max', 'imu_accY_left_std', 'imu_accZ_left_std',
                'imu_accZ_left_max', 'imu_angularZ_left_std'],
    'posture': ['imu_gyroZ_spine_mean_abs_dev',
                'imu_gyroZ_spine_min',
                'imu_gyroZ_spine_max',
                'imu_gyroZ_spine_range',
                'imu_gyroZ_spine_median_abs_dev'],
    'laterality': ['imu_gyroZ_left_std', 'imu_gyroZ_left_min', 'imu_gyroZ_spine_std',
                   'imu_gyroZ_spine_median_abs_dev', 'imu_angleX_left_std',
                   'imu_angleZ_left_std', 'imu_angleZ_left_min', 'imu_angleZ_left_max',
                   'imu_gyroZ_right_std', 'imu_gyroZ_right_min',
                   'imu_angleX_right_std', 'imu_angleZ_right_std', 'imu_angleZ_right_min',
                   'imu_angleZ_right_max'],
    'asa': ['imu_gyroZ_spine_mean_abs_dev', 'imu_gyroZ_spine_median_abs_dev',
            'imu_angleZ_left_range', 'imu_gyroZ_left_min',
            'imu_gyroZ_left_mean_abs_dev', 'imu_accY_left_median',
            'imu_accZ_spine_positive_count', 'imu_angleX_left_min',
            'imu_gyroZ_spine_range', 'imu_angleZ_right_range',
            'imu_gyroZ_right_min', 'imu_gyroZ_right_mean_abs_dev',
            'imu_accY_right_median', 'imu_angleX_right_min'],
    'dysk': ['imu_accX_right_negative_count', 'imu_accX_right_positive_count',
             'imu_accY_right_std', 'imu_accY_right_min',
             'imu_accY_right_median_abs_dev', 'imu_accY_left_std',
             'imu_accY_left_median_abs_dev', 'imu_accZ_left_std',
             'imu_accZ_left_min', 'imu_angularZ_right_min',
             'imu_accZ_right_std', 'imu_accZ_right_min']
}

MODELS_NAMES = {
    'svm': 'model.svm.pkl',
    'knn': 'model.knn.pkl',
    'gboost': 'model.gboost.pkl',
    'dt': 'model.dt.pkl',
    'rf': 'model.rf.pkl'
}

BEST_MODEL_BY_SYMPTOM = {
    'tremor': 'knn',
    'posture': 'knn',
    'laterality': 'knn',
    'asa': 'knn',
    'dysk': 'knn'
}


def predict_symptom(symptom: str, model: str, register: pd.Series):

    features = MODELS_FEATURES[symptom]
    model = _load_model(symptom, model)
    values = register[features].to_numpy()
    values = values.reshape(1, -1)

    symptom_prediction = model.predict_proba(values)
    return symptom_prediction


def _load_model(symptom: str, model: str):

    template_folder = f'{symptom}-features'
    results_model_path = str(RESULTS_DIR) + '/' + template_folder + '/' + MODELS_NAMES[model]

    with open(results_model_path, 'rb') as file:
        model = pickle.load(file)
        file.close()

    return model

