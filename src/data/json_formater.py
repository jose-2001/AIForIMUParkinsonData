import json
import pandas as pd
import numpy as np
from src.utils.imu_extremities import ImuComponents, Joint
from src.utils.basic_imu_data_attributes import IMU_DATA_ATTRIBUTES


def get_joint_data(data: dict, joint: str) -> list:
    """
    Returns all the data of a specific joint (Includes all components). Returns a list of lists.
    Every list represents a component and its time series.
    """
    joint_data = []
    imu_components = ImuComponents.__members__.values()
    for component in imu_components:
        joint_data.append(imu_joint_data_to_array(data, joint, component))
    return joint_data


def imu_joint_data_to_array(data: json, joint: str, component: str) -> np.array:
    """
    Converts all the data from a specific component to a numpy array
    data: key-value structure with the record of a patient
    joint_label: Joint value
    component: The label of the IMU component
    """
    vector = np.array([])
    for i in range(len(data[joint]) - 1):
        vector = np.append(vector, data[joint][i][component])
    return vector


def get_joint_dimension(data: list, dimension: chr) -> np.array:
    """
    0(a): gyroX,
    1(b): gyroY,
    2(g): gyroZ,
    3(x): accX,
    4(y): accY,
    5(z): accZ
    """
    temp = []
    if dimension == 'a':
        temp = data[0]
    elif dimension == 'b':
        temp = data[1]
    elif dimension == 'g':
        temp = data[2]
    elif dimension == 'x':
        temp = data[3]
    elif dimension == 'y':
        temp = data[4]
    elif dimension == 'z':
        temp = data[5]
    elif dimension == 't':
        temp = data[6]
    return temp


def imu_data2dataframe(imu_data: object) -> pd.DataFrame:
    """
    Transforms all the data from an instance of ImuData into a pandas dataframe
    every row is the measur a timestamp of the
    """
    attributes = IMU_DATA_ATTRIBUTES.copy()
    attributes.remove('patient_id')
    attributes.remove('date_measure')

    lower_timestamp_size = get_joint_with_less_timestamps(imu_data)

    imu_dict = {
        'patient_id': [imu_data.patient_id for i in range(lower_timestamp_size)],
        'date_measure': [imu_data.date_measure for i in range(lower_timestamp_size)],
    }

    for attr in attributes:
        if attr == 'time_stamp':
            imu_dict[attr] = getattr(imu_data, 'imu_timestamp_right')[:lower_timestamp_size]
        else:
            imu_dict[attr] = getattr(imu_data, attr)[:lower_timestamp_size]

    imu_pd = pd.DataFrame(imu_dict)

    return imu_pd


def get_joint_with_less_timestamps(data: object) -> int:
    """
    Obtains the len of the smaller ndarray in the data

    data: Instance of ImuData that contains all the ndarrays
    """
    attributes = IMU_DATA_ATTRIBUTES.copy()
    attributes.remove('patient_id')
    attributes.remove('date_measure')
    attributes.remove('time_stamp')

    size = min([len(getattr(data, attr)) for attr in attributes])
    return size


def measurement_has_valid_keys(measurement: dict) -> bool:
    keys = measurement.keys()
    valid_keys = ((Joint.RIGHT in keys) and
                  (Joint.LEFT in keys) and
                  ((Joint.BASE_SPINE in keys) or ('spina_base' in keys)) and
                  (len(keys) == 3)
                  )

    return valid_keys
