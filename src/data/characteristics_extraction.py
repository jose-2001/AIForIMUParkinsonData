import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

default_sensor_columns = ['imu_gyroX_right', 'imu_gyroY_right', 'imu_gyroZ_right', 'imu_accX_right',
                          'imu_accY_right', 'imu_accZ_right', 'imu_gyroX_left', 'imu_gyroY_left', 'imu_gyroZ_left',
                          'imu_accX_left', 'imu_accY_left', 'imu_accZ_left', 'imu_gyroX_spine', 'imu_gyroY_spine',
                          'imu_gyroZ_spine', 'imu_accX_spine', 'imu_accY_spine', 'imu_accZ_spine',
                          'imu_angleX_right', 'imu_angleY_right', 'imu_angleZ_right', 'imu_angleX_left',
                          'imu_angleY_left', 'imu_angleZ_left', 'imu_angleX_spine', 'imu_angleY_spine',
                          'imu_angleZ_spine', 'imu_angularX_left', 'imu_angularY_left', 'imu_angularZ_left',
                          'imu_angularX_right', 'imu_angularY_right', 'imu_angularZ_right', 'imu_angularX_spine',
                          'imu_angularY_spine', 'imu_angularZ_spine']
default_statistical_measures = ['mean', 'std', 'mean_abs_dev',
                                'min', 'max', 'range', 'median', 'median_abs_dev',
                                'interquartile_range', 'negative_count', 'positive_count', 'above_mean_count',
                                'local_maxima_count', 'skewness', 'kurtosis']
default_sampling_frequency = 50  # Hz
default_window_length_s = 1  # seconds
default_overlap = 0.5  # 50%

'''
.. function:: calculate_statistical_measures(column)

   Calculate statistical measures for the given column.

   :param column: A pandas Series representing the column for which statistical measures are calculated.
   :type column: pandas.Series

   :return: A dictionary containing various statistical measures.
   :rtype: dict

   This function calculates the following statistical measures for the given column:

   - **mean**: Mean of the column.
   - **std**: Standard deviation of the column.
   - **mean_abs_dev**: Mean absolute deviation of the column.
   - **min**: Minimum value of the column.
   - **max**: Maximum value of the column.
   - **range**: Range of the column (max - min).
   - **median**: Median of the column.
   - **median_abs_dev**: Median absolute deviation of the column.
   - **interquartile_range**: Interquartile range (75th percentile - 25th percentile) of the column.
   - **negative_count**: Number of negative values in the column.
   - **positive_count**: Number of positive values in the column.
   - **above_mean_count**: Number of values above the mean of the column.
   - **local_maxima_count**: Number of local maxima in the column.
   - **skewness**: Skewness of the column.
   - **kurtosis**: Kurtosis of the column.

'''


def calculate_statistical_measures(column):
    measures = {
        'mean': np.mean(column),
        'std': np.std(column),
        'mean_abs_dev': np.mean(np.abs(column - np.mean(column))),
        'min': np.min(column),
        'max': np.max(column),
        'range': np.max(column) - np.min(column),
        'median': np.median(column),
        'median_abs_dev': np.median(np.abs(column - np.median(column))),
        'interquartile_range': np.percentile(column, 75) - np.percentile(column, 25),
        'negative_count': np.sum(column < 0),
        'positive_count': np.sum(column > 0),
        'above_mean_count': np.sum(column > np.mean(column)),
        'local_maxima_count': len(column) - np.sum((column.shift(-1) < column) & (column.shift(1) < column)),
        'skewness': skew(column),
        'kurtosis': kurtosis(column)
    }
    return measures


'''
.. function:: generate_column_names(sensor_columns=None, statistical_measures=None, preprocessing: bool = True)

   Generate column names for sensor data.

   :param sensor_columns: A list of sensor column names.
   :type sensor_columns: list, optional
   :param statistical_measures: A list of statistical measures to be calculated for each sensor column.
   :type statistical_measures: list, optional
   :param preprocessing: A boolean indicating whether preprocessing columns should be included.
   :type preprocessing: bool, optional
   :return: A list of column names.
   :rtype: list

   This function generates column names for sensor data based on the specified sensor columns and statistical measures. By default, it includes preprocessing columns.
   If ``preprocessing`` is set to ``False``, the function excludes preprocessing columns:

   Example::

      >>> generate_column_names(sensor_columns=['acceleration', 'temperature'], statistical_measures=['mean', 'std'], preprocessing=False)
      ['date_measure', 'window_number', 'first_timestamp', 'acceleration_mean', 'acceleration_std', 'temperature_mean', 'temperature_std']

'''


def generate_column_names(sensor_columns=None, statistical_measures=None, preprocessing: bool = True):
    if sensor_columns is None:
        sensor_columns = default_sensor_columns
    if statistical_measures is None:
        statistical_measures = default_statistical_measures

    column_names = ['date_measure', 'window_number', 'first_timestamp']
    if preprocessing:
        column_names.append('anon_id')

    for column in sensor_columns:
        for measure in statistical_measures:
            column_name = f"{column}_{measure}"
            column_names.append(column_name)

    if preprocessing:
        column_names.append('PD')

    return column_names


'''
.. function:: extract_features(data, window_df, window_length_s=None, sampling_frequency=None, overlap=None, preprocessing: bool = True)

   Extract features from sensor data.

   :param data: The input DataFrame containing sensor data.
   :type data: pandas.DataFrame
   :param window_df: The DataFrame to store the extracted features.
   :type window_df: pandas.DataFrame
   :param window_length_s: The length of the window in seconds.
   :type window_length_s: int, optional
   :param sampling_frequency: The sampling frequency of the sensor data.
   :type sampling_frequency: int, optional
   :param overlap: The overlap between consecutive windows.
   :type overlap: float, optional
   :param preprocessing: A boolean indicating whether preprocessing columns are included.
   :type preprocessing: bool, optional
   :return: The DataFrame containing the extracted features.
   :rtype: pandas.DataFrame

   This function extracts features from the sensor data stored in the input DataFrame. It divides the data into windows of specified length and overlap, calculates statistical measures for each window, and adds the extracted features to the provided DataFrame.

   Example::

      >>> extract_features(data, window_df, window_length_s=60, sampling_frequency=100, overlap=0.5)
      # DataFrame with extracted features

   The function iterates over the input DataFrame and extracts features for each window. It calculates statistical measures for each sensor column in the window data, including mean, standard deviation, median, etc. Preprocessing columns like 'anon_id' and 'PD' are optionally included in the calculation of statistical measures.

'''


def extract_features(data, window_df, window_length_s=None, sampling_frequency=None, overlap=None,
                     preprocessing: bool = True):
    if window_length_s is None:
        window_length_s = default_window_length_s
    if sampling_frequency is None:
        sampling_frequency = default_sampling_frequency
    if overlap is None:
        overlap = default_overlap
    rows_per_window = int(window_length_s * sampling_frequency)
    paso_tiempo = int(rows_per_window * (1 - overlap))
    # Iterate over dataframe data
    window_number = 1
    for i in range(0, len(data) - rows_per_window + 1, paso_tiempo):
        # Select the first row of each window
        window_first_row = data.iloc[i]

        # Select relevant row values
        anon_id, PD = None, None
        date_measure = window_first_row['date_measure']
        first_timestamp = window_first_row['time_stamp']
        if preprocessing:
            anon_id = window_first_row['anon_id']
            PD = window_first_row['PD']

        # Select window including both sensor values and metadata
        ventana = data.iloc[i:i + rows_per_window]

        # Verify the values of the first row match the last in the window
        comparable_columns = ['date_measure']
        if preprocessing:
            comparable_columns = comparable_columns + ['anon_id', 'PD']

        while len(ventana) > 1 and not ventana.iloc[0][comparable_columns].equals(ventana.iloc[-1][comparable_columns]):
            ventana = ventana.iloc[:-1]  # Reduce window length by dropping the last row until values are equal

        # If the window empties but condition is still not met, break loop
        if len(ventana) == 1 and not ventana.iloc[0][comparable_columns].equals(ventana.iloc[-1][comparable_columns]):
            break

        # Remove 'anon_id','date_measure', and 'PD' values from the window before calculating statistical measures.
        if preprocessing:
            window_sensor_values = ventana.drop(['anon_id', 'date_measure', 'time_stamp', 'PD'], axis=1)
        else:
            window_sensor_values = ventana.drop(['patient_id', 'date_measure', 'time_stamp'], axis=1)

        # Calculate the statistical measures of the window for each column.
        window_data = {
            'date_measure': date_measure,
            'window_number': window_number,
            'first_timestamp': first_timestamp
        }

        if preprocessing:
            window_data['anon_id'] = anon_id
            window_data['PD'] = PD

        for sensor_col in window_sensor_values.columns:
            measures = calculate_statistical_measures(window_sensor_values[sensor_col])
            for measure, measure_value in measures.items():
                column_name = f"{sensor_col}_{measure}"
                window_data[column_name] = measure_value

        # Add window data to the DataFrame.
        window_df = pd.concat([window_df, pd.DataFrame(window_data, index=[0])], ignore_index=True)

        window_number += 1
    return window_df
