import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

columnas_sensores_default = ['imu_gyroX_right', 'imu_gyroY_right', 'imu_gyroZ_right', 'imu_accX_right',
                             'imu_accY_right', 'imu_accZ_right', 'imu_gyroX_left', 'imu_gyroY_left', 'imu_gyroZ_left',
                             'imu_accX_left', 'imu_accY_left', 'imu_accZ_left', 'imu_gyroX_spine', 'imu_gyroY_spine',
                             'imu_gyroZ_spine', 'imu_accX_spine', 'imu_accY_spine', 'imu_accZ_spine',
                             'imu_angleX_right', 'imu_angleY_right', 'imu_angleZ_right', 'imu_angleX_left',
                             'imu_angleY_left', 'imu_angleZ_left', 'imu_angleX_spine', 'imu_angleY_spine',
                             'imu_angleZ_spine', 'imu_angularX_left', 'imu_angularY_left', 'imu_angularZ_left',
                             'imu_angularX_right', 'imu_angularY_right', 'imu_angularZ_right', 'imu_angularX_spine',
                             'imu_angularY_spine', 'imu_angularZ_spine']
medidas_estadisticas_default = ['mean', 'std', 'mean_abs_dev',
                                'min', 'max', 'range', 'median', 'median_abs_dev',
                                'interquartile_range', 'negative_count', 'positive_count', 'above_mean_count',
                                'local_maxima_count', 'skewness', 'kurtosis']
frecuencia_muestreo_default = 50  # Hz
longitud_ventana_s_default = 1  # segundos
overlap_default = 0.5  # 50%


def calcular_medidas_estadisticas(columna):
    medidas = {
        'mean': np.mean(columna),
        'std': np.std(columna),
        'mean_abs_dev': np.mean(np.abs(columna - np.mean(columna))),
        'min': np.min(columna),
        'max': np.max(columna),
        'range': np.max(columna) - np.min(columna),
        'median': np.median(columna),
        'median_abs_dev': np.median(np.abs(columna - np.median(columna))),
        'interquartile_range': np.percentile(columna, 75) - np.percentile(columna, 25),
        'negative_count': np.sum(columna < 0),
        'positive_count': np.sum(columna > 0),
        'above_mean_count': np.sum(columna > np.mean(columna)),
        'local_maxima_count': len(columna) - np.sum((columna.shift(-1) < columna) & (columna.shift(1) < columna)),
        'skewness': skew(columna),
        'kurtosis': kurtosis(columna)
    }
    return medidas


def generar_nombres_columnas(columnas_sensores=None, medidas_estadisticas=None):
    if columnas_sensores is None:
        columnas_sensores = columnas_sensores_default
    if medidas_estadisticas is None:
        medidas_estadisticas = medidas_estadisticas_default

    nombres_columnas = ['anon_id', 'date_measure', 'window_number', 'first_timestamp']
    for columna in columnas_sensores:
        for medida in medidas_estadisticas:
            nombre_columna = f"{columna}_{medida}"
            nombres_columnas.append(nombre_columna)
    nombres_columnas.append("PD")
    return nombres_columnas


def extraer_caracteristicas(data, df_ventanas, longitud_ventana_s=None, frecuencia_muestreo=None, overlap=None):
    if longitud_ventana_s is None:
        longitud_ventana_s = longitud_ventana_s_default
    if frecuencia_muestreo is None:
        frecuencia_muestreo = frecuencia_muestreo_default
    if overlap is None:
        overlap = overlap_default
    filas_por_ventana = int(longitud_ventana_s * frecuencia_muestreo)
    paso_tiempo = int(filas_por_ventana * (1 - overlap))
    # Iterar sobre los datos originales en el DataFrame
    window_number = 1
    for i in range(0, len(data) - filas_por_ventana + 1, paso_tiempo):
        # Seleccionar la primera fila de cada ventana
        primera_fila_ventana = data.iloc[i]

        # Seleccionar los valores relevantes de la fila
        anon_id = primera_fila_ventana['anon_id']
        date_measure = primera_fila_ventana['date_measure']
        PD = primera_fila_ventana['PD']
        first_timestamp = primera_fila_ventana['time_stamp']

        # Seleccionar la ventana incluyendo tanto los valores de los sensores como los metadatos
        ventana = data.iloc[i:i + filas_por_ventana]

        # Verificar si los valores en la primera y última fila de la ventana son iguales
        while len(ventana) > 1 and not ventana.iloc[0][['anon_id', 'date_measure', 'PD']].equals(ventana.iloc[-1][['anon_id', 'date_measure', 'PD']]):
            ventana = ventana.iloc[:-1]  # Reducir la longitud de la ventana eliminando la última fila

        # Si la ventana se vacía pero aún no se cumple la condición, salir del bucle
        if len(ventana) == 1 and not ventana.iloc[0][['anon_id', 'date_measure', 'PD']].equals(
            ventana.iloc[-1][['anon_id', 'date_measure', 'PD']]):
            break

        # Eliminar los valores de 'anon_id','date_measure'y'PD' de la ventana antes de calcular las medidas estadísticas
        ventana_valores_sensor = ventana.drop(['anon_id', 'date_measure', 'time_stamp', 'PD'], axis=1)

        # Calcular las medidas estadísticas de la ventana para cada columna
        datos_ventana = {
            'anon_id': anon_id,
            'date_measure': date_measure,
            'window_number': window_number,
            'first_timestamp': first_timestamp,
            'PD': PD
        }
        for sensor_col in ventana_valores_sensor.columns:
            medidas = calcular_medidas_estadisticas(ventana_valores_sensor[sensor_col])
            for medida, medida_valor in medidas.items():
                nombre_columna = f"{sensor_col}_{medida}"
                datos_ventana[nombre_columna] = medida_valor

        # Añadir los datos de la ventana al DataFrame
        df_ventanas = pd.concat([df_ventanas, pd.DataFrame(datos_ventana, index=[0])], ignore_index=True)

        window_number += 1
    return df_ventanas
