import scipy
import numpy as np


def get_asa_laterality(imu_angle_y_left: np.ndarray, imu_angle_y_right: np.ndarray):
    MIN_PEAKS = 20

    left_upper_peaks_idx, _ = scipy.signal.find_peaks(imu_angle_y_left, MIN_PEAKS)
    right_upper_peaks_idx, _ = scipy.signal.find_peaks(imu_angle_y_left, MIN_PEAKS)

    left_lower_peaks_idx, _ = scipy.signal.find_peaks(-imu_angle_y_left, MIN_PEAKS)
    right_lower_peaks_idx, _ = scipy.signal.find_peaks(-imu_angle_y_right, MIN_PEAKS)

    if len(left_upper_peaks_idx) > 0 and len(left_lower_peaks_idx) > 0:
        all_peaks_left = list(left_upper_peaks_idx) + list(left_lower_peaks_idx)
        all_peaks_left.sort()

    if len(right_upper_peaks_idx) > 0 and len(right_lower_peaks_idx) > 0:
        all_peaks_right = list(right_upper_peaks_idx) + list(right_lower_peaks_idx)
        all_peaks_right.sort()

    peaks_left_up = imu_angle_y_left[left_upper_peaks_idx]
    peaks_right_up = imu_angle_y_right[right_upper_peaks_idx]

    peaks_left_down = imu_angle_y_left[left_lower_peaks_idx]
    peaks_right_down = imu_angle_y_right[right_lower_peaks_idx]

    prom_up_left = np.mean(peaks_left_up)
    prom_down_left = np.mean(peaks_left_down)
    prom_up_right = np.mean(peaks_right_up)
    prom_down_right = np.mean(peaks_right_down)

    magnitude_left = prom_up_left - prom_down_left
    magnitude_right = prom_up_right - prom_down_right

    lat = "izq"
    if magnitude_left > magnitude_right:
        lat = "der"

    return lat
