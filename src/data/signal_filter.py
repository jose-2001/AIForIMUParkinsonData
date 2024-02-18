import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt


CONVERSION_FACTOR = 9.8 / 4130


def low_pass_filter(input_signal):
    try:
        order = 20
        cutoff = 7.5  # Hz
        fs = 50  # Hz
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        out = filtfilt(b, a, input_signal)

    except:
        print("Sample too short to be filtered")
        out = input_signal
    return out


def interpolar(xin, yin, zin, tin):
    # Realizar las interpolaciones mediante PCHIP
    pchip_x = PchipInterpolator(tin, xin)
    pchip_y = PchipInterpolator(tin, yin)
    pchip_z = PchipInterpolator(tin, zin)

    xout = pchip_x(tin)
    yout = pchip_y(tin)
    zout = pchip_z(tin)

    return xout, yout, zout, tin


def roll_derivative(wx: np.array, wy: np.array, wz: np.array) -> np.array:
    return wx + wy * np.sin(90) * np.tan(-90) + wz * np.cos(90) * np.tan(-90)


def pitch_derivative(wx: np.array, wy: np.array, wz: np.array) -> np.array:
    return wy * np.cos(90) - wz * np.sin(90)


def yaw_derivative(wx: np.array, wy: np.array, wz: np.array) -> np.array:
    return wy * np.sin(90) / np.cos(90) + wz * np.cos(90) / np.cos(90)


def calculate_angular_acc(gyro_data, time):
    # se multiplica el valor por el factor de conversión a rad/s
    diff_angular_speed = np.diff(gyro_data * (np.pi / 180))
    diff_time = int(sum(np.diff(time)) / len(time))
    angular_acc = diff_angular_speed / diff_time
    return angular_acc


def correct_angular_acc_x(gyro_data: np.array, time: np.array):
    diff_time = int(sum(np.diff(time)) / len(time))
    alpha = 9.8 / 1000  # [1/s²] gravedad / distancia entre el acelerometro fantasma ...
    radian_factor = np.pi / 180

    gyro_data_ = gyro_data * radian_factor  # rad/s
    diff_angular_speed = np.diff(gyro_data_)  # rad/s
    angular_acc = diff_angular_speed / diff_time  # rad/s²

    angular_acc_offset = diff_angular_speed * alpha * diff_time
    angular_acc = angular_acc - angular_acc_offset

    return angular_acc


if __name__ == "__main__":
    pass
