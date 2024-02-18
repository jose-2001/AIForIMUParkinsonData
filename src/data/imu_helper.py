import os
from dotenv import load_dotenv
from src.utils.imu_extremities import Joint, ImuComponents
import numpy as np
from src.data.json_formater import get_joint_data, get_joint_dimension
from src.data.signal_filter import (low_pass_filter, roll_derivative, pitch_derivative, yaw_derivative, interpolar,
                                    correct_angular_acc_x, calculate_angular_acc, CONVERSION_FACTOR)

load_dotenv()
DATABASE_URL = os.environ.get('DATABASE_URL')


class ImuData:
    def __init__(self, patient_id: str, date_measure: str, data: dict) -> None:
        self.patient_id = patient_id
        self.date_measure = date_measure
        self.raw_data = data
        self.structured_data = {}  # {gyroX_joint-> {component: np.array}}

        self.imu_data_der = get_joint_data(self.raw_data, Joint.RIGHT)
        self.imu_data_izq = get_joint_data(self.raw_data, Joint.LEFT)

        try:
            self.imu_data_spine = get_joint_data(self.raw_data, "spina_base")
        except:
            try:
                self.imu_data_spine = get_joint_data(self.raw_data, Joint.BASE_SPINE)
            except:
                print(f"No spine found in patient {self.patient_id}")

        # offSet correction --> get in experimental process
        self.imu_gyroX_right = get_joint_dimension(self.imu_data_der, ImuComponents.A) - 20
        self.imu_gyroY_right = get_joint_dimension(self.imu_data_der, ImuComponents.B) + 8
        self.imu_gyroZ_right = get_joint_dimension(self.imu_data_der, ImuComponents.G) + 20
        self.imu_accX_right = get_joint_dimension(self.imu_data_der, ImuComponents.X) * CONVERSION_FACTOR
        self.imu_accY_right = get_joint_dimension(self.imu_data_der, ImuComponents.Y) * CONVERSION_FACTOR
        self.imu_accZ_right = get_joint_dimension(self.imu_data_der, ImuComponents.Z) * CONVERSION_FACTOR
        self.imu_timestamp_right = get_joint_dimension(self.imu_data_der, ImuComponents.TIME)

        self.imu_gyroX_left = get_joint_dimension(self.imu_data_izq, ImuComponents.A) - 20
        self.imu_gyroY_left = -1 * get_joint_dimension(self.imu_data_izq, ImuComponents.B) + 8
        self.imu_gyroZ_left = -1 * get_joint_dimension(self.imu_data_izq, ImuComponents.G) + 20
        self.imu_accX_left = get_joint_dimension(self.imu_data_izq, ImuComponents.X) * CONVERSION_FACTOR
        self.imu_accY_left = get_joint_dimension(self.imu_data_izq, ImuComponents.Y) * CONVERSION_FACTOR
        self.imu_accZ_left = get_joint_dimension(self.imu_data_izq, ImuComponents.Z) * CONVERSION_FACTOR
        self.imu_timestamp_left = get_joint_dimension(self.imu_data_izq, ImuComponents.TIME)

        self.imu_gyroX_spine = get_joint_dimension(self.imu_data_spine, ImuComponents.A) - 20
        self.imu_gyroY_spine = get_joint_dimension(self.imu_data_spine, ImuComponents.B) + 8
        self.imu_gyroZ_spine = get_joint_dimension(self.imu_data_spine, ImuComponents.G) + 20
        self.imu_accX_spine = get_joint_dimension(self.imu_data_spine, ImuComponents.X) * CONVERSION_FACTOR
        self.imu_accY_spine = get_joint_dimension(self.imu_data_spine, ImuComponents.Y) * CONVERSION_FACTOR
        self.imu_accZ_spine = get_joint_dimension(self.imu_data_spine, ImuComponents.Z) * CONVERSION_FACTOR
        self.imu_timestamp_spine = get_joint_dimension(self.imu_data_spine, ImuComponents.TIME)

        # ---------------------------------
        # -  Acceleration Filter Process  -
        # ---------------------------------

        self.imu_accX_left, self.imu_accY_left, self.imu_accZ_left, self.imu_timestamp_left = interpolar(
            self.imu_accX_left, self.imu_accY_left, self.imu_accZ_left, self.imu_timestamp_left)

        self.imu_accX_left = low_pass_filter(self.imu_accX_left)
        self.imu_accY_left = low_pass_filter(self.imu_accY_left)
        self.imu_accZ_left = low_pass_filter(self.imu_accZ_left)

        self.imu_accX_right, self.imu_accY_right, self.imu_accZ_right, self.imu_timestamp_right = interpolar(
            self.imu_accX_right, self.imu_accY_right, self.imu_accZ_right, self.imu_timestamp_right)

        self.imu_accX_right = low_pass_filter(self.imu_accX_right)
        self.imu_accY_right = low_pass_filter(self.imu_accY_right)
        self.imu_accZ_right = low_pass_filter(self.imu_accZ_right)

        self.imu_accX_spine, self.imu_accY_spine, self.imu_accZ_spine, self.imu_timestamp_spine = interpolar(
            self.imu_accX_spine, self.imu_accY_spine, self.imu_accZ_spine, self.imu_timestamp_spine)

        self.imu_accX_spine = low_pass_filter(self.imu_accX_spine)
        self.imu_accY_spine = low_pass_filter(self.imu_accY_spine)
        self.imu_accZ_spine = low_pass_filter(self.imu_accZ_spine)

        # ------------------------------
        # -  Gyroscope Filter Process  -
        # ------------------------------
        self.imu_gyroX_left, self.imu_gyroY_left, self.imu_gyroZ_left, self.imu_timestamp_left = interpolar(
            self.imu_gyroX_left, self.imu_gyroY_left, self.imu_gyroZ_left, self.imu_timestamp_left)

        self.imu_gyroX_left = low_pass_filter(self.imu_gyroX_left)
        self.imu_gyroY_left = low_pass_filter(self.imu_gyroY_left)
        self.imu_gyroZ_left = low_pass_filter(self.imu_gyroZ_left)

        self.imu_gyroX_right, self.imu_gyroY_right, self.imu_gyroZ_right, self.imu_timestamp_right = interpolar(
            self.imu_gyroX_right, self.imu_gyroY_right, self.imu_gyroZ_right, self.imu_timestamp_right)

        self.imu_gyroX_right = low_pass_filter(self.imu_gyroX_right)
        self.imu_gyroY_right = low_pass_filter(self.imu_gyroY_right)
        self.imu_gyroZ_right = low_pass_filter(self.imu_gyroZ_right)

        self.imu_gyroX_spine, self.imu_gyroY_spine, self.imu_gyroZ_spine, self.imu_timestamp_spine = interpolar(
            self.imu_gyroX_spine, self.imu_gyroY_spine, self.imu_gyroZ_spine, self.imu_timestamp_spine)

        self.imu_gyroX_left = low_pass_filter(self.imu_gyroX_left)
        self.imu_gyroY_left = low_pass_filter(self.imu_gyroY_left)
        self.imu_gyroZ_left = low_pass_filter(self.imu_gyroZ_left)

        self.imu_gyroX_right_d = roll_derivative(self.imu_gyroX_right, self.imu_gyroY_right, self.imu_gyroZ_right)
        self.imu_gyroY_right_d = pitch_derivative(self.imu_gyroX_right, self.imu_gyroY_right, self.imu_gyroZ_right)
        self.imu_gyroZ_right_d = yaw_derivative(self.imu_gyroX_right, self.imu_gyroY_right, self.imu_gyroZ_right)

        self.imu_gyroX_left_d = roll_derivative(self.imu_gyroX_left, self.imu_gyroY_left, self.imu_gyroZ_left)
        self.imu_gyroY_left_d = pitch_derivative(self.imu_gyroX_left, self.imu_gyroY_left, self.imu_gyroZ_left)
        self.imu_gyroZ_left_d = yaw_derivative(self.imu_gyroX_left, self.imu_gyroY_left, self.imu_gyroZ_left)

        self.imu_gyroX_spine_d = roll_derivative(self.imu_gyroX_spine, self.imu_gyroY_spine, self.imu_gyroZ_spine)
        self.imu_gyroY_spine_d = pitch_derivative(self.imu_gyroX_spine, self.imu_gyroY_spine, self.imu_gyroZ_spine)
        self.imu_gyroZ_spine_d = yaw_derivative(self.imu_gyroX_spine, self.imu_gyroY_spine, self.imu_gyroZ_spine)

        # differential time calculation

        self.dt_right = int(sum(np.diff(self.imu_timestamp_right)) / len(self.imu_timestamp_right)) / 1000
        self.dt_left = int(sum(np.diff(self.imu_timestamp_left)) / len(self.imu_timestamp_left)) / 1000
        self.dt_spine = int(sum(np.diff(self.imu_timestamp_spine)) / len(self.imu_timestamp_spine)) / 1000

        self.imu_angleX_right = self.imu_gyroX_right_d + self.imu_gyroX_right_d * self.dt_right
        self.imu_angleY_right = self.imu_gyroY_right_d + self.imu_gyroY_right_d * self.dt_right
        self.imu_angleZ_right = self.imu_gyroZ_right_d + self.imu_gyroZ_right_d * self.dt_right

        self.imu_angleX_left = self.imu_gyroX_left_d + self.imu_gyroX_left_d * self.dt_left
        self.imu_angleY_left = self.imu_gyroY_left_d + self.imu_gyroY_left_d * self.dt_left
        self.imu_angleZ_left = self.imu_gyroZ_left_d + self.imu_gyroZ_left_d * self.dt_left

        self.imu_angleX_spine = self.imu_gyroX_spine_d + self.imu_gyroX_spine_d * self.dt_spine
        self.imu_angleY_spine = self.imu_gyroY_spine_d + self.imu_gyroY_spine_d * self.dt_spine
        self.imu_angleZ_spine = self.imu_gyroZ_spine_d + self.imu_gyroZ_spine_d * self.dt_spine

        # ---------------------------------
        # -  Angular Process Calculation  -
        # ---------------------------------

        self.imu_angularX_left = correct_angular_acc_x(self.imu_gyroX_left_d, self.imu_timestamp_left)
        self.imu_angularY_left = calculate_angular_acc(self.imu_gyroY_left_d, self.imu_timestamp_left)
        self.imu_angularZ_left = calculate_angular_acc(self.imu_gyroZ_left_d, self.imu_timestamp_left)

        self.imu_angularX_right = correct_angular_acc_x(self.imu_gyroX_right_d, self.imu_timestamp_right)
        self.imu_angularY_right = calculate_angular_acc(self.imu_gyroY_right_d, self.imu_timestamp_right)
        self.imu_angularZ_right = calculate_angular_acc(self.imu_gyroZ_right_d, self.imu_timestamp_right)

        self.imu_angularX_spine = correct_angular_acc_x(self.imu_gyroX_spine_d, self.imu_timestamp_spine)
        self.imu_angularY_spine = calculate_angular_acc(self.imu_gyroY_spine_d, self.imu_timestamp_spine)
        self.imu_angularZ_spine = calculate_angular_acc(self.imu_gyroZ_spine_d, self.imu_timestamp_spine)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


if __name__ == '__main__':
    pass
