import pandas as pd
import matplotlib.pyplot as plt

FULL_NAMES_DICT = {
    'acc': 'Acceleration',
    'gyro': 'Gyro',
    'angle': 'Angle',
    'angular': 'Angular Velocity'
}

LIFTS = ['right', 'left', 'spine']


def plot_lifts_component_signal(record: pd.DataFrame, component: str, measure: str,
                                title: str = '', trim_value: int = -1):
    fig, ax = plt.subplots(nrows=3, figsize=(8, 10))

    column = f'imu_{measure}{component}_'
    subtitle = FULL_NAMES_DICT[measure]

    if title == '':
        date = record['date_measure'].unique().tolist()[0]
        title = f"Sample: {date}"

    if trim_value == -1:
        trim_value = len(record['time_stamp'])
    else:
        title += f'\nTrimmed at: {trim_value} stamps'

    time = record['time_stamp'].to_numpy()[:trim_value]

    for index, lift in enumerate(LIFTS):
        ax[index].plot(time, record[column + lift].to_numpy()[:trim_value])
        ax[index].set(xlabel='time (s)',
                      ylabel=f'{measure} {component} - {lift}',
                      title=f'{subtitle} in {lift} - {component}')

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
