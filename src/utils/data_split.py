import os
import pickle
from pathlib import Path

import tensorflow as tf
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
from src.settings import ROOT_DIR

TRAIN_PERCENTAGE = 0.75
VAL_PERCENTAGE = 0.15
TEST_PERCENTAGE = 0.1

SEED = 11

PROCESSED_PATH = ROOT_DIR / 'data' / 'processed'


def save_and_split(df: DataFrame, module: str):
    train, test, val = __get_train_test_val(df)

    train = DataFrame(train)
    test = DataFrame(test)
    val = DataFrame(val)

    __write_files('train', module, train)
    __write_files('test', module, test)
    __write_files('val', module, val)


def save_and_split_sequences(sequence: list, module: str):
    train, test, val = __get_train_test_val(sequence)
    sets = {
        'train': train,
        'test': test,
        'val': val
    }

    for value in sets:
        filename_pkl: Path = Path(value + '.pkl')
        path: Path = PROCESSED_PATH / module / filename_pkl

        if os.path.exists(path):
            with open(str(path), 'wb') as f:
                pickle.dump(sets[value], f)
        else:
            with open(str(path), 'wb') as f:
                pickle.dump(sets[value], f)


def get_features_target(sequence: list) -> tuple:
    features = [values for values, _ in sequence]
    target = np.array([label for _, label in sequence])

    return features, target


def __get_train_test_val(data: any):
    train, test = train_test_split(data, test_size=1 - TRAIN_PERCENTAGE, random_state=SEED, shuffle=True)
    val, test = train_test_split(test, test_size=TEST_PERCENTAGE/(TEST_PERCENTAGE + VAL_PERCENTAGE),
                                 random_state=SEED, shuffle=True)
    return train, test, val


def __write_files(type_set: str, module: str, df: DataFrame):
    filename_pkl: Path = Path(type_set + '.pkl')
    filename_csv = type_set + '.csv'

    path_pkl = PROCESSED_PATH / module / filename_pkl
    path_csv = PROCESSED_PATH / module / filename_csv

    df.to_pickle(path_pkl)
    df.to_csv(path_csv)
