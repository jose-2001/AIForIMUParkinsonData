import pickle
from pathlib import Path

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
    train, test = train_test_split(df, test_size=1 - TRAIN_PERCENTAGE, random_state=SEED)
    val, test = train_test_split(test, test_size=TEST_PERCENTAGE/(TEST_PERCENTAGE + VAL_PERCENTAGE),
                                 random_state=SEED)

    train = DataFrame(train)
    test = DataFrame(test)
    val = DataFrame(val)

    __write_files('train', module, train)
    __write_files('test', module, test)
    __write_files('val', module, val)


def __write_files(type_set: str, module: str, df: DataFrame):
    filename_pkl: Path = Path(type_set + '.pkl')
    filename_csv = type_set + '.csv'

    path_pkl = PROCESSED_PATH / module / filename_pkl
    path_csv = PROCESSED_PATH / module / filename_csv

    df.to_pickle(path_pkl)
    df.to_csv(path_csv)
