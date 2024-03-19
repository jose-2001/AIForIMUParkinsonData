import pickle

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

    filename = module + '/train'
    write_pickle(filename + '.pkl', train)
    DataFrame(train).to_csv(PROCESSED_PATH / filename + '.csv')

    filename = module + '/test'
    write_pickle(filename + '.pkl', test)
    DataFrame(test).to_csv(PROCESSED_PATH / filename + '.csv')

    filename = module + '/val'
    write_pickle(filename + '.pkl', val)
    DataFrame(val).to_csv(PROCESSED_PATH / filename + '.csv')


def write_pickle(filename: str, variable: np.array) -> None:
    with open(PROCESSED_PATH / filename, 'x') as file:
        pickle.dump(variable, file)
