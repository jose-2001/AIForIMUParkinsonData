import pickle

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from src.settings import ROOT_DIR

TRAIN_PERCENTAGE = 0.75
VAL_PERCENTAGE = 0.15
TEST_PERCENTAGE = 0.1


def save_and_split(df: DataFrame, module: str):
    train, test = train_test_split(df, test_size=1 - TRAIN_PERCENTAGE)
    val, test = train_test_split(test, test_size=TEST_PERCENTAGE/(TEST_PERCENTAGE + VAL_PERCENTAGE))

    filename = module + '/train'
    write_pickle(filename + '.pkl', train)
    write_pickle(filename + '.csv', train)

    filename = module + '/test'
    write_pickle(filename + '.pkl', test)
    write_pickle(filename + '.csv', test)

    filename = module + '/val'
    write_pickle(filename + '.pkl', val)
    write_pickle(filename + '.csv', val)


def write_pickle(filename: str, variable: any) -> None:
    processed_path = ROOT_DIR / 'data' / 'processed'
    with open(processed_path / filename, 'wb') as file:
        pickle.dump(variable, file)
