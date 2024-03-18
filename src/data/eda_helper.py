import pandas as pd
from src.settings import ROOT_DIR

raw_data_path = ROOT_DIR / 'data' / 'raw'
ID_MAP = None
PATIENTS_IDS = None


def get_id(anon_id: str, id_map: pd.DataFrame = ID_MAP) -> str:
    print(anon_id, end="\r")
    if anon_id is None or anon_id == "":
        return ""

    if id_map is None:
        id_map = load_id_map()

    real_id_list: list = id_map.loc[id_map['anon_id'] == anon_id, 'patient_id'].tolist()

    return real_id_list[0] if len(real_id_list) > 0 else ""


def load_id_map():
    # Read file with all the anonymized ids
    global ID_MAP
    ID_MAP = pd.read_csv(raw_data_path / 'id_map.csv')
    ID_MAP['anon_id'] = ID_MAP['anon_id'].astype(str)
    ID_MAP['patient_id'] = ID_MAP['patient_id'].astype(str)

    return ID_MAP


def load_patients_ids():
    # Read file with patients
    global PATIENTS_IDS
    PATIENTS_IDS = pd.read_excel(raw_data_path / 'patients_ids.xlsx')
    PATIENTS_IDS['CC'] = PATIENTS_IDS['CC'].astype(str)

    return PATIENTS_IDS


if __name__ == '__main__':
    print(get_id('1'))
