import pandas as pd
from src.data.eda_helper import load_id_map

ID_MAP = None


def get_anon_id(ced: str, id_map: pd.DataFrame = ID_MAP) -> str:
    if ced is None or ced == "":
        return ""

    if id_map is None:
        id_map = load_id_map()
    real_id_list = id_map.loc[id_map['patient_id'] == ced, 'anon_id'].tolist()
    return real_id_list[0] if len(real_id_list) > 0 else ""
