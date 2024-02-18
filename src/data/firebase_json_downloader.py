import os
import numpy as np
import json
from urllib.request import urlopen
from dotenv import load_dotenv


load_dotenv()
DATABASE_URL = os.environ.get('DATABASE_URL')


def get_data_summary(url: str = None) -> object | dict:
    """
        url: Database url where the summary is stored. By default, uses the one on the .env
        returns all the records in the database
   """
    if url is None:
        url = DATABASE_URL

    response = urlopen(url + "/resumen.json")
    json_data = response.read().decode('utf-8', 'replace')
    json_ = json.loads(json_data)
    return json_


def get_measurement(patient_id: str, date: str, url: str = None) -> object | dict:
    """
        url: Database url where the summary is stored. By default, uses the one on the .env
        returns a specific record of the database in a specific date
   """

    if url is None:
        url = DATABASE_URL

    response = urlopen(url + "/pruebas/" + patient_id + "/pruebas/" + date + ".json")
    json_data = response.read().decode('utf-8', 'replace')
    json_ = json.loads(json_data)
    return json_

