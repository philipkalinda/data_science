#Work In Progress

import requests
from bs4 import BeautifulSoup
import json
import pandas as pd

response = requests.get('https://money.comparethemarket.com/savings-accounts/?AFFCLIE=CM01')
parser = BeautifulSoup(response.content, 'html.parser')

savings_json_data = parser.find_all('div', id='savings-json')[0]
savings_py_obj = json.loads(savings_json_data.text)

cols = []
for key,value in savings_py_obj[0].items():
    if key not in cols:
        cols.append(key)
    else:
        continue

raw_data = pd.DataFrame()


for bank in savings_py_obj:
    for key,value in bank.items():
        pass
