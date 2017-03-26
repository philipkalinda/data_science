##Work In Progress##

import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import sqlite3

#Web Scraping and Parsing
response = requests.get('https://money.comparethemarket.com/savings-accounts/?AFFCLIE=CM01')
parser = BeautifulSoup(response.content, 'html.parser')
savings_json_data = parser.find_all('div', id='savings-json')[0]

#Python Object Conversion
savings_py_obj = json.loads(savings_json_data.text)

#Column Naming
cols = []
for bank in savings_py_obj:
    for key,value in bank.items():
        if key not in cols:
            cols.append(key)
        else:
            continue

#DataFrame Creation
raw_data = pd.DataFrame(columns=cols)
for bank in savings_py_obj:
    raw_data = raw_data.append(bank, ignore_index=True)

print(raw_data)

#Database Management

conn = sqlite3.connect('savings_db.db')
c = conn.cursor()
c.execute("""drop table if exists towns""")
conn.commit()

# Create table
c.execute('''CREATE TABLE savings_table
             (Name TEXT, Id INTEGER, ProviderBrandName TEXT, IsInstantAccess TEXT, IsCashIsa TEXT, IsFixedRateBond TEXT,
             ProductImage TEXT, ApplyLink TEXT, GrossAer REAL, RateType TEXT, IntroBonusNote TEXT, MinimumInvestment INTEGER,
             MaximumInvestment INTEGER, ProductApplicationUrl TEXT, IsCommercial TEXT, HasInternetAccess TEXT, HasBranchAccess TEXT,
             HasTelephoneAccess TEXT, HasPostAccess TEXT, SavingsType TEXT, AdditionalInfo TEXT, MinimumAge INTEGER,
             MaximumAge INTEGER, Pros TEXT, Cons TEXT, GrossAerString TEXT, MinimumInvestmentString TEXT, MaximumInvestmentString TEXT,
             EligibilityAgeString TEXT, HasImage TEXT)''')

c.executemany("INSERT INTO savings_table VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", raw_data) # <- Enter your values here
conn.commit()
conn.close()
