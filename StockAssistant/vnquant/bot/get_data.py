# get_data.py

import vnquant.data as dt
import pandas as pd

# E1VFVN30
loader_E1VFVN30 = dt.DataLoader(symbols="E1VFVN30",
           start="2022-01-01",
           end="2024-11-08",
           minimal=False,
           data_source="VND")

data_E1VFVN30 = loader_E1VFVN30.download()
print(data_E1VFVN30.head(5))
E1VFVN30 = data_E1VFVN30.to_csv("E1VFVN30.csv")


# # Assuming 'E1VFVN30.csv' exists in the current directory
# try:
#     df = pd.read_csv("E1VFVN30.csv")
#     print(df.columns)
# except FileNotFoundError:
#     print("Error: 'E1VFVN30.csv' not found. Please ensure the file exists in the current directory.")

# FUEDCMID
loader_FUEDCMID = dt.DataLoader(symbols="FUEDCMID",
           start="2022-01-01",
           end="2024-11-08",
           minimal=False,
           data_source="VND")

data_FUEDCMID = loader_FUEDCMID.download()
print(data_FUEDCMID.head(5))
FUEDCMID = data_FUEDCMID.to_csv("FUEDCMID_02.csv")

# ENF
loader_ENF = dt.DataLoader(symbols="ENF",
           start="2022-01-01",
           end="2024-11-08",
           minimal=False,
           data_source="VND")

data_ENF = loader_ENF.download()
print(data_ENF.head(5))
ENF = data_ENF.to_csv("ENF.csv")
