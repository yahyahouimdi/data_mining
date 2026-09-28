import os
import pandas as pd
from scipy.io import loadmat

folder_path = os.path.dirname(os.path.abspath(__file__))

mat_files = [
    f for f in os.listdir(folder_path)
    if f.lower().endswith(".mat")
]

for fname in mat_files:
    fpath = os.path.join(folder_path, fname)

    try:
        data = loadmat(fpath)

        keys = [k for k in data.keys() if not k.startswith("__")]

        if not keys:
            print(f"ECHEC : {fname} -> aucune donnée trouvée")
            continue

        print(f"{fname} : variables trouvées -> {keys}")

        raw_data = data[keys[0]]

        # Temporary inspection
        print(f"Type : {type(raw_data)}")
        print(f"Shape : {getattr(raw_data, 'shape', None)}")

    except Exception as e:
        print(f"ECHEC : {fname} -> {e}")