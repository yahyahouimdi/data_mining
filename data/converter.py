from matio import load_from_mat
import os

folder_path = '/content/drive/MyDrive/data_mining/Projet data mining 3AINFO'
mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

for fname in mat_files:
    fpath = os.path.join(folder_path, fname)
    try:
        data = load_from_mat(fpath, add_table_attrs=True)
        keys = [k for k in data.keys() if not k.startswith('__')]
        raw_df = data[keys[0]]

        df_flat = flatten_timetable(raw_df)

        outname = fname.replace('.mat', '.csv')
        df_flat.to_csv(os.path.join(folder_path, outname), index=False)
        print(f"OK : {fname} -> {outname} ({df_flat.shape})")
    except Exception as e:
        print(f"ECHEC : {fname} -> {e}")