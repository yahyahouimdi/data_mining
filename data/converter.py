import os
import numpy as np
import pandas as pd
from matio import load_from_mat


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing this Python script and the .mat files
folder_path = os.path.dirname(os.path.abspath(__file__))

# Sampling frequency
FS = 256  # Hz


# ============================================================
# FUNCTION: FLATTEN MATLAB TIMETABLE
# ============================================================

def flatten_timetable(df, fs=256):
    """
    Convert the DataFrame returned by mat-io into a normal
    sample-by-sample DataFrame.

    Input:
        1575 rows x 23 columns
        Each cell contains multiple samples.

    Output:
        ~403,200 rows x 24 columns
        Time + 23 signal channels
    """

    flattened = {}

    # --------------------------------------------------------
    # Flatten each signal/channel
    # --------------------------------------------------------

    for column in df.columns:

        samples = []

        for cell in df[column]:

            # Convert cell to NumPy array
            arr = np.asarray(cell)

            # Remove dimensions such as (256,1)
            # and make it a simple 1D array
            arr = arr.squeeze().reshape(-1)

            samples.extend(arr)

        flattened[column] = np.asarray(samples)

    # --------------------------------------------------------
    # Check that all channels have the same number of samples
    # --------------------------------------------------------

    lengths = {
        column: len(values)
        for column, values in flattened.items()
    }

    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        print("WARNING: Channels do not have the same number of samples.")

        for column, length in lengths.items():
            print(f"  {column}: {length} samples")

        raise ValueError(
            "Cannot create the CSV because channel lengths are different."
        )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    result = pd.DataFrame(flattened)

    # --------------------------------------------------------
    # Create Time column
    #
    # 256 Hz -> 0.00390625 seconds between samples
    # --------------------------------------------------------

    result.insert(
        0,
        "Time",
        np.arange(len(result)) / fs
    )

    return result


# ============================================================
# FIND ALL MAT FILES
# ============================================================

mat_files = [
    f for f in os.listdir(folder_path)
    if f.lower().endswith(".mat")
]

print(f"Found {len(mat_files)} MAT files.\n")


# ============================================================
# CONVERT EACH MAT FILE
# ============================================================

for fname in mat_files:

    fpath = os.path.join(folder_path, fname)

    print("=" * 70)
    print(f"Processing: {fname}")

    try:

        # ----------------------------------------------------
        # Load MATLAB file
        # ----------------------------------------------------

        data = load_from_mat(
            fpath,
            add_table_attrs=True
        )

        # ----------------------------------------------------
        # Find actual data variable
        # Ignore MATLAB metadata variables
        # ----------------------------------------------------

        keys = [
            k for k in data.keys()
            if not k.startswith("__")
        ]

        if not keys:
            raise ValueError("No data variable found in MAT file.")

        # Use the first data variable
        key = keys[0]

        raw_df = data[key]

        print(f"Variable: {key}")
        print(f"Original shape: {raw_df.shape}")
        print(f"Channels: {len(raw_df.columns)}")

        # ----------------------------------------------------
        # Flatten timetable
        # ----------------------------------------------------

        df_flat = flatten_timetable(
            raw_df,
            fs=FS
        )

        # ----------------------------------------------------
        # Create CSV filename
        # ----------------------------------------------------

        outname = os.path.splitext(fname)[0] + ".csv"
        outpath = os.path.join(folder_path, outname)

        # ----------------------------------------------------
        # Save CSV
        # ----------------------------------------------------

        df_flat.to_csv(
            outpath,
            index=False
        )

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        print(f"Final shape: {df_flat.shape}")
        print(f"Saved: {outname}")

        print("\nFirst 5 rows:")
        print(df_flat.head())

    except Exception as e:

        print(f"ERROR: {fname}")
        print(f"Reason: {e}")


print("\n" + "=" * 70)
print("CONVERSION FINISHED")
print("=" * 70)