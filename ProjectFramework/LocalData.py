import pandas as pd
import glob

# ------------------- Data gathered by Hila -------------------- #

PATH = r'C:\Users\ortal\Documents\FinalProject\data\raw data'

list_of_df = []
for file_path in glob.glob(f"{PATH}\\*.csv"):
    list_of_df.append(pd.read_csv(file_path))

# Print just for sanity check
print(list_of_df[2])
print(list_of_df[-1])

