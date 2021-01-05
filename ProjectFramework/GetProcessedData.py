import pandas as pd
import glob
import os
from pathlib import Path

path = Path(os.path.abspath(__file__))
DATA_PATH = os.path.dirname(path.parent) + '\\ProcessedData'


def get_answer_names(data_frame):
    col_names = ['Worker ID', 'Problem', 'Age',	'Gender', 'Hand', 'Strong hand', 'Education', 'Answer',	'Confidence', 'Subjective Difficulty', 'Objective Difficutly', 'Psolve', 'Class', 'group_number']
    data_frame = data_frame.drop(col_names, axis=1, errors='ignore')
    return pd.Series(data_frame.columns).apply(lambda x: str(x))


def get_question_dfs():
    list_of_df = []
    for file_path in glob.glob(f"{DATA_PATH}\\*.csv"):
        list_of_df.append(pd.read_csv(file_path, index_col=0))
    for file_path in glob.glob(f"{DATA_PATH}\\Article\\*.csv"):
        list_of_df.append(pd.read_csv(file_path, index_col=0))
    return list_of_df