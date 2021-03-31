import pandas as pd
import glob
import os
from pathlib import Path

path = Path(os.path.abspath(__file__))
DATA_PATH = os.path.dirname(path.parent) + '\\ProcessedData'


def get_answer_names(data_frame):
    col_names = ['Worker ID', 'Problem', 'Age',	'Gender', 'Hand', 'Strong hand', 'Education', 'Answer',	'Confidence', 'Subjective Difficulty', 'Objective Difficutly', 'Psolve', 'Class', 'group_number', 'sum_prediction']
    data_frame = data_frame.drop(col_names, axis=1, errors='ignore')
    return pd.Series(data_frame.columns).apply(lambda x: str(x))


def get_question_dfs():
    list_of_df = []
    for file_path in glob.glob(f"{DATA_PATH}\\*.csv"):
        list_of_df.append(get_df(file_path))
    for file_path in glob.glob(f"{DATA_PATH}\\Article\\*.csv"):
        list_of_df.append(get_df(file_path))
    for file_path in glob.glob(f"{DATA_PATH}\\NewData\\*.csv"):
        list_of_df.append(get_df(file_path))
    return list_of_df


def get_file_name(file_path):
    return file_path.split(sep="\\")[-1].split(sep='.')[0]


def get_question_dicts():
    dict_of_df = {}
    for file_path in glob.glob(f"{DATA_PATH}\\*.csv"):
        dict_of_df[get_file_name(file_path)] = get_df(file_path)
    for file_path in glob.glob(f"{DATA_PATH}\\Article\\*.csv"):
        dict_of_df[get_file_name(file_path)] = get_df(file_path)
    for file_path in glob.glob(f"{DATA_PATH}\\NewData\\*.csv"):
        dict_of_df[get_file_name(file_path)] = get_df(file_path)
    return dict_of_df


def get_df(file_path):
    df = pd.read_csv(file_path, index_col=0)
    df['Answer'] = df['Answer'].astype(str)
    return df
