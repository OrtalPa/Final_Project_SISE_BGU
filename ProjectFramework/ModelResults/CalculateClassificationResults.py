import pandas as pd
import glob
import os
from pathlib import Path

from Pipeline.MultiLabelPipeline import get_data, get_label_names

path = Path(os.path.abspath(__file__))
DATA_PATH = os.path.dirname(path.parent) + '\\ModelResults\\ResultFiles'


def get_dfs():
    list_of_df = []
    for file_path in glob.glob(f"{DATA_PATH}\\*.csv"):
        list_of_df.append(get_df(file_path))
    return list_of_df


def get_file_name(file_path):
    return file_path.split(sep="\\")[-1].split(sep='.')[0]


def get_df_dict():
    dict_of_df = {}
    # key: filename value: df
    for file_path in glob.glob(f"{DATA_PATH}\\*.csv"):
        dict_of_df[f'{get_file_name(file_path)}'] = get_df(file_path)
    return dict_of_df


def get_df(file_path):
    df = pd.read_csv(file_path, index_col=0)
    return df


def calculate_when_method_answers_correctly(method, df_test, df_train):
    sum_r = 0
    for index, row in df_test.iterrows():
        real = row[method]
        pred = df_train.iloc[df_train.index == index][method]
        pred = pred[index]
        if real == 1 and pred == 1:
            sum_r = sum_r + 1 if real == 1 and pred == 1 else 0
    return sum_r


if __name__ == "__main__":
    # create a file with:
    # get the results of the model -> for each method the prediction



    print(DATA_PATH)
    df_dict = get_df_dict()
    df_correct = get_data()
    df_correct = df_correct[get_label_names()]
    result_dict = {}
    for df_name in df_dict:
        try:
            if 'wnone' in df_name:
                df = df_dict[df_name]
                for method in get_label_names():
                    sum_of_correct = calculate_when_method_answers_correctly(method=method, df_test=df, df_train=df_correct)
                    print(sum_of_correct)
                    print(df_correct[method].sum())
                    result_dict[method+"_"+df_name] = sum_of_correct/df_correct[method].sum()
        except Exception as e:
            print("error in "+df_name)
            print(e)
    print(result_dict)
