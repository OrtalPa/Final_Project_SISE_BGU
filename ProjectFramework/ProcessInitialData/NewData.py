import glob
import pandas as pd

DATA_PATH = r'C:\Users\ortal\Documents\FinalProject\data\NewData'


def create_csvs():
    for file_path in glob.glob(f"{DATA_PATH}\\*.csv"):
        data = pd.read_csv(file_path)
        data_problem_range = data['Problem'].unique()
        data_problem_range = data_problem_range.tolist()
        for i, value in enumerate(data_problem_range):
            data[data['Problem'] == value].to_csv(DATA_PATH + '\\SplitV2\\' + get_file_name(file_path) + '\\' + str(value) + r'.csv', index=False, na_rep='N/A')


def get_file_name(file_path):
    return file_path.split(sep="\\")[-1].split(sep='.')[0]


def get_list_of_questions_for_dataset():
    dataset_dict = {}
    index = 2
    for file_path in glob.glob(f"{DATA_PATH}\\*.csv"):
        data = pd.read_csv(file_path)
        data_problems = data['Problem'].unique()
        dataset_dict[get_file_name(file_path)+'_'+str(index)] = data_problems.tolist()
        index += 1
    return dataset_dict


#create_csvs()
