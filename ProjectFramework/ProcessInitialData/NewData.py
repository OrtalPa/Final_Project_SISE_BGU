import pandas as pd

data = pd.read_csv(r'C:\Users\Pnina\PycharmProjects\Final_Project_SISE_BGU\ProcessedData\NewData\politics_responses.csv')

data_problem_range = data['Problem'].unique()
data_problem_range = data_problem_range.tolist()

for i, value in enumerate(data_problem_range):
    data[data['Problem'] == value].to_csv(str(value) + r'.csv', index=False, na_rep='N/A')
