import sys
from collections import defaultdict

import pandas as pd
import glob

# ------------------- Data gathered by Hila -------------------- #

PATH = r'C:\Users\ortal\Documents\FinalProject\data\NewData'


def arrange_prediction(x):
    if type(x) != str:
        if x > 1:
            x = x/100
        return float(x)
    else:
        # It is a percentage, 50% for example
        percentage = int(x.replace('%', ''))
        return float(percentage/100)


def get_answer_names(data_frame):
    col_names = ['Worker ID', 'Problem', 'Age',	'Gender', 'Hand', 'Strong hand', 'Education', 'Answer',	'Confidence', 'Subjective Difficulty', 'Objective Difficutly', 'Psolve', 'Class', 'group_number']
    data_frame = data_frame.drop(col_names, axis=1, errors='ignore')
    return pd.Series(data_frame.columns).apply(lambda x: str(x))


for file_path in glob.glob(f"{PATH}\\*.csv"):
    try:
        df = pd.read_csv(file_path)
        cols = ['Answer', 'Confidence', 'Class']
        answer_names = get_answer_names(df)
        cols.extend(answer_names)
        # Get only the relevant columns from the df
        df = df[cols]
        # Prediction to range 0 to 1
        for answer in answer_names:
            df[answer] = df[answer].apply(lambda x: arrange_prediction(x))
        # Remove prediction if it does not some up to 1
        df['sum_prediction'] = df[answer_names].sum(axis=1)
        df = df[df['sum_prediction'] > 0.999]
        # Confidence to range 0 to 1
        df['Confidence'] = df['Confidence'].apply(lambda x: x/10)
        # Class to 1 if correct and 0 if not
        df['Class'] = df['Class'].apply(lambda x: 1 if str(x).lower() == 'solver' else 0)
        # Write the processed df to csv
        file_name = file_path.split(sep="\\")[-1].split(sep='.')[0]
        df.to_csv(f'{PATH}\\ProcessedData\\{file_name}.csv')
    except:
        print('exception in ' + file_path, sys.exc_info()[0])

