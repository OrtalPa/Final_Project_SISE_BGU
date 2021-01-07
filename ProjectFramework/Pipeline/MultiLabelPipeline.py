import pandas as pd
from GetProcessedData import get_question_dfs
from AggregationMethods.ConfidenceMethods import highest_average_confidence
from AggregationMethods.SurprisinglyPopular import surprisingly_pop_answer
from AggregationMethods.MajorityRule import majority_answer


def run_pipeline():
    data = create_data_df()
    # train model


def create_data_df():
    question_dfs = get_question_dfs()
    all_data = pd.DataFrame()
    for df in question_dfs:
        correct_answer = str(df[df['Class'] == 1]['Answer'].iloc[0])
        d = {
            'HAC': 1 if correct_answer == highest_average_confidence(df) else 0,
            'SP': 1 if correct_answer == surprisingly_pop_answer(df) else 0,
            'MR': 1 if correct_answer == majority_answer(df) else 0
        }
        all_data.append(d, ignore_index=True)
    return all_data
