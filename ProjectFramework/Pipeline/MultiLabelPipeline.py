import os
import traceback
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from skmultilearn.problem_transform import LabelPowerset, ClassifierChain
from sklearn.model_selection import train_test_split
from GetProcessedData import get_question_dfs
from AggregationMethods.ConfidenceMethods import highest_average_confidence
from AggregationMethods.SurprisinglyPopular import surprisingly_pop_answer
from AggregationMethods.MajorityRule import majority_answer
from FeaturesExtraction.AnswerFeatures import AnswerF
from FeaturesExtraction.AnswerFeaturesSubgroups import AnswerSubF
from FeaturesExtraction.ConfidenceFeatures import ConfidenceF
import FeaturesExtraction.ConfidenceFeaturesSubgroups

METHOD_NAMES = ['HAC', 'SP', 'MR']


def run_pipeline():
    data = create_data_df()
    feature_df = data.drop(METHOD_NAMES.extend(['correct_answer','MR_ans','SP_ans','HAC_ans']), axis=1, errors='ignore')
    feats = data[list(feature_df.columns)]
    target = data[METHOD_NAMES]
    X_train, X_test, y_train, y_test = train_test_split(data[list(feature_df.columns)], data[METHOD_NAMES], test_size=0.2, random_state=0)
    # initialize LabelPowerset multi-label classifier with a RandomForest
    classifier = ClassifierChain(
        classifier=RandomForestClassifier(n_estimators=100),
        require_dense=[False, True]
    )
    # train
    print("training")
    classifier.fit(X_train, y_train)
    # predict
    print("predicting")
    predictions = classifier.predict(X_test)
    print(accuracy_score(y_test, predictions))
    print(predictions)


def create_data_df():
    all_data = pd.DataFrame()
    question_dfs = get_question_dfs()
    for df in question_dfs:
        try:
            answers = AnswerF(df)
            answers_subs = AnswerSubF(df)
            confidence = ConfidenceF(df)
            correct_answer = str(df.loc[df['Class'] == 1, 'Answer'].iloc[0])
            d = {
                'HAC_ans' : highest_average_confidence(df),
                'SP_ans' : surprisingly_pop_answer(df),
                'MR_ans' : majority_answer(df),
                'correct_answer':correct_answer,
                'HAC': 1 if correct_answer == str(highest_average_confidence(df)) else 0,
                'SP': 1 if correct_answer == str(surprisingly_pop_answer(df)) else 0,
                'MR': 1 if correct_answer == str(majority_answer(df)) else 0,
                'A_num': answers.feature_get_num_of_answers(),
                'A_var': answers.get_total_var(),
                #'A_entropy': answers.feature_entropy(),
                'A_distance1-2': answers.feature_distance_between_first_and_second_answer(),
                'A_distance1-last': answers.feature_distance_between_first_and_last_answer(),
                'A_above50%': answers.feature_above_50_percent(),
                'A_wasser_uniform': answers.feature_wasserstein_distance_between_uniform_distribution(),
                'A_entropy_eliminate': answers.feature_entropy_without_low_rate_answers(),
                'AS_distribution_most_pop_ans': answers_subs.feature_distribution_of_most_popular_answer(),
                'AS_entropy_distance_highest_lowest': answers_subs.feature_groups_distance_between_highest_to_lowest_entropy(),
                'AS_std_distance_highest_lowest': answers_subs.feature_groups_distance_between_highest_to_lowest_std(),
                'AS_var_distance_highest_lowest': answers_subs.feature_groups_distance_between_highest_to_lowest_var(),
                'AS_has_pop_ans_changed': answers_subs.feature_if_most_popular_answer_changed(),
                'C_highest': confidence.feature_get_highest(),
                'C_lowest': confidence.feature_get_lowest(),
                'C_mean': confidence.feature_get_total_mean(),
                'C_std': confidence.feature_get_total_std(),
                'C_var': confidence.feature_get_total_var(),
                'C_least_pop': confidence.feature_confidence_of_least_popular_answer(),
                'C_most_pop': confidence.feature_confidence_of_most_popular_answer(),
                'C_above_90': confidence.feature_count_highest_above_90(),
                'C_above_95': confidence.feature_count_highest_above_95(),
                'C_above_98': confidence.feature_count_highest_above_98(),
                'C_under_15': confidence.feature_count_lowest_under_15(),
                'C_distance_mean': confidence.feature_get_distance_highest_from_mean(),
                'C_if_highest_pop': confidence.feature_is_highest_confidence_is_most_popular(),
            }
            all_data = all_data.append(d, ignore_index=True)
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)
            continue
    path = Path(os.path.abspath(__file__))
    all_data.to_csv(os.path.dirname(path.parent))
    return all_data



run_pipeline()
