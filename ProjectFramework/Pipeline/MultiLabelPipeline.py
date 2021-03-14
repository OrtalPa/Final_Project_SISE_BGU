import os
import traceback
from pathlib import Path
from sklearn import metrics
import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from skmultilearn.problem_transform import ClassifierChain
from sklearn.model_selection import train_test_split
from GetProcessedData import get_question_dfs, get_question_dicts
from AggregationMethods.ConfidenceMethods import *
from AggregationMethods.SurprisinglyPopular import surprisingly_pop_answer
from AggregationMethods.MajorityRule import majority_answer
from FeaturesExtraction.AnswerFeatures import AnswerF
from FeaturesExtraction.AnswerFeaturesSubgroups import AnswerSubF
from FeaturesExtraction.ConfidenceFeatures import ConfidenceF
from FeaturesExtraction.ConfidenceFeaturesSubgroups import *
from FeaturesExtraction.PredictionFeatures import PredictionsF

# Highest average confidence, surprisingly popular, majority rule, weighted confidence
METHOD_NAMES = ['HAC', 'SP', 'MR', 'WC']
path = Path(os.path.abspath(__file__))
RESULT_FILE_NAME = os.path.dirname(path.parent)+"\\results.csv"


def run_pipeline(data):
    feature_df = data.drop(METHOD_NAMES, axis=1, errors='ignore')
    X_train, X_test, y_train, y_test = train_test_split(data[list(feature_df.columns)], data[METHOD_NAMES], test_size=0.2, random_state=0)
    classifier = classifier_chain_rf(X_train, X_test, y_train, y_test)
    results = get_model_results(classifier, X_test, y_test)
    print_model(results)


def get_model_results(clf, X_test, y_test):
    prediction = clf.predict(X_test)
    r_square = clf.score(X_test, y_test)
    accuracy = accuracy_score(y_test, prediction)
    rounded_pred = np.around(prediction)
    metrics_cls_report = metrics.classification_report(y_test, rounded_pred, zero_division=0)
    return [prediction, r_square, accuracy, rounded_pred, metrics_cls_report]


# multi layer perceptron classifier
def mlp_cls(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    # clf.predict_proba(X_test[:1])
    clf.fit(X_test)
    return clf


def classifier_chain_rf(X_train, X_test, y_train, y_test):
    # initialize LabelPowerset multi-label classifier with a RandomForest
    clf = ClassifierChain(
        classifier=RandomForestClassifier(n_estimators=100),
        require_dense=[False, True]
    )
    # train
    clf.fit(X_train, y_train)
    return clf


def random_forest_cls(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=0)
    # train
    clf.fit(X_train, y_train)
    return clf


def print_model(model_results):
    for r in model_results:
        print(r)


'''Confusion Matrix'''
# # Construct the Confusion Matrix
# labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# cm = confusion_matrix(y_test, prediction, labels)
# print(cm)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(cm)
# plt.title('Confusion matrix')
# fig.colorbar(cax)
# ax.set_xticklabels([''] + labels)
# ax.set_yticklabels([''] + labels)
# plt.xlabel('Predicted Values')
# plt.ylabel('Actual Values')
# plt.show()


def create_data_df():
    # gets the questions data from the files
    # each question has a df of its own, each row is an answer of a person
    ## question_dfs = get_question_dfs()
    question_dict_df = get_question_dicts()
    # will contain a row for each question at the end
    all_data = pd.DataFrame()
    for df_name in question_dict_df:
        try:
            df = question_dict_df[df_name]
            # create the classes that create the features
            answers = AnswerF(df)
            answers_subs = AnswerSubF(df)
            confidence = ConfidenceF(df)
            confidence_subs = ConfidenceSubF(df)
            predictions = PredictionsF(df)
            correct_answer = str(df[df['Class'] == 1]['Answer'].iloc[0])
            d = {
                'HAC': 1 if correct_answer == highest_average_confidence(df) else 0,
                'SP': 1 if correct_answer == surprisingly_pop_answer(df) else 0,
                'MR': 1 if correct_answer == majority_answer(df) else 0,
                'WC': 1 if correct_answer == weighted_confidence(df) else 0,
                'A_num': answers.feature_get_num_of_answers(),
                'A_var': answers.get_total_var(),
                'A_entropy': answers.feature_entropy(),
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
                'CS_var_most_pop': confidence_subs.feature_variance_of_most_popular_answer_confidence_in_subgroups(),
                'CS_var_least_pop': confidence_subs.feature_variance_of_least_popular_answer_confidence_in_subgroups(),
                'CS_mean_high_low_diff': confidence_subs.feature_groups_distance_between_highest_to_lowest_confidence_mean(),
                'CS_var_high_low_diff': confidence_subs.feature_groups_distance_between_highest_to_lowest_confidence_var(),
                'P_highest_mean': predictions.feature_get_highest_mean_prediction_for_answer(),
                'P_highest_std': predictions.feature_get_highest_std_prediction_for_answer(),
                'P_highest_var': predictions.feature_get_highest_var_prediction_for_answer(),
                'P_lowest_mean': predictions.feature_get_lowest_mean_prediction_for_answer(),
                'P_lowest_var': predictions.feature_get_lowest_var_prediction_for_answer(),
                'P_lowest_std': predictions.feature_get_lowest_std_prediction_for_answer(),
                'P_above_80': predictions.feature_count_highest_above_80(),
                'P_prediction': predictions.feature_count_prediction_equal_to_votes(),
                'P_higher_then_vote': predictions.feature_count_prediction_higher_then_votes(),
                'P_lower_then_votes': predictions.feature_count_prediction_lower_then_votes(),
                'P_high_con_low_p': predictions.feature_count_high_confidence_low_prediction(),
                'P_low_con_high_p': predictions.feature_count_low_confidence_high_prediction(),
            }
            all_data = all_data.append(d, ignore_index=True)
        except Exception as e:
            print("error in "+df_name)
            print(e)
            traceback.print_tb(e.__traceback__)
            continue
    all_data.to_csv(RESULT_FILE_NAME)
    return all_data


# creates a csv file containing a row for each question with features
# result = create_data_df()
# read the result file once it's created
result = pd.read_csv(RESULT_FILE_NAME, index_col=0)
run_pipeline(result)
