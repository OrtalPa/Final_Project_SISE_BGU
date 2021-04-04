import os
from pathlib import Path
from sklearn import metrics
import operator

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import BinaryRelevance

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
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset


# Highest average confidence, surprisingly popular, majority rule, weighted confidence
from Pipeline.feature_selection import get_features_with_high_var

METHOD_NAMES = {0: 'HAC', 1: 'MR', 2: 'NONE', 3: 'SP', 4: 'WC'}  # maps the method name to an index. DO NOT REPLACE ORDER
path = Path(os.path.abspath(__file__))
RESULT_FILE_NAME = os.path.dirname(path.parent)+"\\results.csv"
# skip questions that have only 10 respondents and on one answered correctly
FILES_TO_SKIP = ["RawData_Pills", "W10_0", "W10_15", "W10_16", "W11_18", "W12_12", "W12_31",
                 "W12_35", "W4_16", "W5_31", "W6_16", "W6_23", "W6_24", "W7_26", "W7_45", "W7_5",
                 "W8_20", "W8_27", "W9_27"]


def normalize_df(feature_df):
    scaler = MinMaxScaler()
    # Fit and transform the data
    df_norm = pd.DataFrame(scaler.fit_transform(feature_df), columns=feature_df.columns)
    return df_norm


def run_pipeline(data):
    # drop label names and dataset id
    feature_df = data.drop(get_label_names(), axis=1, errors='ignore')
    feature_df = feature_df.drop('dataset_id', axis=1, errors='ignore')
    df_norm = normalize_df(feature_df)
    df_norm = pd.DataFrame(get_features_with_high_var(df_norm))
    X_train, X_test, y_train, y_test = train_test_split(df_norm, data[get_label_names()], test_size=0.2, random_state=0)
    classifier = classifier_chain(X_train, y_train, random_forest_cls(X_train, y_train))
    results = get_chain_model_results(classifier, X_test)
    acc = get_accuracy(results, y_test)
    print(str(acc))


# receives results in the form of a dictionary, key: question index ; value: selected method by name
# y_test is a df with index of questions:
# for each method there is a column with 1 value if the method was correct for the question and 0 if not
def get_accuracy(results, y_test):
    count_true = 0
    for res in results.items():
        q_index = res[0]  # question index
        selected_method = str(res[1][0])  # selected method
        # here real_result is a tuple of (index,result)
        real_result = y_test.iloc[y_test.index == q_index][selected_method]
        # now we take the result of the index which is 0 or 1
        real_result = real_result[q_index]
        # add to the count so finally it will be the sum of all correct results
        count_true += real_result
    return count_true / len(y_test)


# receives trained multilabel classifier and returns a dictionary with the results
def get_chain_model_results(clf, X_test):
    prediction_prob = clf.predict_proba(X_test)

    # maps the question index to an array of the methods suitable to solve it
    prediction_by_question_index = {}
    selected_method_for_q = {}
    i = 0
    for p in prediction_prob:
        try:
            # p.indices is the array of predicted methods
            question_index = X_test.index[i]
            answered_by = {}
            method_index = 0
            for method in p.indices:
                answered_by[METHOD_NAMES[method]] = p.data[method_index]  # data [v, v, v] indices [1,3,4]
                method_index += 1
            prediction_by_question_index[question_index] = answered_by
            selected_method_for_q[question_index] = max(answered_by.items(), key=operator.itemgetter(1))
            i += 1
        except Exception as e:
            print("error in get_chain_model_results")
            print(e)
            continue

    # other methods to test the model
    # r_square = clf.score(X_test, y_test)
    # accuracy = accuracy_score(y_test, prediction)
    # rounded_pred = np.around(prediction)
    # metrics_cls_report = metrics.classification_report(y_test, rounded_pred, zero_division=0)
    return selected_method_for_q


# receives trained multilabel classifier and returns a dictionary with the results
def get_binary_model_results(clf, X_test):
    prediction_prob = clf.predict_proba(X_test)

    # maps the question index to an array of the methods suitable to solve it
    prediction_by_question_index = {}
    selected_method_for_q = {}
    i = 0
    for p in prediction_prob:
        try:
            # p.indices is the array of predicted methods
            question_index = X_test.index[i]
            answered_by = {}
            for method in range(len(p.rows[0])):
                answered_by[METHOD_NAMES[method]] = p.data[0][method]
            prediction_by_question_index[question_index] = answered_by
            selected_method_for_q[question_index] = max(answered_by.items(), key=operator.itemgetter(1))
            i += 1
        except Exception as e:
            print("error in get_binary_model_results")
            print(e)
            continue

    return selected_method_for_q


def classifier_chain(X_train, y_train, classifier):
    # initialize ClassifierChain multi-label classifier with a RandomForest
    clf = ClassifierChain(
        classifier=classifier,
    )
    # train
    clf.fit(X_train, y_train)
    return clf


def binary_relevance(X_train, y_train, classifier):
    # initialize BinaryRelevance multi-label classifier
    clf = BinaryRelevance(
        classifier=classifier,
    )
    # train
    clf.fit(X_train, y_train)
    return clf


def label_powerset(X_train, y_train, classifier):
    # initialize LabelPowerset multi-label classifier
    clf = LabelPowerset(
        classifier=classifier,
    )
    # train
    clf.fit(X_train, y_train)
    return clf


def random_forest_cls(X_train, y_train):
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


def no_method_succeeded(df, correct_ans):
    return (correct_ans != highest_average_confidence(df))\
           and (correct_ans != surprisingly_pop_answer(df))\
           and (correct_ans != weighted_confidence(df))\
           and (correct_ans != majority_answer(df))


def create_data_df():
    # gets the questions data from the files
    # each question has a df of its own, each row is an answer of a person
    question_dict_df = get_question_dicts()
    # will contain a row for each question at the end
    all_data = pd.DataFrame()
    for df_name in question_dict_df:
        if any(name in df_name for name in FILES_TO_SKIP):
            # skip this file
            continue
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
                'dataset_id': df_name.split('_')[-1],
                'HAC': 1 if correct_answer == highest_average_confidence(df) else 0,
                'MR': 1 if correct_answer == majority_answer(df) else 0,
                'SP': 1 if correct_answer == surprisingly_pop_answer(df) else 0,
                'WC': 1 if correct_answer == weighted_confidence(df) else 0,
                'NONE': 1 if no_method_succeeded(df, correct_answer) else 0,
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


def get_data():
    # creates a csv file containing a row for each question with features
    # result = create_data_df()
    # read the result file once it's created
    result = pd.read_csv(RESULT_FILE_NAME, index_col=0)
    return result


def get_label_names():
    return METHOD_NAMES.values()


run_pipeline(get_data())
