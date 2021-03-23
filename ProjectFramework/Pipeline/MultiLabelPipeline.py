import os
from pathlib import Path
from sklearn import metrics
import operator

from sklearn.ensemble import RandomForestClassifier
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
METHOD_NAMES = {0: 'HAC', 1: 'MR', 2: 'SP', 3: 'WC'}  # maps the method name to an index. DO NOT REPLACE ORDER
path = Path(os.path.abspath(__file__))
RESULT_FILE_NAME = os.path.dirname(path.parent)+"\\results.csv"


def run_pipeline(data):
    feature_df = data.drop(METHOD_NAMES.values(), axis=1, errors='ignore')
    X_train, X_test, y_train, y_test = train_test_split(data[list(feature_df.columns)], data[METHOD_NAMES.values()], test_size=0.2, random_state=0)
    classifier = classifier_chain_rf(X_train, X_test, y_train, y_test)
    results = get_model_results(classifier, X_test, y_test)
    acc = get_accuracy(results, y_test)
    print(acc)


def get_accuracy(results, y_test):
    count_true = 0
    for res in results.items():
        q_index = res[0]  # question index
        selected_method = str(res[1][0])  # selected method
        real_result = y_test.iloc[y_test.index == q_index][selected_method]
        real_result = real_result[q_index]
        count_true += real_result
    return str(count_true / len(y_test))


def get_model_results(clf, X_test, y_test):
    prediction_prob = clf.predict_proba(X_test)

    # maps the question index to an array of the methods suitable to solve it
    prediction_by_question_index = {}
    selected_method_for_q = {}
    i = 0
    for p in prediction_prob:
        # p.indices is the array of predicted methods
        question_index = X_test.index[i]
        answered_by = {}
        for method in p.indices:
            answered_by[METHOD_NAMES[method]] = p.data[method]
        prediction_by_question_index[question_index] = answered_by
        selected_method_for_q[question_index] = max(answered_by.items(), key=operator.itemgetter(1))
        i += 1

    print(prediction_by_question_index)
    print(selected_method_for_q)

    # other methods to test the model
    # r_square = clf.score(X_test, y_test)
    # accuracy = accuracy_score(y_test, prediction)
    # rounded_pred = np.around(prediction)
    # metrics_cls_report = metrics.classification_report(y_test, rounded_pred, zero_division=0)
    return selected_method_for_q


def classifier_chain_rf(X_train, X_test, y_train, y_test):
    # initialize ClassifierChain multi-label classifier with a RandomForest
    clf = ClassifierChain(
        classifier=RandomForestClassifier(n_estimators=100),
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
            res_from_agg_methods = ""
            res_from_agg_methods += 'HAC' if correct_answer == highest_average_confidence(df) else ''
            res_from_agg_methods += 'SP' if correct_answer == surprisingly_pop_answer(df) else ''
            res_from_agg_methods += 'MR' if correct_answer == majority_answer(df) else ''
            res_from_agg_methods += 'WC' if correct_answer == weighted_confidence(df) else ''
            d = {
                'HAC': 'HAC' if correct_answer == highest_average_confidence(df) else 'NOT_HAC',
                'MR': 'MR' if correct_answer == majority_answer(df) else 'NOT_MR',
                'SP': 'SP' if correct_answer == surprisingly_pop_answer(df) else 'NOT_SP',
                'WC': 'WC' if correct_answer == weighted_confidence(df) else 'NOT_WC',
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
