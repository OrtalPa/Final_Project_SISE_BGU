import pandas as pd
from GetProcessedData import get_question_dfs
from AggregationMethods.ConfidenceMethods import highest_average_confidence
from AggregationMethods.SurprisinglyPopular import surprisingly_pop_answer
from AggregationMethods.MajorityRule import majority_answer
from FeaturesExtraction.AnswerFeatures import AnswerF
from FeaturesExtraction.AnswerFeaturesSubgroups import AnswerSubF
from FeaturesExtraction.ConfidenceFeatures import ConfidenceF
from FeaturesExtraction.ConfidenceFeaturesSubgroups import ConfidenceSubF
from FeaturesExtraction.PredictionFeatures import PredictionsF


def run_pipeline():
    data = create_data_df()
    # train model


def create_data_df():
    question_dfs = get_question_dfs()
    all_data = pd.DataFrame()
    for df in question_dfs:
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
            'P_predicition': predictions.feature_count_prediction_equal_to_votes(),
            'P_higher_then_vote': predictions.feature_count_prediction_higher_then_votes(),
            'P_lower_then_votes': predictions.feature_count_prediction_lower_then_votes(),
            'P_high_con_low_p': predictions.feature_count_high_confidence_low_prediction(),
            'P_low_con_high_p': predictions.feature_count_low_confidence_high_prediction(),

        }
        all_data.append(d, ignore_index=True)
    return all_data
