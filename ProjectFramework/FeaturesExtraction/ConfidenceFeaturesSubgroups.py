import pandas as pd
from scipy.stats import entropy
import numpy as np
from sklearn.utils import shuffle
from ProjectFramework.GetProcessedData import get_answer_names

NUM_OF_GROUPS = 3


class ConfidenceSubF:

    def __init__(self, df):
        self.unique_answers = get_answer_names(df)
        self.num_of_ans = self.unique_answers.size
        self.subs = self.build_sub_groups(df)

    # get the mean value of all answers confidence
    def get_total_mean(self, df_conf):
        var = float(np.mean(df_conf))
        return var

    # get the variance value of all answers confidence
    def get_total_var(self, df_conf):
        var = float(np.var(df_conf))
        return var

    # get the standard deviation value of all answers confidence
    def get_total_std(self, df_conf):
        std = float(np.std(df_conf))
        return std

    # get the highest value of confidence
    def get_highest(self, df_conf):
        highest = float(sorted(df_conf, reverse=True)[0])
        return highest

    # get the count of confidence with value over 0.90
    def count_highest_above_90(self, df_conf):
        counter = 0
        highest = sorted(df_conf, reverse=True)
        for i in highest:
            if i >= 0.90:
                counter += 1
        return float(counter / df_conf.size)

    # get the count of confidence with value over 0.95
    def count_highest_above_95(self, df_conf):
        counter = 0
        highest = sorted(df_conf, reverse=True)
        for i in highest:
            if i >= 0.95:
                counter += 1
        return float(counter / df_conf.size)

    # get the count of confidence with value over 0.98
    def count_highest_above_98(self, df_conf):
        counter = 0
        highest = sorted(df_conf, reverse=True)
        for i in highest:
            if i >= 0.98:
                counter += 1
        return float(counter / df_conf.size)

    # get the distance from highest confidence to mean
    def get_distance_highest_from_mean(self, df_conf, total_mean):
        highest = float(sorted(df_conf, reverse=True)[0])
        return float(highest - total_mean)

    # get the lowest value of confidence
    def get_lowest(self, df_conf):
        lowest = float(sorted(df_conf, reverse=False)[0])
        return lowest

    # get the count of confidence with value under 0.15
    def count_lowest_under_15(self, df_conf):
        counter = 0
        lowest = sorted(df_conf, reverse=False)
        for i in lowest:
            if i <= 0.15:
                counter += 1
        return float(counter / df_conf.size)

    # This function builds the array of how many people chose each answer
    def build_answers_distribution_array(self, df):
        dictionary = {}
        for answer_name in self.unique_answers:
            answer_number = (df['Answer'] == answer_name).sum() / self.num_of_ans
            if answer_number != 0:
                dictionary[answer_name] = answer_number
        return dictionary

    # create NUM_OF_GROUPS subsets of the existing data frame, no overlaps, return list of data frames
    def build_sub_groups(self, big_df):
        shuffled = shuffle(big_df)
        sub_groups = []
        start = 0
        row_num = big_df.shape[0]
        jumps = int(row_num / NUM_OF_GROUPS)
        for i in range(NUM_OF_GROUPS):
            if start + jumps > row_num:
                sub_groups.append(shuffled.iloc[start:row_num])
                break
            sub_groups.append(shuffled.iloc[start:start + jumps])
            start = start + jumps
        return sub_groups

    ######################################## Subgroups Features ##########################################

    # get the difference between the subgroup with highest confidence mean and subgroup with lowest
    def feature_groups_distance_between_highest_to_lowest_confidence_mean(self):
        confidence_list = []
        for frame in self.subs:
            confidence_list.append(self.get_total_mean(frame['Confidence']))
        sorted_by_value = sorted(confidence_list, reverse=True)
        for value in sorted_by_value:
            if value != 0:
                last_value = value
        difference = float(sorted_by_value[0]) - float(last_value)
        return difference

    # get the difference between the subgroup with highest confidence var and subgroup with lowest
    def feature_groups_distance_between_highest_to_lowest_confidence_var(self):
        confidence_list = []
        for frame in self.subs:
            confidence_list.append(self.get_total_var(frame['Confidence']))
        sorted_by_value = sorted(confidence_list, reverse=True)
        for value in sorted_by_value:
            if value != 0:
                last_value = value
        difference = float(sorted_by_value[0]) - float(last_value)
        return difference

    # feature: returns the variance of the mean of the most popular answer's confidence
    def feature_variance_of_most_popular_answer_confidence_in_subgroups(self):
        confidence_array = []
        for sub_group in self.subs:
            answers_distribution = self.build_answers_distribution_array(sub_group)
            sorted_distribution_by_value = {k: answers_distribution[k] for k in
                                            sorted(answers_distribution, key=answers_distribution.get,
                                                   reverse=True)}
            most_popular_answer = list(sorted_distribution_by_value.keys())[0]
            most_popular_answer_confidence = np.mean(
                sub_group[sub_group['Answer'] == most_popular_answer]['Confidence'])
            confidence_array.append(most_popular_answer_confidence)
        return self.get_total_var(confidence_array)

    # feature: returns the variance of the mean of the least popular answer's confidence
    def feature_variance_of_least_popular_answer_confidence_in_subgroups(self):
        confidence_array = []
        for sub_group in self.subs:
            answers_distribution = self.build_answers_distribution_array(sub_group)
            sorted_distribution_by_value = {k: answers_distribution[k] for k in
                                            sorted(answers_distribution, key=answers_distribution.get,
                                                   reverse=False)}
            least_popular_answer = list(sorted_distribution_by_value.keys())[0]
            least_popular_answer_confidence = np.mean(
                sub_group[sub_group['Answer'] == least_popular_answer]['Confidence'])
            confidence_array.append(least_popular_answer_confidence)
        return self.get_total_var(confidence_array)


# test function
def main():
    df = pd.read_csv(
        "C:\\Users\\Pnina\\PycharmProjects\\Final_Project_SISE_BGU\\ProcessedData\\RawData_Apple.csv")
    conf = ConfidenceSubF(df)
    subgroups = conf.build_sub_groups(df)
    a = conf.feature_groups_distance_between_highest_to_lowest_confidence_mean(subgroups)
    print(f'std: {a}')
    b = conf.feature_groups_distance_between_highest_to_lowest_confidence_var(subgroups)
    print(f'std: {b}')
    c = conf.feature_variance_of_most_popular_answer_confidence_in_subgroups(subgroups)
    print(f'var: {c}')
    e = conf.feature_variance_of_least_popular_answer_confidence_in_subgroups(subgroups)
    print(f'mean: {e}')


if __name__ == "__main__":
    main()
