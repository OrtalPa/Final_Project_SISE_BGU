import pandas as pd
from scipy.stats import entropy
import numpy as np
from sklearn.utils import shuffle

NUM_OF_GROUPS = 3


class AnswerSubF:

    def __init__(self, df):
        # todo self init
        self.df = df
        self.unique_answers = df['answer'].unique()
        self.num_of_ans = df['answer'].nunique()

    # get number of solvers how chose answer "ans_name"
    def get_answers_number(self, df, ans_name):
        return (df['answer'] == ans_name).sum()

    # This function builds the array of how many people chose each answer
    def build_answers_distribution_array(self, df):
        dictionary = {}
        for answer_name in self.unique_answers:
            answer_number = self.get_answers_number(df, answer_name) / self.num_of_ans
            dictionary[answer_name] = answer_number
        return dictionary

    # This function builds the array of how many people chose each answer
    def build_answers_count_array(self, df):
        dictionary = {}
        for answer_name in self.unique_answers:
            answer_number = self.get_answers_number(df, answer_name)
            dictionary[answer_name] = answer_number
        return dictionary

    # get list - for every answer, it calculates the distance between uniform distribution
    # and the amount of solvers how chose this answer
    def get_answers_distance_from_uniform(self, df):
        dfu_dict = {}
        for answer_name in self.unique_answers:
            distance_from_avg = pow(
                self.build_answers_distribution_array(df)[answer_name] - df['answer'].size / self.num_of_ans, 2)
            dfu_dict[answer_name] = distance_from_avg
        return dfu_dict

    # get the variance value of all answers distribution
    def get_var(self, answers_count):
        var = np.var(list(answers_count.values()))
        return var

    # get the standard deviation value of all answers distribution
    def get_std(self, answers_count):
        std = np.std(list(answers_count.values()))
        return std

    # get entropy of answers distribution
    def feature_entropy(self, df):
        value, counts = np.unique(df["answer"], return_counts=True)
        return entropy(counts, base=None)

    # get distance between two highest answers and subtract std
    def feature_distance_between_first_and_second_answer(self, answers_count, std):
        sorted_distribution_by_value = sorted(answers_count.values(), reverse=True)
        difference = float(sorted_distribution_by_value[0]) - float(sorted_distribution_by_value[1])
        return difference - std

    # calculates the distance between first and last answer someone picked, divided by the std
    def feature_distance_between_first_and_last_answer(self, answers_count, std):
        sorted_distribution_by_value = sorted(answers_count.values(), reverse=True)
        for value in sorted_distribution_by_value:
            if value != 0:
                last_value = value
        difference = float(sorted_distribution_by_value[0]) - float(last_value)
        return difference - std

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

    def feature_groups_distance_between_first_and_last_highest_distribution(self):
        subsets = self.build_sub_groups()
        for frame in subsets:
            into = 5

    # feature: the distribution of the most popular answer in each subgroup
    def feature_distribution_of_most_popular_answer(self, subs):
        most_popular_distribution = []
        for sub_group in subs:
            answers_distribution = self.build_answers_distribution_array(sub_group)
            sorted_distribution_by_value = sorted(answers_distribution.values(), reverse=True)
            most_popular_distribution.append(sorted_distribution_by_value[1] / sum(answers_distribution))
        return np.var(most_popular_distribution)

    # feature: if the most popular answer is in each subgroup id equal - 1, else- 0
    def feature_if_most_popular_answer_changed(self, subs):
        last_value = ""
        for sub_group in subs:
            answers_distribution = self.build_answers_distribution_array(sub_group)
            sorted_distribution_by_value = {k: answers_distribution[k] for k in
                                            sorted(answers_distribution, key=answers_distribution.get, reverse=True)}
            the_key = list(sorted_distribution_by_value.keys())[0]
            if last_value == "":
                last_value = the_key
            elif last_value != the_key:
                return 0
        return 1


# test function
def main():
    cereal_df = pd.read_csv("C:\\Users\\school & work\\PycharmProjects\\Final_Project_SISE_BGU\\test.csv")
    a = AnswerSubF(cereal_df)
    subs = a.build_sub_groups(cereal_df)
    group_num = 0;
    for sub in subs:
        ans_count = a.build_answers_count_array(sub)
        b = a.get_std(ans_count)
        print(f'std {group_num}: {b}')
        c = a.get_var(ans_count)
        print(f'var {group_num}: {c}')
        d = a.feature_distance_between_first_and_last_answer(ans_count, b)
        print(f'first and last {group_num}: {d}')
        e = a.feature_distance_between_first_and_second_answer(ans_count, b)
        print(f'first and second {group_num}: {e}')
        f = a.feature_entropy(sub)
        print(f'Entropy {group_num}: {f}')
        group_num += 1


if __name__ == "__main__":
    main()

    """
    תתי קבוצות:
    מרחקים בין התפלגויות בכל תת קבוצה
    השונות של אחוז הפפולריות של התשובה הפפולרית ביותר
    האם השתנה התשובה במקום הראשון בין תתי קבוצות
    הורדת 10 אחוז באנתרופיה והתפלגות אחידה
    """
