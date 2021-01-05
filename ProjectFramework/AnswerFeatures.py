import pandas as pd
from scipy.stats import entropy
import numpy as np
from sklearn.utils import shuffle
from scipy.stats import wasserstein_distance

class AnswerF:

    def __init__(self, df):
        # todo self init
        self.df = df
        self.unique_answers = df['answer'].unique()
        self.num_of_ans = df['answer'].nunique()
        self.avg_for_answer = df['answer'].size / self.num_of_ans
        self.answers_distribution = self.build_answers_distribution_array()
        self.answers_count = self.build_answers_count_array()

        self.total_std = self.get_total_std()
        self.total_var = self.get_total_var()

    # get number of solvers how chose answer "ans_name"
    def get_answers_number(self, ans_name):
        return (self.df['answer'] == ans_name).sum()

    # This function builds the array of how many people chose each answer
    def build_answers_distribution_array(self):
        dictionary = {}
        for answer_name in self.unique_answers:
            answer_number = self.get_answers_number(answer_name) / self.num_of_ans
            dictionary[answer_name] = answer_number
        return dictionary

    # This function builds the array of how many people chose each answer
    def build_answers_count_array(self):
        dictionary = {}
        for answer_name in self.unique_answers:
            answer_number = self.get_answers_number(answer_name)
            dictionary[answer_name] = answer_number
        return dictionary

    # get list - for every answer, it calculates the distance between uniform distribution
    # and the amount of solvers how chose this answer
    def get_answers_distance_from_uniform(self):
        dfu_dict = {}
        for answer_name in self.unique_answers:
            distance_from_avg = pow(self.answers_distribution[answer_name] - self.avg_for_answer, 2)
            dfu_dict[answer_name] = distance_from_avg
        return dfu_dict

    # get the variance value of all answers distribution
    def get_total_var(self):
        var = np.var(list(self.answers_count.values()))
        return var

    # get the standard deviation value of all answers distribution
    def get_total_std(self):
        std = np.std(list(self.answers_count.values()))
        return std

    # get entropy of answers distribution
    def feature_entropy(self):
        return entropy(np.array(self.answers_count.values()), base=None)

    # get distance between two highest answers and subtract std
    def feature_distance_between_first_and_second_answer(self):
        sorted_distribution_by_value = sorted(self.answers_count.values(),reverse=True)
        difference = float(sorted_distribution_by_value[0]) - float(sorted_distribution_by_value[1])
        return difference - self.total_std

    # calculates the distance between first and last answer someone picked, divided by the std
    def feature_distance_between_first_and_last_answer(self):
        sorted_distribution_by_value = sorted(self.answers_count.values(),reverse=True)
        for value in sorted_distribution_by_value:
            if value != 0:
                last_value = value
        difference = float(sorted_distribution_by_value[0]) - float(last_value)
        return difference - self.total_std

    # return 1 if the most popular answer was picked more than 50%, otherwise 0
    def feature_above_50_percent(self):
        if self.sorted_distribution_by_value[0] / self.num_of_ans >= 0.5:
            return 1
        return 0

    # return the wasserstein between the uniform distribution
    def feature_wasserstein_distance_between_uniform_distribution(self):
        uniform_distribution = np.full(1, self.num_of_ans, self.avg_for_answer)
        return wasserstein_distance(self.answers_distribution, uniform_distribution)

    # calculates entropy of the data when eliminating low rate answers (under defined threshold)
    def feature_entropy_without_low_rate_answers(self):
        THRESHOLD = 15/100
        updated_list = []
        all_answers = self.df['answer'].size
        for value in self.answers_count.values():
            if value >= all_answers * THRESHOLD:
                updated_list.append(value)
        return entropy(np.array(updated_list), base=None)

# test function
def main():
    cereal_df = pd.read_csv("C:\\Users\\school & work\\PycharmProjects\\Final_Project_SISE_BGU\\test.csv")
    a = AnswerF(cereal_df)
    # b = a.get_total_std()
    # print(f'std: {b}')
    # c = a.get_total_var()
    # print(f'var: {c}')
    # d = a.feature_distance_between_first_and_last_answer()
    # print(f'first and last: {d}')
    # e = a.feature_distance_between_first_and_second_answer()
    # print(f'first and second: {e}')
    # f = a.feature_entropy()
    # print(f'Entropy: {f}')
    # a.build_sub_groups()
    g = a.feature_entropy_without_low_rate_answers()
    print(f'Entropy without low rate: {g}')


if __name__ == "__main__":
    main()
