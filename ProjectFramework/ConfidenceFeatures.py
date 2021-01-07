import pandas as pd
from scipy.stats import entropy
import numpy as np
from sklearn.utils import shuffle
from scipy.stats import wasserstein_distance
from GetProcessedData import get_answer_names


class ConfidenceF:

    def __init__(self, df):
        self.df = df
        self.df_conf = df['Confidence']
        self.unique_answers = get_answer_names(df)
        self.num_of_ans = self.unique_answers.size

    # get the mean value of all answers confidence
    def get_total_mean(self):
        var = float(np.mean(self.df_conf))
        return var

    # get the variance value of all answers confidence
    def get_total_var(self):
        var = float(np.var(self.df_conf))
        return var

    # get the standard deviation value of all answers confidence
    def get_total_std(self):
        std = float(np.std(self.df_conf))
        return std

    # get the standard deviation value of all answers confidence
    def get_highest(self):
        highest = float(sorted(self.df_conf, reverse=True)[0])
        return highest

    # get the standard deviation value of all answers confidence
    def count_highest_above_90(self):
        counter = 0
        highest = sorted(self.df_conf, reverse=True)
        for i in highest:
            if i >= 0.90:
                counter += 1
        return float(counter/self.df_conf.size)

    def count_highest_above_95(self):
        counter = 0
        highest = sorted(self.df_conf, reverse=True)
        for i in highest:
            if i >= 0.95:
                counter += 1
        return float(counter/self.df_conf.size)

    def count_highest_above_98(self):
        counter = 0
        highest = sorted(self.df_conf, reverse=True)
        for i in highest:
            if i >= 0.98:
                counter += 1
        return float(counter/self.df_conf.size)

    # get the standard deviation value of all answers confidence
    def get_distance_highest_from_mean(self):
        highest = float(sorted(self.df_conf, reverse=True)[0])
        return float(highest-self.get_total_mean())

    # This function builds the array of how many people chose each answer
    def build_answers_count_array(self):
        dictionary = {}
        for answer_name in self.unique_answers:
            answer_number = (self.df['Answer'] == answer_name).sum()
            if answer_number != 0:
                dictionary[answer_name] = answer_number
        return dictionary

    # This function gets the avg confidence of the most popular answer
    def get_confidence_of_most_popular_answer(self):
        answers_count = self.build_answers_count_array()
        sorted_distribution_by_value = {k: v for k, v in sorted(answers_count.items(), key=lambda item: item[1], reverse=True)}
        most_popular_answer = next(iter(sorted_distribution_by_value))
        confidence_list = self.df[self.df['Answer'] == most_popular_answer]['Confidence']
        return np.mean(confidence_list)

    # This function gets the avg confidence of the least popular answer
    def get_confidence_of_least_popular_answer(self):
        answers_count = self.build_answers_count_array()
        sorted_distribution_by_value = {k: v for k, v in sorted(answers_count.items(), key=lambda item: item[1])}
        least_popular_answer = next(iter(sorted_distribution_by_value))
        confidence_list = self.df[self.df['Answer'] == least_popular_answer]['Confidence']
        return np.mean(confidence_list)


# test function
def main():
    df = pd.read_csv("C:\\Users\\Pnina\\PycharmProjects\\Final_Project_SISE_BGU\\ProcessedData\\RawData_Apple.csv")
    a = ConfidenceF(df)
    b = a.get_total_std()
    print(f'std: {b}')
    c = a.get_total_var()
    print(f'var: {c}')
    d = a.get_total_mean()
    print(f'mean: {d}')
    e = a.get_highest()
    print(f'high: {e}')
    f = a.get_distance_highest_from_mean()
    print(f'high minus mean: {f}')
    g = a.count_highest_above_90()
    print(f'count high 90: {g}')
    h = a.count_highest_above_95()
    print(f'count high 95: {h}')
    i = a.count_highest_above_98()
    print(f'count high 98: {i}')
    j = a.get_confidence_of_most_popular_answer()
    print(f'count most popular: {j}')
    k = a.get_confidence_of_least_popular_answer()
    print(f'count least popular: {k}')



if __name__ == "__main__":
    main()