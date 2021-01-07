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
    def feature_get_total_mean(self):
        var = float(np.mean(self.df_conf))
        return var

    # get the variance value of all answers confidence
    def feature_get_total_var(self):
        var = float(np.var(self.df_conf))
        return var

    # get the standard deviation value of all answers confidence
    def feature_get_total_std(self):
        std = float(np.std(self.df_conf))
        return std

    # get the highest value of confidence
    def feature_get_highest(self):
        highest = float(sorted(self.df_conf, reverse=True)[0])
        return highest

    # get the count of confidence with value over 0.90
    def feature_count_highest_above_90(self):
        counter = 0
        highest = sorted(self.df_conf, reverse=True)
        for i in highest:
            if i >= 0.90:
                counter += 1
        return float(counter/self.df_conf.size)

    # get the count of confidence with value over 0.95
    def feature_count_highest_above_95(self):
        counter = 0
        highest = sorted(self.df_conf, reverse=True)
        for i in highest:
            if i >= 0.95:
                counter += 1
        return float(counter/self.df_conf.size)

    # get the count of confidence with value over 0.98
    def feature_count_highest_above_98(self):
        counter = 0
        highest = sorted(self.df_conf, reverse=True)
        for i in highest:
            if i >= 0.98:
                counter += 1
        return float(counter/self.df_conf.size)

    # get the distance from highest confidence to mean
    def feature_get_distance_highest_from_mean(self):
        highest = float(sorted(self.df_conf, reverse=True)[0])
        return float(highest-self.feature_get_total_mean())

    # get the lowest value of confidence
    def feature_get_lowest(self):
        lowest = float(sorted(self.df_conf, reverse=False)[0])
        return lowest

    # get the count of confidence with value under 0.15
    def feature_count_lowest_under_15(self):
        counter = 0
        lowest = sorted(self.df_conf, reverse=False)
        for i in lowest:
            if i <= 0.15:
                counter += 1
        return float(counter / self.df_conf.size)

    ######################################## confidence && answers features ##########################################

    # This function builds the array of how many people chose each answer
    def build_answers_count_array(self):
        dictionary = {}
        for answer_name in self.unique_answers:
            answer_number = (self.df['Answer'] == answer_name).sum()
            if answer_number != 0:
                dictionary[answer_name] = answer_number
        return dictionary

    # This function gets the avg confidence of the most popular answer
    def feature_confidence_of_most_popular_answer(self):
        answers_count = self.build_answers_count_array()
        sorted_distribution_by_value = {k: v for k, v in sorted(answers_count.items(), key=lambda item: item[1], reverse=True)}
        most_popular_answer = next(iter(sorted_distribution_by_value))
        confidence_list = self.df[self.df['Answer'] == most_popular_answer]['Confidence']
        return np.mean(confidence_list)

    # This function gets the avg confidence of the least popular answer
    def feature_confidence_of_least_popular_answer(self):
        answers_count = self.build_answers_count_array()
        sorted_distribution_by_value = {k: v for k, v in sorted(answers_count.items(), key=lambda item: item[1])}
        least_popular_answer = next(iter(sorted_distribution_by_value))
        confidence_list = self.df[self.df['Answer'] == least_popular_answer]['Confidence']
        return np.mean(confidence_list)

    # This function returns 1 if the most popular answer has the highest confidence
    def feature_is_highest_confidence_is_most_popular(self):
        answers_count = self.build_answers_count_array()
        sorted_distribution_by_value = {k: v for k, v in sorted(answers_count.items(), key=lambda item: item[1], reverse=True)}
        most_popular_answer = ""
        most_popular_answer_confidence = ""
        for answer in sorted_distribution_by_value:
            if most_popular_answer_confidence=="":
                most_popular_answer_confidence = np.mean(self.df[self.df['Answer'] == answer]['Confidence'])
            else:
                current_confidence = np.mean(self.df[self.df['Answer'] == answer]['Confidence'])
                if current_confidence > most_popular_answer_confidence:
                    return 1
        return 0

# test function
def main():
    df = pd.read_csv("C:\\Users\\school & work\\PycharmProjects\\Final_Project_SISE_BGU\\ProcessedData\\RawData_Apple.csv")
    a = ConfidenceF(df)
    b = a.feature_get_total_std()
    print(f'std: {b}')
    c = a.feature_get_total_var()
    print(f'var: {c}')
    d = a.feature_get_total_mean()
    print(f'mean: {d}')
    e = a.feature_get_highest()
    print(f'high: {e}')
    f = a.feature_get_distance_highest_from_mean()
    print(f'high minus mean: {f}')
    g = a.feature_count_highest_above_90()
    print(f'count high 90: {g}')
    h = a.feature_count_highest_above_95()
    print(f'count high 95: {h}')
    i = a.feature_count_highest_above_98()
    print(f'count high 98: {i}')
    j = a.feature_get_lowest()
    print(f'lowest: {j}')
    k = a.feature_count_lowest_under_15()
    print(f'lowest under 15: {k}')
    l = a.feature_is_highest_confidence_is_most_popular()
    print(f'is most popular has biggest confidence: {l}')


if __name__ == "__main__":
    main()