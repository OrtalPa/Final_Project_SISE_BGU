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

    # get the highest value of confidence
    def get_highest(self):
        highest = float(sorted(self.df_conf, reverse=True)[0])
        return highest

    # get the count of confidence with value over 0.90
    def count_highest_above_90(self):
        counter = 0
        highest = sorted(self.df_conf, reverse=True)
        for i in highest:
            if i >= 0.90:
                counter += 1
        return float(counter/self.df_conf.size)

    # get the count of confidence with value over 0.95
    def count_highest_above_95(self):
        counter = 0
        highest = sorted(self.df_conf, reverse=True)
        for i in highest:
            if i >= 0.95:
                counter += 1
        return float(counter/self.df_conf.size)

    # get the count of confidence with value over 0.98
    def count_highest_above_98(self):
        counter = 0
        highest = sorted(self.df_conf, reverse=True)
        for i in highest:
            if i >= 0.98:
                counter += 1
        return float(counter/self.df_conf.size)

    # get the distance from highest confidence to mean
    def get_distance_highest_from_mean(self):
        highest = float(sorted(self.df_conf, reverse=True)[0])
        return float(highest-self.get_total_mean())

    # get the lowest value of confidence
    def get_lowest(self):
        lowest = float(sorted(self.df_conf, reverse=False)[0])
        return lowest

    # get the count of confidence with value under 0.15
    def count_lowest_under_15(self):
        counter = 0
        lowest = sorted(self.df_conf, reverse=False)
        for i in lowest:
            if i <= 0.15:
                counter += 1
        return float(counter / self.df_conf.size)

    ######################################## confidence && answers features ##########################################




# test function
def main():
    df = pd.read_csv("C:\\Users\\school & work\\PycharmProjects\\Final_Project_SISE_BGU\\ProcessedData\\RawData_Apple.csv")
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
    j = a.get_lowest()
    print(f'lowest: {j}')
    k = a.count_lowest_under_15()
    print(f'lowest under 15: {k}')


if __name__ == "__main__":
    main()