import pandas as pd
from scipy.stats import entropy
import numpy as np
from sklearn.utils import shuffle
from GetProcessedData import get_answer_names

NUM_OF_GROUPS = 3


class ConfidenceSubF:

    def __init__(self, df):
        self.unique_answers = get_answer_names(df) # pd.unique(df['Answer'])
        self.num_of_ans = self.unique_answers.size

    # get the mean value of all answers confidence
    def feature_get_total_mean(self,df_conf):
        var = float(np.mean(df_conf))
        return var

    # get the variance value of all answers confidence
    def feature_get_total_var(self,df_conf):
        var = float(np.var(df_conf))
        return var

    # get the standard deviation value of all answers confidence
    def feature_get_total_std(self,df_conf):
        std = float(np.std(df_conf))
        return std

    # get the highest value of confidence
    def feature_get_highest(self,df_conf):
        highest = float(sorted(df_conf, reverse=True)[0])
        return highest

    # get the count of confidence with value over 0.90
    def feature_count_highest_above_90(self,df_conf):
        counter = 0
        highest = sorted(df_conf, reverse=True)
        for i in highest:
            if i >= 0.90:
                counter += 1
        return float(counter/df_conf.size)

    # get the count of confidence with value over 0.95
    def feature_count_highest_above_95(self,df_conf):
        counter = 0
        highest = sorted(df_conf, reverse=True)
        for i in highest:
            if i >= 0.95:
                counter += 1
        return float(counter/df_conf.size)

    # get the count of confidence with value over 0.98
    def feature_count_highest_above_98(self,df_conf):
        counter = 0
        highest = sorted(df_conf, reverse=True)
        for i in highest:
            if i >= 0.98:
                counter += 1
        return float(counter/df_conf.size)

    # get the distance from highest confidence to mean
    def feature_get_distance_highest_from_mean(self,df_conf,total_mean):
        highest = float(sorted(df_conf, reverse=True)[0])
        return float(highest-total_mean)

    # get the lowest value of confidence
    def feature_get_lowest(self,df_conf):
        lowest = float(sorted(df_conf, reverse=False)[0])
        return lowest

    # get the count of confidence with value under 0.15
    def feature_count_lowest_under_15(self,df_conf):
        counter = 0
        lowest = sorted(df_conf, reverse=False)
        for i in lowest:
            if i <= 0.15:
                counter += 1
        return float(counter / df_conf.size)

    ######################################## confidence && answers features ##########################################


