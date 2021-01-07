import pandas as pd
from scipy.stats import entropy
import numpy as np
from sklearn.utils import shuffle
from scipy.stats import wasserstein_distance
from GetProcessedData import get_answer_names


class PredictionsF:

    def __init__(self, df):
        self.df = df
        self.unique_answers = get_answer_names(df)
        self.df_pred = df[[self.unique_answers]]
        self.num_of_ans = self.unique_answers.size

    # get the mean value of each answers predictions
    def get_mean_list(self):
        mean_list = []
        for value in self.unique_answers:
            mean_list.append(float(np.mean(self.df_pred[value])))
        return mean_list

    # get the variance value of each answers predictions
    def get_var_list(self):
        var_list = []
        for value in self.unique_answers:
            var_list.append(float(np.var(self.df_pred[value])))
        return var_list

    # get the standard deviation value of each answers predictions
    def get_std_list(self):
        std_list = []
        for value in self.unique_answers:
            std_list.append(float(np.std(self.df_pred[value])))
        return std_list

    # get the highest value of Prediction mean
    def feature_get_highest_mean_prediction_for_answer(self):
        highest = float(sorted(self.get_mean_list(), reverse=True)[0])
        return highest

    # get the highest value of Prediction var
    def feature_get_highest_var_prediction_for_answer(self):
        highest = float(sorted(self.get_var_list(), reverse=True)[0])
        return highest

    # get the highest value of Prediction std
    def feature_get_highest_std_prediction_for_answer(self):
        highest = float(sorted(self.get_std_list(), reverse=True)[0])
        return highest

    # get the lowest value of Prediction mean
    def feature_get_lowest_mean_prediction_for_answer(self):
        lowest = float(sorted(self.get_mean_list(), reverse=False)[0])
        return lowest

    # get the lowest value of Prediction var
    def feature_get_lowest_var_prediction_for_answer(self):
        lowest = float(sorted(self.get_var_list(), reverse=False)[0])
        return lowest

    # get the lowest value of Prediction std
    def feature_get_lowest_std_prediction_for_answer(self):
        lowest = float(sorted(self.get_std_list(), reverse=False)[0])
        return lowest

    # get the count of prediction with value over 0.80
    def feature_count_highest_above_80(self):
        counter = 0
        for value in self.unique_answers:
            highest = sorted(self.df_pred[value], reverse=True)
            for i in highest:
                if i >= 0.80:
                    counter += 1
        return float(counter/len(self.df.index))

    ######################################## prediction && confidence features ##########################################

    # get the count of low prediction with value under 0.30 and confidence over 0.90
    def feature_count_high_confidence_low_prediction(self):
        counter = 0
        for answer in self.unique_answers:
            df_for_ans = self.df.loc[self.df['Answer'] == answer, 'Confidence', answer]
            ans_list = df_for_ans.apply(lambda row: 1 if row['Confidence'] > 0.8 and row[answer] < 0.2 else 0)
            counter = sum(ans_list)
        return counter


# test function
def main():
    cereal_df = pd.read_csv("C:\\Users\\school & work\\PycharmProjects\\Final_Project_SISE_BGU\\test.csv")
    a = PredictionsF(cereal_df)
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


if __name__ == "__main__":
    main()