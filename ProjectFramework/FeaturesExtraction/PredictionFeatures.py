import pandas as pd
from scipy.stats import entropy
import numpy as np
from sklearn.utils import shuffle
from scipy.stats import wasserstein_distance
from GetProcessedData import get_answer_names


class PredictionsF:

    def __init__(self, df):
        self.df = df
        self.df['Answer'] = self.df['Answer'].astype(str)
        self.unique_answers = get_answer_names(df)
        self.df_pred = df[self.unique_answers]
        self.num_of_ans = self.unique_answers.size

    # get number of solvers how chose answer "ans_name"
    def get_answers_number(self, ans_name):
        return (self.df['Answer'] == ans_name).sum()

    # This function builds the array of how many people chose each answer
    def build_answers_distribution_array(self):
        dictionary = {}
        for answer_name in self.unique_answers:
            answer_number = self.get_answers_number(answer_name) / self.num_of_ans
            dictionary[answer_name] = answer_number
        return dictionary

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
        return float(counter / len(self.df.index))

    ######################################## prediction && confidence features ##########################################

    # get the count of low prediction and high confidence
    def feature_count_high_confidence_low_prediction(self):
        counter = 0
        for answer in self.unique_answers:
            test_df = self.df[self.df['Answer'] == answer]
            for index,row in test_df.iterrows():
                if row['Confidence'] > 0.8 and row[answer] < 0.2:
                    counter += 1
        return counter

    # get the count of high prediction and low confidence
    def feature_count_low_confidence_high_prediction(self):
        counter = 0
        for answer in self.unique_answers:
            test_df = self.df[self.df['Answer'] == answer]
            for index, row in test_df.iterrows():
                if row['Confidence'] < 0.5 and row[answer] > 0.5:
                    counter += 1
        return counter

    # get the count of high prediction and low votes
    def feature_count_prediction_higher_then_votes(self):
        votes = self.build_answers_distribution_array()
        counter = 0
        for answer in self.unique_answers:
            test_df = self.df[self.df['Answer'] == answer]
            for index, row in test_df.iterrows():
                if row[answer] > votes[answer]:
                    counter += 1
        return counter


# test function
def main():
    cereal_df = pd.read_csv("C:\\Users\\school & work\\PycharmProjects\\Final_Project_SISE_BGU\\test.csv", index_col=0)
    a = PredictionsF(cereal_df)
    b = a.feature_get_highest_mean_prediction_for_answer()
    print(f'high mean: {b}')
    c = a.feature_get_highest_var_prediction_for_answer()
    print(f'high var: {c}')
    d = a.feature_get_highest_std_prediction_for_answer()
    print(f'high std: {d}')
    e = a.feature_get_lowest_mean_prediction_for_answer()
    print(f'low mean: {e}')
    f = a.feature_get_lowest_var_prediction_for_answer()
    print(f'low var: {f}')
    g = a.feature_get_lowest_std_prediction_for_answer()
    print(f'low std: {g}')
    h = a.feature_count_highest_above_80()
    print(f'above_80: {h}')
    i = a.feature_count_high_confidence_low_prediction()
    print(f'this is what i need : {i}')
    j = a.feature_count_low_confidence_high_prediction()
    print(f'this is what i need : {j}')
    k = a.feature_count_prediction_higher_then_votes()
    print(f'this is what i need : {k}')


if __name__ == "__main__":
    main()
