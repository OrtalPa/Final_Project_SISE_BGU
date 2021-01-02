import pandas as pd
import math

class AnswerF:
    def __init__(self,df):
        # todo self init
        self.df = df
        self.total_var = self.get_total_var()
        self.total_std = self.get_total_std()
        self.unique_answers = df['answer'].unique()
        self.num_of_ans = df['answer'].nunique()
        self.avg_for_answer = df['answer'].size / self.num_of_ans
        self.answers_distribution = self.build_answers_distribution_array(self)

    # get number of solvers how chose answer "ans_name"
    def get_answers_number(self,ans_name):
        return (self.df['answer'] == ans_name).sum()

    # This function builds the array of how many people chose each answer
    def build_answers_distribution_array(self):
        dictionary = {}
        for answer_name in self.unique_answers:
            answer_number = self.get_answers_number(self, answer_name)/self.num_of_ans
            dictionary[answer_name] = answer_number
        return dictionary

    # get list - for every answer, it calculates the distance between uniform distribution
    # and the amount of solvers how chose this answer
    def get_answers_distance_from_uniform(self):
        dfu_dict = {}
        for answer_name in self.unique_answers:
            distance_from_avg = pow(self.answers_distribution(answer_name) - self.avg_for_answer,2)
            dfu_dict[answer_name] = distance_from_avg
        return dfu_dict

    # get the variance value of all answers distribution
    def get_total_var(self):
        var = 0
        for answer_name in self.unique_answers:
            var = var + (pow(self.answers_distribution(answer_name) - self.avg_for_answer, 2) / self.num_of_ans)
        return var

    # get the standard deviation value of all answers distribution
    def get_total_std(self, var_list):
        std = math.sqrt(self.total_var)
        return std



    """
    המרחק של כל בנאדם מהממוצע שקרה בפועל,
    var, std, mean
    הפרש מהתפלגות אחידה
    מרחקים והפרשים בין המקום הראשון למקום השני.
     האם יש תשובה שקיבלה יותר מ50 אחוז
    האם ההפרש ממקום ראשון לשני הוא גדול יותר מהסטיית תקן
    התפלגות בפעול - כמה ענו כל תשובה
    אנטרופיה
    המרחק בין מקום ראשון למקום אחרון(מקום אחרון שמישהו בחר בו)
    
    תתי קבוצות:
    מרחקים בין התפלגויות בכל תת קבוצה
    השונות של אחוז הפפולריות של התשובה הפפולרית ביותר
    האם השתנה התשובה במקום הראשון בין תתי קבוצות
    
    """