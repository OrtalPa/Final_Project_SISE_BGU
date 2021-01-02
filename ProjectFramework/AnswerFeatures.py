import pandas as pd


class AnswerF:
    def __init__(self,df):
        # todo self init
        self.df = df
        self.unique_answers = df['answer'].unique()
        self.num_of_ans = df['answer'].nunique()
        self.avg_for_answer = df['answer'].size / self.num_of_ans

    # get number of solvers how chose answer "ans_name"
    def get_answers_number(self,ans_name):
        return (self.df['answer'] == ans_name).sum()

    # get list - for every answer, it calculates the
    def get_answers_distance_from_avg(self):
        dfa_list = []
        for i in self.unique_answers:
            distance_from_avg = pow(self.get_answers_number(i) - self.avg_for_answer,2)
            dfa_list.append(distance_from_avg)
        return dfa_list

    def get_total_var(self):
        var = 0
        for i in self.unique_answers:
            var = var + ( pow(self.get_answers_number(i) - self.avg_for_answer,  self.num_of_ans2) / self.num_of_ans )
        return var

    def get_total_std(self, var_list):
        std_list = []
        for i in self.unique_answers:
            var = pow(self.get_answers_number(i) - self.avg_for_answer,2) / self.num_of_ans
            list.append(var)
        return std_list


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