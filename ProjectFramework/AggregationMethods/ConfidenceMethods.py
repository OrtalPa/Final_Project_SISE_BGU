import traceback
from GetProcessedData import get_answer_names


def highest_average_confidence(df):
    try:
        answers = get_answer_names(df)
        answer_to_conf = {}
        for answer in answers:
            answer_to_conf[answer] = df.loc[df['Answer'] == answer, 'Confidence'].mean()
        highest = max(answer_to_conf.values())
        keys = [k for k, v in answer_to_conf.items() if v == highest]
        return None if len(keys) != 1 else keys[0]
    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)


# from GetProcessedData import get_question_dfs
#
# dfs = get_question_dfs()
# print(highest_average_confidence(dfs[11]))


def weighted_confidence(df):
    try:
        answers = get_answer_names(df)
        answer_to_conf = dict()
        for answer in answers:
            answer_to_conf[answer] = df[df.Answer == answer]['Confidence'].sum()
        return max(answer_to_conf, key=answer_to_conf.get)
    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)
