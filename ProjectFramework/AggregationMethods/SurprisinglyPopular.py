import traceback
from GetProcessedData import get_answer_names


def surprisingly_pop_answer(df):
    try:
        # df = df_tuple[1]
        # file_name = df_tuple[0]
        answers = get_answer_names(df)
        answer_to_pred_diff = {}
        for answer in answers:
            mean_prediction = df[answer].mean()
            vote_percentage = len(df[df['Answer'] == answer])/len(df)
            if mean_prediction < vote_percentage:
                answer_to_pred_diff[answer] = vote_percentage - mean_prediction
        highest = max(answer_to_pred_diff.values())
        keys = [k for k, v in answer_to_pred_diff.items() if v == highest]
        return None if len(keys) != 1 else keys[0]
        # correct_answer = df[df['Class'] == 1]['Answer'].iloc[0]
        # Get prediction for each answer
        # answer_to_prediction_avg[answer] = df[answer].mean(), len(df[df['Answer'] == str(answer)])
        # Get the votes for each answer
        # answer_to_sum[answer] = len(df[df['Answer'] == answer])
    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)


from GetProcessedData import get_question_dfs

dfs = get_question_dfs()
print(surprisingly_pop_answer(dfs[0]))