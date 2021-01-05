import traceback
from GetProcessedData import get_answer_names


def surprisingly_pop_answer(df):
    try:
        # df = df_tuple[1]
        # file_name = df_tuple[0]
        answer_names = get_answer_names(df)
        max_difference = 0
        the_answer = "empty"
        # correct_answer = df[df['Class'] == 1]['Answer'].iloc[0]
        for answer in answer_names:
            mean = df[answer].mean()
            actual = len(df[df['Answer'] == answer])/len(df)
            if mean < actual:
                diff = actual - mean
                if diff > max_difference:
                    max_difference = diff
                    the_answer = answer
        return the_answer
        # Get prediction for each answer
        # answer_to_prediction_avg[answer] = df[answer].mean(), len(df[df['Answer'] == str(answer)])
        # Get the votes for each answer
        # answer_to_sum[answer] = len(df[df['Answer'] == answer])
    except Exception as e:
        print(f"Exception in {file_name}")
        print(e)
        traceback.print_tb(e.__traceback__)


