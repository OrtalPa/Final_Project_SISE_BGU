import traceback


def majority_answer(df):
    try:
        mode = df['Answer'].mode()
        return None if len(mode) != 1 else mode.values[0]
    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)


from GetProcessedData import get_question_dfs

dfs = get_question_dfs()
print(majority_answer(dfs[10]))