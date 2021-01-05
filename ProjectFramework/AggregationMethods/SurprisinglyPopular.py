import pandas as pd
from GetProcessedData import get_answer_names


def suprisingly_pop_answer(df):
    answer_names = get_answer_names(df)
    # Get prediction for each answer

    # Get the votes for each answer
    # Find the suprinsingly popular answer