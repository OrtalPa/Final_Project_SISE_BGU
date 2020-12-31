import pandas as pd

# --------------- Data from article ----------------- #

PATH = r'C:\Users\ortal\Documents\FinalProject\journal.pone.0232058.s003\AnalysisCode\Raws'
list_of_df = [pd.read_csv(f"{PATH}\\r2a_rawdata.csv"),
              pd.read_csv(f"{PATH}\\r2b_rawdata.csv"),
              pd.read_csv(f"{PATH}\\r2c_rawdata.csv"),
              pd.read_csv(f"{PATH}\\r2d_rawdata.csv"),
              pd.read_csv(f"{PATH}\\r2e_rawdata.csv")]

# Column names:
# answer = whether the statement presented was more likely to be true or false
# prediction_true = what percentage of other forecasters would predict the statement to be true
# prob_true = the probability that the statement was true
# prediction_prob_true = what the average probability estimated by other forecasters would be

# Create a list - question_tables: in each index in te list, there is a table representing the question answers
question_tables = []
for df in list_of_df:
    i = 0
    while i < 400:
        # Get 4 columns from the df
        q_df = df[[f'{i}', f'{i+1}', f'{i+2}', f'{i+3}']]
        # Create new dataframe from the selected columns and rename the column titles
        q_df = q_df.rename(columns={f'{i}': 'answer', f'{i+1}': 'prediction_true', f'{i+2}': 'prob_true', f'{i+3}': 'prediction_prob_true'})
        question_tables.append(q_df)
        i = i + 4
    print(question_tables[-1])  # printing just for sanity check

