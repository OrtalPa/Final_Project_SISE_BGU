import pandas as pd

# --------------- Data from article ----------------- #

PATH = r'C:\Users\ortal\Documents\FinalProject\journal.pone.0232058.s003\AnalysisCode\Raws'
PATH_TO_WRITE = r'C:\Users\ortal\Documents\FinalProject\data\UpdatedData'
list_of_df = [pd.read_csv(f"{PATH}\\r2a_rawdata.csv"),
              pd.read_csv(f"{PATH}\\r2b_rawdata.csv"),
              pd.read_csv(f"{PATH}\\r2c_rawdata.csv"),
              pd.read_csv(f"{PATH}\\r2d_rawdata.csv"),
              pd.read_csv(f"{PATH}\\r2e_rawdata.csv")]
question_list_df = pd.read_csv(f'{PATH}\\QuestionsList.csv')

# Column names:
# i = whether the statement presented was more likely to be true or false
# i+1 = what percentage of other forecasters would predict the statement to be true
# i+2 = the probability that the statement was true
# i+3 = what the average probability estimated by other forecasters would be

question_index = 1
for q_df in list_of_df:
    i = 0
    # Each loop is for one question -> question index increases by 1, and i by 4 for 4 columns per question
    while i < 400:
        # Get 4 columns from the df
        df = q_df[[f'{i}', f'{i+1}', f'{i+2}', f'{i+3}']]
        # Create new dataframe from the selected columns and rename the column titles -> one df for a question
        df = df.rename(columns={f'{i}': 'Answer', f'{i+1}': 'True', f'{i+2}': 'Confidence', f'{i+3}': 'to_drop'})
        df = df.drop('to_drop', axis=1)
        # Rename the answers to True and False
        df[['Answer']] = df[['Answer']].replace({1: 'False', 2: 'True'})
        # Make confidence express the actual confidence and do it in range 0 to 1
        df['Confidence'] = df['Confidence'].apply(lambda x: float(float(x)/100))
        copy_df = df.copy()
        df.loc[df['Answer'] == 'False', 'Confidence'] = copy_df.loc[copy_df['Answer'] == 'False', 'Confidence'].apply(lambda x: 1-x)
        # Get prediction for false and make all prediction in range 0 to 1
        df['True'] = df['True'].apply(lambda x: float(x/100))
        df['False'] = df['True'].apply(lambda x: float(1-x))
        # Get the class for each question: 1 if Answer is correct 0 if not
        correct_answer = question_list_df[question_list_df['Order'] == question_index]['Outcome'].iloc[0]
        df['Class'] = df['Answer'].apply(lambda x: 1 if str(x).lower() == str(correct_answer).lower() else 0)
        # Write to csv
        file_name = f'Article_{question_index}'
        question_index = question_index + 1
        df.to_csv(f'{PATH_TO_WRITE}\\ProcessedData\\Article\\{file_name}.csv')
        i = i + 4

