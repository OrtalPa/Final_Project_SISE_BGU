from GetProcessedData import get_answer_names


def na_answer(df, sp_answer, mr_answer, hac_answer):
    answers = get_answer_names(df)
    # verify input
    if len(answers) > 2:
        raise Exception(f'received question with more than 2 answers: {answers}')
    # if sp_answer not in answers or mr_answer not in answers or hac_answer not in answers:
    #     raise Exception(f'{sp_answer} or {mr_answer} or {hac_answer} is not in {answers}')
    # count for each answer how many methods selected it
    count_ans = 4  # more than the number of methods
    selected = 'NONE'
    for ans in answers:
        count = 0
        if sp_answer == ans:
            count += 1
        if mr_answer == ans:
            count += 1
        if hac_answer == ans:
            count += 1
        if count < count_ans:
            count_ans = count
            selected = ans
    # return the answer the least methods selected
    if selected == 'NONE':
        raise Exception('selected none')
    return selected

