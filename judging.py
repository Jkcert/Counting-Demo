# Judging a test wave is a qualified action or not
def ave_error_judging1d(sample, template):
    error = 0
    n = len(sample)
    # if n < 5:
    #     return 1
    for item in sample:
        basic = template(item[0])
        error += abs(item[1] - basic)
    error = error / n
    return error
