import numpy as np
import pandas as pd

def makeStatsarr(df):
    mu_arr = np.array(df.mean(axis=1)).reshape(-1, 1)
    std_arr = np.array(df.std(axis=1)).reshape(-1, 1)
    return np.hstack((mu_arr, std_arr))


def getPDFprior(statsArr):
    mu = np.mean(statsArr, axis=0)
    var = np.var(statsArr, axis=0)
    return np.hstack((mu, var))

def dictAddStats(d):
    # find the variance and the mean of each sample from every number
    # test data d[i][0] and train data d[i][2] in the dictionary crea
    for i in range(len(d)):
        train_stats = makeStatsarr(d[i][0])
        pdf_variables = getPDFprior(train_stats)
        d[i].append(pdf_variables)

        test_data_stats = makeStatsarr(d[i][2])
        d[i].append(test_data_stats)
    return d




