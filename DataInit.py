import numpy as np
import pandas as pd



def createNumDf(df, num):
    # from the dataframe containing all the digits 0-9 create a
    # dataframe containing pictures (1x784) of a single num
    #print (f'creating number {num} df')

    return df.loc[df['label'] == num]

def SeperateNum(df, num):
    # from the dataframe containing all the digits 0-9 create a
    # dataframe containing pictures (1x784) of a single num
    #print (f'creating number {num} df')

    return df.loc[df['label'] == num] , df.loc[df['label'] != num]




def splitData(df, splitPercent=0.75, seed=123):
    # randomly shuffle the dataframe before splitting based on the
    # % of data you want to have in your train set. defaults to 75% train 25% test
    rand = np.random
    rand.seed(123)

    [n, _] = df.shape
    #print(df.shape)

    n = int(n * splitPercent)
    train_data, train_lab, test_data, test_lab = df.iloc[:n, 1:], df.iloc[:n, :1], \
                                                 df.iloc[n:, 1:], df.iloc[n:, :1],
    lst=[train_data, train_lab, test_data, test_lab]
    return lst


def createDictNumSplit(df):
    # loop through all 9 digits and create a dictionary for each digit containing
    # d[digit#]=(train_data, train_lab, test_data, test_lab)

    dict_num_df = {}
    for i in range(10):
        dict_num_df[i] = createNumDf(df, i)
        dict_num_df[i] = splitData(dict_num_df[i], 0.75)

    return dict_num_df



