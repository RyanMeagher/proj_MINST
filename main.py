import math

import DataInit
import CreateStats
import numpy as np
import pandas as pd
import CreatePDFPredict


df = pd.read_csv('train.csv')

# create a dict and then add stats so that
# d[digit#]=(train_data, train_lab, test_data, test_lab, pdf_variables, test_data_stats)
d = DataInit.createDictNumSplit(df)
CreateStats.dictAddStats(d)

meanPredArr, stdPredArr = CreatePDFPredict.mean_std_predict(d)

# Predict what the number is Based on the probability scores given by based on the probability distribution of the
# different numbers in the training dataset. what is returned is the # of predictions for each individual prediction
# creates a list of lists where the position in the list corresponds to the num, and the num elements
# [predict test data 0 ->[class, number of predictions, classification_rate]  ... for every individual numbers
# test dataset
meanPredictions, stdPredictions= CreatePDFPredict.predictNum(meanPredArr, stdPredArr)

print(meanPredictions[4])
print(stdPredictions[4])



