import math
import numpy as np


def createPDF(x: float, mean: float, var=float) -> float:
    eps = 1e-4  # Added in denominator to prevent division by zero
    coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
    return coeff * exponent


def mean_std_predict(d):
    # use the priori knowledge of the mean, std, var(mean),var(std) of the distribtuiton of all training images for a particular num
    # to predict the probability that an image belongs to a certain number given the 28x28 pixel image mean and std

    # initialize necessary variables for prediction of image num based on the pixels mean and std
    meanPredArr, stdPredArr = {}, {}
    pdf_priori_mean, pdf_priori_meanvar, pdf_priori_std, pdf_priori_stdvar = [], [], [], []

    # setup the appropriate prior probabilities for a given num in their appropriate list [0-9] so that we can compute the pdf
    for i in range(len(d)):
        pdf_priori_mean.append(d[i][4][0])
        pdf_priori_meanvar.append(d[i][4][2])
        pdf_priori_std.append(d[i][4][1])
        pdf_priori_stdvar.append(d[i][4][3])

    for i in range(len(d)):
        testArr = d[i][5]
        for j in range(len(pdf_priori_mean)):
            meanPredict = [createPDF(testArr[z, 0], pdf_priori_mean[j], pdf_priori_meanvar[j]) for z in
                           range(testArr.shape[0])]

            if i not in meanPredArr.keys():
                meanPredArr[i] = []
                meanPredArr[i].append(meanPredict)

            else:

                meanPredArr[i].append(meanPredict)

            stdPredict = [createPDF(testArr[z, 1], pdf_priori_std[j], pdf_priori_stdvar[j]) for z in
                          range(testArr.shape[0])]

            if i not in stdPredArr.keys():
                stdPredArr[i] = []
                stdPredArr[i].append(stdPredict)

            else:
                stdPredArr[i].append(stdPredict)

        meanPredArr[i] = np.array(meanPredArr[i])
        stdPredArr[i] = np.array(stdPredArr[i])

    return meanPredArr, stdPredArr


def predictNum(meanPredArr, stdPredArr):
    meanPredictions = []
    stdPredictions = []
    for i in range(len(meanPredArr)):
        meanPredictions.append(np.argmax(meanPredArr[i], axis=0))
        unique, counts = np.unique(meanPredictions[i], return_counts=True)
        num_samp = int(np.sum(counts))

        # predicting the percent of classifications for the different num classes
        meanPredictions[i] = np.array((unique, counts, counts / num_samp * 100)).astype(int).T

        stdPredictions.append(np.argmax(stdPredArr[i], axis=0))
        unique, counts = np.unique(stdPredictions[i], return_counts=True)
        stdPredictions[i] = np.array((unique, counts, counts / num_samp * 100)).astype(int).T

    return meanPredictions, stdPredictions
