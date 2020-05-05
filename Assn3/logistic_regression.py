import cv2
import os
import glob
import pandas as pd
import numpy as np
from numpy.linalg import eig
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import sys

def extractDataTrain(trainFile):
    dt=pd.read_csv(trainFile, sep=' ', names=['file','label'])
    data = []
    for f in dt['file']:
        img = cv2.imread(f,0)
        img = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA) 
        data.append(img)
    data = np.array(data)
    labels = np.array(dt['label'].values)
    data = data.reshape(data.shape[0], data.shape[1]*data.shape[2]) 
    return data, labels

def extractDataTest(testFile):
    dt=pd.read_csv(testFile, sep=' ', names=['file'])
    data = []
    for f in dt['file']:
        img = cv2.imread(f,0)
        img = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA) 
        data.append(img)
    data = np.array(data)
    data = data.reshape(data.shape[0], data.shape[1]*data.shape[2]) 
    return data

def PCAManual(data):
    dataMean = np.mean(data.T, axis=1)
    dataCov = data - dataMean
    dataVar = np.cov(dataCov.T)
    values, vectors = eig(dataVar)

    indEig = values.argsort()[-500:][::-1]
    vectors = vectors[:,indEig]
    dataPCA = np.dot(dataCov, vectors)

    dataF = np.c_[ np.ones(len(dataPCA)), dataPCA ]
    return dataF

def predict(data , coef):
    coef = coef.reshape(len(coef),1)
    ypred = np.dot(data, coef)
    return 1.0 / (1.0 + np.exp(-1*ypred))

def coefficients_sgd(data , labels, alpha, itr):
    coef = np.zeros((len(data[0]),1))
    labels = labels.reshape(len(labels),1)
    for i in range(itr):
        ypred = predict(data, coef)
        error = labels - ypred
        p = alpha * np.dot(data.T, error)
        coef = coef + p
    return coef

def stat(testLabel, labelsPred):
    print ("\nAccuracy score for Logistic regression\n", accuracy_score(testLabel, labelsPred))
    print("\nConfusion matrix for Logistic regression\n",confusion_matrix(testLabel, labelsPred))
    print("\nClassification report for Logistic regression\n",classification_report(testLabel, labelsPred))

def LogisticRegression(trainFile, testFile):
    alpha = 5e-10
    itr = 10000

    trainData, trainLabel = extractDataTrain(trainFile)
    testData = extractDataTest(testFile)
    trainData = PCAManual(trainData)
    testData = PCAManual(testData)

    labelsUNQ = np.unique(trainLabel)
    shape = (trainLabel.size, labelsUNQ.size)
    labels1H = np.zeros(shape)
    for i in range(len(labelsUNQ)):
        temp = np.where(trainLabel == labelsUNQ[i])
        labels1H[temp[0],i] = 1
    predictions = np.zeros((len(testData),len(labelsUNQ)))
    coefficients = np.zeros((len(labels1H[0]), len(trainData[0])))

    for i in range(len(labels1H[0])):
        coefficients[i] = coefficients_sgd(trainData, labels1H[:,i] , alpha, itr).reshape(coefficients[i].shape)
    for i in range(len(labelsUNQ)):
        temp = predict(testData, coefficients[i])
        predictions[:,i] = temp.reshape(len(temp))
    labelsInd = np.argmax(predictions, axis=1)
    labelsPred = labelsUNQ[labelsInd]
    for x in labelsPred:
        print(x)
        print(" ")

if __name__ == "__main__": 
    LogisticRegression(str(sys.argv[1]), str(sys.argv[2]))