import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import metrics

class AuthorClassifier:
    subclf = svm.SVC(kernel='linear', C=0.001)
    def stat(self, labelst, labelsp):
        print("--------------------------------------------------")
        print("STATISTICS")
        print("Accuracy:",metrics.accuracy_score(labelst, labelsp))
        print("Confusion matrix:\n",metrics.confusion_matrix(labelst, labelsp))

    def prep(self, txt):
        # print("train Data shape", trainData.shape)
        # print("text ", txt[0])
        vectorizer  = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word', stop_words = 'english')
        txt_fitted = vectorizer.fit(txt)
        x = txt_fitted.transform(txt)
        # x = x.todense()
        #x = np.array(x)
        #x = x.astype(np.float)
        # print("2 ", x.shape)
        # print("3 ", y.shape )

        svd = TruncatedSVD(n_components=100)
        xSVD = svd.fit_transform(x)
        # print("SVD reduced shape ",xSVD.shape)
        return xSVD

    def train(self, DataFile):
        with open(DataFile,'r') as fD:
            readerD = csv.reader(fD)
            trnData = np.array(list(readerD))
            np.random.shuffle(trnData)
            # Remove the header row and id column
            trnData = np.delete(trnData, 0, axis=0) 
            trnData = np.delete(trnData, 0, axis=1) 
            txt = trnData[:,0]
            # print("text shape", txt.shape)
            txt = txt.reshape(txt.shape[0])
            y = trnData[:,1]
            y = y.reshape(len(y), 1)
            x= self.prep(txt)
            # print("y shape ", y.shape)
            # xTest = x[0:100]
            # xTrain = x[100:]
            # yTest = y[0:100]
            # yTrain = y[100:]
            self.subclf.fit(x,y)
            # yPredict = subclf.predict(xTest)
            # print("predicted length:",len(yPredict))

            # self.stat(yTest, yPredict)

    def predict(self, TestFile):
        with open(TestFile,'r') as fT:
            readerT = csv.reader(fT)
            testData = np.array(list(readerT))
            np.random.shuffle(testData)
            testData = np.delete(testData, 0, axis=0) 
            testData = np.delete(testData, 0, axis=1) 
            xTest = self.prep(testData)
            yPredict = self.subclf.predict(xTest)
            return yPredict

if __name__ == "__main__": 
    auth_classifier = AuthorClassifier()
    auth_classifier.train('./Question-5/Train(1).csv')