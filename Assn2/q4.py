# Example of making predictions
import csv
import numpy as np 
import pandas as pd
import collections
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class Weather:
    itNum = 500
    alpha = 0.00001   #learning rate
    np.random.seed(10)
    theta = np.random.rand(10)

    def meanSquareError(self, x, y, m):
        prediction = np.dot(x, self.theta) 
        error = prediction - y
        cost = 1/(2*m) * np.dot(error.T, error)
        derivative = np.dot(x.T, error)
        return cost, derivative, "Mean square error"

    def meanAbsError(self, x, y):
        prediction = np.dot(x, self.theta) 
        error = prediction - y
        cost = np.sum(np.abs(error))
        derivative = np.dot(x.T, np.sign(error))
        return cost, derivative, "Mean absolute error"

    def gradient_descent(self, trainData):
        n1 = trainData.shape[1]-1
        x = trainData[:,0:n1]
        y = trainData[:,n1]
        m, n = trainData.shape
        cost_list = []
        for i in range(0, self.itNum):
            # Mean square error
            cost, derivative, errType = self.meanSquareError(x, y, m)
            # Mean absolute error
            # cost, derivative, errType = self.meanAbsError(x, y)
            # Mean absolute percentage error not tried due to zero division
            cost_list.append(cost)
            self.theta = self.theta - (self.alpha * (1/m) * derivative) 

        return cost_list, errType

    def prepData(self, trainData):
        trainData = trainData.drop(['Formatted Date', 'Summary', 'Daily Summary'], axis=1)
        trainData['Precip Type'] = trainData['Precip Type'].fillna('null')
        dummy = pd.get_dummies(trainData['Precip Type'])
        df = dummy['snow']
        n1 =(df-df.mean())/df.std()
        df = dummy['rain']
        n2 =(df-df.mean())/df.std()
        df = dummy['null']
        n3 =(df-df.mean())/df.std()

        rtrainData = trainData.drop(['Precip Type'], axis = 1) 
        trnData = pd.concat([rtrainData,n1,n2,n3], axis=1)
        return trnData

    def validate(self, validateData):
        x = validateData[:,0:10]
        y = validateData[:,10]
        predictions = np.dot(x, self.theta)
        self.stats(y, predictions)

    def stats(self, test_labels, predictions ):
        print ("Mean squared error", mean_squared_error(test_labels, predictions))
        # print ("Accuracy score", accuracy_score(test_labels, predictions))
        print ("Mean absolute error", mean_absolute_error(test_labels, predictions))

    def train(self, DataFile):
        trainData = pd.read_csv(DataFile) 
        y = trainData['Apparent Temperature (C)']
        trainData = trainData.drop(['Apparent Temperature (C)'], axis=1)
        trainData = self.prepData(trainData)
        Data = pd.concat([trainData,y], axis=1) 
        data = np.array(Data.values)
        data = np.delete(data, 0, axis=0)
        data = np.c_[ np.ones(len(data)),data]
        print("trainData shape", trainData.shape)
        validateData = trainData[0:2000]
        trainData = trainData[2000:]
        cost_list, errType = self.gradient_descent(data)

        # print("No of iterations ", self.itNum)
        # fig,ax =  plt.subplots()
        # ax.plot(np.arange(1,self.itNum+1), cost_list)
        # ax.set_title(errType)
        # plt.show()

        self.validate(validateData)

    def predict(self, TestFile):
        with open(TestFile,'r') as fT:
            readerT  = csv.reader(fT, quoting=csv.QUOTE_NONNUMERIC)
            TestData = np.array(list(readerT))
            TestData = self.prepData(TestData)
            data = np.array(TestData.values)
            data = np.delete(data, 0, axis=0)
            x = np.c_[ np.ones(len(data)),data]
            predictions = np.dot(x, self.theta)
            return predictions

if __name__ == "__main__": 
    model3 = Weather()
    model3.train('./weather_test.csv')