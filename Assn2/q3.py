# Example of making predictions
import csv
import numpy as np 
import collections
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error


class Airfoil:
    itNum = 500000
    alpha = 0.00001   #learning rate
    np.random.seed(10)
    coeff = np.random.rand(6)

    def gradient_descent(self, trainData):
        x = trainData[:,0:6]
        y = trainData[:,6]
        m, n = trainData.shape
        cost_list = []   #to record all cost values to this list
        for i in range(0, self.itNum):
            prediction = np.dot(x, self.coeff)   #predicted y values theta_0*x0+theta_1*x1
            # print("prediction shape", prediction.shape)
            # print("coeff shape", self.coeff)
            # print("x shape", x.shape)
            # print("y shape", y.shape)
            error = prediction - y
            cost = 1/(2*m) * np.dot(error.T, error)
            # print("cost", cost)   #  (1/2m)*sum[(error)^2]
            cost_list.append(cost)
            self.coeff = self.coeff - (self.alpha * (1/m) * np.dot(x.T, error)) 
            # exit(0)  # self.alpha * (1/m) * sum[error*x]
            # if cost_list[itNum]-cost_list[itNum+1] < 1e-9:   #checking if the change in cost function is less than 10^(-9)
            #     break
         
        return cost_list

    def prepData(self, x):
        x_normed = (x - x.min(0)) / x.ptp(0)
        trainData = np.c_[ np.ones(len(x_normed)), x_normed ] 
        # trainData = np.concatenate((trainData, np.ones(r).T),axis=1)
        return trainData

    def validate(self, validateData):
        x = validateData[:,0:6]
        y = validateData[:,6]
        predictions = np.dot(x, self.coeff)
        self.stats(y, predictions)

    def stats(self, test_labels, predictions ):
        print ("Mean squared error", mean_squared_error(test_labels, predictions))
        # print ("Accuracy score", accuracy_score(test_labels, predictions))
        print ("Mean absolute error", mean_absolute_error(test_labels, predictions))

    def predict(self, TestFile):
        # print("predict")
        with open(TestFile,'r') as fT:
            readerT  = csv.reader(fT, quoting=csv.QUOTE_NONNUMERIC)
            TestData = np.array(list(readerT))
            x = self.prepData(TestData)
            predictions = np.dot(x, self.coeff)
            return predictions

    def train(self, DataFile):
        with open(DataFile,'r') as fD:
            readerD = csv.reader(fD, quoting=csv.QUOTE_NONNUMERIC)
            trainData = np.array(list(readerD))
            np.random.shuffle(trainData)
            x = trainData[:,0:5]
            y = trainData[:,5]
            # print(trainData[0:1])
            print("trainData shape", trainData.shape)
            trainData = self.prepData(x)
            trainData = np.c_[ trainData,y ] 
            # print(trainData[0:1])
            # print("coeff init", self.coeff)
            # return 
            # # print("Train length", len(trainData), len(trainData[0]))
            validateData = trainData[0:200]
            trainData = trainData[200:]
            cost_list = self.gradient_descent(trainData)
            # print("itnum ", self.itNum)
            self.validate(validateData)
       
if __name__ == "__main__": 
    model3 = Airfoil()
    model3.train('./airfoil_test.csv')