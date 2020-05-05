import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error


def extract_Data(datafile):
    dt=pd.read_csv(datafile, sep=';', header =[0])
    dt['D'] = dt['Date'] + " " + dt['Time']
    dt['D'] = pd.to_datetime(dt['D'], infer_datetime_format=True)
    dt = dt[dt['Global_active_power'] != '?']
    dt['Global_active_power'] = pd.to_numeric(dt['Global_active_power'], downcast="float")
    return dt

def prepData(dt):
    dt = dt[:10000] 
    data = np.zeros((dt.shape[0]-60,60))
    i = 0
    for index, row in dt[::-1].iterrows():
        if(index - 61 >= 0):
            b = dt[index-61:index]['Global_active_power']
            data[i,:] = np.array(b)[:60].reshape(60)
            i = i+1
        else :
            break
        
    data = np.array(data)
    data =data.reshape(data.shape[0],60)
    l = int(data.shape[0]*0.8)
    trainData = data[:l,:-1].astype(np.float32)
    trainLabel = data[:l,-1].astype(np.float32)
    testData = data[l:,:-1].astype(np.float32)
    testLabel = data[l:,-1].astype(np.float32)
    return trainData, trainLabel, testData, testLabel


def regression(datafile):
    dt = extract_Data(datafile)
    trainData, trainLabel, testData, testLabel = prepData(dt)
    reg = LinearRegression().fit(trainData, trainLabel)
    predictions = reg.predict(testData)
    # stat(testLabel, predictions)
    for x in predictions:
        print(x)
        print(" ")


def stat(testLabel, predictions):
    print ("Mean squared error", mean_squared_error(testLabel, predictions))
    print ("Mean absolute error", mean_absolute_error(testLabel, predictions))

if __name__ == "__main__": 
    regression(str(sys.argv[1]))