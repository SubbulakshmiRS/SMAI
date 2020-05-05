# Example of making predictions
import csv
import numpy as np 
import collections
from sklearn.metrics import accuracy_score

class KNNClassifier:
    DataFile = ""
    num_neighbors = 1
    def train(self, DataFile):
        # print("train")
        self.DataFile = DataFile

    def mode(self, num_list):
        # print("len of num_list", len(num_list))
        # print("num_list1", num_list)
        num_list = np.array(list(filter(lambda a: a != '?', num_list)))
        # print("num_list2", num_list)
        # exit(0)
        modev = collections.Counter(num_list).most_common(1)
        mode_val = modev[0][0]
        return mode_val

    # calculate the distance between 2 vectors 
    def distanceChar(self, test_row, trainData):
        # a = trainData - test_row
        # print("train data", len(trainData), len(trainData[0]))
        # print("test_row", len(test_row))
        b = trainData!=test_row
        c = b.astype(np.int)
        distances = np.sum(c, axis = 1)
        return distances.reshape(-1,1)

    def prepData(self, trainData):
        for i in range(len(trainData[0])):
            mode_val = self.mode(trainData[:,i])
            for j in range(len(trainData[:,i])):
                if trainData[j][i] == '?':
                    trainData[j][i] = mode_val
        return trainData

    # Locate the most similar neighbors
    def get_neighbors(self, trainData, test_row):
        distances = self.distanceChar(test_row, trainData[:,1:23])
        dist = np.append(trainData, distances, axis=1)
        dist = np.array(sorted(dist, key=lambda a_entry: a_entry[-1]))
        # print(distances[0,:])
        neighbors = dist[0:self.num_neighbors,0]
        # print("NRIGH", neighbors[0])
        return neighbors

    def predict(self, TestFile):
        # print("predict")
        with open(self.DataFile,'r') as fD:
            readerD = csv.reader(fD)
            trainData = np.array(list(readerD))
            np.random.shuffle(trainData)
            trainData = self.prepData(trainData)
            # print("Train length", len(trainData), len(trainData[0]))
            with open(TestFile,'r') as fT:
                readerT  = csv.reader(fT)
                TestData = np.array(list(readerT))
                TestData = self.prepData(TestData)
                # print("Testdata length", len(TestData), len(TestData[0]))
                # tags = np.empty(len(TestData), dtype = "<U10") 
                predictions = np.empty(len(TestData), dtype = "<U10")
                for index in range(len(TestData)):
                    # print(index)
                    test_row = TestData[index]
                    # print("test_row ", test_row)
                    # print("len", len(test_row))
                    # exit(0)
                    output_values = self.get_neighbors(trainData, test_row)
                    b = collections.Counter(output_values)
                    prediction = b.most_common()[0][0]
                    predictions[index] = prediction
                #     tags[index] = test_row[0]
                # print("Accuracy",accuracy_score(tags, predictions))
                return list(predictions)