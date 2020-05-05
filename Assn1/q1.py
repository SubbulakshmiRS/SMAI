# Example of making predictions
import csv
import numpy as np 
from collections import Counter

class KNNClassifier:
    DataFile = ""
    num_neighbors = 1
    def train(self, DataFile):
        self.DataFile = DataFile

    # calculate the Euclidean distance between two vectors
    def euclidean_distance(self, test_row, trainData):
        a = trainData - test_row
        b =  a**2
        distances = np.sum(b, axis = 1)
        return distances.reshape(-1,1)

    # calculate the Manhattan distance between two vectors
    def manhattan_distance(self, test_row, trainData):
        # print("train data length", len(trainData), len(trainData[0]))
        a = trainData - test_row
        b = np.absolute(a)
        distances = np.sum(b, axis = 1)
        return distances.reshape(-1,1)

    # Locate the most similar neighbors
    def get_neighbors(self, trainData, test_row, num_neighbors):
        distances = self.manhattan_distance(test_row, trainData[:,1:785])
        dist = np.append(trainData, distances, axis=1)
        dist = np.array(sorted(dist, key=lambda a_entry: a_entry[-1]))
        # print(distances[0,:])
        neighbors = dist[0:self.num_neighbors,0]
        # print("NRIGH", neighbors[0])
        return neighbors

    def predict(self, TestFile):
        num_neighbors = 1
        with open(self.DataFile,'r') as fD:
            readerD = csv.reader(fD, quoting=csv.QUOTE_NONNUMERIC)
            trainData = np.array(list(readerD))
            # print("Train Data length", len(trainData), len(trainData[0]))
            with open(TestFile,'r') as fT:
                readerT  = csv.reader(fT, quoting=csv.QUOTE_NONNUMERIC)
                TestData = np.array(list(readerT))
                # print("Testdata length", len(TestData), len(TestData[0]))
                predictions = np.zeros(len(TestData))
                for index in range(len(TestData)):
                    # print(index)
                    test_row = TestData[index]
                    # print("test_row ", test_row)
                    # print("len", len(test_row))
                    # exit(0)
                    output_values = self.get_neighbors(trainData, test_row, num_neighbors)
                    b = Counter(output_values)
                    prediction = b.most_common()[0][0]
                    predictions[index] = prediction
                return list(predictions)