import csv
import collections
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np 

class DecisionTree:
    root = None
    numCols = list()

    def mode(self, num_list):
        # print("len of num_list", len(num_list))
        # print("num_list1", num_list)
        num_list = np.array(list(filter(lambda a: a != 'NA', num_list)))
        # print("num_list2", num_list)
        modev = collections.Counter(num_list).most_common(1)
        mode_val = modev[0][0]
        return mode_val

    def prepData(self, trainData):
        # trainData.sort(key=lambda x:x[-1])
        self.numCols = list()
        for i in range(len(trainData[0])):
            for j in range(len(trainData)):
                if trainData[j][i] != 'NA' and str.isdigit(trainData[j][i]) :
                    self.numCols.append(i)
                    break
        # print("NUMCOLS")
        # print(self.numCols)
        # print("train data ", len(trainData), len(trainData[0]))
        for i in self.numCols:
            mode_val = self.mode(trainData[:,i])
            # print("mode",mode_val)
            for j in range(len(trainData)):
                if trainData[j][i] == 'NA':
                    trainData[j][i] = mode_val
            #Data normalization, specific for numerical data
            if i!= 79:
                a = trainData[:,i].astype(np.float)
                a = a - np.mean(a)
                a = a / np.std(a)
                trainData[:,i] = a.astype(np.str)
        return trainData

    # Split a dataset based on an attribute and an attribute value
    def splitDataset(self, split_index, split_value, dataset):
        left, right = dataset, dataset
        if split_index in self.numCols :
            a = dataset[:,split_index]
            b = a.astype(np.float)
            c = b<=float(split_value)
            d = c.astype(np.float)
            indexes = np.where(d == 1)[0]
            left = dataset[np.r_[indexes],:]
            indexes = np.where(d == 0)[0]
            right = dataset[np.r_[indexes],:]
        else :
            a = dataset[:,split_index]
            c = a==split_value
            d = c.astype(np.float)
            indexes = np.where(d == 1)[0]
            left = dataset[np.r_[indexes],:]
            indexes = np.where(d == 0)[0]
            right = dataset[np.r_[indexes],:]
        return left, right

    # Calculate the Mean squared error for a splitNode dataset
    def meanSquaredError(self, left,right):
        mse = 0.0
        sizeC = 0
        groups = np.array([left, right])
        for group in groups:
            size = float(len(group))
            sizeC += size
            # avoid divide by zero
            if size == 0:
                continue
            groupVal = group[:,-1]
            groupVal = groupVal.astype(np.float)
            a = np.full(len(groupVal),np.average(groupVal))
            mse += size*mean_squared_error(groupVal, a)
        if sizeC != 0:
            mse = mse/sizeC
        return mse

    # Create a terminal node(predictPrice price)
    def terminalNode(self, groupVal):
        outcomes = groupVal[:,-1].astype(np.float)
        return float(np.sum(outcomes) / len(outcomes)) 

    # Find split point of least MSE
    def findSplit(self, dataset):
        #intializing
        split_index = 0
        split_value = np.unique(dataset[:,0])[0]
        split_left, split_right = self.splitDataset(0, split_value, dataset)
        split_mse = self.meanSquaredError(split_left, split_right)
        
        for index in range(len(dataset[0])-1):
            uniqValue = np.unique(dataset[:,index])
            for value in uniqValue:
                left, right = self.splitDataset(index, value, dataset)
                mse = self.meanSquaredError(left, right)
                if mse < split_mse:
                    split_index, split_value, split_mse, split_left, split_right = index, value, mse, left, right
        return {'indexAttr':split_index, 'valueAttr':split_value, 'left':split_left, 'right':split_right}

    # Make node terminal or split further
    def splitNode(self, node, max_depth, min_size, depth):
        left = node['left']
        right = node['right']

        # check if already terminal node
        # assumption made to terminate when the one of the nodes are None, thus the nodes are not further split
        # this is to prevent a node becoming None
        if left is None or right is None:
            node['left'] = self.terminalNode(left + right)
            node['right'] = node['left']
            return
        # check for max depth
        if depth >= max_depth:
            node['left'] = self.terminalNode(left)
            node['right'] = self.terminalNode(right)
            return
    
        if len(left) > min_size:
            node['left'] = self.findSplit(left)
            self.splitNode(node['left'], max_depth, min_size, depth+1)
        else:
            node['left'] = self.terminalNode(left)

        if len(right) > min_size:
            node['right'] = self.findSplit(right)
            self.splitNode(node['right'], max_depth, min_size, depth+1)
        else:
            node['right'] = self.terminalNode(right)

    # Build a decision tree
    def buildTree(self, trainData, max_depth, min_size):
        self.root = self.findSplit(trainData)
        self.splitNode(self.root, max_depth, min_size, 1)
        return self.root

    def train(self, trainFile):
        # print("train here")
        with open(trainFile,'r') as f:
            reader  = csv.reader(f)
            trainData = np.array(list(reader))
            #remove id column and labels
            trainData = np.delete(trainData, 0, 1)
            trainData = np.delete(trainData,0,0)
            trainData = self.prepData(trainData)
            self.root = self.buildTree(trainData, 8,8)

    # Predict with a decision tree
    def predictPrice(self, node, test_row):
        if node['indexAttr'] in self.numCols :
            # print(test_row)
            # print(node['indexAttr'])
            if float(test_row[node['indexAttr']]) <= float(node['valueAttr']):
                if isinstance(node['left'], float):
                    return node['left']
                else :
                    return self.predictPrice(node['left'], test_row)               
            else:
                if isinstance(node['right'], float):
                    return node['right']
                else :
                    return self.predictPrice(node['right'], test_row)       
        else:
            if test_row[node['indexAttr']] == node['valueAttr']:
                if isinstance(node['right'], float):
                    return node['right']
                else :
                    return self.predictPrice(node['right'], test_row)
            else:
                if isinstance(node['right'], float):
                    return node['right']
                else :
                    return self.predictPrice(node['right'], test_row)

    def predict(self, testFile):
        with open(testFile,'r') as f:
            reader  = csv.reader(f)
            testData = np.array(list(reader))
            #remove id column and labels
            testData = np.delete(testData, 0, 1)
            testData = np.delete(testData,0,0)
            testData = self.prepData(testData)
            predictions = np.empty(len(testData), dtype = "float")
            for index in range(len(testData)):
                # print("predict ", index)
                test_row = testData[index]
                predictions[index] = self.predictPrice(self.root, test_row)
            return predictions
                
