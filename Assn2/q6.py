# Example of making predictions
import csv
import numpy as np 
from collections import Counter
import glob
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.decomposition import PCA


class Cluster:
    K = 5
    m = 0
    n = 0
    centroids = []
    itNum = 100

    def stat(self, labels_test, predictions):
        print("RAND score ", adjusted_rand_score(labels_test, predictions))
        print("Homogeneity score ", homogeneity_score(labels_test, predictions))
  
    # calculate the Euclidean distance between two vectors
    def euclidean_distance(self, trainData, test_row):
        a = trainData - test_row
        b =  a**2
        distances = np.sum(b, axis = 1)
        return distances.reshape(-1,1)

    def predict(self, testData):
        distances = np.zeros([testData.shape[0],self.K])
        for k in range(self.K):
            d= self.euclidean_distance(testData, self.centroids[k])
            distances[:,k] = d.reshape(testData.shape[0])
        predictions = np.argmin(distances, axis = 1)
        predictions = predictions+1   
        return predictions       

    def clustering(self, trainData):
        for i in range(self.itNum):
            distances = np.zeros([self.m,self.K])
            for k in range(self.K):
                d= self.euclidean_distance(trainData, self.centroids[k])
                distances[:,k] = d.reshape(self.m)
            predictions = np.argmin(distances, axis = 1)
            predictions = predictions+1
            labels = np.unique(predictions)

            # print("should be 5 check", labels)
            for l in range(len(labels)):
                pool = np.where(predictions == labels[l])[0]
                if len(pool) is not 0:
                    self.centroids[l] = np.mean(trainData[pool], axis = 0)

        
    def prepData(self, fileList):
        vectorizer = TfidfVectorizer(input='filename', decode_error='ignore', 
                                    lowercase=True, token_pattern=r'\b[^\d\W]+\b', 
                                    stop_words='english')
        x = vectorizer.fit_transform(fileList).toarray()
        x = np.array(x)
        x = x.astype(np.float)
        pca = PCA(n_components=1000)
        xPCA = pca.fit_transform(x)
        # x_normed = (x - x.min(0)) / x.ptp(0)
        # print("X shape", xPCA.shape)
        # print("no of files is ", i)
        return xPCA


    def cluster(self, TestFile):
        fileList = []
        for dirpath,_,filenames in os.walk(TestFile):
            for f in filenames:
                fileList.append(os.path.abspath(os.path.join(dirpath, f)))
        print(fileList[0])
        x = self.prepData(fileList)

        self.m, self.n = x.shape
        i = np.random.randint(0,self.m-1, size=(1, self.K))
        index = list(itertools.chain.from_iterable(i))
        print("index = ", index)
        self.centroids = x[index]
        self.clustering(x)
        predictions =  self.predict(x)
        return predictions

# if __name__ == "__main__": 
#     cluster_algo = Cluster()
#     predictions = cluster_algo.cluster('./dataset/') 