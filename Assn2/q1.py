import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import metrics
import warnings
from q1_old import KNNClassifier as knc

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

class SVMclassifier:
    featuresd_scale = []
    labelsd = []
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    pca = PCA(n_components=500)
    knn_classifier = knc()


    def stat(self, labelst, labelsp):
        print("Accuracy:",metrics.accuracy_score(labelst, labelsp))
        print("F1 Score",metrics.f1_score(labelst, labelsp, average='micro'))
        print("Confusion matrix")
        print(metrics.confusion_matrix(labelst, labelsp))
        print("-----------------------------------------------------")

    def svmC(self, c):
        subclf = SVC(kernel='linear', C=c)
        subclf.fit(self.featuresd_scale,self.labelsd)
        return subclf
    
    def preprocess(self, file):
        with open(file, 'rb') as fo:
            batch = pickle.load(fo, encoding='latin1')
            features = batch['data']
            labels = batch['labels']
            return features, labels

    def trainGivenFile(self):
        features1, labels1 = self.preprocess("./cifar-10-batches-py/data_batch_1")
        features2, labels2 = self.preprocess("./cifar-10-batches-py/data_batch_2")
        features3, labels3 = self.preprocess("./cifar-10-batches-py/data_batch_3")
        features4, labels4 = self.preprocess("./cifar-10-batches-py/data_batch_4")
        features5, labels5 = self.preprocess("./cifar-10-batches-py/data_batch_5")
        featuresd = np.concatenate((features1, features2, features3, features4, features5))
        labelsd = np.concatenate((labels1, labels2, labels3, labels4, labels5))
        featuresd = featuresd[0:5000]
        labelsd = labelsd[0:5000]
        # print("features shape",featuresd.shape)
        # print("labels shape", labelsd.shape)
        featuresd_pca = self.pca.fit_transform(featuresd)
        self.featuresd_scale = self.scaler.fit_transform(featuresd_pca)
        self.labelsd = np.array(labelsd)
        # print("features scale shape",self.featuresd_scale.shape)
        # print("training done")

    def testGivenFile(self):
        featurest, labelst = self.preprocess("./cifar-10-batches-py/test_batch")
        featurest = featurest[0:1000]
        labelst = labelst[0:1000]
        featurest_pca = self.pca.transform(featurest)
        featurest_scale = self.scaler.transform(featurest_pca)
        # print("featurest scale shape", featurest_scale.shape)
        labelst = np.array(labelst)

        subclf = self.svmC(0.001)
        l = subclf.support_vectors_
        print("length of support vector images ",len(l))
        print("Support vectors are ",l)
        labelsp = subclf.predict(featurest_scale)
        print("-----------------------------------------------------")
        print("SVM : For c value 0.001")
        self.stat(labelst, labelsp)

        # subclf = self.svmC(0.01)
        # labelsp = subclf.predict(featurest_scale)
        # print("-----------------------------------------------------")
        # print("SVM : For c value 0.01")
        # self.stat(labelst, labelsp)
    
        # subclf = self.svmC(0.001)
        # labelsp = subclf.predict(featurest_scale)
        # print("-----------------------------------------------------")
        # print("SVM : For c value 0.001")
        # self.stat(labelst, labelsp)

        labelspKNN = self.knn_classifier.test(self.featuresd_scale, self.labelsd, featurest_scale)
        print("-----------------------------------------------------")
        print("Comparison with KNN")
        self.stat(labelst, labelspKNN)

    def train(self, trainFile):
        # Assumption that the file is a pickle one
        featuresd, labelsd = self.preprocess(trainFile)
        featuresd_pca = self.pca.fit_transform(featuresd)
        self.featuresd_scale = self.scaler.fit_transform(featuresd_pca)
        self.labelsd = np.array(labelsd)

    def predict(self, testFile):
        # Assumption that the file is a pickle one
        featurest, labelst = self.preprocess(testFile)
        featurest_pca = self.pca.fit_transform(featurest)
        featurest_scale = self.scaler.fit_transform(featurest_pca)
        labelst = np.array(labelst)
        subclf = self.svmC(0.001)
        labelsp = subclf.predict(featurest_scale)
        self.stat(labelst, labelsp)
        return labelsp

if __name__ == "__main__": 
    svm_classifier = SVMclassifier()
    svm_classifier.trainGivenFile()
    svm_classifier.testGivenFile()
