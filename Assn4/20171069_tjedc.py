import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
import keras.backend as K
from PIL import Image
from tensorflow.keras import datasets, layers, models
from sklearn.preprocessing import StandardScaler
import os
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.utils import np_utils
from keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping

class SentimentClassifier:
    model = svm.SVC()

    def stat(self, labelst, labelsp):
        print("--------------------------------------------------")
        print("test shape",labelst.shape)
        print("predict ", labelsp.shape)
        print("STATISTICS")
        print("Accuracy:",metrics.accuracy_score(labelst, labelsp))
        print("Confusion matrix:\n",metrics.confusion_matrix(labelst, labelsp))
        print("F1 score:\n",metrics.f1_score(labelst, labelsp, average='macro'))

    # using Image to load and resize the images in PIL format. Image.ANTIALIAS a high-quality downsampling filter
    def load(self, imgName, folder):
        fileName = str('./'+folder+'/'+imgName+'.jpg')
        image = Image.open(fileName)
        image = image.resize((320, 320), Image.ANTIALIAS)
        x = np.asarray(image)
        return x

    #convert images to numpy array 
    def prep(self, trainData, trainFolder):
        x = []
        for i in tqdm(range(trainData.shape[0])):
            tmp = self.load(trainData['image_file'][i], trainFolder)
            x.append(tmp.reshape(320, 320, 3))
#         tStd = self.sc.fit_transform(temp)
        x = np.asarray(x)
#         x = StandardScaler().fit_transform(x)
        return x

    # A simple 2d CNN implementation using keras
    def model_CNN(self):
        # print("model 1")
        model = models.Sequential()
        #convolution filters of size 3*3
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(320, 320, 3)))
        #choose the best features via pooling
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        #flatten since too many dimensions, we only want a classification output
        model.add(layers.Flatten())
        #fully connected to get all relevant data
        model.add(layers.Dense(64, activation='relu'))
        #output a softmax to squash the matrix into output probabilities for 5 categories
        model.add(layers.Dense(5, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                    loss=tf.keras.losses.categorical_crossentropy,
                    metrics=['accuracy'])
        return model

    def train(self, DataFile, trainFolder):
        # arr = [os.path.abspath(x) for x in os.listdir()]
        # print(arr)
        data = pd.read_csv(DataFile)
        # print("training data shape ",data.shape)
        x = self.prep(data, trainFolder) 
        # print("x shape", x.shape)
        y = np.asarray(data['emotion'].values)
        y = np_utils.to_categorical(y, num_classes=5)
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.05, random_state=42)
        
        self.model = self.model_CNN()
#         self.model.fit(xTrain,yTrain)
        model_log = self.model.fit(xTrain, yTrain,
                batch_size=64,
                epochs=30,
                verbose=1,
                validation_split = 0.05,
                callbacks = [EarlyStopping(monitor='accuracy', mode='max', restore_best_weights = 'True')])

        yPredict = np.argmax(self.model.predict(xTest), axis=-1)
        yTest_NH = np.argmax(yTest, axis=-1)
        self.stat(yTest_NH, yPredict)

    #Save output in a csv file
    def test(self, TestFile, testFolder):
        data = pd.read_csv(TestFile)
        x = self.prep(data, testFolder)
        yPredict = np.argmax(self.model.predict(x), axis=-1)
        yPredict = yPredict.astype('int') 
        np.savetxt("submissionq2.csv", yPredict, delimiter=",", header='emotion', comments='')

if __name__ == '__main__':
	print("Here for testing and training, uncomment the below commands")

	#training
	# senti_classifier = SentimentClassifier()
	# senti_classifier.train('./train.csv', 'ntrain')

	#testing
	# senti_classifier.test('./ttest.csv', 'ttest')