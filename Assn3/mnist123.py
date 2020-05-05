import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import keras
# from keras.datasets import mnist
from mnist import MNIST
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def extractData(datafile):
    mnist = MNIST(datafile)
    x_train, y_train = mnist.load_training()
    x_test, y_test = mnist.load_testing() 
    return x_train, y_train, x_test, y_test

def prepData(x_train, y_train, x_test, y_test) :
    sc = StandardScaler()

    x_train = np.asarray(x_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.int32)
    x_test = np.asarray(x_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.int32)

    x_trainf = sc.fit_transform(x_train)
    x_testf = sc.transform(x_test)
    x_trainf = x_trainf.reshape(x_trainf.shape[0], 28, 28,1)
    x_testf = x_testf.reshape(x_testf.shape[0], 28, 28,1)

    return x_trainf, y_train, x_testf, y_test

def CNNModel():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model

def predict(model, x_train, y_train, x_test, y_test):
    model_log = model.fit(x_train, y_train,
            batch_size=128,
            epochs=5,
            verbose=1,
            validation_data=(x_test, y_test))

    y_pred = model.predict_classes(x_test)
    return y_pred


def mnistPredict(datafile):
    x_train1, y_train1, x_test1, y_test1 = extractData(datafile)
    x_train, y_train, x_test, y_test = prepData(x_train1, y_train1, x_test1, y_test1)
    model = CNNModel()
    predictions = predict(model, x_train, y_train, x_test, y_test)
    # stat(y_test, predictions)
    for x in predictions:
        print(x)
        print(" ")


def stat(y_test, y_pred):
    print("\nAccuracy score for MLP\n",accuracy_score(y_test, y_pred))
    print("\nConfusion matrix for MLP\n",confusion_matrix(y_test,y_pred))
    print("\nClassification report for MLP\n",classification_report(y_test,y_pred))

if __name__ == "__main__": 
    mnistPredict(str(sys.argv[1]))
