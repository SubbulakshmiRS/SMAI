import nltk
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import rundocs
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
from gensim.models import Word2Vec 
from nltk.tokenize import TweetTokenizer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
import tensorflow as tf
import keras.backend as K
from keras.utils import np_utils
from keras.regularizers import l1, l2


class HateClassifier:
    model = svm.SVC()
    tokenizer = TweetTokenizer()
    plim = 200

    #Download corpus for stopwords, specifically 
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')

    def stat(self, labelst, labelsp):
        print("--------------------------------------------------")
        print("STATISTICS")
        print("Accuracy:",metrics.accuracy_score(labelst, labelsp))
        print("Confusion matrix:\n",metrics.confusion_matrix(labelst, labelsp))
        print("F1 score:\n",metrics.f1_score(labelst, labelsp))

    def f1(self, y_true, y_pred):
        y_pred = K.round(y_pred)
        tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
        tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2*p*r / (p+r+K.epsilon())
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
        return K.mean(f1)

    #tokenize the sentence/tweet, removing all mentions, hashtags and links
    def tokenize(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        return list(tokens)

    #Vectorizing tweets using word2vec
    def prep_W2V(self, trainData):         
        trainData['token'] = trainData['text'].apply(lambda x : self.tokenize(x))
        #flatten the full list for learning
        flat_list = [item for sublist in list(trainData['token']) for item in sublist]
        #Learning the dictionary with context
        tweet_w2v = Word2Vec([flat_list], size=self.plim, min_count=0)
        #Vectorize the data(collection of words) to get a weighted meaning for each word and thus more relevant vectors for each tweet 
        vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1)
        matrix = vectorizer.fit_transform([flat_list])
        tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
        #Vectorize the tweets 
        x = np.empty([trainData.shape[0], self.plim])
        for i in range(trainData.shape[0]):
            temp = np.zeros((self.plim,1))
            for word in trainData['token'][i]:
                a = (tweet_w2v[word]*tfidf[word]).reshape(self.plim,1)
                temp  = np.add(temp, a).reshape(self.plim,1)
#             temp = temp/trainData['token'].shape[0]
            x[i] = temp.reshape(self.plim)
        # print("vectors data type",x.shape )
        # print("vector size", x[0].shape)
        # print("vector size", x[3].shape)
        x = x.astype('float64')
        return x
    
    #Vectorizing words using TfidfVectorizer
    def prep_TfidfVectorizer(self, trainData): 
        #remove mentions 
        trainData['text'] = trainData['text'].apply(lambda x : ' '.join([w for w in x.split() if not w.startswith('@') ])  ) 
        #remove stopwords and symbols
        trainData['text'] = [i.lower() for i in trainData['text'] if i not in self.stopwords and i not in ['.',',','/','@','"','&amp','<br />','+/-','zzzzzzzzzzzzzzzzz',':-D',':D',':P',':)','!',';']]
        txt = np.asarray(trainData['text'].values)
        txt = txt.reshape(txt.shape[0])
        #Vectorize using TfidfVectorizer
        vectorizer  = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word', stop_words = 'english')
        txt_fitted = vectorizer.fit(txt)
        x = txt_fitted.transform(txt)
        #Truncate the number of components to plim and standardize it
        svd = TruncatedSVD(n_components=self.plim)
        xPCA = svd.fit_transform(x)
        # print("PCA reduced shape from tfid ",xPCA.shape)
        return xPCA

    #SVC model
    def model_SVM(self, kernel, c):
        model = svm.SVC(kernel=kernel, C=c, probability=True, class_weight="balanced")
        return model

    #Logistic regression model
    def model_LR(self, c):
        model = LogisticRegression(C=c)
        return model

    #Using basic regression neural network
    def model_NN(self):
        dim = self.plim
        model = Sequential()
        model.add(Dense(500,input_dim=dim,kernel_initializer='normal',activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(2, kernel_initializer='normal', activation='softmax'))
#         model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=[self.f1, 'accuracy'])
        return model

    def prep(self, data):
        x = self.prep_W2V(data)
        # x2 = self.prep_TfidfVectorizer(data) 
        # x = np.concatenate((x1, x2),axis=1)     
        return x
    
    def train(self, DataFile,kernel, c):
        data = pd.read_csv(DataFile)
        x = self.prep(data)       
        # print("x shape", x.shape)
        y = np.asarray(data['labels'].values)
        y = np_utils.to_categorical(y, num_classes=2)
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.1, random_state=42)
        self.model = self.model_LR(c)
        # self.model = self.model_SVM(kernel, c)

        # self.model.fit(xTrain,yTrain)
        # yPredict = self.model.predict(xTest)

        self.model = self.model_NN()
        model_log = self.model.fit(xTrain, yTrain,
                batch_size=128,
                epochs=500,
                verbose=0,
                validation_split = 0.1)
        yPredict = np.argmax(self.model.predict(xTest), axis=-1)
        yTest_NH = np.argmax(yTest, axis=-1)
        self.stat(yTest, yPredict)

    #Save output in a csv file
    def test(self, TestFile):
        data = pd.read_csv(TestFile)
        x = self.prep(data)  
        # yPredict = self.model.predict(xTest)
        yPredict = np.argmax(self.model.predict(x), axis=-1)
        final = np.c_[ np.arange(len(yPredict)), yPredict ]
        df = pd.DataFrame(final, columns=[' ','labels']) 
        df.to_csv("submissionq1.csv", index=False)

if __name__ == '__main__':
	print("Here for testing and training, uncomment the below commands")

	#training
	# hate_classifier = HateClassifier()
	# hate_classifier.train('./train.csv','linear',0.01)

	#testing
	# hate_classifier.test('./ttest.csv')