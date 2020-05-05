import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.linear_model import RidgeCV
from numpy.testing import rundocs
from sklearn import metrics
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tensorflow import keras
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import svm
from sklearn.linear_model import BayesianRidge
from keras.regularizers import l1, l2

class AffinityRegression:
    model = RidgeCV(cv=2)
    plim = 200

    def stat(self, labelst, labelsp):
        print("--------------------------------------------------")
        print("STATISTICS")
        print("RMSE :",metrics.mean_squared_error(labelst, labelsp, squared=False))
        print("MAE :",metrics.mean_absolute_error(labelst, labelsp))
    
    #calculate number of atoms for each chemical symbol like 'C', 'S'
    def number_of_atoms(self, atom_list, df):
        for i in atom_list:
            df['num_of_{}_atoms'.format(i)] = df['mol'].apply(lambda x: len(x.GetSubstructMatches(Chem.MolFromSmiles(i))))
    
    #manual extraction of data such as number of atoms, number of valence electrons, etc. 
    def prep_RDKIT(self, trainData): 
        #Convert the SMILES sequence into a Mol format using rdkit
        trainData['mol'] = trainData['SMILES sequence'].apply(lambda x: Chem.MolFromSmiles(x)) 
        trainData['mol'] = trainData['mol'].apply(lambda x: Chem.AddHs(x))
        trainData['num_of_atoms'] = trainData['mol'].apply(lambda x: x.GetNumAtoms())
        trainData['num_of_heavy_atoms'] = trainData['mol'].apply(lambda x: x.GetNumHeavyAtoms())
        self.number_of_atoms(['C','O', 'N', 'Cl', 'S'], trainData)
        trainData['tpsa'] = trainData['mol'].apply(lambda x: Descriptors.TPSA(x))
        trainData['mol_w'] = trainData['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
        trainData['num_valence_electrons'] = trainData['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
        trainData['num_heteroatoms'] = trainData['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))
        # print("train Data\n", trainData.columns)
        x = np.asarray(trainData.drop(columns=['SMILES sequence', 'mol', 'Binding Affinity']))
        return x

    #using a pre trained mol2vec algorithm for extracting features about the molecule
    def prep_m2v(self, trainData):
        #300 dimension feature extracted using this model
        model = word2vec.Word2Vec.load('./model_300dim.pkl')
        #Convert the SMILES sequence into a Mol format using rdkit
        trainData['mol'] = trainData['SMILES sequence'].apply(lambda x: Chem.MolFromSmiles(x)) 
        trainData['mol'] = trainData['mol'].apply(lambda x: Chem.AddHs(x))
        #Convert from mol format to mol sentence format using sentence2vec and applying the model
        trainData['sentence'] = trainData.apply((lambda x: MolSentence(mol2alt_sentence(x['mol'], 1))), axis=1)
        trainData['mol2vec'] = [DfVec(x) for x in sentences2vec(trainData['sentence'], model, unseen='UNK')]
        x = np.array([x.vec for x in trainData['mol2vec']])
        return x

    #Using Bayesian Ridge 
    def model_BRidge(self):
        model = BayesianRidge()
        return model

    #Using ridgeCV, with different cv values
    def model_RidgeCV(self):
        model = RidgeCV(alphas=[1e-6, 1e-3, 1, 1e3, 1e6])
        return model
    
    #Using ridge, with different alpha values
    def model_Ridge(self):
        model = linear_model.Ridge(solver='sparse_cg')
        return model
    
    #Using regression model of svm
    def model_SVR(self):
        model = svm.SVR()
        return model

    #Using basic regression neural network
    def model_NN(self):
        dim = 311
        model = Sequential()
        model.add(Dense(600,input_dim=dim,kernel_initializer='normal',activation='relu',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)))
        model.add(Dropout(0.3))
        model.add(Dense(1, kernel_initializer='normal'))
#         model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
        return model

    #Linear regression model
    def model_LR(self):
        model = LinearRegression()
        return model 

    def train(self, DataFile):
        data = pd.read_csv(DataFile)
        x2 = self.prep_RDKIT(data)
        x1 = self.prep_m2v(data)
        # Combining both set of features, extracted in different ways for the model to train on
        x3 = np.concatenate((x1, x2),axis=1)
        x = StandardScaler().fit_transform(x3)
        # print("x shape", x.shape)
        y = np.asarray(data['Binding Affinity'].values)
        y = y.reshape(len(y), 1)
        l = x.shape[0]
        l = int(0.9*l)
        xTrain = x[0:l]
        xTest = x[l:]
        yTrain = y[0:l]
        yTest = y[l:]

#         self.model = self.model_BRidge()
#         self.model.fit(xTrain,yTrain)

#         self.model = self.model_Ridge()
#         self.model.fit(xTrain,yTrain)

#         self.model = self.model_RidgeCV()
#         self.model.fit(xTrain,yTrain)
        
#         self.model = self.model_LR()
#         self.model.fit(xTrain,yTrain)

        self.model = self.model_NN()
        history = self.model.fit(xTrain, yTrain,
                batch_size=128,
                epochs=500,
                verbose=0,
                validation_split = 0.1)

        yPredict = self.model.predict(xTest)
        self.stat(yTest, yPredict)


    #Save output in a csv file
    def test(self, TestFile):
        data = pd.read_csv(TestFile)
        x2 = self.prep_RDKIT(data)
        x1 = self.prep_m2v(data)
        x3 = np.concatenate((x1, x2),axis=1)
        x = StandardScaler().fit_transform(x3)
        yPredict = self.model.predict(x)
        final = np.c_[ data['SMILES sequence'], yPredict ]
        df = pd.DataFrame(final, columns=['SMILES sequence','Binding Affinity']) 
#         np.savetxt("submission.csv", final, delimiter=",", header='SMILES sequence, Binding Affinity', comments='')
        df.to_csv("submissionq3.csv", index=False)


if __name__ == '__main__':
	print("Here for testing and training, uncomment the below commands")
	# For training 

	# affinity_reg = AffinityRegression()
	# affinity_reg.train('./train.csv')

	# For testing 
	# affinity_reg.test('./ttest.csv')