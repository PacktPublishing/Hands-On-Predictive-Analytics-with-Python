# -*- coding: utf-8 -*-
"""
@author: Alvaro Fuentes
Chapter 9. Hands-On Predictive Analytics with Python
Producing the predictive model's objects
"""
## Imports
import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense
from sklearn.externals import joblib

## Loading the dataset
DATA_DIR = '../data'
FILE_NAME = 'diamonds.csv'
data_path = os.path.join(DATA_DIR, FILE_NAME)
diamonds = pd.read_csv(data_path)


## Preparing the dataset
diamonds = diamonds.loc[(diamonds['x']>0) | (diamonds['y']>0)]
diamonds.loc[11182, 'x'] = diamonds['x'].median()
diamonds.loc[11182, 'z'] = diamonds['z'].median()
diamonds = diamonds.loc[~((diamonds['y'] > 30) | (diamonds['z'] > 30))]
diamonds = pd.concat([diamonds, pd.get_dummies(diamonds['cut'], prefix='cut', drop_first=True)], axis=1)
diamonds = pd.concat([diamonds, pd.get_dummies(diamonds['color'], prefix='color', drop_first=True)], axis=1)
diamonds = pd.concat([diamonds, pd.get_dummies(diamonds['clarity'], prefix='clarity', drop_first=True)], axis=1)

## Dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=1, random_state=123)
diamonds['dim_index'] = pca.fit_transform(diamonds[['x','y','z']])
diamonds.drop(['x','y','z'], axis=1, inplace=True)

## Creating X and y
X = diamonds.drop(['cut','color','clarity','price'], axis=1)
y = np.log(diamonds['price'])

## Standarization: centering and scaling
numerical_features = ['carat', 'depth', 'table', 'dim_index']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X.loc[:, numerical_features] = scaler.fit_transform(X[numerical_features])

## Building the neural network
n_input = X.shape[1]
n_hidden1 = 32
n_hidden2 = 16
n_hidden3 = 8

nn_reg = Sequential()
nn_reg.add(Dense(units=n_hidden1, activation='relu', input_shape=(n_input,)))
nn_reg.add(Dense(units=n_hidden2, activation='relu'))
nn_reg.add(Dense(units=n_hidden3, activation='relu'))
# output layer
nn_reg.add(Dense(units=1, activation=None))

## Training the neural network
batch_size = 32
n_epochs = 40
nn_reg.compile(loss='mean_absolute_error', optimizer='adam')
nn_reg.fit(X, y, epochs=n_epochs, batch_size=batch_size)

## Serializing:
# PCA
joblib.dump(pca, './Model/pca.joblib') 

# Scaler
joblib.dump(scaler, './Model/scaler.joblib')

# Trained model
nn_reg.save("./Model/diamond-prices-model.h5")