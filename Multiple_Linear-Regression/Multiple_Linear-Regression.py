# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 23:22:50 2024

@author: USER
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(x)

#Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
c = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])],remainder='passthrough')  
X=np.array(c.fit_transform(x))
print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

