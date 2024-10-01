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


