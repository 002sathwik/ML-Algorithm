# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 

import pandas as pd

dataset = pd.read_csv("Data.csv")
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values
 
print(x)
print(y)

# Halndling missing data
from sklearn.impute import SimpleImputer
impute=SimpleImputer(missing_values=np.nan , strategy="mean") 
impute.fit(x[:,1:3])
(x[:,1:3]) = impute.transform((x[:,1:3]))
print(x)


#Encode categorical Data ,Independent Data
from sklearn.compose import ColumnTransformer
from  sklearn.preprocessing import OneHotEncoder
ct =ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X=np.array(ct.fit_transform(x))
print(X)

#Encoding dependent Variable 
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
Y=l.fit_transform(y)
print(Y) 

#Splitting to traing and testing set 

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
s=StandardScaler()
X_train[:,3:]=s.fit_transform(X_train[:,3:])
X_test[:,3:]=s.fit_transform(X_test[:,3:])
print(X_train[:,3:])
print(X_test[:,3:])

