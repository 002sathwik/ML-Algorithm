# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:28:34 2024

@author: USER
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
dataset = pd.read_csv("Social_Network_Ads.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#splitting data Train,Test 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
print(x_train)
print(x_test)
print(y_test)
print(y_train)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
s=StandardScaler()
X_test=s.fit_transform(x_test)
X_train=s.fit_transform(x_train)
print(X_train)
print(X_test)


#TraningModel/prepurning
from sklearn.tree import DecisionTreeClassifier
treemodel=DecisionTreeClassifier(criterion='entropy',random_state=0,max_depth=3)
treemodel.fit(X_train,y_train)

#predicting result /accuracy score

from sklearn.metrics import confusion_matrix, accuracy_score
print(treemodel.predict(s.transform([[30,87000]])))
y_pred=treemodel.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)




  