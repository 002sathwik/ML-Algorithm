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



