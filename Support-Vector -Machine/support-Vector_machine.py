# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 08:03:06 2024

@author: USER
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

data= pd.read_csv("Position_Salaries.csv")
x=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values
y=y.reshape(len(y),1)
print(y)

from sklearn.preprocessing import StandardScaler
sx=StandardScaler()
sy=StandardScaler()
X=sx.fit_transform(x)
y=sy.fit_transform(y)

print(x)
print(y)