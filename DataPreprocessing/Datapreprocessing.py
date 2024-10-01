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