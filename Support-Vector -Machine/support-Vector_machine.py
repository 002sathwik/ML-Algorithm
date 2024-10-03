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
Y=sy.fit_transform(y)




from sklearn.svm import SVR
reg=SVR(kernel='rbf')
reg.fit(X,Y)

# Predict the value for X=6.5
predicted_value = reg.predict(sx.transform([[6.5]]))

# Reshape the predicted value to 2D (n_samples, 1) before applying inverse_transform
predicted_value_reshaped = predicted_value.reshape(-1, 1)

# Apply inverse transformation to get the predicted value back in the original scale
original_scale_value = sy.inverse_transform(predicted_value_reshaped)

print(original_scale_value)
