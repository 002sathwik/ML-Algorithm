# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:59:17 2024

@author: USER
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=1/3, random_state=0)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)

#train Data
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train , reg.predict(x_train), color ="blue")
plt.title('salary vs Experience')
plt.xlabel('exp')
plt.ylabel('Salary')
plt.show()



# test Data
plt.scatter(x_test, y_test, color = "red")
plt.plot(x_train , reg.predict(x_train), color ="blue")
plt.title('salary vs Experience')
plt.xlabel('exp')
plt.ylabel('Salary')
plt.show()
