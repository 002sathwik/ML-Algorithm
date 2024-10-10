# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:45:15 2024

@author: USER
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values  # Position levels (independent variable)
y = dataset.iloc[:, -1].values    # Salaries (dependent variable)

# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)  # Polynomial of degree 4
x_poly = poly_reg.fit_transform(x)       # Transform the original features to polynomial features

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Plotting the Linear Regression results
plt.scatter(x, y, color='red')            # Actual data points
plt.plot(x, lin_reg.predict(x), color='blue')  # Predicted salaries using linear regression
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Plotting the Polynomial Regression results
plt.scatter(x, y, color='red')            # Actual data points
plt.plot(x, lin_reg_2.predict(x_poly), color='blue')  # Predicted salaries using polynomial regression
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

