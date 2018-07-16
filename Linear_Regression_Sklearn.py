# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:07:13 2018

@author: Bhoomeendra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("C:/Users/Admin/Desktop/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression/Simple_Linear_Regression/Salary_Data.csv")
X = dataset.iloc[:,:1].values
y = dataset.iloc[:,1:].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 2)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

print(X_train,X_test,y_train,y_test)
from sklearn.linear_model import LinearRegression
regressor  = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)


plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.show()

plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.show()
