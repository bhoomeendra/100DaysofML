# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 05:43:30 2018

@author: Admin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def model(X_train,y_train,X_test):
    
    b1 = 0
    X_mean = np.mean(X_train)
    y_mean = np.mean(y_train)
    for i in range(X_train.size):
        b1 = b1 + (y_train[i] - y_mean)*(X_train[i] -X_mean)
    sumd = 0
    for i in range(X_train.size):
        sumd = sumd +((X_train[i] -X_mean)**2)
    b1 =  b1 / sumd
    
    b0 = y_mean - b1 * X_mean
    
    print(b0,b1,X_mean,y_mean)
    y_pre = np.empty(X_test.size)
    for i in range(X_test.size):
       y_pre[i] =  b1*X_test[i] + b0
    y_pre = y_pre.reshape(6,1)
    return y_pre 


dataset = pd.read_csv("C:/Users/Admin/Desktop/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression/Simple_Linear_Regression/Salary_Data.csv")
X = dataset.iloc[:,:1].values
y = dataset.iloc[:,1:].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 2)

plt.scatter(X_train,y_train, color = 'red')
plt.show()

plt.scatter(X_test,y_test, color = 'red')
plt.show()
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)
y_pre = model(X_train,y_train,X_test)

plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_test,y_pre, color = 'blue')
plt.show()

plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_test,y_pre, color = 'blue')
plt.show()



