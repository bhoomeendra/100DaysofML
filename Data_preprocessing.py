import numpy as np
import matplotlib.pyplot as ptl
import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder
dataset=pd.read_csv("C:/Users/Admin/Desktop/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/Data_Preprocessing/Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1:].values
z = dataset.loc[:, ["Country", "Salary"]]
print(z)


#for the missing value in the dataset
imputer=Imputer(missing_values = 'NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:3])
print(imputer)
X[:,1:3] = imputer.transform(X[:,1:3])
print(X)


#for making the data into numbers
lable_X = LabelEncoder()
X[:,0] = lable_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
lable_y = LabelEncoder()
y = lable_y.fit_transform(y)
print(y)

#for splitting the data into traning and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train)
#Scaling data so no coloumn dominate
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(X_train)
