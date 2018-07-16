import matplotlib as plt
import pandas as pd
import numpy as np

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:4].values
y = dataset.iloc[:,4:].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_X = LabelEncoder()
X[:,3] = label_X.fit_transform(X[:,3])
onehotencoder= OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

from sklearn.linear_model import LinearRegression
regressor  = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
print(regressor.coef_)
diff = []
for  i in range(len(y_pred)):
    diff.append(abs(y_pred[i] - y_test[i]))
mean_diff =sum(diff)/len(y_pred)
print(mean_diff)