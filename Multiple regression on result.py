import matplotlib as plt
import pandas as pd
import numpy as np

dataset = pd.read_excel('RESULT_DATA.xls')

X = dataset.iloc[:,4:15].values
y = dataset.iloc[:,16:].values


from sklearn.linear_model import LinearRegression
regressor  = LinearRegression()
regressor.fit(X,y)
coff = regressor.coef_
print(sum(coff[0]))
for i in coff[0]:
    print( 100 * i)
