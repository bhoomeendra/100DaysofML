import numpy as np
import matplotlib as plt
import pandas as pd

def cost(loss):
    costs = np.sum(loss**2)/2000
    return costs
    
def Gradient_decent(X,y,coff):
    sz = 1000
    alpha = 0.0001
    for i in range(100000):
        #hypothetical values
        hp = np.around(X.dot(coff),decimals = 10)#(250,1)
        #print("sum of hp",np.sum(hp))
        #error in predection
        loss = np.around(hp - y,decimals = 10)#(250,1)
        #print("Sum of loss",np.sum(loss))
        #gradient
        gradient = np.around(loss.T.dot(X)/sz,decimals = 10)#(1,3)
        #modification
        coff = np.around(coff - alpha * gradient.T,decimals =10)
        #print("Cost is", cost())
        
    return coff

df = pd.read_csv("C:/Users/Admin/Desktop/student.csv")
X = np.array([np.ones(1000),df.iloc[:,0],df.iloc[:,1]]).T#(50,3)
y = np.array(df.iloc[:,2]).reshape(1000,1)#(50,)

coff = np.zeros(3).reshape(3,1)

coff_L = Gradient_decent(X,y,coff)
print(coff_L)


