import numpy as np
import matplotlib as plt
import pandas as pd

def cost(loss,sz):
    costs = np.sum(loss**2)/2*sz
    return costs
    
def Gradient_decent(X,y,coff,sz):
    
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
        if(i%1000 == 0):
            print("Cost is", cost(loss,sz))
        
    return coff

df = pd.read_excel("C:/Users/Admin/Documents/RESULT_DATA.xls")

rows = df.shape[0]
col  = df.shape[1]
const = np.expand_dims(np.ones(rows),axis = 1)

X = df.iloc[:,4:15].values
X = np.append(X,const,axis = 1)

y = np.array(df.iloc[:,-1]).reshape(rows,1)#(50,)


coff = np.zeros(12).reshape(12,1)


coff_L = Gradient_decent(X,y,coff,rows)
print(coff_L)


