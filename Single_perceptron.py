import numpy as np
import pandas as pd

def acc(X1,X2,w):
    count = 0;
    total = X1.shape[0]+X2.shape[0]
    Y1 = X1.dot(w)
    for i in range(X1.shape[0]):
        if(Y1[i]<0):
            count+=1
    Y2 = X2.dot(w)
    for i in range(X2.shape[0]):
        if(Y2[i]>0):
            count+=1
    return (1 - count/total)

def train(X1,X2,w):
    for i in range(10000):
        Y1 = X1.dot(w)
        for i in range(X1.shape[0]):
            if(Y1[i]<0):
                w = w + X1[i]
        Y2 = X2.dot(w)
        for i in range(X2.shape[0]):
            if(Y2[i]>0):
                w = w - X2[i]
    return w    

df = pd.read_csv(r"C:\Users\Admin\Downloads\Assignment1_Q4_data.txt" ,delim_whitespace = True,header = None)


#Class 1
X1 = df.iloc[:500,:].values
y = np.ones(500).reshape(500,1)
X1 = np.append(y,X1,axis = 1)
#Class 2
X2 = df.iloc[500:,:].values
X2 = np.append(y,X2,axis = 1)
w = np.array([-3.5,1000,-700]) 
w.reshape(3,1)

w = train(X1,X2,w)

print("Accurecy : {}".format(acc(X1,X2,w)))
print(w)
