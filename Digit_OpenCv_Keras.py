import tensorflow as tf

mnist  = tf.keras.datasets.mnist

(X_train,y_train) , (X_test , y_test)  = mnist.load_data()

#Normalize Data

X_train = tf.keras.utils.normalize(X_train,axis = 1)
X_test = tf.keras.utils.normalize(X_test,axis = 1)

#Model 

model  = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))

model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))

model.add(tf.keras.layers.Dense(10,activation = tf.nn.softmax))

model.compile(optimizer ='adam' ,loss ='sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train,y_train,epochs = 3)
'''
val_loss, val_acc  = model.evaluate(X_test,y_test)

print(val_loss,val_acc)

import matplotlib.pyplot as plt

plt.imshow(X_train[0])
plt.show() 

'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
cap  = cv2.VideoCapture(0)
while(True):
    add_img = np.zeros(480 * 640,dtype = 'uint8').reshape(480,640)
    
    while(True):
        ret ,frame = cap.read()
        #cv2.imshow("Camer",frame)
        
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        lower_b = np.array([110,150,65])
        upper_b = np.array([120,255,255])
        ##
        mask = cv2.inRange(frame,lower_b,upper_b)
        
        
        g_mask = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,9,2)
        
        add_img = cv2.add(add_img,g_mask)
        
        cv2.imshow("Add",cv2.flip(add_img , 1))
        if( cv2.waitKey(1) & 0xFF == ord('q')):
            break
    
    final = cv2.resize(add_img,(28,28),interpolation = cv2.INTER_AREA)
    final = cv2.flip(final,1)
    plt.imshow(final)
    
    pre = model.predict(np.array([final]))
    num = np.argmax(pre)
    
    predick = cv2.imread(r"E:\Deep leaning\TensorFlow and Keras\Digits\{}.png".format(num),1)
    
    cv2.imshow("Prediction",predick)
    
    if(cv2.waitKey(0) & 0xFF == ord('w')):
        break
    cv2.destroyAllWindows()
cv2.destroyAllWindows()
cap.release()
