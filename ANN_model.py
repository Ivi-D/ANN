import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io as sc
import seaborn as sns

#Importing the dataset
re = pd.read_pickle("x_re.pickle")
im = pd.read_pickle("x_im.pickle")
Y = pd.read_pickle("y.pickle")
#Concatenate Real and Imaginary part
re_im = np.concatenate((im, re), axis=1)
del re, im

#SD of real_imaginary data
sd=np.std(re_im)
print(sd)
#Add Gaussian noise
noise = np.random.normal(0, sd*0.01, (re_im.shape))
noisy_signal = re_im + noise

##Data Pre-processing

#Set matrix of features and dependent variable vector
X = noisy_signal #re_im
y = Y[:, 0:2]

#Spliting the dataset into Training and Testing
from sklearn.model_selection import train_test_split
#Train - Test set split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Train - Validation set split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=0)

del X, y
del Y,re_im
del noise, noisy_signal, sd

#Feature normalization
from sklearn import preprocessing
norm = preprocessing.Normalizer()
X_train = norm.fit_transform(X_train)
X_val = norm.transform(X_val) 
X_test = norm.transform(X_test)


#Custom metrics for T1 and T2 accuracy 
def cma1(y_train, y_pred):
    return 1-tf.math.reduce_mean(tf.math.abs(y_train[:,0]-y_pred[:,0])/(y_train[:,0]), axis=0, keepdims=False)
def cma2(y_train, y_pred):
    return 1-tf.math.reduce_mean(tf.math.abs(y_train[:,1]-y_pred[:,1])/(y_train[:,1]), axis=0, keepdims=False)

import time
start_time = time.time()

#Build the ANN

#Initializing the ANN as sequence of layers
ann = tf.keras.models.Sequential()

#Add the input layer and the first hidden layer
# The input layer will automaticaly contain all the different columns as nodes
ann.add(tf.keras.layers.Dense(units = 300, activation = 'tanh', input_shape=(2000,))) 

#Second hidden layer
ann.add(tf.keras.layers.Dense(units = 300, activation = 'tanh'))

#Output Layer
ann.add(tf.keras.layers.Dense(units = 2, activation = 'relu'))

#Compiling the ANN 
ann.compile(optimizer = 'adam', loss = 'mse', metrics = [cma1, cma2])

#Training the ANN on the Training set 
history = ann.fit(X_train, y_train, batch_size = 800, epochs = 235, validation_data=(X_val, y_val)) 

print("--- %s Seconds ---" % (time.time() - start_time))

#Save trained model
ann.save('Final_ann_235_epochs_Noise.h5')

#Evaluate on the test set
score = ann.evaluate(X_test, y_test, verbose=0)
print("\n")
print("Test loss:", score[0])
print("Test Accuracy:", score[1])

#Predicting the Test set results
y_pred = ann.predict(X_test)

#MSE for T1 and T2 
from sklearn.metrics import mean_squared_error
T1_mse = mean_squared_error(y_pred[:,0], y_test[:,0])
T2_mse = mean_squared_error(y_pred[:,1], y_test[:,1])

print("\n")
print("T1 mse:", T1_mse)
print("T2 mse:", T2_mse)


import matplotlib.pyplot as plt
#Epochs - Accuracy
T1_acc = history.history['cma1']
T2_acc = history.history['cma2']
epochs = range(1,236)
plt.plot(epochs, T1_acc, 'm', label = 'T1 accuracy (cma1)')
plt.plot(epochs, T2_acc, 'c', label = 'T2 accuracy (cma2)')
plt.title('T1 and T2 accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Epochs - Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#correlation coefficient - R^2
correlation_matrix1 = np.corrcoef(y_test[:,0], y_pred[:,0])
correlation_xy1 = correlation_matrix1[0,1]
r_squared1 = correlation_xy1**2
print(r_squared1)

correlation_matrix2 = np.corrcoef(y_test[:,1], y_pred[:,1])
correlation_xy2 = correlation_matrix2[0,1]
r_squared2 = correlation_xy2**2
print(r_squared2)

#(T1)Reference values - Estimated values 
plt.plot(y_test[:,0], y_pred[:,0], 'm.', label='Predictions')
plt.plot(y_test[:,0], y_test[:,0], 'k-', label='Reference Line')
plt.plot(r_squared2, label='R^2 = 0.999')
#plt.xlim(0.1,1.05)
plt.xlabel('Reference T1 (ms)')
plt.ylabel('Estimated T1 (ms)')
#plt.title('T1 Predictions')
plt.legend(loc='upper left')
plt.show()

#(T2)Reference values - Estimated values 
plt.plot(y_test[:,1], y_pred[:,1], 'c.', label='Predictions')
plt.plot(y_test[:,1], y_test[:,1], 'k-', label='Reference Line')
plt.plot(r_squared2, label='R^2 = 0.999')
plt.xlabel('Reference T2 (ms)')
plt.ylabel('Estimated T2 (ms)')
#plt.title('T2 Predictions')
plt.legend(loc='upper left')
plt.show()











