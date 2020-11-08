import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.io import loadmat
from sklearn import preprocessing
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Import helper functions
from Phantom_helper_functions import brain_predictions, import_phantom_dataset, replace_zero_with_NaN 
from Phantom_helper_functions import load_phantom_reference, plot_phantom_reference, get_phantom_predicted
from Phantom_helper_functions import plot_phantom_predicted, print_absolute_error, print_errors

# Import phantom dataset
phantom_re_im = import_phantom_dataset()

# Reshape array to normalize fingerprints
phantom_re_im = phantom_re_im.reshape(65536, 2000)

# Normalize fingerprints
norm = preprocessing.Normalizer()
phantom_re_im = norm.transform(phantom_re_im)

# Reshape back to make predictions
phantom_re_im = phantom_re_im.reshape(256, 256, 2000)

# Replace Zero fingerprints with NaN to predict
replace_zero_with_NaN(phantom_re_im)

#Custom metrics for T1 and T2 accuracy 
def cma1(y_train, y_pred):
    return 1 - tf.math.reduce_mean(tf.math.abs(y_train[:, 0] - y_pred[:, 0]) / (y_train[:, 0]), axis = 0, keepdims = False)
def cma2(y_train, y_pred):
    return 1 - tf.math.reduce_mean(tf.math.abs(y_train[:, 1] - y_pred[:, 1]) / (y_train[:, 1]), axis = 0, keepdims = False)
#Load trained model
ann = load_model('ANN_trained_model.h5', custom_objects={'cma1': cma1, 'cma2': cma2})
#Compiling the ANN 
ann.compile(optimizer = 'adam', loss = 'mse', metrics = [cma1, cma2])

#Phantom predictions
brain_pred = brain_predictions(ann, phantom_re_im)

# Plot T1_true phantom
T1_true = load_phantom_reference("T1")
plot_phantom_reference(T1_true)

# Plot T1_pred phantom
T1_pred = get_phantom_predicted(brain_pred, 0)
plot_phantom_predicted(T1_pred)

# Plot T2_true phantom
T2_true = load_phantom_reference("T2")
plot_phantom_reference(T2_true)

# Plot T2_pred phantom
T2_pred = get_phantom_predicted(brain_pred, 1)
plot_phantom_predicted(T2_pred)

# Plot absolute error between true an predicted phantom
print_absolute_error(T1_true, T1_pred)
print_absolute_error(T2_true, T2_pred)

# Print errors
print_errors(T1_pred, T1_true, T2_pred, T2_true)


