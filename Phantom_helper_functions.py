import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.io import loadmat
from sklearn import preprocessing
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Import Phantom Dataset
def import_phantom_dataset():
  phantom_imaginary = loadmat("phantom_im.mat")
  phantom_im = phantom_imaginary["Im"]
  phantom_real = loadmat("phantom_re.mat")
  phantom_re = phantom_real["re"]
  return np.concatenate((phantom_im, phantom_re), axis = 2)

# Replace zero values to NaN
def replace_zero_with_NaN(fingerprints):
  fingerprints[np.where(fingerprints[:, :, 0] == 0)] = np.nan
  fingerprints[np.where(fingerprints[:, :, 1] == 0)] = np.nan
  fingerprints[np.where(fingerprints[:, :, 2] == 0)] = np.nan
  
# Brain predictions
def brain_predictions(model, fingerprints):
    y_predicted = []

    for i in range(len(fingerprints)):
        y_predicted.append(model.predict([fingerprints[i]]))
    return np.array(y_predicted)

# Load phantom reference
def load_phantom_reference(value):
  phantom = loadmat("{}_phantom.mat".format(value))
  return phantom["{}_phantom".format(value)]

# Plot phantom reference
def plot_phantom_reference(phantom):
  plt.figure()
  plt.imshow(phantom, cmap = "gist_heat") 
  plt.axis("off")
  plt.colorbar()

# Load phantom reference
def get_phantom_predicted(brain_prediction, column):
  pred = brain_prediction[:, :, column]
  return np.nan_to_num(pred) 

# Plot phantom predicted
def plot_phantom_predicted(phantom):
  plt.figure()
  plt.imshow(phantom, cmap = 'gist_heat')
  plt.axis('off')
  plt.colorbar()

# Print absolute error between true and predicted phantom
def print_absolute_error(reference_phantom, predicted_phantom):
  error = abs(reference_phantom - predicted_phantom)
  plt.figure()
  plt.imshow(error, cmap = 'gist_heat')
  plt.axis('off')
  plt.colorbar()

def print_errors(T1_pred, T1_true, T2_pred, T2_true):
  T1_mse = mean_squared_error(T1_pred, T1_true)
  T2_mse = mean_squared_error(T2_pred, T2_true)
  print("\n")
  print("T1 mse:", T1_mse)
  print("T2 mse:", T2_mse)

  T1_mae = mean_absolute_error(T1_pred, T1_true)
  T2_mae = mean_absolute_error(T2_pred, T2_true)
  print("\n")
  print("T1 mae:", T1_mae)
  print("T2 mae:", T2_mae)
