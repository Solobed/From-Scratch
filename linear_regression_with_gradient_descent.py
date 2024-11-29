import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error

class LinearRegression():
  def __init__(self):
    self.params = []
    self.intercept = 0
    self.scaler = StandardScaler()
  
  def fit_transform(self, X, Y, alpha, iterations, costs):
    """
      Function to train the model.

      Args:
          X (numpy.ndarray) : The features
          Y (numpy.ndarray): The labels
          alpha (float) : Learning rate
          iterations (int):  Number of iterations

    """
    # Standardiation des données
    
    X_scaled = self.scaler.fit_transform(X)
    
    # Initialisation des paramètres
    self.params = [0] * len(X[0])
    iteration = 0
    cost = 10000

    # Apprentissage
    while (iteration < iterations):

      y_pred = np.dot(X_scaled, self.params) + self.intercept

      self.update_params(X_scaled, Y, y_pred, alpha)
      
      cost = mean_squared_error(Y, y_pred)

      iteration +=1
    
  
  def update_params(self, X, Y, Y_pred, alpha):
    """
    Function to update the parameters.

    Args:
        X (numpy.ndarray) : The features
        Y (numpy.ndarray): The labels
        Y_pred (numpy.ndarray): The predicted labels
        alpha (float) : Learning rate
          
    """
    # Mis à jour des gradients
    gradients = [0] * len(self.params)
    difference = []
    for i in range(len(X)):
      for j in range(len(self.params)):
        gradients[j] += (Y_pred[i] - Y[i]) * X[i][j]
        difference.append(Y_pred[i] - Y[i])

    for i in range(len(self.params)):
      self.params[i] = self.params[i] - (alpha * (1/len(X) * gradients[i]))
    
    # Mis à jour de l'intercept  
    self.intercept = self.intercept - alpha * (1/len(X) * np.sum(difference)) 
  
  def predict(self, X):
    """
    Function to predict the label.

    Args:
        X (numpy.ndarray) : The features
    """
    X_scaled = self.scaler.fit_transform(X)
    y_pred = np.dot(X_scaled, self.params) + self.intercept
    return y_pred

