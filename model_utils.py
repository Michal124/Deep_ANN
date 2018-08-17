"""
Created on Tue Aug  7 16:15:12 2018

@author: michalgorski
"""

import numpy as np


# =============================================================================
# Activation functions
# =============================================================================

#Forward propagation

def relu(Z):
    A = np.maximum(0,Z)
    return A

def leaky_realu(Z):
    A = np.maximum(0.01,Z)
    return A

def tanh(Z):
    A = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    return A

# Last Layer functions

def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    return A

def softmax(Z):
    A = np.exp(Z)/np.sum(np.exp(Z))
    return A


#Backpropagation
    
