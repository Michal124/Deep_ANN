"""
Created on Tue Aug  7 16:15:12 2018

@author: michalgorski
"""

import numpy as np


# =============================================================================
#                           Activation functions
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
    

# =============================================================================
#                          Neural network functions
# =============================================================================


def initialize_parameters(L_dims):
   
    np.random.seed(3)
    parameters = {}
    L = len(L_dims)
    
    for l in range(1,L):
        
        parameters["W" + str(l)] = np.random.randn(L_dims[l], L_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((L_dims[l],1))
    
    return parameters


   
    
    
def compute_cost(AL,Y):
    
    m = Y.shape()
    
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    
    cost = np.squeeze(cost)      
    
    return cost
