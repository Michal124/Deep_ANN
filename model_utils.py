"""
Created on Tue Aug  7 16:15:12 2018

@author: michalgorski
"""

import numpy as np
from activations import *


    
# =============================================================================
#                          INITIALIZE PARAMETERS
# =============================================================================


def initialize_parameters(L_dims):
   
    '''
    Description: Initialize weight & bias for neural network
    
    Input : 
        * L_dims -- array of layers ex. L_dims = [5,12,10,5]
        
    Output :
        * parameters -- initialize parameters of wei==ghts and bias 
     
    '''
    
    np.random.seed(3)
    parameters = {}
    L = len(L_dims)
    
    for l in range(1,L):
        
        parameters["W" + str(l)] = np.random.randn(L_dims[l], L_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((L_dims[l],1))
    
    return parameters


# =============================================================================
#                       Forward propagation module
# =============================================================================
   
def forward_propagation(A_prev,W,b,activation):

    '''
    Description:  Compute linear and activation functions for each layer.
    
    
    Input:
        * A_prev -- Previous layer activation or Input features
        * W -- weights
        * b -- bias
        * activation -- activation function of previous layer 

    Output:
        * A -- activation function of next layer
        * cache -- memory for parameters(A_prev,W,b) in each layer 
        
    '''
    
    #First step of neuron computation (linear)
    Z = np.dot(W.A_prev) + b  
    
    #Second step of neuron computation (activation) 
    if activation == "relu":
        A = relu(Z)
    
    elif activation == "leaky_relu":
        A = leaky_realu(Z)
    
    elif activation == "tanh" :
        A = tanh(Z)
        
    elif activation == "sigmoid": # binary output
        A = sigmoid(Z)
        
   # elif activation == "softmax": # Categorical output
   #     A = softmax(Z)
    
    
    # Create linear and activation cache
    linear_cache = (A_prev,W,b)
    activation_cache = Z
    
    # Add both steps of computation to cache
    cache = (linear_cache,activation_cache)
    return A, cache


def forward_model(X,parameters):
    
    caches = []
    A = X 
    L = len(parameters) // 2
    
    # compute hidden layers activation
    for l in (1,L):
        A_prev = A 
        A, cache = forward_propagation(A_prev,
                                       parameters["W" + str(l)],
                                       parameters["b" + str(l)],
                                       activation = "relu")
        caches.append(cache)
    
    
    # compute output layer activation
    AL, cache = forward_propagation(A,
                                    parameters["W" + str(L)],
                                    parameters["b" + str(L)],
                                    activation = "sigmoid")
    
    caches.append(cache)
    
    return AL, caches

    

# =============================================================================
#                          Backpropagation module
# =============================================================================



# =============================================================================
#                           Compute cost function
# =============================================================================

def compute_cost(AL,Y):
    
    '''
    
    Description : Compute-cross entropy cost
    
    Input:
      * AL -- Activation value in last layer
      
      * Y -- Outputs .
    
    Output:
      * cost -- result of cost function.
    '''
    
    m = Y.shape[1]
    
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    
    cost = np.squeeze(cost)      
    
    return cost
