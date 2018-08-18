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
        
        
    assert(parameters['W' + str(l)].shape == (L_dims[l], L_dims[l-1]))
    assert(parameters['b' + str(l)].shape == (L_dims[l], 1))
    
    return parameters


# =============================================================================
#                       Forward propagation module
# =============================================================================
   
def forward_propagation_layer(A_prev,W,b,activation):

    '''
    Description: Compute linear and activation functions for each layer.
    
    
    Input:
        * A_prev -- Previous layer activation or Input features
        * W -- weights
        * b -- bias
        * activation -- activation function of previous layer 

    Output:
        * A -- activation function of next layer
        * cache -- memory for parameters(A_prev,W,b) in each layer 
        
    '''
    Z = np.dot(W,A_prev) + b  
    
    #Second step of neuron computation (activation) 
    if activation == "relu":
        A = relu(Z)
    
    elif activation == "leaky_relu":
        A = leaky_realu(Z)
    
    elif activation == "tanh" :
        A = tanh(Z)
        
    elif activation == "sigmoid": # binary output
        A = sigmoid(Z)
        
    #elif activation == "softmax": # Categorical output
     #   A = softmax(Z)
    
    
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
    for l in range(1,L):
        A_prev = A 
        A, cache = forward_propagation_layer(A_prev,
                                       parameters["W" + str(l)],
                                       parameters["b" + str(l)],
                                       activation = "relu")
        caches.append(cache)
    
    # compute output layer activation
    AL, cache = forward_propagation_layer(A,
                                    parameters["W" + str(L)],
                                    parameters["b" + str(L)],
                                    activation = "sigmoid")
    
    caches.append(cache)
    
    return AL, caches


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
    
    #binary cross_entropy
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    
    cost = np.squeeze(cost)      
    
    return cost

    

# =============================================================================
#                          Backpropagation module
# =============================================================================


def back_propagation_layer(dA, cache ,activation):
    
    '''
    Description: Compute derivatives for selected layer
    
    
    Input:
        * dA -- derivative with respect to activation of last layer
        * cache - data from each layer from forward propagation, contain:
 
               * linear cache - A_prev, W, b  ;
               * activation cache - Z .
               
        * activation -- activation function
    

    Output:
        
        * dA_prev -- derivative with respect to previous layer
        * dW -- derivative with respect to Weights
        * db -- derivative with respect to bias
        
    '''
    
    
    
    linear_cache,activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        
    elif activation == "leaky_relu":
        dZ = leaky_relu_backward(dA,activation_cache)
        
    elif activation == "tanh":
        dZ = tanh_backward(dA,activation_cache)
    
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
    
    A_prev , W , b = linear_cache 
    
    m = A_prev.shape[1] # traning examples
    
    dW = (1/m) * np.dot(dZ,A_prev.T)
    db = (1/m) * np.sum(dZ,axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    
    #Check derivative's shapes
    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape  == b.shape)


    return dA_prev, dW, db
    


def backward_model(AL,Y,caches):
    
    
    '''
    Description: Compute derivatives of all layers
    
    
    Input:
        * AL -- activation vector of last layer of forward propagation
        * Y -- output vector
        * caches -- list of caches contains
               * linear cache - A_prev, W, b  ;
               * activation cache - Z .
               

    Output:
        * grads -- a dictionary of gradients
    
    '''
    
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = back_propagation_layer(dAL, current_cache, "sigmoid")
    ### END CODE HERE ###
    
    for l in reversed(range(L-1)):

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = back_propagation_layer(grads["dA" + str(L)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# =============================================================================
#                               Update parameters
# =============================================================================
    
def update_parameters(parameters, grads, learning_rate):
    
    '''
    Description: Update layer parameters 
    
    
    Input:
        * parameters -- dictionary parameter of each layer 
        * grads -- dictionary containing computed gradients

    Output:
        * parameters -- dictionary parameter of each layer s
    
    '''
    
    L = len(parameters) // 2 # number of layers in the neural network
    # Update each parameter
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    
    
    return parameters
    
    



