#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 14:48:28 2018

@author: michal
"""

import numpy as np


# =============================================================================
#                           Forward propagation
# =============================================================================

#Forward propagation

def relu(Z):
    return np.maximum(0,Z)

def leaky_realu(Z):
    return np.maximum(0.01,Z)

def tanh(Z):
    A = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    return A

# Last Layer functions

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

#def softmax(Z):
 #   return np.exp(Z)/np.sum(np.exp(Z))


# =============================================================================
#                           Back propagation
# =============================================================================
    

def relu_backward(dA,Z):

    dZ = np.array(dA, copy=True)      
    dZ[Z <= 0] = 0
    
    return dZ

def leaky_relu_backward(dA,Z):

    dZ = np.array(dA, copy=True)      
    dZ[Z <= 0] = 0.01
    
    return dZ

def tanh_backward(dA,Z):
    return dA * (1 - np.square(tanh(Z)))

def sigmoid_backward(dA,Z):
    return dA * (sigmoid(Z)*(1-sigmoid(Z)))


#def softmax_backward(Z):
 #   return(Z)