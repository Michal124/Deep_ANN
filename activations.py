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

def softmax(Z):
    return np.exp(Z)/np.sum(np.exp(Z))


# =============================================================================
#                           Back propagation
# =============================================================================
    

def relu_backward(Z):

    if Z < 0 :
        return 0
    elif Z >= 0:
        return 1

def leaky_relu_backward(Z):

    if Z<0 :
        return 0.01
    elif Z >= 0:
        return 1

def tanh_backward(Z):
    return 1 - np.square(tanh(Z))

def sigmoid_backward(Z):
    return sigmoid(Z)*(1-sigmoid(Z))


def softmax_backward(Z):
    return(Z)