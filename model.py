#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 21:59:37 2018

@author: michal
"""

import numpy as np
from model_utils import *

def train(X_train, y_train, L_dims, learning_rate = 0.0075,num_iterations = 2000):
    
    np.random.seed(1)
    costs = []
    
    # initialize parameters
    
    parameters = initialize_parameters(L_dims)
    
    for i in range(0,num_iterations):
        
        #Forward propagation
        AL, caches = forward_model(X_train,parameters)
        
        #Compute cost
        cost = compute_cost(AL,y_train)
        
        #Backward propagation
        grads = backward_model(AL,y_train,caches)
        
        #Update parameters
        parameters = update_parameters(parameters,grads,learning_rate)
        
                
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
                
    return parameters

def predict(X,y, parameters):
    
    m = X.shape[1]
    n_layers = len(parameters) // 2
    pred = np.zeros((1,m)) # initialize shape
    
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            pred[0,i] = 1
        else:
            pred[0,i] = 0
    
    print("Accuracy: "  + str(np.sum((pred == y)/m)))
        
    return pred
    