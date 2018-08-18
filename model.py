#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 21:59:37 2018

@author: michal
"""

from model_utils import *

def train(X_train, y_train, L_dims, learning_rate = 0.0075,num_iterations = 2000):
    
    np.random.seed(1)
    costs = []
    s
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