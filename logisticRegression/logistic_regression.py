# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:22:29 2016

@author: sampepose
"""

import numpy as np
from scipy import optimize

def log_softmax(z):
    return np.log(softmax(z))
    
# credits: https://github.com/scikit-learn/scikit-learn/blob/935877798271db6e01ca337e9fe71d885c55f2af/sklearn/utils/extmath.py#L676
def softmax(z):
    max_prob = np.max(z, axis=1).reshape((-1, 1))
    z -= max_prob
    np.exp(z, z)
    sum_prob = np.sum(z, axis=1).reshape((-1, 1))
    z /= sum_prob
    return z

class LogisticRegression:
    
    def __init__(self, C):
        self.C = C
    
    def __cost(self, theta, X, y):
        M, N = X.shape        
        theta = np.reshape(theta, (N, self.K))        
        
        X_theta = X.dot(theta)
        prob = softmax(X_theta)
        log_prob = log_softmax(X_theta)
        
        cost = (-1.0 / M) * np.sum(self.binary_labels * log_prob)
        cost += 0.5 * (1.0 / self.C) * (1.0 / M) * np.sum(theta ** 2)
        
        grad = (-1.0 / M) * X.T.dot(self.binary_labels - prob)
        grad += (1.0 / self.C) * (1.0 / M) * theta

        return cost, grad.flatten()
        
    def fit(self, X, y, n_classes):
        self.K = n_classes

        # Added biases (col of 1s)
        X = np.insert(X, 0, 1, axis=1)

        M, N = X.shape        
        print(y.shape)
        # convert labels to (K, M) binary matrix
        self.binary_labels = np.zeros((M, self.K))
        self.binary_labels[range(M), y] = 1.0

        # randomly generate initial thetas
        self.theta = np.random.rand(X.shape[1], self.K)
        
        res = optimize.minimize(
            self.__cost,
            self.theta,
            method='L-BFGS-B',
            args=(X, y),
            jac=True,
            options={'maxiter': 500, 'disp': True}
        )   
        
        self.theta = res.x
                                
    def predict(self, X):
        # Added biases
        X = np.insert(X, 0, 1, axis=1)
        M, N = X.shape
        theta = np.reshape(self.theta, (N, self.K))
        return softmax(X.dot(theta)).argmax(axis=1)