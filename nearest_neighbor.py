# -*- coding: utf-8 -*-

import numpy as np

class NearestNeighbor:

    def train(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, X):
        count = 0
        yPredict = np.zeros(X.shape[0])    

        for i in range(X.shape[0]):
            a = np.abs(self.X - X[i])
            s = np.sum(a, axis=1)
            yPredict[i] = self.y[np.argmin(s)]
            
            count += 1
            print('{:,}/28,000'.format(count))

        return yPredict