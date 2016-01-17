# -*- coding: utf-8 -*-

import numpy as np
from kdtree import KdTree
from collections import Counter

class NearestNeighbor:

    def train(self, X, y):
        self.tree = KdTree(X, y)
        self.X = X
        self.y = y
        
    def predict(self, X, k=5, method='tree'):
        count = 0
        yPredict = np.zeros(X.shape[0])    

        for i in range(X.shape[0]):
            if method == 'tree':
                values = self.tree.kNN(X[i], k)
                c = Counter([pair[0].value for pair in values]).most_common()                            
                yPredict[i] = c[0][0]
            else:
                a = np.abs(self.X - X[i])
                s = np.sum(a, axis=1)
                
                y_idxs = np.argsort(s)[:k]
                c = Counter(self.y[y_idxs]).most_common()
                yPredict[i] = c[0][0]
            
            count += 1
            print('{:}: {:,}/28,000'.format(yPredict[i], count))

        return yPredict