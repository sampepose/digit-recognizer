# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:43:42 2016

@author: sampepose
"""

import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata("MNIST original")
print('Fetched data')

# use the traditional train/test split
X, y = mnist.data, mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Construct k-nearest neighbor classifier and 'fit' it
kNeigh = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, algorithm='kd_tree') # use all the CPU cores in parallel!
kNeigh.fit(X_train, y_train)
print('Trained classifier')

# predict the test data
prediction = kNeigh.score(X_test, y_test)
print('accuracy: ' + prediction)