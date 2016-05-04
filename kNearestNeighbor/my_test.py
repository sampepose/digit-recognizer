# -*- coding: utf-8 -*-

import csv
import numpy as np
from nearest_neighbor import NearestNeighbor as NearestN
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata("MNIST original")
print('Fetched data')

# use the traditional train/test split
X, y = mnist.data, mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Construct k-nearest neighbor classifier and 'fit' it
nn = NearestN()
nn.train(X_train, y_train)

# predict the test data
predict = nn.predict(X_test)

correct = 0.0
count = len(y_test)
accuracy = 0.0
for i, p in enumerate(predict):
	if p == y_test[i]
		correct += 1

print('accuracy:', float(correct) / count)