# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:43:42 2016

@author: sampepose
"""

import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

data = []

# Read the training data
f = open('data/train.csv')
reader = csv.reader(f)
next(reader, None)
for row in reader:
    data.append(row)
f.close()
        
X = np.array([x[1:] for x in data])
y = np.array([x[0] for x in data])
del data # free up the memory
print('loaded training data')

validation_X = X[-12000:]
validation_y = y[-12000:]

X = X[:-12000]
y = y[:-12000]

x_plot = []
y_plot = []

maxK = 10
for k in range(1, maxK + 1):
    # Construct k-nearest neighbor classifier and 'fit' it
    kNeigh = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    kNeigh.fit(X, y)
            
    # predict the test data
    predict = kNeigh.predict(validation_X)

    correct = 0    
    for r in range(0, validation_y.shape[0]):
        if predict[r] == validation_y[r]:
            correct += 1
            
    x_plot.append(k)
    y_plot.append(100.0 * (correct / validation_y.shape[0]))

    print('finished k=',k)

print(x_plot)
print(y_plot)
plt.axis([1, maxK + 1, 85, 100])
plt.xlabel('k')
plt.ylabel('percent accuracy')
plt.scatter(x_plot, y_plot, marker='o')
plt.show()
