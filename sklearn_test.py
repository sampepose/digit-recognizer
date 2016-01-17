# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:43:42 2016

@author: sampepose
"""

import csv
from sklearn.neighbors import KNeighborsClassifier

data = []
TestData = []

# Read the training data
f = open('data/train.csv')
reader = csv.reader(f)
next(reader, None)
for row in reader:
    data.append(row)
f.close()
        
X = [x[1:] for x in data]
y = [x[0] for x in data]
del data # free up the memory
print('loaded training data')

# Construct k-nearest neighbor classifier and 'fit' it
kNeigh = KNeighborsClassifier(n_neighbors=5, n_jobs=1, algorithm='kd_tree') # use all the CPU cores in parallel!
kNeigh.fit(X, y)

# Read the test data
f = open('data/test.csv')
reader = csv.reader(f)
next(reader, None)
for row in reader:
    TestData.append(row)
f.close()
print('loaded test data')
        
# predict the test data
predict = kNeigh.predict(TestData)
del TestData # free up the memory
print('predicted test data')

# write predictions to csv
with open('data/out-sklearn-nearest-neighbor.csv.csv', 'w') as writer:
    writer.write('"ImageId", "Label"\n')
    count = 0
    for p in predict:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')