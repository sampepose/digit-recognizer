# -*- coding: utf-8 -*-

import csv
import numpy as np
from nearest_neighbor import NearestNeighbor as NearestN

# Read the training data
f = open('data/train.csv')
reader = csv.reader(f)
next(reader, None) # skip header
data = [data for data in reader]
f.close()
        
X = np.asarray([x[1:] for x in data], dtype=np.int16)
y = np.asarray([x[0] for x in data], dtype=np.int16)
del data # free up the memory
print('loaded training data')

# Construct k-nearest neighbor classifier and 'fit' it
nn = NearestN()
nn.train(X, y)

# Read the test data
f = open('data/test.csv')
reader = csv.reader(f)
next(reader, None) # skip header
TestData = np.asarray([data for data in reader], dtype=np.int16)
f.close()
print('loaded test data')
        
# predict the test data
predict = nn.predict(TestData)
del TestData # free up the memory
print('predicted test data')

# write predictions to csv
with open('data/out-my-nearest-neighbor.csv', 'w') as writer:
    writer.write('"ImageId", Label\n')
    count = 0
    for p in predict:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')