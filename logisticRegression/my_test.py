import csv
import numpy as np
from logistic_regression import LogisticRegression
from numpy.random import RandomState
from sklearn.cross_validation import train_test_split

# Read the training data
f = open('../data/train.csv')
reader = csv.reader(f)
next(reader, None) # skip header
data = [data for data in reader]
f.close()
        
X = np.asarray([x[1:] for x in data], dtype=np.int16)
y = np.asarray([x[0] for x in data], dtype=np.int16)

X = np.true_divide(X, 255); # normalize image data to 0-1

del data # free up the memory
print('loaded training data')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RandomState())
    
lr = LogisticRegression(C=0.35)
lr.fit(X_train, y_train, 10)
guesses = lr.predict(X_test)

score = 0.0
for g in range(guesses.shape[0]):
    if guesses[g] == y_test[g]:
        score += 1
        
print('Score: ', score / len(guesses))