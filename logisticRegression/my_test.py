import csv
import numpy as np
from logistic_regression import LogisticRegression
from numpy.random import RandomState
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata("MNIST original")
print('Fetched data')

# use the traditional train/test split
X, y = mnist.data / 255.0, mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

lr = LogisticRegression(C=0.35)
lr.fit(X_train, y_train, 10)
guesses = lr.predict(X_test)

score = 0.0
for g in range(guesses.shape[0]):
    if guesses[g] == y_test[g]:
        score += 1
        
print('Score: ', score / len(guesses))