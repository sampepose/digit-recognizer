import csv
import numpy as np
from sklearn.svm import SVC
from numpy.random import RandomState
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import fetch_mldata

class RUN_TYPE:
    hyperparams, validate, test = range(3)
    
runType = RUN_TYPE.validate    
    
def find_hyperparams(svc, X, y):
    # Set the parameters by cross-validation
    param_grid = [{'kernel': ['rbf'], 'gamma': [2**x for x in range(-15, 4)], 'C': [2**x for x in range(-5, 16)]},
              {'kernel': ['sigmoid'], 'gamma': [2**x for x in range(-15, 4)], 'C': [2**x for x in range(-5, 16)]}]
    grid = GridSearchCV(svc, param_grid, n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_

mnist = fetch_mldata("MNIST original")
print('Fetched data')

# use the traditional train/test split
X, y = mnist.data / 255.0, mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

RS = RandomState() 

if runType == RUN_TYPE.hyperparams:
    # use less data or the process will outlive us    
    X = X_train[1:1500, :]
    y = y_train[1:1500]

    lr = SVC(random_state=RS, cache_size=8192)
    print(find_hyperparams(lr, X, y))
else:
    # Optimal hyperparams found via cross-validation
    kernel = 'rbf'
    gamma = 0.03125
    C = 3.0
    
    lr = SVC(C=C, kernel=kernel, gamma=gamma, random_state=RS, cache_size=8192)    
    
    if runType == RUN_TYPE.validate:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RandomState())
        lr.fit(X_train, y_train)
        print('Score of {}'.format(lr.score(X_test, y_test)))
    else:
        lr.fit(X_train, y_train)
        
        acc = lr.score(X_test, y_test)
        
        print("acc: ", acc)