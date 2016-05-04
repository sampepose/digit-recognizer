import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from numpy.random import RandomState
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import fetch_mldata

class RUN_TYPE:
    hyperparams, validate, test = range(3)
    
runType = RUN_TYPE.test    
    
def find_hyperparams(lr, X, y):
    # Set the parameters by cross-validation
    param_grid = [{'C': [0.05 * x for x in range(1, 40)]}]
    grid = GridSearchCV(lr, param_grid, n_jobs=-1)
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

    lr = LogisticRegression(random_state=RS, solver='lbfgs', multi_class='multinomial', n_jobs=-1)
    print(find_hyperparams(lr, X, y))
else:
    # Optimal hyperparams found via cross-validation
    C = 0.35
    
    lr = LogisticRegression(C=C, random_state=RS, solver='lbfgs', multi_class='multinomial', n_jobs=-1)    
    
    if runType == RUN_TYPE.validate:
        lr.fit(X_train, y_train)
        print('Score of {}'.format(lr.score(X_test, y_test)))
    else:
        lr.fit(X_train, y_train)
            
        acc = lr.score(X_test, y_test)
        
        print("acc: ", acc)