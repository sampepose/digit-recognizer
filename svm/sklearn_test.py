import csv
import numpy as np
from sklearn.svm import SVC
from numpy.random import RandomState
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

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

RS = RandomState() 

if runType == RUN_TYPE.hyperparams:
    # use less data or the process will outlive us    
    X = X[1:1500, :]
    y = y[1:1500]

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
        lr.fit(X, y)
        
        # Read the test data
        f = open('../data/test.csv')
        reader = csv.reader(f)
        next(reader, None) # skip header
        TestData = np.asarray([data for data in reader], dtype=np.int16)
        f.close()
        print('loaded test data')
        
        TestData = np.true_divide(TestData, 255);
        
        predict = lr.predict(TestData)
        
        # write predictions to csv
        with open('out/out-sklearn.csv', 'w') as writer:
            writer.write('"ImageId", Label\n')
            count = 0
            for p in predict:
                count += 1
                writer.write(str(count) + ',"' + str(p) + '"\n')