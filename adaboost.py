import pdb
import numpy as np
from sklearn.externals.six.moves import zip
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import cross_val_score
import pylab as pl
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

''' This script is used for finding best parameters of adaboost classifier and
plotting the relationship between error rate on the testing data and number of
trees.
'''

X_train, y_train = load_svmlight_file('data/vehicle_train.scale')
X_test, y_test = load_svmlight_file('data/vehicle_test.scale')
X_train = X_train.toarray()
X_test = X_test.toarray()
n_features = X_train.shape[1]


# Adaboost classifier using SAMME.R
bdt_real = AdaBoostClassifier(
    #    SVC(C=44, kernel='linear', probability=True),
     RandomForestClassifier(),
   # DecisionTreeClassifier(max_depth=2),
   # n_estimators=num_estimators,
    learning_rate=1)

# Adaboost classifier using SAMME
bdt_discrete = AdaBoostClassifier(
   # DecisionTreeClassifier(max_depth=2),
    RandomForestClassifier(),
   #SVC(C=44, kernel='linear', probability=True),
  #  n_estimators=num_estimators,
    learning_rate=1,
    algorithm="SAMME")

'''
# Find best value of n_estimator using grid search
param_grid = {"n_estimators" : [1, 10, 50, 100, 200, 300, 400, 500, 600]}
grid_search = GridSearchCV(bdt_real, param_grid=param_grid, cv=10)
grid_search.fit(X_train, y_train)
print 'Best parameters of Adaboost SAMME.R:' , grid_search.best_params_
print 'Best scrore of Adaboost SAMME.R:', grid_search.best_score_

pdb.set_trace()

grid_search = GridSearchCV(bdt_discrete, param_grid=param_grid, cv=10)
grid_search.fit(X_train, y_train)
print 'Best parameters of Adaboost SAMME:' , grid_search.best_params_
print 'Best scrore of Adaboost SAMME:', grid_search.best_score_

pdb.set_trace()
'''

# Train on the training data set
num_estimators = 600;

bdt_real.set_params(n_estimators=num_estimators)
bdt_discrete.set_params(n_estimators=num_estimators)

bdt_real.fit(X_train, y_train)
bdt_discrete.fit(X_train, y_train)

real_test_errors = []
discrete_test_errors = []

# Test on the testing data set and display the accuracies
ypred_r = bdt_real.predict(X_test)
ypred_e = bdt_discrete.predict(X_test)
print 'Accuracy of SAMME.R = ', accuracy_score(ypred_r, y_test)
print 'Accuracy of SAMME = ', accuracy_score(ypred_e, y_test)

# Plot the relationship between error rates and number of trees
for real_test_predict, discrete_train_predict in zip(
        bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):
    real_test_errors.append(
        1. - accuracy_score(real_test_predict, y_test))
    discrete_test_errors.append(
        1. - accuracy_score(discrete_train_predict, y_test))


n_trees = xrange(1, num_estimators + 1)

pl.figure(figsize=(15, 5))

pl.subplot(131)
pl.plot(xrange(1, len(discrete_test_errors)+1), discrete_test_errors, c='black', label='SAMME')
pl.plot(xrange(1, len(real_test_errors)+1), real_test_errors, c='black',\
        linestyle='dashed', label='SAMME.R')
pl.legend()
pl.ylim(0.18, 0.62)
pl.ylabel('Test Error')
pl.xlabel('Number of Trees')

'''
pl.subplot(132)
pl.plot(n_trees, bdt_discrete.estimator_errors_, "b", label='SAMME', alpha=.5)
pl.plot(n_trees, bdt_real.estimator_errors_, "r", label='SAMME.R', alpha=.5)
pl.legend()
pl.ylabel('Error')
pl.xlabel('Number of Trees')
pl.ylim((.2,
        max(bdt_real.estimator_errors_.max(),
            bdt_discrete.estimator_errors_.max()) * 1.2))
pl.xlim((-20, len(bdt_discrete) + 20))
'''

'''
pl.subplot(133)
pl.plot(n_trees, bdt_discrete.estimator_weights_, "b", label='SAMME')
pl.legend()
pl.ylabel('Weight')
pl.xlabel('Number of Trees')
pl.ylim((0, bdt_discrete.estimator_weights_.max() * 1.2))
pl.xlim((-20, len(bdt_discrete) + 20))
'''

# prevent overlapping y-axis labels
pl.subplots_adjust(wspace=0.25)
pl.show()


