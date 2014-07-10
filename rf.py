import pdb
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import pylab as pl

""" This script is used for finding best parameters of random forest classifier
and ploting feature importance figure
"""

X_train, y_train = load_svmlight_file('data/vehicle_train.scale')
X_test, y_test = load_svmlight_file('data/vehicle_test.scale')
X_train = X_train.toarray()
X_test = X_test.toarray()
n_features = X_train.shape[1]

# Cross validate random forest classifier using best parameters found
rf = RandomForestClassifier(n_estimators=14, max_depth=None,\
        min_samples_split=1, min_samples_leaf=1, criterion='entropy',
        max_features=3, random_state=None, compute_importances=True)

scores = cross_val_score(rf, X_train, y_train, cv=10)
print '10-fold cross validation accuracy: ', scores.mean()

'''
# Find best parameters using grid search
rf = RandomForestClassifier(n_estimators=14,  compute_importances=True)
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

grid_search = GridSearchCV(rf, param_grid=param_grid, cv=10)
grid_search.fit(X_train, y_train)
print 'Best parameters:' , grid_search.best_params_
print 'Best scrore:', grid_search.best_score_
'''

# Train random forest classifier and compute feature importances
rf = rf.fit(X_train, y_train)
importances = rf.feature_importances_
f_indices = np.argsort(importances)[::-1]
std = np.std([tree.feature_importances_ for tree in rf.estimators_],\
                     axis=0)

# Make prediction on test data
ypred = rf.predict(X_test)
acc = rf.score(X_test, y_test)
print 'Accuracy on test data: ', acc

# Plot feature importances
pl.figure()
pl.title("Feature importances")
pl.bar(range(n_features), importances[f_indices],
               color="r", yerr=std[f_indices], align="center")
pl.xticks(range(n_features), f_indices)
pl.xlim([-1, n_features])
pl.show()
