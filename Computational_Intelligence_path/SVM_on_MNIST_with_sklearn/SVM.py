#!/usr/bin/env python
# coding: utf-8

Author: Konstantinos Nikopoulos


import keras
import numpy as np
import datetime as dt
from keras.datasets import mnist
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

################################ Preprocess dataset ###########################################

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("shape of data:", x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print("shape of labels:", y_train.shape, y_test.shape) #(60000,) (10000,) 

# Reshape the arrays
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Make the values float so we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize the RGB codes by dividing it to the max RGB value
x_train /= 255
x_test /= 255

# Scaling to [-1,1]
x_train = 2 * x_train - 1
x_test = 2 * x_test - 1

# Change labels to 0 and 1 for even and odd
y_train = y_train % 2
y_test = y_test % 2

print("new shape of data:", x_train.shape, x_test.shape) 
print("new shape of labels:", y_train.shape, y_test.shape) 

# PCA for reduction of dimensions (sklearn)
pca = PCA(n_components=100)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

print("new shape of data:", x_train.shape, x_test.shape) 
print("new shape of labels:", y_train.shape, y_test.shape) 


##################################### SVM ############################################

# All models

clf = svm.SVC(C=1, kernel='linear')
clf.fit(x_train, y_train)
x= clf.predict(x_test)
print("kernel: linear, C=1:")
print (accuracy_score(x, y_test))

clf = svm.SVC(C=10, kernel='linear')
clf.fit(x_train, y_train)
x= clf.predict(x_test)
print("kernel: linear, C=10:")
print (accuracy_score(x, y_test))

clf = svm.SVC(C=100, kernel='linear')
clf.fit(x_train, y_train)
x= clf.predict(x_test)
print("kernel: linear, C=100:")
print (accuracy_score(x, y_test))

clf = svm.SVC(C=1, kernel='poly', degree= 2)
clf.fit(x_train, y_train)
x= clf.predict(x_test)
print("kernel: polynomial, degree=2, C=1:")
print (accuracy_score(x, y_test))

clf = svm.SVC(C=10, kernel='poly', degree= 2)
clf.fit(x_train, y_train)
x= clf.predict(x_test)
print("kernel: polynomial, degree=2, C=10:")
print (accuracy_score(x, y_test))

clf = svm.SVC(C=100, kernel='poly', degree= 2)
clf.fit(x_train, y_train)
x= clf.predict(x_test)
print("kernel: polynomial, degree=2, C=100:")
print (accuracy_score(x, y_test))

clf = svm.SVC(C=1, kernel='poly', degree= 3)
clf.fit(x_train, y_train)
x= clf.predict(x_test)
print("kernel: polynomial, degree=3, C=1:")
print (accuracy_score(x, y_test))

clf = svm.SVC(C=10, kernel='poly', degree= 3)
clf.fit(x_train, y_train)
x= clf.predict(x_test)
print("kernel: polynomial, degree=3, C=10:")
print (accuracy_score(x, y_test))

clf = svm.SVC(C=100, kernel='poly', degree= 3)
clf.fit(x_train, y_train)
x= clf.predict(x_test)
print("kernel: polynomial, degree=3, C=100:")
print (accuracy_score(x, y_test))


# Grid search
param_grid = [
             {'kernel': ['poly'],
              'degree': [2, 3],
              'C': [1, 10, 100],
             },
             {'kernel': ['linear'],
              'C': [1, 10, 100]}
             ]
svr = svm.SVC()
grid = GridSearchCV(svr, param_grid, scoring='accuracy', cv=5)
grid.fit(x_train, y_train)
best_parameters = grid.best_params_ 
print(best_parameters)


##################################### Final SVM ############################################

# Apply best parameters to model
clf = svm.SVC(C=1, kernel='poly', degree= 3)
start_time = dt.datetime.now()
clf.fit(x_train, y_train)
end_time = dt.datetime.now() 
elapsed_time= end_time - start_time
x= clf.predict(x_test)
print("Final kernel: polynomial, degree=3, C=1:")
print (accuracy_score(x, y_test))
print('Elapsed learning {}'.format(str(elapsed_time)))








