#!/usr/bin/env python
# coding: utf-8

#Author: Konstantinos Nikopoulos

from __future__ import print_function
import keras
import datetime as dt
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score


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

# Convert one hot encoding
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

print("new shape of data:", x_train.shape, x_test.shape) 
print("new shape of labels:", y_train.shape, y_test.shape) 

# PCA for reduction of dimensions (sklearn)
pca = PCA(n_components=100)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

print("new shape of data:", x_train.shape, x_test.shape) 
print("new shape of labels:", y_train.shape, y_test.shape)


##################################### RBF ############################################

# K-means to find centers on train data
kmeans_model = KMeans(20) #find 20 centers, 2 for each digit
kmeans_model.fit(x_train)
centers = kmeans_model.cluster_centers_
x = kmeans_model.predict(kmeans_model.cluster_centers_)
x = keras.utils.to_categorical(x, 20)

# RBF layer
kernel = PairwiseKernel(gamma=0.1, metric= 'linear')
#kernel = PairwiseKernel(gamma=1, metric= 'linear')
#kernel = PairwiseKernel(gamma=10, metric= 'linear')
#kernel = PairwiseKernel(gamma=0.1, metric= 'poly') #degree=3
#kernel = PairwiseKernel(gamma=1, metric= 'poly')
#kernel = PairwiseKernel(gamma=10, metric= 'poly')
rbf_model = GaussianProcessRegressor(kernel=kernel, alpha=0.1).fit(centers, x)
temp1 = rbf_model.predict(x_train)
temp2 = rbf_model.predict(x_test)

# MLP
batch_size = 128
epochs = 30

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(20,)))
#model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()
adam=Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

history = model.fit(temp1, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(temp2, y_test))

score = model.evaluate(temp2, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])







