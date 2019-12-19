#!/usr/bin/env python
# coding: utf-8

Author: Konstantinos Nikopoulos


import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU,LeakyReLU



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

# Encode the labels to one-hot vectors
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

print("new shape of data:", x_train.shape, x_test.shape) #(60000, 784) (10000, 784)
print("new shape of labels:", y_train.shape, y_test.shape) #(60000, 10) (10000, 10) 

# PCA for reduction of dimensions (sklearn)
pca = PCA(n_components=100)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
pca_std = np.std(x_train)

print("new shape of data:", x_train.shape, x_test.shape) #(60000, 100) (10000, 100)
print("new shape of labels:", y_train.shape, y_test.shape) #(60000, 10) (10000, 10)

############################### Train neural network ############################################

model = Sequential()

# First  hidden layer 
model.add(Dense(128, activation=LeakyReLU(0.01), input_dim=100))
# Second hidden layer
model.add(Dense(128, activation=LeakyReLU(0.01)))
# Output layer 
model.add(Dense(10, activation='softmax'))

# Optimizer
adam = Adam(lr=0.0001)
# Compile model
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

results = model.fit(x_train, y_train, epochs=50, batch_size=128, verbose=2, validation_data=(x_test, y_test))

print("Train accuracy: ", model.evaluate(x_train, y_train, batch_size=128))
print("Test accuracy: ", model.evaluate(x_test, y_test, batch_size=128))

############################### Evaluate neural network ############################################

y_out = model.predict(x_train)
#print(y_out[0], np.argmax(y_out[0]))

model.summary()

plt.figure(1)
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.legend(['train loss', 'test loss'])

plt.figure(2)
plt.plot(results.history['acc'])
plt.plot(results.history['val_acc'])
plt.legend(['train acc', ' acc'])






