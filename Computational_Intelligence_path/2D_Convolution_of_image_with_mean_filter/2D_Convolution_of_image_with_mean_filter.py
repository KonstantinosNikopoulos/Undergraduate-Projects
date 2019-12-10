#!/usr/bin/env python
# coding: utf-8


#  2D convolution of image with mean filter

#  author: Konstantinos Nikopoulos

import numpy as np
import cv2
import os


# 2D Convolution
def myConv2(A, B):
    Ah, Aw = A.shape
    Bh, Bw = B.shape
    if (Bh != Bw):
        raise ValueError("Passed kernel is not of the right shape")
    Ch = Ah + Bh - 1 # height of result
    Cw = Aw + Bw - 1 # width of result
    x = (Bh-1) // 2
    y = (Bw-1) // 2
    B = np.flipud(np.fliplr(B)) # flip the kernel
    C = np.zeros((Ch, Cw)) # result 
    A_padded = np.zeros((Ch, Cw)) # add zero padding 
    A_padded[x : Ah + x, y : Aw + y] = A
    for i in range(x, Ch - x): 
        for j in range(y, Cw - y):
            sum = 0
            for n in range(0, Bh):
                for m in range(0, Bw):
                    sum = sum + A_padded[n + i - x, m + j - y] * B[n, m]
            C[i,j] = sum 
    return C


# Add gaussian noise
def gaussian_noise(A):
    row,col = A.shape
    mean = 0.0
    var = 1000
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    noisy_image = np.zeros(A.shape, np.float32)
    noisy_image = A + gauss
    return noisy_image
    


image = cv2.imread('test.jpg', 0)  # read image - black and white

noisy = gaussian_noise(image) # add gaussian noise

cv2.imwrite('noisy.png', noisy)  # save image to disk

kernel = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]) # mean filter

filtered = myConv2(noisy, kernel) # apply mean filter
cv2.imwrite('filtered.png', filtered)  # save image to disk






