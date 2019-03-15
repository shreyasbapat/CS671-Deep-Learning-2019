import numpy as np
import math

def relu(arr):
    #y = []
    #for i in range(np.size(arr)):
     #      y.append(max(0, arr[i]))
    #arr[arr<0] = 0

    #return np.array(y)
    arr[arr<0] = 0
    return arr

def sigmoid(arr):
    sigm = 1. / (1. + np.exp(-arr))
    return sigm

def softmax(arr):
    y = np.exp(arr)
    norm_factor = np.sum(y)
    y = (1 / norm_factor) * y
    return y

def cross_entropy(y, y_pred):
    index = np.argmax(y)
    return -1 * np.log(y_pred[index])
