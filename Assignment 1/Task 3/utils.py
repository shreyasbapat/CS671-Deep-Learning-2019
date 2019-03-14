import numpy as np

def relu(arr):

    arr[arr<0] = 0

    return arr

def sigmoid(arr):
    sigm = 1. / (1. + np.exp(-arr))
    return sigm

def softmax(arr):
    y = np.exp(arr)
    norm_factor = np.sum(y)
    return norm_factor * y

def cross_entropy(y, y_pred):
    return -1 * np.log(y_pred[y - 1])
