import numpy as np

def relu(arr):

    arr[arr<0] = 0

    return arr

def sigmoid(arr, derivative=True):
    sigm = 1. / (1. + np.exp(-arr))
    if derivative:
        return sigm * (1. - sigm)
    return sigm
