import numpy as np

from layers import Layer

class DenseNetwork:

    def __init__(self, input_n, input_val, activation):

        self.input_layer = Layer.input(input_n, input_val, activation)

        self.layers = [self.input_layer,]

    def add(self, n, activation):

        layer = Layer.from_prev_layer(n, self.layers[-1], activation)

        self.layers.append(layer)

    def train(self):

        pass