import numpy as np

from layers import Layer

class DenseNetwork:

    def __init__(self, activation):

        self.layers = [] # list containg layer objects
        self.num_layers = 0 # number of layers in network
    
       
    def func_derivative(a, activation):
           if activation == 'sigmoid':
                  sig_value = 1. / (1. + np.exp(-a))
                  return sig_value * (1 - sig_value)
           elif activation == 'relu':
                  if a >= 0:
                         return 1
                  else:
                         return 0
           else:
                  raise NotImplementedError("Not implemented as of now!")
       
       
    def derivative(vec, activation):
           derivative_vec = vec
           for i in range(np.size(vec)):
                  derivative_vec[i] = func_derivative(vec[i], activation)
           return derivative_vec
       
       
    def forward_prop(self):
           #
           
    def back_prop(self, x, y, learning_rate):
           output_layer = self.layers[self.num_layers-1]
           output_dim = output_layer.n
           y_pred = output_layer.values
           e_y = np.zeros((output_dim, 1))
           e_y[y-1] = 1 # making one-hot-encoding
           output_gradient = -1 * (e_y - y_pred)   
           gradient = output_gradient # gradient for the last layer 
           for i in range(1, self.num_layers):
                  
                  # layer objects stored in layers of DenseNetwork
                  prev_layer = self.layers[self.num_layers-i-1] # object of previous layer
                  prev_values = prev_layer.values # output values in previous layer
                  prev_input = prev_layer._prev_val # input values in previous layer
                  curr_layer = self.layers[self.num_layers-i] # current layer
                  curr_weights = curr_layer.weights # current layer weights
                  
                  # gradients needed to update parameters
                  weight_gradient = np.outer(gradient, prev_values)
                  bias_gradient = gradient
                  
                  # calculation for calculating gradients for next iteration
                  pre_activate_gradient = np.transpose(curr_weights).dot(gradient)
                  derivative_vec = derivative(prev_input, activation)
                  gradient = np.multiply(pre_activate_gradient, derivative_vec)
                  
                  # update the weights and bias in current layer
                  curr_layer.update_params(weight_gradient, bias_gradient, learning_rate)
    
    
    def cross_entropy(self, y, y_pred):
        return -1 * np.log(y_pred[y-1])


    def input(self, dim, activation='relu'):
        input_n = dim[1] * dim[0]
        self.input_layer = Layer.input(input_n, activation)
        self.layers.append(self.input_layer)


    def add(self, n, activation):

        layer = Layer.from_prev_layer(n, self.layers[-1], activation)

        self.layers.append(layer)
        self.num_layers += 1


    def fit(self, x_train, y_train, max_epochs, error_tol, learning_rate):
        
        counter = 0
        self.initialize() # initialize all weights and biases of all layers
        for i in range(max_epochs):
               index = 0
               loss_func = 0
               # stochastic gradient descent with batch size = 1
               for x in x_train:
                      y = y_train[index]
                      self.forward_prop()
                      self.back_prop(x, y, learning_rate)
                      loss_func += cross_entropy(y, y_pred)
                      print("iter no. : %d, loss func: %f" % (counter, loss_func))
                      index += 1
        