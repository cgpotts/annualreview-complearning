#!/usr/bin/env python

"""
Implements the simple supervised learning example from table 3 of the
paper. Use

python distributed.py

to run the example from the paper. Some additional examples are
included as well.
"""


__author__ = "Christopher Potts and Percy Liang"
__credits__ = []
__license__ = "GNU general public license, version 2"
__version__ = "2.0"
__maintainer__ = "Christopher Potts"
__email__ = "See the authors' websites"


import sys
import random
import math
import numpy
from numpy import array, matrix, dot, outer
from copy import deepcopy
from random import shuffle


def randfloat(lower=-0.5, upper=0.5):
    """Return a random value x such that `lower <= x <= upper`"""
    return random.uniform(lower, upper)

def randmatrix(m, n, lower=-0.5, upper=0.5):
    """Return an `m` x `n` matrix of random values of `x`
    such that `lower <= x <= upper`"""
    vals = numpy.array([randfloat(lower, upper) for i in range(m*n)])
    return vals.reshape(m, n)

def sigmoid(z):
    """Inverse logistic function; scales all values to `0 <= x <= 1`"""
    return 1.0 / (1.0 + numpy.exp(-z))

def sigmoid_prime(z):
    """Derivative of the inverse logistic"""
    return z * (1.0 - z)

def tanh(z):
    """Hyperbolic tangent function; scales all values to `-1 <= x <= 1`"""
    return numpy.tanh(z)

def tanh_prime(z):
    """Derivative of the hyperbolic tangent"""
    return 1.0 - z**2

class ShallowNeuralNetwork:
    def __init__(self,
            input_dim=0,
            hidden_dim=0,
            output_dim=0,
            activation_func=tanh,
            activation_func_prime=tanh_prime):
        self.input_dim = input_dim + 1 # +1 for the bias, in final position
        self.hidden_dim = hidden_dim + 1 # +1 for the bias, in final position
        self.output_dim = output_dim
        self.activation_func = activation_func
        self.activation_func_prime = activation_func_prime
        self.input_layer = numpy.ones(self.input_dim)                                            
        self.hidden_layer = numpy.ones(self.hidden_dim)        
        self.output_layer = numpy.ones(self.output_dim)
        # input weights ignore the bias in final position:
        self.input_weights = randmatrix(self.input_dim, self.hidden_dim-1)      
        self.output_weights = randmatrix(self.hidden_dim, self.output_dim)
        self.output_errors = numpy.zeros(self.output_dim)
        self.input_errors = numpy.zeros(self.input_dim)
        
    def forward_propagation(self, inputs):
        # ignore the bias in final position:
        self.input_layer[ :-1] = inputs
        # ignore the bias in final position:
        self.hidden_layer[ : -1] = self.activation_func(dot(self.input_layer, self.input_weights)) 
        self.output_layer = self.activation_func(dot(self.hidden_layer, self.output_weights))
        return deepcopy(self.output_layer)
        
    def backward_propagation(self, labels, alpha=0.2):
        labels = array(labels)       
        self.output_errors = (labels - self.output_layer) * \
          self.activation_func_prime(self.output_layer)
        self.hidden_errors = dot(self.output_errors, self.output_weights.T) * \
          self.activation_func_prime(self.hidden_layer)
        self.output_weights += alpha * outer(self.hidden_layer, self.output_errors)
        # ignore the bias in final position:
        self.input_weights += alpha * outer(self.input_layer, self.hidden_errors[:-1]) 
        error = sum(0.5 * (labels - self.output_layer)**2)
        return error

    def train(self,
            training_data,
            maxiter=5000,
            alpha=0.5,
            epsilon=1.5e-8,
            display_progress=True):       
        iteration = 0
        error = sys.float_info.max
        while error > epsilon and iteration < maxiter:            
            error = 0.0
            shuffle(training_data)
            for ex, labels in training_data:
                self.forward_propagation(ex)
                error += self.backward_propagation(labels, alpha=alpha)           
            if display_progress:
                sys.stderr.write('\r')
                sys.stderr.write('Error at iteration {}: {}'.format(iteration, error))
                sys.stderr.flush()                
            iteration += 1
        if display_progress:
            sys.stderr.write('\n')
        
    def predict(self, inputs):
        self.forward_propagation(inputs)
        return deepcopy(self.output_layer)
        
    def hidden_representation(self, inputs):
        self.forward_propagation(inputs)
        return self.hidden_layer
   
######################################################################
######################################################################
    
if __name__ == '__main__':

    def generic_demo(training_data):
        net = ShallowNeuralNetwork(input_dim=len(training_data[0][0]),
                                   hidden_dim=2,
                                   output_dim=len(training_data[0][1]),
                                   activation_func=sigmoid,
                                   activation_func_prime=sigmoid_prime)
        net.train(training_data, maxiter=5000, display_progress=True)
        print('Inputs', 'Gold', 'Predicted')
        for inputs, labels in training_data:
           print(inputs, labels, net.predict(inputs), net.hidden_representation(inputs))        
        print()
        print('Input weights')
        print(net.input_weights)
        print()
        print('Output weights')
        print(net.output_weights)
        
    def boolean_xor():
         training_data = [            
            ([1,1], [0]),
            ([1,0], [1]),
            ([0,1], [1]),
            ([0,0], [0])
            ]
         print('XOR')
         generic_demo(training_data)

    def boolean_iff():
         training_data = [            
            ([1,1], [1]),
            ([1,0], [0]),
            ([0,1], [0]),
            ([0,0], [1])
            ]
         print('IFF')
         generic_demo(training_data)

    def exactly_one():
         training_data = [            
            ([1,1,1], [0]),
            ([1,1,0], [0]),
            ([1,0,1], [0]),
            ([0,0,0], [0]),
            ([0,1,1], [0]),
            ([1,0,0], [1]),
            ([0,1,0], [1]),
            ([0,0,1], [1]),            
            ]
         print('Exactly one (xor over three terms)')
         generic_demo(training_data)

    def modified_nouns():
        rollercoaster = array([1.0,1.0])        
        textbook =      array([0.0,1.0])
        airplane =      array([1.0,0.0])
        movie =         array([0.0,0.0])
        
        training_data = [
            (movie,         [1]),
            (textbook,      [0]),
            (airplane,      [0]),
            (rollercoaster, [1]),            
            ]
        print('Modified nouns')
        generic_demo(training_data)  
        

        
    
    #boolean_xor()
    #boolean_iff()
    #exactly_one()
    modified_nouns()

                
                
                
        
        
        
