# CS 221 Neural Netork Implementation
'''
Erick Fidel Siavichay-Velasco |
This file contains a neural network implementation using PyTorch v1.1.0
'''

# Packages needed
import sys
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

class Neural_Network(nn.Module):
    def __init__(self, ):
        super(Neural_Network, self).__init__()
        # Parameters
        # Current state: 
        # 3 layer model; input layer with 300 neurons (number of features)
        # 1 hidden layer, with 3 neurons
        # 1 output layer, with 1 neuron
        self.inputSize = 300
        self.hiddenSize = 3
        self.outputSize = 1

        # Learning rate
        # bigger means faster convergence but less accuracy
        # smaller means slower convergence but more accuracy
        self.alpha = 0.1 

        
        # Thetas, randomly initialize
        self.W1 = torch.randn(self.hiddenSize, self.inputSize) # Should be (3x300)
        self.W2 = torch.randn(self.outputSize, self.hiddenSize) # Should be (1x3)
        
    # Performs forward propagation with input X        
    def forward(self, X):
        self.z1 = torch.matmul(self.W1, X) 
        self.a1 = self.sigmoid(self.z1) #Should be (3xNUM_FEATURES)
        self.z2 = torch.matmul(self.W2, self.a1)
        a2 =self.sigmoid(self.z2) # y hat, actual hypthesis
        return a2

        # a2 should be the prediction vector, and it should just be a scalar...not a vector
        # This is returning 1x3 for some reason. unclear what is happening to the matrix multiplication in forward()


        # if a2.item().item() <= 0.5:
        #     return 0
        # else:
        #     return 1
        

    # activation functions, can change later    
    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))
    
    def sigmoidPrime(self, s):
        return s * (1 - s)
   
    # Performs back prop 
    def backward(self, X, y, a2):
        #Debug statements: why aren't a2 and y same dimension???
        print("y size: ", y.size())
        print("a2 size: ", a2.size())

        self.a2_error = y - a2 # error in output
        self.a2_delta = self.a2_error * self.sigmoidPrime(a2) # derivative of sig to error
        self.a1_error = torch.matmul(torch.t(self.W2),self.a2_delta)
        self.a1_delta = self.a1_error * self.sigmoidPrime(self.a1)
        self.W1 += torch.matmul(self.a1_delta, torch.t(X))*(self.alpha)
        self.W2 += torch.matmul(self.a2_delta, torch.t(self.a1))*(self.alpha)
        
    def train(self, X, y):
        # forward + backward pass for training
        a2 = self.forward(X)
        self.backward(X, y, a2)
        
    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")
        
    def predict(self, xPredicted):
        print ("Predicted data based on trained weights: ")
        # print ("Input (scaled): \n" + str(xPredicted))
        print ("Prediction: \n" + str(self.forward(xPredicted)))
