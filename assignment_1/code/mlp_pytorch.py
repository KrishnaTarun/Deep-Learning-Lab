"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    super(MLP, self).__init__()
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_classes = n_classes

    
    # includes input layer, hidden_layer, output_layer

    self.Linear_in = nn.Linear(n_inputs, n_hidden[0]) #from input to first hidden layer
    self.Linear_in.weight.data.normal_(mean=0, std=0.0001)
    self.Linear_in.bias.data.zero_()
    
    self.batchnorm1 = nn.BatchNorm1d(n_hidden[0])
    
    self.Linear1 = nn.Linear(n_hidden[0], n_hidden[1])
    self.Linear1.weight.data.normal_(mean=0, std=0.0001)
    self.Linear1.bias.data.zero_()
    self.batchnorm2 = nn.BatchNorm1d(n_hidden[1])

    self.Linear2 = nn.Linear(n_hidden[1], n_hidden[-1])
    self.Linear2.weight.data.normal_(mean=0, std=0.0001)
    self.Linear2.bias.data.zero_()
    self.batchnorm3 = nn.BatchNorm1d(n_hidden[2]) 

    self.Linear_out = nn.Linear(n_hidden[-1], n_classes)
    self.Linear_out.weight.data.normal_(mean=0, std=0.0001)
    self.Linear_out.bias.data.zero_() 

    self.Relu = nn.ReLU()     
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = self.Linear_in(x)
    out = self.Relu(self.batchnorm1(out))
    out = self.Linear1(out)
    out = self.Relu(self.batchnorm2(out))
    out = self.Linear2(out)
    out = self.Relu(self.batchnorm3(out))
    out = self.Linear_out(out)

    ########################
    # END OF YOUR CODE    #
    #######################

    return out
