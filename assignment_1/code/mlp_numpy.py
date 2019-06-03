"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
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
    self.n_layers = len(n_hidden)
    
    # ---------------Initialize all the weight and biases-------------------
    # includes input layer, hidden_layer, output_layer

    self.Linear_in = LinearModule(n_inputs, n_hidden[0]) #from input to first hidden layer

    #intermediate hidden layers
    self.Linear_inter = [LinearModule(in_, out_) for in_, out_ in \
                                              zip(n_hidden[:-1], n_hidden[1:])]
    #from last hidden layer to ouput layer
    self.Linear_out = LinearModule(n_hidden[-1], n_classes)
    #---------------------End--------------
    self.ReLU = ReLUModule()
    self.softmax = SoftMaxModule()

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
    out = self.Linear_in.forward(x) 
    # print("First Input Linear Layer: {}".format(out.shape))
    out = self.ReLU.forward(out)
    # print("Apply Relu; {}".format(out.shape))
    #loop in through hidden_layers
    if len(self.Linear_inter)!=0:

      for modules in self.Linear_inter:
          out = modules.forward(out)
          out = self.ReLU.forward(out)

    out = self.Linear_out.forward(out) 
    # print("Last hidden to output: {}".format(out.shape))
    out = self.softmax.forward(out)
    # print("softmax output".format(out.shape))  
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dout = self.softmax.backward(dout)
    dout = self.Linear_out.backward(dout)
    
    if len(self.Linear_inter)!=0:
      for idx, _ in enumerate(self.Linear_inter):
          dout = self.ReLU.backward(out)
          dout = self.Linear_inter[idx].backward(out)
    
    dout = self.ReLU.backward(dout)
    dout = self.Linear_in.backward(dout)
    ########################
    # END OF YOUR CODE    #
    #######################

    return 
  
  def sgd_update(self, lr, batch_size):

    
    self.Linear_in.params['weight'] = self.Linear_in.params['weight'] -\
                                        (lr)*self.Linear_in.grads['weight']
    self.Linear_in.params['bias'] = self.Linear_in.params['bias'] -\
                                        (lr)*self.Linear_in.grads['bias']
    if len(self.Linear_inter)!=0:

      for idx, _ in enumerate(self.Linear_inter):
          
          self.Linear_inter[idx].params['weight'] = self.Linear_inter[idx].params['weight']-\
                                       (lr)*self.Linear_inter[idx].grads['weight']

          self.Linear_inter[idx].params['bias'] = self.Linear_inter[idx].params['bias']-\
                                       (lr)*self.Linear_inter[idx].grads['bias']

    self.Linear_out.params['weight'] = self.Linear_out.params['weight'] -\
                                          (lr)*self.Linear_out.grads['weight']
    
    self.Linear_out.params['bias'] = self.Linear_out.params['bias'] -\
                                          (lr)*self.Linear_out.grads['bias']

    return