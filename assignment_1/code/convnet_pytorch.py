"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet,self).__init__()

    self.conv1 = nn.Conv2d(n_channels,64,3,stride=1,padding=1)
    self.batchnorm1 = nn.BatchNorm2d(64)
    self.maxpool1 = nn.MaxPool2d(3,stride=2,padding=1)
    self.conv2 = nn.Conv2d(64,128,3,stride=1,padding=1)
    self.batchnorm2 = nn.BatchNorm2d(128)
    self.maxpool2 = nn.MaxPool2d(3,stride=2,padding=1)
    self.conv3_a = nn.Conv2d(128,256,3,stride=1,padding=1)
    self.batchnorm3_a = nn.BatchNorm2d(256)
    self.conv3_b = nn.Conv2d(256,256,3,stride=1,padding=1)
    self.batchnorm3_b = nn.BatchNorm2d(256)
    self.maxpool3 = nn.MaxPool2d(3,stride=2,padding=1)
    self.conv4_a = nn.Conv2d(256,512,3,stride=1,padding=1)
    self.batchnorm4_a = nn.BatchNorm2d(512)
    self.conv4_b = nn.Conv2d(512,512,3,stride=1,padding=1)
    self.batchnorm4_b = nn.BatchNorm2d(512)
    self.maxpool4 = nn.MaxPool2d(3,stride=2,padding=1)
    self.conv5_a = nn.Conv2d(512,512,3,stride=1,padding=1)
    self.batchnorm5_a = nn.BatchNorm2d(512)
    self.conv5_b = nn.Conv2d(512,512,3,stride=1,padding=1)
    self.batchnorm5_b = nn.BatchNorm2d(512)
    self.maxpool5 = nn.MaxPool2d(3,stride=2,padding=1)
    self.avgpool = nn.AvgPool2d(1,stride=1,padding=0)
    self.linear = nn.Linear(512,n_classes)

    self.ReLU = nn.ReLU()
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
    out = self.ReLU(self.batchnorm1(self.conv1(x)))
    out = self.maxpool1(out)
    out = self.ReLU(self.batchnorm2(self.conv2(out)))
    out = self.maxpool2(out)
    out = self.ReLU(self.batchnorm3_a(self.conv3_a(out)))
    out = self.ReLU(self.batchnorm3_b(self.conv3_b(out)))
    out = self.maxpool3(out)
    out = self.ReLU(self.batchnorm4_a(self.conv4_a(out)))
    out = self.ReLU(self.batchnorm4_b(self.conv4_b(out)))
    out = self.maxpool4(out)
    out = self.ReLU(self.batchnorm5_a(self.conv5_a(out)))
    out = self.ReLU(self.batchnorm5_b(self.conv5_b(out)))
    out = self.maxpool5(out)
    out = self.avgpool(out)
    out = self.linear(torch.squeeze(out))
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
