"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'weight': np.random.normal(loc = 0, scale=0.0001, size=(out_features,in_features)),\
                   'bias': np.zeros((1, out_features))}
    
    self.grads = {'weight': np.zeros((out_features,in_features)),\
                  'bias': np.zeros((1, out_features))}
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = np.matmul(x, self.params['weight'].transpose()) + self.params['bias']
    #basically store activations from from previous layers
    self.in_x = x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.grads['weight'] = np.matmul(dout.transpose(), self.in_x)
    self.grads['bias'] = np.sum(dout, axis=1, keepdims=True)
    dx = np.matmul(dout, self.params['weight'])
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.out = np.maximum(x,0)
    ########################
    # END OF YOUR CODE    #
    #######################

    return self.out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.dx = dout*(self.out>0)
    
    ########################
    # END OF YOUR CODE    #
    #######################    
    
    return self.dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = x
    u = np.max(self.x, axis=1, keepdims=True)
    z = np.exp(self.x-u)
    self.out = z/np.sum(z, axis=1, keepdims=True)
    ########################
    # END OF YOUR CODE    #
    #######################

    return self.out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    m,n = self.out.shape
    a = np.reshape(self.out, (m,n,1))
    e = np.ones(a.shape)
    identity = np.identity(n)
    identity = identity[np.newaxis,:,:]
    identity = np.repeat(identity,m,axis=0)

    da = np.matmul(a, e.transpose(0,2,1))*(identity-np.matmul(e, a.transpose(0,2,1)))
    dout = np.reshape(dout, (m,n,1))
    self.dx = np.matmul(da,dout).reshape(m,n)
    ########################
    # END OF YOUR CODE    #
    #######################

    return self.dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.out = -1*y*np.log(x)
    #keep it for unittest
    self.out = np.sum(self.out, axis=1, keepdims=True)

    ########################
    # END OF YOUR CODE    #
    #######################
    return self.out.mean()

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.dx = -1*y/x
    ########################
    # END OF YOUR CODE    #
    #######################

    return self.dx/x.shape[0]
