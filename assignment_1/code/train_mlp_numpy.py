"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = 'cifar10/cifar-10-batches-py'

FLAGS = None

def evaluate(model, cifar10, batch_size):
  x,y = cifar10['test'].images , cifar10['test'].labels
  updates = 0
  acc=0
  for iter_ in np.arange(y.shape[0]/batch_size):
    batch_x, batch_y = cifar10['test'].next_batch(batch_size)
    
    #reshape x into vectors
    batch_x = np.reshape(batch_x, (200, 3072))
    predictions = model.forward(batch_x)
    
    acc += accuracy(predictions, batch_y)
    updates+=1
  print("test accuracy: {}".format("%.3f"%(acc/updates)))
  return  acc/updates



def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  pred_index = np.argmax(predictions,axis=1)
  target_index = np.argmax(targets, axis=1)
  correct = np.count_nonzero(np.equal(pred_index,target_index))
  accuracy = correct/pred_index.size
    ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  #create MLP model
  net = MLP(3072, dnn_hidden_units, 10)
  loss = CrossEntropyModule()

  #Load cifar10
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  print()
  print()
  print("----------------------------------------------")
  print("\t \t Training")
  print("----------------------------------------------\n")
  pl_loss =[]
  average_loss =[]
  moving_average=0.0
  acc =[]
  for iter_ in np.arange(0, FLAGS.max_steps):

    #Load batches 
    x , y = cifar10['train'].next_batch(FLAGS.batch_size)

    #reshape x into vectors
    x = np.reshape(x, (200, 3072))
    
    #perform forward pass
    out = net.forward(x)
    iter_loss = loss.forward(out, y)

    pl_loss.append(iter_loss)
    moving_average+=iter_loss
    average_loss.append(np.mean(np.mean(pl_loss[:-100:-1])))

    print("iter: {}, training loss: {}".format(iter_, "%.3f"%iter_loss))
    
    # perform bckward pass
    
    dout = loss.backward(out, y)
    net.backward(dout)
    
    # update weights and biases
    net.sgd_update(FLAGS.learning_rate, FLAGS.batch_size)

    if (iter_+1)%FLAGS.eval_freq==0:
      print("\n--------Testing-------- \n")
      acc.append(evaluate(net, cifar10, FLAGS.batch_size))
    # break
        

  plt.plot(pl_loss,'r-', label="Batch loss", alpha=0.5)
  plt.plot(average_loss,'g-', label="Average loss", alpha=0.5)
  plt.legend()
  plt.xlabel("Iterations")
  plt.ylabel("Loss")
  plt.title("Training Loss")
  plt.grid(True)
  plt.show()
  plt.close()

  plt.plot(acc,'g-', alpha=0.5)
  plt.xlabel("Iterations")
  plt.ylabel("Accuracy")
  plt.title("Test Accuracy")
  plt.grid(True)
  plt.show()
  plt.close()

  print()
  print("TRAINING COMPLETED")


  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()