"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

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
  correct = np.count_nonzero(np.equal(pred_index,target_index),axis=0)
  accuracy = correct/targets.shape[0]
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)


  ########################
  # PUT YOUR CODE HERE  #
  #######################
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)
  print()
  cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)
  train = cifar10['train']

  input_channels = 3
  output_class = 10

  net = ConvNet(input_channels, output_class)
  net.to(device)



  if OPTIMIZER_DEFAULT == 'ADAM':
    optimizer = torch.optim.Adam(net.parameters(),lr=FLAGS.learning_rate)
  CrossEntropyLoss = nn.CrossEntropyLoss()  
  loss_iter = []
  loss_mean = []
  acc_iter = []

  loss_sum = 0.0

  for iter_n in np.arange(0,FLAGS.max_steps):
    x, labels = train.next_batch(FLAGS.batch_size)
    x = torch.tensor(x).to(device)
    
    labels = np.argwhere(labels>0)
    labels = torch.from_numpy(labels[:,1]).to(device)

    optimizer.zero_grad()

    net.train()
    
    predictions = net.forward(x)
    loss = CrossEntropyLoss(predictions,labels.long())

    loss_iter.append(loss.item())
    loss_sum += loss.item()
    loss_mean.append(np.mean(loss_iter[:-50:-1]))

    loss.backward()
    optimizer.step()
    

    print("Iter: {}, training Loss: {}".format(iter_n, "%.3f"%loss.item()))
    
    if (iter_n+1) % int(FLAGS.eval_freq) == 0:
      print("....Testing.....\n")
      test = cifar10['test']

      acc = []
      with torch.no_grad():
        for _ in np.arange(0,(test.num_examples//FLAGS.batch_size)):
          x, t = test.next_batch(FLAGS.batch_size)
          x = torch.tensor(x).to(device)
          t = torch.tensor(t).to(device)

          net.eval()
          y = net.forward(x)
          acc.append(accuracy(y,t))
        acc_iter.append(np.mean(acc))
        print("Testing accuracy: {}".format("%.3f"%np.mean(acc)))
        print()
        
  plt.plot(loss_iter,'r-',alpha=0.1, label="Batch loss")
  plt.plot(loss_mean,'b-', label="Average loss")
  plt.legend()
  plt.xlabel("Iterations")
  plt.ylabel("Loss")
  plt.title("Training Loss")
  plt.grid(True)
  plt.show()
  plt.close()

  plt.plot(acc_iter,'g-')
  plt.xlabel("Iterations")
  plt.ylabel("Accuracy")
  plt.grid(True)
  plt.title("Test Accuracy")
  plt.show()
  plt.close()
  print()
  print("COMPLETED")


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