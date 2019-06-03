"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '300, 300, 300'
LEARNING_RATE_DEFAULT = 2e-2
MAX_STEPS_DEFAULT = 6000
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

#check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print()
print(device)
print()

FLAGS = None

def evaluate(model, cifar10, batch_size):
  
  x,y = cifar10['test'].images , cifar10['test'].labels
  updates = 0
  acc=0
  with torch.no_grad():
    for iter_ in np.arange(y.shape[0]/batch_size):
        batch_x, batch_y = cifar10['test'].next_batch(batch_size)

             
        #reshape x into vectors
        batch_x = np.reshape(batch_x, (200, 3072))
        inputs = torch.from_numpy(batch_x).to(device)
        predictions = model.forward(inputs)
        
        acc += accuracy(predictions.cpu().numpy(), batch_y)
        updates+=1
  print("Test accuracy: {}".format("%.3f"%(acc/updates)))
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
  torch.manual_seed(42)
  
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
  net = MLP(3072, dnn_hidden_units, 10)
  net.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr = FLAGS.learning_rate)

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
  count = 1
  acc =[]
  check =0
  for iter_ in np.arange(0, FLAGS.max_steps):

    #Load batches 
    x , y = cifar10['train'].next_batch(FLAGS.batch_size)
    
    labels = np.argmax(y, axis=1)
    
    #reshape x into vectors
    x = np.reshape(x, (200, 3072))
    inputs, labels = torch.from_numpy(x), torch.LongTensor(torch.from_numpy(labels))
    
    inputs, labels = inputs.to(device), labels.to(device)

    # # labels = torch.LongTensor(labels)
    
    # # zero the parameter gradients
    optimizer.zero_grad()

    # # forward + backward + optimize
    outputs = net(inputs)
    print("output: {}, labels:{}".format(outputs.size(),labels.size()))
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # # print statistics
    running_loss = loss.item()
    pl_loss.append(running_loss)
    moving_average+=running_loss
    average_loss.append(np.mean(np.mean(pl_loss[:-100:-1])))
    print("iter: {} | training loss: {} ".format(iter_,"%.3f"%running_loss))

    
    if (iter_+1)%FLAGS.eval_freq==0:
      net.eval()
      acc.append(evaluate(net, cifar10, FLAGS.batch_size))

  #######################
  # END OF YOUR CODE    #
  #######################
  
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
