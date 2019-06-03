################################################################################
# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM
import torch.nn as nn
import torch.optim as optim
import pickle
# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def train(config):
    

    np.random.seed(42)
    torch.manual_seed(42)
    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    print(device)

    # Initialize the model that we are going to use
    if config.model_type=="RNN":
        
        print("Training VanillaRNN")
        print()
        model = VanillaRNN(config.input_length, config.input_dim,\
                                config.num_hidden, config.num_classes, config.batch_size, config.device)  # fixme
    else:
        print("Training LSTM")
        print()
        model = LSTM(config.input_length, config.input_dim,\
                                config.num_hidden, config.num_classes, config.batch_size, config.device)

    model = model.to(device)
    
    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    
    # Setup the loss and optimizer
    criterion =  nn.CrossEntropyLoss()  #fixme
    if config.optimizer=="adam":
        optimizer = optim.Adam(model.parameters(), lr = config.learning_rate) # fixme
    else: 
        optimizer = optim.RMSprop(model.parameters(), lr = config.learning_rate)   
    pl_loss =[]
    average_loss =[]
    acc =[]
    
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()
        
        batch_targets = torch.LongTensor(batch_targets)
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        
        
        # zero the parameter gradients
        model.zero_grad()
        
        # Add more code here ...
        output = model(batch_inputs)

        out_loss = criterion(output, batch_targets)
        out_loss.backward()
        
        ############################################################################
        # QUESTION: what happens here and why?
        # ANSWER: helps prevent the exploding gradient problem in RNNs / LSTMs.
        ############################################################################
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        ############################################################################
        optimizer.step()
        
        # Add more code here ...

        loss = out_loss.item()   # fixme
        # get argmax
        softmax = torch.nn.Softmax(dim=1)
        predictions = torch.argmax(softmax(output), dim=1)
        predictions = config.batch_size-len(torch.nonzero(predictions - batch_targets))
        accuracy = predictions/config.batch_size              
        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)
        
        pl_loss.append(loss)
        average_loss.append(np.mean(pl_loss[:-100:-1]))
        acc.append(accuracy)
        
        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

        # if step%100==0:
        #     # save training loss
        #     plt.plot(pl_loss,'r-', label="Batch loss", alpha=0.5)
        #     plt.plot(average_loss,'g-', label="Average loss", alpha=0.5)
        #     plt.legend()
        #     plt.xlabel("Iterations")
        #     plt.ylabel("Loss")  
        #     plt.title("Training Loss")
        #     plt.grid(True)
        #     # plt.show()
        #     plt.savefig(config.optimizer+"_loss_"+config.model_type+"_"+str(config.input_length)+".png")
        #     plt.close()
    ################################training##################################################
    # plt.plot(acc,'g-', alpha=0.5)
    # plt.xlabel("Iterations")
    # plt.ylabel("Accuracy")
    # plt.title("Train Accuracy")
    # plt.grid(True)
    # plt.savefig("accuracy_"+config.sampling+"_"+str(config.temp)+".png")
    #  plt.close()
    # fl = config.optimizer+"_acc_"+config.model_type+"_"+str(config.input_length)
   
    
    # np.savez(fl, acc=acc)
    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--optimizer', type=str, default="adam", help="adam or rmsprop")
    config = parser.parse_args()

    # Train the model
    train(config)
    