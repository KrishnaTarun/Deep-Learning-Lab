# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import TextDataset
from model import TextGenerationModel
import shutil
################################################################################
np.random.seed(42)
torch.manual_seed(42)

# save checkpoint
def save_checkpoint(state, config):
    
    
    if config.sampling=="greedy":
        
        path = "model_"+config.sampling+".pth.tar"
    else:
        path = "model_"+config.sampling + "_" + str(config.temp) + ".pth.tar"

    torch.save(state,path)

def greedy(in_tensor):

    _, out = torch.max(in_tensor, dim=2)
    return out

def random(in_tensor, T):
    
    softmax = nn.Softmax(1)
    in_tensor = in_tensor.squeeze(0)
    out = torch.multinomial(softmax(in_tensor/T),1).squeeze(-1)
    # _,out = torch.max(softmax(in_tensor/T),dim=1)

    return out.unsqueeze(-1)

def generate_sample(model, vocab_size, sequences, device, config):
    
    in_char_index = torch.randint(0,vocab_size, (1,1,1)).long()
    in_char = torch.zeros(1, 1, vocab_size).scatter_(2, in_char_index, 1)
    
    chars_ix = [in_char_index.item()]
    model.hidden = model.init_hidden(1)
    in_char = in_char.to(device)
    
    for i in range(sequences):
        
        out, model.hidden = model(in_char)
        
        if config.sampling=="greedy":
            out = greedy(out)
        else:
            out = random(out, config.temp)

        chars_ix.append(out.item())

        in_char = torch.zeros(1, 1, vocab_size).to(device).scatter_(2,out.unsqueeze(-1),1)
       

    return chars_ix


def train(config):
    
    
    # Initialize the device which to run the model on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)   # fixme
    data_loader = DataLoader(dataset, batch_size = config.batch_size, shuffle=True, num_workers=1)
    vocab_size = dataset.vocab_size
    # char2i = dataset._char_to_ix
    # i2char = dataset._ix_to_char
    # ----------------------------------------
    
    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, vocab_size, \
                                config.lstm_num_hidden, config.lstm_num_layers, device)  # fixme
    model.to(device)

    # Setup the loss and optimizer
    criterion = nn.NLLLoss()  # fixme
    optimizer = optim.RMSprop(model.parameters(), lr = config.learning_rate)  # fixme
    logSoftmax = nn.LogSoftmax(dim=2)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
                  step_size=config.learning_rate_step, gamma=config.learning_rate_decay)
    step = 1
    
    if config.resume:
        if os.path.isfile(config.resume):
            print("Loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            step = checkpoint['step']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("Checkpoint loaded '{}', steps {}".format(config.resume, checkpoint['step']))

    if not os.path.isdir(config.summary_path):
            os.makedirs(config.summary_path)

    if config.sampling =="greedy":
        
        f = open(os.path.join(config.summary_path,"sampled_"+config.sampling+".txt"), "w+")
    else:
        f = open(os.path.join(config.summary_path,"sampled_"+config.sampling+"_"+str(config.temp)+".txt"), "w+")



    
   
    best_accuracy = 0.0
    pl_loss =[]
    average_loss =[]
    acc =[]

    for epochs in range(30):

        if step == config.train_steps:
            print('Done training.')
            break

        for (batch_inputs, batch_targets) in data_loader:

            if config.batch_size!=batch_inputs.size()[0]:
                print("batch mismatch")
                break

            # Only for time measurement of step through network
            t1 = time.time()
            model.hidden = model.init_hidden(config.batch_size)

            model.zero_grad()
            #######################################################
            # Add more code here ...
            
            #convert batch inputs to one-hot vector
            batch_inputs= torch.zeros(config.batch_size, config.seq_length, vocab_size).scatter_(2,batch_inputs.unsqueeze(-1),1.0)
            
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            predictions, _ = model(batch_inputs)
            if config.sampling=="greedy":
                predictions = logSoftmax(predictions)
            else:
                predictions = logSoftmax(predictions/config.temp)

            loss = criterion(predictions.transpose(2,1), batch_targets)   # fixme

            _, predictions = torch.max(predictions, dim=2, keepdim=True)
            predictions = (predictions.squeeze(-1) == batch_targets).float()
            accuracy = torch.mean(predictions)
            
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
            
            optimizer.step()
            lr_scheduler.step()

            #######################################################

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)
            pl_loss.append(loss.item())
            average_loss.append(np.mean(pl_loss[:-100:-1]))
            acc.append(accuracy)


            if step % config.print_every == 0:

                print("[{}] Train Step {}/{}, Batch Size = {}, Examples/Sec = {:.2f}, "
                    "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss.item()
                ))
                
                

            if step % config.sample_every == 0:
                               
                model.eval()
               
                with torch.no_grad():
                   char_ix = generate_sample(model, vocab_size, config.seq_length, device, config)
                   sentence = dataset.convert_to_string(char_ix) 
                           
            
                f.write("--------------"+str(step)+"----------------\n")
                f.write(sentence+"\n")
                print(sentence)
                print()
                model.train()
                # ###########################################################################
                # save training loss
                plt.plot(pl_loss,'r-', label="Batch loss", alpha=0.5)
                plt.plot(average_loss,'g-', label="Average loss", alpha=0.5)
                plt.legend()
                plt.xlabel("Iterations")
                plt.ylabel("Loss")  
                plt.title("Training Loss")
                plt.grid(True)
                # plt.show()
                if config.sampling == "greedy":
                    plt.savefig("loss_"+config.sampling+".png")
                else:
                    plt.savefig("loss_"+config.sampling+"_"+str(config.temp)+".png")

                plt.close()
                ################################training##################################################
                plt.plot(acc,'g-', alpha=0.5)
                plt.xlabel("Iterations")
                plt.ylabel("Accuracy")
                plt.title("Train Accuracy")
                plt.grid(True)
                if config.sampling == "greedy":
                    plt.savefig("accuracy_"+config.sampling+".png")
                else:
                    plt.savefig("accuracy_"+config.sampling+"_"+str(config.temp)+".png")
                plt.close()

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break
            
            step+=1
            
        save_checkpoint({
            'epoch': epochs + 1,
            'step': step,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler':lr_scheduler.state_dict(),
            'accuracy': accuracy
                }, config)
        
    f.close()


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='assets/book_EN_grimms_fairy_tails.txt', help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--resume', type=str, default=None, help="Path to latest checkpoint")
    parser.add_argument('--sampling', type=str, default='greedy', help="sampling strategy(greedy and random)")
    parser.add_argument('--temp', type=float, default='0.5', help="Temperature parameters")

    config = parser.parse_args()

    # Train the model
    train(config)
