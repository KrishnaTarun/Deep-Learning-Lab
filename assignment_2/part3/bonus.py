import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import TextDataset
from model import TextGenerationModel
import shutil

def greedy(in_tensor):

    _, out = torch.max(in_tensor, dim=2)
    return out

# Parse training configuration
parser = argparse.ArgumentParser()

# Model params
parser.add_argument('--model_file', type=str, default="model_greedy.pth.tar", help="read model weights")
parser.add_argument('--txt_file', type=str, default='assets/book_EN_grimms_fairy_tails.txt', help="Path to a .txt file to train on")
parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence to be ')
parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')
parser.add_argument('--batch_size', type=int, default=1, help='Number of examples to process in a batch')
parser.add_argument('--generate_length', type=int, default=500, help='Length of character sequence to be generated')


config = parser.parse_args()


# Initialize the device which to run the model on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.isdir(config.summary_path):
    os.makedirs(config.summary_path)

f = open(os.path.join(config.summary_path,"sampled_bonus.txt"), "a+")

dataset = TextDataset(config.txt_file, config.seq_length) 
vocab_size = dataset.vocab_size

# Initialize the model that we are going to use
model = TextGenerationModel(config.batch_size, config.seq_length, vocab_size, \
                                config.lstm_num_hidden, config.lstm_num_layers, device)  # fixme
model.to(device)

if config.model_file:

    print("Loading checkpoint '{}'".format(config.model_file))
    checkpoint = torch.load(config.model_file)
    step = checkpoint['step']
    model.load_state_dict(checkpoint['state_dict'])
    
else:
    print("No model found")
    exit()

model.eval()
               
with torch.no_grad():
    # Generate a initial sequence
    # text = "The old woman had only pretend"
    text = "Then first came two white dove"
    # text = "At first Rapunzel was terribly"
    print(text)
    init_seq = torch.tensor(dataset.convert_to_index(text))
    
    
    # init_seq = t
    init_seq = torch.zeros(config.batch_size, config.seq_length, vocab_size).scatter_(2,torch.unsqueeze(torch.unsqueeze(init_seq,0),2),1).to(device)
    
    model.hidden = model.init_hidden(config.batch_size)
    
    # send the first sequence   
    predictions, hidden = model(init_seq)
    
    _, idx = torch.max(predictions, dim=2)
    
    idx = idx.squeeze(0)[-1]
    idx = idx.reshape(1,1,1)
    chars_ix = [idx.item()]
    in_char = torch.zeros(1, 1, vocab_size, device=device).scatter_(2, idx, 1)
    
    
    # model.hidden = hidden
    in_char = in_char.to(device)
    
    for i in range(config.generate_length):

        out, model.hidden = model(in_char)
                     
        out = greedy(out)
        
        chars_ix.append(out.item())

        in_char = torch.zeros(1, 1, vocab_size).to(device).scatter_(2,out.unsqueeze(-1),1)
 

    sentence = dataset.convert_to_string(chars_ix) 
           

f.write("------------------------------\n")
f.write(text)
f.write(sentence+"\n")
print(sentence)
print()
f.close()
