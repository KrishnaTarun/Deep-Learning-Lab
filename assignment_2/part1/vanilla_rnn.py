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

import torch
import torch.nn as nn
import numpy as np
################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.hidden_size = num_hidden
        self.device = device
        
        
        self.W_hx = nn.Parameter(torch.FloatTensor(num_hidden, input_dim))
        nn.init.normal_(self.W_hx, 0.0,1e-2)
        self.W_hh = nn.Parameter(torch.FloatTensor(num_hidden, num_hidden))
        nn.init.normal_(self.W_hh, 0.0,1e-2)
        self.W_out = nn.Parameter(torch.FloatTensor(num_classes, num_hidden))
        nn.init.normal_(self.W_out, 0.0,1e-2)

        self.tanh = nn.Tanh()

    def initHidden(self):
        
        return torch.zeros(self.batch_size, self.hidden_size).to(self.device)
        
    
    def forward(self, x):
        # Implementation here ...
        self.hidden = self.initHidden()
        for indx_ in np.arange(self.seq_length):
            in_seq = x[:,indx_].unsqueeze(-1)
            ht = self.tanh(torch.mm(in_seq, self.W_hx.transpose(1,0)) + torch.mm(self.hidden, self.W_hh))
            pt = torch.mm(ht, self.W_out.transpose(1,0))

            self.hidden = ht
        return pt 


        
