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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        
        # Initialization here ...
        # Initialization here ...
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.hidden_size = num_hidden
        self.device = device
        
        
        # modulation gate
        self.W_gx = nn.Parameter(torch.FloatTensor(num_hidden, input_dim))
        self.W_gh = nn.Parameter(torch.FloatTensor(num_hidden, num_hidden))
        self.b_g  = nn.Parameter(torch.FloatTensor(num_hidden)) 
        nn.init.normal_(self.W_gx, 0.0,1e-2)
        nn.init.normal_(self.W_gh, 0.0,1e-2)
        nn.init.constant_(self.b_g, 0)

        # input gate
        self.W_ix = nn.Parameter(torch.FloatTensor(num_hidden, input_dim))
        self.W_ih = nn.Parameter(torch.FloatTensor(num_hidden, num_hidden))
        self.b_i  = nn.Parameter(torch.FloatTensor(num_hidden)) 
        nn.init.normal_(self.W_ix, 0.0,1e-2)
        nn.init.normal_(self.W_ih, 0.0,1e-2)
        nn.init.constant_(self.b_i, 0)

        # forget gate
        self.W_fx = nn.Parameter(torch.FloatTensor(num_hidden, input_dim))
        self.W_fh = nn.Parameter(torch.FloatTensor(num_hidden, num_hidden))
        self.b_f  = nn.Parameter(torch.FloatTensor(num_hidden)) 
        nn.init.normal_(self.W_fx, 0.0,1e-2)
        nn.init.normal_(self.W_fh, 0.0,1e-2)
        nn.init.constant_(self.b_f, 0)

        #output gate
        self.W_ox = nn.Parameter(torch.FloatTensor(num_hidden, input_dim))
        self.W_oh = nn.Parameter(torch.FloatTensor(num_hidden, num_hidden))
        self.b_o  = nn.Parameter(torch.FloatTensor(num_hidden)) 
        nn.init.normal_(self.W_ox, 0.0,1e-2)
        nn.init.normal_(self.W_oh, 0.0,1e-2)
        nn.init.constant_(self.b_o, 0)
        
        #w_ph 
        self.W_ph = nn.Parameter(torch.FloatTensor(num_classes, num_hidden))
        self.b_p  = nn.Parameter(torch.FloatTensor(num_classes))
        nn.init.normal_(self.W_ph, 0.0,1e-2)
        nn.init.constant_(self.b_p, 0)
        

        # non-linear functions
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def initHidden(self):
        return torch.zeros(self.batch_size, self.hidden_size).to(self.device)


        
    def forward(self, x):
        
        self.hidden = self.initHidden()
        self.cell = self.initHidden()
        
        for indx_ in np.arange(self.seq_length):
            in_seq = x[:,indx_].unsqueeze(-1)
            
            g_t = self.tanh(torch.mm(in_seq, self.W_gx.transpose(1,0)) +\
                            torch.mm(self.hidden, self.W_gh) +self.b_g)
            
            i_t = self.sigmoid(torch.mm(in_seq, self.W_ix.transpose(1,0)) +\
                            torch.mm(self.hidden, self.W_ih) +self.b_i)
            
            f_t = self.sigmoid(torch.mm(in_seq, self.W_fx.transpose(1,0)) +\
                            torch.mm(self.hidden, self.W_fh) + self.b_f)
            
            o_t = self.sigmoid(torch.mm(in_seq, self.W_ox.transpose(1,0)) +\
                            torch.mm(self.hidden, self.W_oh) + self.b_o)

            cell_t = g_t * i_t + self.cell * f_t 
            h_t = self.tanh(cell_t) * o_t

            p_t = torch.mm(h_t, self.W_ph.transpose(1,0)) + self.b_p

            self.cell = cell_t
            self.hidden = h_t
        
        return p_t


 