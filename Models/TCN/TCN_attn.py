#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    '''
    I think this removes padding but not sure
    '''
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(n_inputs)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.ReLU1 = nn.SELU()
        self.dropout1 = nn.Dropout(dropout)

        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.ReLU2 = nn.SELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.bn1, self.conv1, self.chomp1, self.ReLU1, self.dropout1,
                                 self.bn2, self.conv2, self.chomp2, self.ReLU2, self.dropout2)

        # Lower Dimension of output data if necessary
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.ReLU = nn.SELU()

        for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.ReLU(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x) 

class Attention(nn.Module):
    '''
    Attention Network
    input -> [B, n, hidden_size] TCN output Tensor
    output -> [B, n] sepsis label tensor
    '''
    def __init__(self, hidden_size, attention_dim=1, droprate=0.5):
        super(Attention, self).__init__()
        self.attn_mat = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.attn_vecs = nn.Linear(hidden_size, attention_dim)
        self.drop = nn.Dropout(p = droprate)

    def forward(self, X):
        # compute attention coefficient
        a = self.attn_vecs(self.tanh(self.attn_mat(X)))
        # create context vector
        c = torch.zeros(X.shape[0], X.shape[1], X.shape[2], a.shape[2])
        for t in range(X.shape[1]):
            # normalize attention coefficients up to timestep
            softmaxed = F.softmax(a[:,:t], dim = 1)
            w_avg = torch.sum(softmaxed, dim = 1)
            a_mat = torch.bmm(X[:,t,:].unsqueeze(2), w_avg.unsqueeze(1))
            # Fill context vector with cummulative normalized attention
            c[:,t,:,:] += a_mat
        c = c.view((X.shape[0],X.shape[1], -1))
        c = self.drop(c)
        return c

class TCN(nn.Module):
    '''
    TODO: write description here
    TODO: add regression head
    '''
    def __init__(self, input_size=74, output_size=1, num_channels=[16,16],
                 attention=1, fcl=32, static=181, embedding=64, kernel_size=2, dropout=0.25):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.attn = Attention(hidden_size=num_channels[-1], attention_dim=attention)

        self.s_embed = nn.Linear(static, embedding)
        self.selu1 = nn.SELU()
        self.drop = nn.Dropout(dropout)

        self.fcl = nn.Linear(embedding + ((1 + attention)* num_channels[-1]), fcl)
        self.selu2 = nn.SELU()
        self.drop_fcl = nn.Dropout(dropout)
        self.out = nn.Linear(fcl, output_size)

    def forward(self, static, dynamic):
        """Inputs dims (N, C_in, L_in)"""
        hidden_out = self.tcn(dynamic).permute(0,2,1)       # (B, n, num_channels[-1])
        context = self.attn(hidden_out)                     # (B, n, attn_heads * num_channels[-1])
        static = self.drop(self.selu1((self.s_embed(static))))
        out = self.out(self.drop_fcl(self.selu2(self.fcl(torch.cat((static, context, hidden_out), dim=2)))))
        return out.squeeze()

class DynamicTCN(nn.Module):
    '''
    All static variables treated as dynamic
    '''
    def __init__(self, input_size=255, output_size=10, num_channels=[16,16],
                 attention=1, fcl=32, kernel_size=2, dropout=0.25, single_out=True):
        super(DynamicTCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        if attention:
            self.attn = Attention(hidden_size=num_channels[-1], attention_dim=attention)
        else: self.attn = False

        self.fcl = nn.Linear((1 + attention)* num_channels[-1], fcl)
        self.selu2 = nn.SELU()
        self.drop_fcl = nn.Dropout(dropout)
        self.out = nn.Linear(fcl, output_size)
        if single_out:
            # softmax for single output
            self.prob = nn.Softmax(dim=2)
        else:
            # sigmoid for multi-label
            self.prob = nn.Sigmoid()

    def forward(self, dynamic):
        """Inputs dims (N, C_in, L_in)"""
        hidden_out = self.tcn(dynamic).permute(0,2,1) # (B, n, num_channels[-1])
        if self.attn:
            context = self.attn(hidden_out)           # (B, n, attn_heads * num_channels[-1])
            out = self.out(self.drop_fcl(self.selu2(self.fcl(torch.cat((context, hidden_out), dim=2)))))
        else:
            out = self.out(self.drop_fcl(self.selu2(self.fcl(hidden_out))))
        out = self.prob(out.view(out.shape[0], out.shape[1], out.shape[2]//2, 2))

        return out.view(out.shape[0], out.shape[1], out.shape[2]*2)
