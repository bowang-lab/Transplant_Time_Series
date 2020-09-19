import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

class Attention(nn.Module):
    '''
    Attention Network
    input -> [B, n, hidden_size] output from LSTM
    output -> [B, n] label tensor
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


class lstm(nn.Module):
    '''
    lstm prototype
    input -> [B, n, 40] physiological variable time series tensor
          -> [B, n, 80] if missing markers are included
    output -> [B, n] sepsis label tensor
    '''
    def __init__(self, embedding, hidden_size, input_size=404,  output_size=10,
                 num_layers=2, batch_size=1, attention=0, embed=False, droprate=0.5, clip=5):
        super(lstm, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm_layers = num_layers
        self.embed = embed
        self.nonlinear = nn.ReLU()
        self.grad_clipping = clip

        self.inp = nn.Linear(input_size, embedding) # input embedding - can be changed
        self.drop1 = nn.Dropout(p=droprate)
        self.rnn = nn.LSTM(embedding, hidden_size, num_layers=num_layers, batch_first=True, dropout=droprate) # RNN structure
        self.drop2 = nn.Dropout(p=droprate)

        if attention:
            self.attn = Attention(hidden_size=hidden_size, attention_dim=attention)
        else: self.attn = False

        self.out = nn.Linear((1 + attention)* hidden_size, output_size) # output linear
        self.prob = nn.Softmax(dim=2)

        for m in self.modules():
           if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def init_hidden(self):
        hidden_a = torch.randn(self.lstm_layers, self.batch_size, self.hidden_size)
        hidden_b = torch.randn(self.lstm_layers, self.batch_size, self.hidden_size)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)
    
    def forward(self, X, seq_len, max_len, hidden_state=None): 
        self.hidden = self.init_hidden()

        if self.embed:
            X = self.inp(X)
        X = self.drop1(X)

        X = hotfix_pack_padded_sequence(X, seq_len, batch_first=True, enforce_sorted=False)
        X, self.hidden = self.rnn(X, self.hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True, padding_value=-1, total_length=max_len)
        if X.requires_grad:
            X.register_hook(lambda x: x.clamp(min=-self.grad_clipping, max=self.grad_clipping))
        X = self.drop2(X)

        if self.attn:
            context = self.attn(X)
            X = torch.cat((context, X), dim=2)

        X = self.out(X)

        X = self.prob(X.view(X.shape[0], X.shape[1], X.shape[2]//2, 2))

        return X.view(X.shape[0], X.shape[1], X.shape[2]*2)


class RNN(nn.Module):
    '''
    '''
    def __init__(self, embedding, hidden_size, input_size=140,  output_size=10,
                 num_layers=2, batch_size=1, attention=0, embed=False, droprate=0.5, clip=5):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.layers = num_layers
        self.embed = embed
        self.nonlinear = nn.ReLU()
        self.grad_clipping = clip

        self.inp = nn.Linear(input_size, embedding) # input embedding - can be changed
        self.drop1 = nn.Dropout(p=droprate)
        self.rnn = nn.RNN(embedding, hidden_size, num_layers=num_layers, batch_first=True, dropout=droprate) # RNN structure
        self.drop2 = nn.Dropout(p=droprate)

        if attention:
            self.attn = Attention(hidden_size=hidden_size, attention_dim=attention)
        else: self.attn = False

        self.out = nn.Linear((1 + attention)* hidden_size, output_size) # output linear
        self.prob = nn.Softmax(dim=2)

        for m in self.modules():
           if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def init_hidden(self, batch_size=False):
        hidden_a = torch.randn(self.layers, batch_size, self.hidden_size)
        hidden_a = Variable(hidden_a)

        return (hidden_a)
    
    def forward(self, X, seq_len, max_len, hidden_state=None): 
        batch_size = X.shape[0]
        self.hidden = self.init_hidden(batch_size)

        if self.embed:
            X = self.inp(X)
        X = self.drop1(X)

        X = hotfix_pack_padded_sequence(X, seq_len, batch_first=True, enforce_sorted=False)
        X, self.hidden = self.rnn(X, self.hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True, padding_value=-1, total_length=max_len)
        if X.requires_grad:
            X.register_hook(lambda x: x.clamp(min=-self.grad_clipping, max=self.grad_clipping))
        X = self.drop2(X)

        if self.attn:
            context = self.attn(X)
            X = torch.cat((context, X), dim=2)

        X = self.out(X)

        X = self.prob(X.view(X.shape[0], X.shape[1], X.shape[2]//2, 2))

        return X.view(X.shape[0], X.shape[1], X.shape[2]*2)

def hotfix_pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    '''
    #TODO: if ever fixed just go back to original
    GPU errors with orig func:
    torch.nn.utils.rnn.pack_padded_sequence()
    this fix was provided on pytorch board
    ''' 
    lengths = torch.as_tensor(lengths, dtype=torch.int64)
    lengths = lengths.cpu()
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    data, batch_sizes = \
        torch._C._VariableFunctions._pack_padded_sequence(input, lengths, batch_first)
    return PackedSequence(data, batch_sizes, sorted_indices)