import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)
         self.prob = nn.Softmax(dim=1)
     def forward(self, x):
         d1,d2,d3 = x.shape
         x = x.view(-1, x.shape[2])
         out = self.linear(x)
         out = self.prob(out)
         out = out.view(d1, d2, 10)
         return out

class FFNN(torch.nn.Module):
     def __init__(self, input_dim, hidden, output_dim, drop_prob=0.1):
        super(FFNN, self).__init__()
        self.l1 = torch.nn.Linear(input_dim, hidden)
        self.l2 = torch.nn.Linear(hidden, output_dim)
        self.drop = nn.Dropout(drop_prob)
        self.prob = nn.Softmax(dim=1)
     def forward(self, x):
         d1,d2,d3 = x.shape
         x = x.view(-1, x.shape[2])
         out = self.l2(self.drop(self.l1((x))))
         out = self.prob(out)
         out = out.view(d1, d2, 10)
         return out

