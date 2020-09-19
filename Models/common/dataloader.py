import numpy as np
import random

import torch
from torch.utils import data
from torch.utils.data import DataLoader

class Dataset(data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, IDs, path, inc_year=False):
            self.IDs = IDs #list of IDs in dataset
            self.path = path
            self.inc_year = inc_year

      def __len__(self):
            return len(self.IDs)

      def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            ID = str(self.IDs[index])
            x = torch.load(self.path + ID + '.pt')

            #y = x[:, -13:-8] # 5 year outlook
            #y = x[:, -8:-3] # 1 year outlook
            y = x[:, -13:-3] # 5 year and 1 year outlook
            x = x[:,:-13]

            ''' for future use '''
            if self.inc_year:
                  x = torch.cat((x, x[:, -1].unsqueeze(1)), dim=1)
            #d = x[:, -1] # year of transplant - for performance analysis
            #t = x[:, -2] # time to death

            return x, y

def collate_fn(data):
      ''' Creates mini-batch tensors from the list of tuples (data, label). '''

      data.sort(key=lambda x: len(x[1]), reverse=True) #sort by descending length w/in mini-batch
      inputs, labels = zip(*data)

      seq_len = torch.as_tensor([inputs[i].shape[0] for i in range(len(inputs))], dtype=torch.double).cpu()

      out_data= torch.zeros((len(inputs), inputs[0].shape[0], inputs[0].shape[1])) # (B, max, input_dim) tensor of zeros
      out_labels = -torch.ones((len(inputs), labels[0].shape[0], labels[0].shape[1]))                  # (B, input_dim) tensor of -1s

      for i in range(len(inputs)): # fill in available data

            out_data[i, :inputs[i].shape[0], :] = inputs[i]
            out_labels[i, :labels[i].shape[0]] = labels[i]

      return out_data, out_labels, seq_len

def make_train_loader(train_path, batch_size, shuffle, collate_fn, small=False, cv=1):
      if cv==1:
            end = 33715
      elif cv==2 or cv ==3:
            end = 33717
      elif cv==4:
            end = 33716
      elif cv==5:
            end = 33718
      if small:
            # equal amount of patients from each class = |smallest class|
            train_indices = list(range(27499, 29371))                       # class 1 & 2
            train_indices.extend(random.sample(range(0, 27499), 1000))       # class 0
            train_indices.extend(random.sample(range(29371, 31623), 1000))  # class 3
            train_indices.extend(random.sample(range(31623, end), 1000))  # class 4

            train_data = Dataset(train_indices, train_path)
            return DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
      else:
            # equal amount of patients from each class = |largest class|
            train_indices = list(range(27499, 29371)) * 2                  # class 1 & 2
            train_indices.extend(random.sample(range(0, 27499), 2000))      # class 0
            train_indices.extend(list(range(29371, 31623)))                 # class 3
            train_indices.extend(list(range(31623, end)))                 # class 4

            train_data = Dataset(train_indices, train_path)
            return DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def get_train_weights(train_path, precomputed=True):
      if precomputed:
            neg_weights = torch.tensor([0.8739, 0.1465, 0.1311, 0.1773, 0.1617, 6.048,
                  0.0291, 0.0366, 0.0428, 0.0386])
            multiplier = torch.tensor([0.3573, 1.304, 1.438, 1.107, 1.198, 0.1943,
                  5.889, 4.725, 4.060, 4.488])
            return neg_weights, multiplier

      pos_l = np.zeros((10))
      neg_l = np.zeros((10))
      weights = np.zeros((10))

      for epoch in range(3):
            train_loader = make_train_loader(train_path, batch_size=1024, shuffle=True, collate_fn=collate_fn)
            for batch, labels, seq_len in train_loader:
                  neg_l += torch.sum(torch.sum((labels == 0), dim=0),dim=0).cpu().numpy()
                  pos_l += torch.sum(torch.sum((labels == 1), dim=0),dim=0).cpu().numpy()

      neg_weights = pos_l/neg_l 

      for epoch in range(3):
            train_loader = make_train_loader(train_path, batch_size=1024, shuffle=True, collate_fn=collate_fn)
            for batch, labels, seq_len in train_loader:

                  t_weights = torch.sum(torch.sum((labels == 0), dim=0),dim=0).cpu().numpy() * neg_weights
                  t_weights += torch.sum(torch.sum((labels == 1), dim=0),dim=0).cpu().numpy()
                  weights += t_weights / ((torch.sum(labels == 1).cpu().numpy()+ torch.sum(labels == 0).cpu().numpy())/10)

    
      weights /= len(train_loader)
      multiplier = (1/(weights / max(weights)))
      multiplier = (1/np.mean(weights * multiplier)) * multiplier
      #print('t_neg_weights', neg_weights)
      #print('t_class_weights', multiplier) 
      return torch.from_numpy(neg_weights), torch.from_numpy(multiplier)
    
def get_valid_weights(valid_indices, valid_path, precomputed=True):
      if precomputed:
            neg_weights = torch.tensor([7.519, 1.574e-02, 1.863e-02, 4.513e-02,
                  4.209e-02, 3.0501e+01, 3.811e-03, 5.994e-03, 1.184e-02, 1.038e-02])
            multiplier = torch.tensor([0.5648, 31.35, 28.14, 11.50, 12.31,
                  0.52244, 132.8, 89.43, 43.15, 47.95])
            return neg_weights, multiplier

      val_data = Dataset(valid_indices, valid_path)
      val_loader = DataLoader(val_data, batch_size=1024, shuffle=True, collate_fn=collate_fn)

      pos_l = np.zeros((10))
      neg_l = np.zeros((10))
      weights = np.zeros((10))

      for batch, labels, seq_len in val_loader:
            neg_l += torch.sum(torch.sum((labels == 0), dim=0),dim=0).cpu().numpy()
            pos_l += torch.sum(torch.sum((labels == 1), dim=0),dim=0).cpu().numpy()
      neg_weights = pos_l/neg_l

      for batch, labels, seq_len in val_loader:
            t_weights = torch.sum(torch.sum((labels == 0), dim=0),dim=0).cpu().numpy() * neg_weights
            t_weights += torch.sum(torch.sum((labels == 1), dim=0),dim=0).cpu().numpy()
            weights += t_weights / ((torch.sum(labels == 1).cpu().numpy()+ torch.sum(labels == 0).cpu().numpy())/10)

      weights /= len(val_loader)
      multiplier = (1/(weights / max(weights)))
      multiplier = (1/np.mean(weights * multiplier)) * multiplier
      #print('v_neg_weights', neg_weights)
      #print('v_class_weights', multiplier)    
      return torch.from_numpy(neg_weights), torch.from_numpy(multiplier)