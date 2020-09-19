import numpy as np
import os
import torch
from dataloader import Dataset, collate_fn, make_train_loader
from torch.utils.data import DataLoader
import random

from dataloader import Dataset, collate_fn, make_train_loader

train_path = '/home/osvald/Projects/Diagnostics/github/srtr_data/multi_label/n_train_tensors/'
valid_path = '/home/osvald/Projects/Diagnostics/github/srtr_data/multi_label/n_valid_tensors/'

t_neg_weights = torch.Tensor([0.147, 0.344, 0.287, 0.249, 0.240, 0.531, 0.251, 0.188, 0.181, 0.165]) # for balance between pos & neg
t_class_weights = torch.Tensor([3.94, 1.95, 2.24, 2.50,  2.58,  1.44, 2.49, 3.16, 3.26, 3.53]) # for balance between classes

v_neg_weights = torch.Tensor([2.69, 0.0445, 0.0411, 0.0875, 0.121, 3.97, 0.0338, 0.0286, 0.0645,  0.0873])
v_class_weights = torch.Tensor([ 0.686, 11.7, 12.7,  6.21, 4.63, 0.626, 15.3, 18.0, 8.25, 6.23])


def get_train_weights():

    pos_l = np.zeros((10))
    neg_l = np.zeros((10))
    weights = np.zeros((10))

    for epoch in range(5):
        train_loader = make_train_loader(train_path, batch_size=1024, shuffle=True, collate_fn=collate_fn)
        for batch, labels, seq_len in train_loader:
            neg_l += torch.sum(torch.sum((labels == 0), dim=0),dim=0).cpu().numpy()
            pos_l += torch.sum(torch.sum((labels == 1), dim=0),dim=0).cpu().numpy()
            t_weights = torch.sum(torch.sum((labels == 0), dim=0),dim=0).cpu().numpy() * t_neg_weights.cpu().numpy()
            t_weights += torch.sum(torch.sum((labels == 1), dim=0),dim=0).cpu().numpy()
            weights += t_weights / ((torch.sum(labels == 1).cpu().numpy()+ torch.sum(labels == 0).cpu().numpy())/10)

    neg_weights = pos_l/neg_l 
    weights /= len(train_loader)
    multiplier = (1/(weights / max(weights)))
    multiplier = (1/np.mean(weights * multiplier)) * multiplier
    return neg_weights, multiplier
    
def get_valid_weights():
    
    val_data = Dataset(valid_indices, valid_path)
    val_loader = DataLoader(val_data, batch_size=1024, shuffle=True, collate_fn=collate_fn)

    pos_l = np.zeros((10))
    neg_l = np.zeros((10))
    weights = np.zeros((10))

    for batch, labels, seq_len in val_loader:
        neg_l += torch.sum(torch.sum((labels == 0), dim=0),dim=0).cpu().numpy()
        pos_l += torch.sum(torch.sum((labels == 1), dim=0),dim=0).cpu().numpy()
        t_weights = torch.sum(torch.sum((labels == 0), dim=0),dim=0).cpu().numpy() * v_neg_weights.cpu().numpy()
        t_weights += torch.sum(torch.sum((labels == 1), dim=0),dim=0).cpu().numpy()
        weights += t_weights / ((torch.sum(labels == 1).cpu().numpy()+ torch.sum(labels == 0).cpu().numpy())/10)

    neg_weights = pos_l/neg_l
    weights /= len(val_loader)
    multiplier = (1/(weights / max(weights)))
    multiplier = (1/np.mean(weights * multiplier)) * multiplier
    return neg_weights, multiplier

if __name__ == '__main__':
    
    valid_indices = list(range(4214))


    # val data same every epoch
    val_data = Dataset(valid_indices, valid_path)
    val_loader = DataLoader(val_data, batch_size=1024, shuffle=True, collate_fn=collate_fn)

    v_neg_weights, v_class_weights = get_valid_weights()
    t_neg_weights, t_class_weights = get_train_weights()

    print('v_neg_weights', v_neg_weights)
    print('v_class_weights', v_class_weights)    
    print('t_neg_weights', t_neg_weights)
    print('t_class_weights', t_class_weights)