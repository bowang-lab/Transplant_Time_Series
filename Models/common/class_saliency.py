import argparse
import torch
import os
import numpy as np
#from TCN_attn import DynamicTCN

import sys
sys.path.append('/home/osvald/Projects/Diagnostics/github/Multi-Class/two_outlook/')
sys.path.append('/home/osvald/Projects/Diagnostics/github/Multi-Class/two_outlook/Transformer/')

from model import RT
from torch.utils.data import DataLoader
from common.dataloader import Dataset, collate_fn, make_train_loader, get_train_weights, get_valid_weights
from copy import deepcopy
import pickle

#root = '/home/osvald/Projects/Diagnostics/github/srtr_data/multi_label/'
root = '/home/osvald/Projects/Diagnostics/github/srtr_data/immuno/CV4/'
train_path = root + 'n_train_tensors/'
valid_path = root + 'n_valid_tensors/'
test_path = root + 'n_test_tensors/'


class Saliency(object):
    """ Abstract class for saliency """

    def __init__(self, model):
        self.model = deepcopy(model)
        self.model.train()

    def generate_saliency(self, model, input, target):
        raise "Method not implemented!"

class VanillaSaliency(Saliency):
    """Vanilla Saliency to visualize plain gradient information"""

    def __init__(self, model):
        super(VanillaSaliency, self).__init__(model)


    def generate_saliency(self, input, target):
        input.requires_grad = True

        self.model.zero_grad()

        output = self.model(input)

        grad_outputs = target

        output.backward(gradient = grad_outputs)

        input.requires_grad = False

        return (input.grad.clone()[0] * input)

def get_inputs():
    inputs = np.load('/home/osvald/Projects/Diagnostics/github/Multi-Class/two_outlook/present_vars.npy', allow_pickle=True)
    return inputs


if __name__ == '__main__':
    ''' if in interactive mode on VScode'''
    class Args():
        disable_cuda = False
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        absolute = False
    args=Args()


    #load model
    indices = list(range(33715))
    path = train_path
    data = Dataset(indices, path)
    loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_fn)

    folder='/home/osvald/Projects/Diagnostics/github/models/Transformer/CV4/IS/dim64_heads4_levels4/lr0.000848_b1_0.800277_b2_0.959436_drop0.212179_l2_0.003331'
    model = RT(input_size=190, d_model=64, output_size=10, h=4, rnn_type='RNN',
                ksize=3, n=1, n_level=4, dropout=0).to(args.device)
    model.load_state_dict(torch.load(folder + '/best_auc_model'))
    model.train()

    # I have no idea why running without this causes an error
    with torch.no_grad():
        for batch, labels, seq_len in loader:
            # pass to GPU if available
            batch, labels = batch.to(args.device), labels.to(args.device)
            out = model(batch)
            break
    heatmap = VanillaSaliency(model)

    salience_map = np.zeros([10, 190])
    counts = np.zeros(10)
    for batch, labels, seq_len in loader:

            # pass to GPU if available
            batch, labels = batch.to(args.device), labels.to(args.device)

            for i in range(1,labels.shape[1]):
                
                if args.absolute:
                    salience = heatmap.generate_saliency(batch[:,:i+1,:].abs(), labels[:,:i+1,:])
                else:
                    salience = heatmap.generate_saliency(batch[:,:i+1,:], labels[:,:i+1,:])

                for l in (labels[0,i,:] != 0).nonzero():
                    
                    salience_map[l] += salience[0, i, :].cpu().numpy()
                    counts[l] += 1
    

    main_dir = '/home/osvald/Projects/Diagnostics/github/Multi-Class/Salience/'
    groups =['survival/','cardiac/', 'graft/', 'cancer/', 'infection/']
    outlook = ['5','1']
    suffix = ['signed', 'absolute']

    salience_map = salience_map / np.expand_dims(counts, axis=1)
    inputs = np.load('/home/osvald/Projects/Diagnostics/github/Multi-Class/two_outlook/present_vars.npy', allow_pickle=True)

    for c in range(10):

        importance=[]

        for i in range(len(inputs)):
            importance.append([inputs[i], salience_map[c][i]])

        importance = sorted(importance, key=lambda x: abs(x[1]))

        location = main_dir + groups[c%5] + outlook[int(c>4)] + '_year_' + suffix[int(args.absolute)] + '.pickle'

        with open(location,  'wb') as handle:
            pickle.dump(importance, handle)

        with open(location, 'rb') as handle:
            test = pickle.load(handle)

        if importance == test:
            print(location, 'saved successfully')
        else:
            print('error saving', location)