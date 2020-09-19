import argparse
import torch
import os
import numpy as np
from copy import deepcopy
from class_saliency import *
import pickle

root = '/home/osvald/Projects/Diagnostics/github/srtr_data/immuno/CV4/'
train_path = root + 'n_train_tensors/'
valid_path = root + 'n_valid_tensors/'
test_path = root + 'n_test_tensors/'

if __name__ == '__main__':
    ''' if in interactive mode on VScode'''
    class Args():
        disable_cuda = False
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        absolute = False
    args=Args()


    #load model
    indices = list(range(4215))
    path = valid_path
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

    salience_map = np.zeros([190])
    count=0

    for batch, labels, seq_len in loader:

        # pass to GPU if available
        batch, labels = batch.to(args.device), labels.to(args.device)
        salience = heatmap.generate_saliency(batch, labels)
        for i in range(labels.shape[0]):
            if args.absolute:
                salience_map += np.mean(salience[i, :int(seq_len[i]), :].abs().cpu().numpy(), axis=0)
            else:
                salience_map += np.mean(salience[i, :int(seq_len[i]), :].cpu().numpy(), axis=0)
            count += 1
    
    salience_map = salience_map / count
    inputs = np.load('/home/osvald/Projects/Diagnostics/github/Multi-Class/two_outlook/present_vars.npy', allow_pickle=True)
    importance = []

    # sort input variables by order of importance
    for i in range(len(inputs)):
        importance.append([inputs[i], salience_map[i]])
    importance = sorted(importance, key=lambda x: abs(x[1]))

    for i in range(len(inputs)):
        print(importance[i])

    suffix = ['signed', 'absolute']
    location = '/home/osvald/Projects/Diagnostics/github/Multi-Class/Salience/total/importance_' + suffix[int(args.absolute)] + '.pickle'

    with open(location,  'wb') as handle:
        pickle.dump(importance, handle)

    with open(location, 'rb') as handle:
        test = pickle.load(handle)

    if importance == test:
        print(location, 'saved successfully')
    else:
        print('error saving', location)