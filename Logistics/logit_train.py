import numpy as np
import time
import argparse
import os
import torch
import torch.nn as nn
from torch import optim
from logit import LogisticRegression
from util import auc_plotter, show_prog, save_prog, Dataset, collate_fn
from torch.utils.data import DataLoader
from sklearn import metrics
import random

#train_path = '/home/osvald/Projects/Diagnostics/github/srtr_data/multi_label/backup/n_train_tensors/'
#valid_path = '/home/osvald/Projects/Diagnostics/github/srtr_data/multi_label/backup/n_valid_tensors/'
#save_path = '/home/osvald/Projects/Diagnostics/github/models/TCN/normalized/'

######## __GENERAL__ ########
parser = argparse.ArgumentParser(description='training control')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-epochs', action='store', default=30, type=int,
                    help='num epochs')
parser.add_argument('-batch', action='store', default=1024, type=int,
                    help='batch size')
parser.add_argument('-nosave', action='store_true',
                    help='do not save flag')
parser.add_argument('-prog', action='store_true',
                    help='show progress')

######## __OPTIM__ ########
parser.add_argument('-lr', action='store', default=0.01, type=float,
                    help='learning rate')
parser.add_argument('-b1', action='store', default=0.9, type=float,
                    help='momentum')
parser.add_argument('-b2', action='store', default=0.999, type=float,
                    help='momentum')
args = parser.parse_args()

######## __GPU_SETUP__ ########
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    args.device = torch.device('cpu')
    torch.set_default_tensor_type('torch.DoubleTensor')

def make_train_loader(small=False):
    #TODO update with new amounts of tensors
    if small:
        # equal amount of patients from each class = |smallest class|
        train_indices = list(range(26750, 29876))                       # class 1 & 2
        train_indices.extend(random.sample(range(0, 26750), 1600))       # class 0
        train_indices.extend(random.sample(range(29876, 32750), 1600))  # class 3
        train_indices.extend(random.sample(range(32750, 36798), 1400))  # class 4

        train_data = Dataset(train_indices, train_path)
        return DataLoader(train_data, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    else:
        # equal amount of patients from each class = |largest class|
        train_indices = list(range(26750, 29876)) * 3                   # class 1 & 2
        train_indices.extend(random.sample(range(0, 26750), 2505))      # class 0
        train_indices.extend(list(range(29876, 32750)))                 # class 3
        train_indices.extend(random.sample(range(29876, 32750), 762))   # class 3
        train_indices.extend(list(range(32750, 36798)))                 # class 4

        train_data = Dataset(train_indices, train_path)
        return DataLoader(train_data, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

def get_aucs(actual, predictions):
    '''
    in: list of np array with each class
    '''
    predictions = np.array(predictions)
    actual = np.array(actual)
    aucs = np.zeros([10])
    for i in range(10):
        fpr, tpr, t= metrics.roc_curve(actual[i], predictions[i])
        roc_auc = metrics.auc(fpr, tpr)
        aucs[i]= roc_auc
    return aucs

save = not args.nosave

model_name = 'logistic_regression'

t_neg_weights = torch.Tensor([0.685, 0.175,  0.183,  0.171,  0.169,  4.776, 0.0377,  0.053,   0.042,  0.048 ]) # for balance between pos & neg
t_class_weights = torch.Tensor([0.089, 0.244, 0.234, 0.248, 0.251, 0.0438, 1, 0.721, 0.895, 0.783]) # for balance between classes

v_neg_weights = torch.Tensor([6.001, 0.0207, 0.0201, 0.0482, 0.0602, 22.06, 0.0049,  0.0074,  0.013,  0.019 ])
v_class_weights = torch.Tensor([0.0642, 0.321, 0.311, 0.295, 0.284, 0.0525, 1.458, 1.0325, 1.133, 0.938])


def train():

    running_loss  = 0
    correct = np.zeros((10))
    pos = np.zeros((10))
    total = 0
    predictions = [np.array([])] * 10 
    actual = [np.array([])] * 10

    for batch, labels, seq_len in train_loader:
        # pass to GPU if available
        batch, labels = batch.to(args.device), labels.to(args.device)

        # run network
        optimizer.zero_grad()
        outputs = model(batch)

        multiplier = ( ((labels==0).double() * t_neg_weights) + (labels==1).double() ) * t_class_weights
        mask = (multiplier > 0) * (labels <= 1) * (labels >= 0)

        loss = torch.mean( criterion(outputs.masked_select(mask), labels.masked_select(mask)) * multiplier.masked_select(mask) )

        # adjust weights and record loss
        loss.backward()
        optimizer.step()
        running_loss += loss.cpu().data.numpy()

        # train accuracy
        for i in range(labels.shape[0]):
            targets = labels.data[i][:int(seq_len[i])].cpu().numpy()
            prob = outputs.data[i][:int(seq_len[i])].cpu().numpy()

            prediction = np.zeros(targets.shape)
            prediction[np.arange(prediction.shape[0]), np.argmax(prob, axis=1)] = 1
            match = (targets == prediction)
            
            pos += np.sum(prediction,axis=0).astype(int)
            correct += np.sum(match,axis=0).astype(int)
            total += targets.shape[0]

            # for AUC
            for j in range(10):
                actual[j] = np.concatenate((actual[j], labels[i][:int(seq_len[i]),j].view(-1).cpu().numpy()))
                predictions[j] = np.concatenate((predictions[j], outputs[i][:int(seq_len[i]),j].detach().view(-1).cpu().numpy()))

    train_losses[epoch] = running_loss/len(train_loader) * 10
    train_acc[epoch] = correct/total
    train_freq[epoch] = pos / sum(pos)
    train_auc[epoch] = get_aucs(actual, predictions)
    
def valid():
    
    running_loss  = 0
    correct = np.zeros(10)
    pos = np.zeros(10)
    total = 0
    predictions = [np.array([])] * 10
    actual = [np.array([])] * 10

    with torch.no_grad():
        for batch, labels, seq_len in val_loader:
            # pass to GPU if available
            batch, labels = batch.to(args.device), labels.to(args.device)

            # run network
            optimizer.zero_grad()
            outputs = model(batch)

            multiplier = ( ((labels==0).double() * v_neg_weights) + (labels==1).double() ) * v_class_weights
            mask = (multiplier > 0) * (labels <= 1) * (labels >= 0)

            loss = torch.mean( criterion(outputs.masked_select(mask), labels.masked_select(mask)) * multiplier.masked_select(mask) )

            running_loss += loss.cpu().data.numpy()
            
            # Validation accuracy
            for i in range(labels.shape[0]):
                targets = labels.data[i][:int(seq_len[i])].cpu().numpy()
                prob = outputs.data[i][:int(seq_len[i])].cpu().numpy()

                prediction = np.zeros(targets.shape)
                prediction[np.arange(prediction.shape[0]), np.argmax(prob, axis=1)] = 1
                match = (targets == prediction)
                
                pos += np.sum(prediction,axis=0).astype(int)
                correct += np.sum(match,axis=0).astype(int)
                total += targets.shape[0]

                # for AUC
                for j in range(10):
                    actual[j] = np.concatenate((actual[j], labels[i][:int(seq_len[i]),j].view(-1).cpu().numpy()))
                    predictions[j] = np.concatenate((predictions[j], outputs[i][:int(seq_len[i]),j].view(-1).cpu().numpy()))


        val_losses[epoch] = running_loss/len(val_loader) * 10
        val_acc[epoch] = correct/total
        val_freq[epoch] = pos / sum(pos)
        val_auc[epoch] = get_aucs(actual, predictions)


if __name__ == '__main__':
    
    # Create target Directory if it doesn't already exist
    # TODO: make architecture folder and hp folder within it
    if args.nosave: print('WARNING: MODEL AND DATA ARE NOT BEING SAVED')
    elif not args.nosave:
        if not os.path.exists(save_path+model_name):
            os.mkdir(save_path+model_name)
        else:
            print('WARNING: overwriting existing directory:', model_name)
    save_path = save_path + model_name + '/'

    ''' training data setup '''
    #train_indices = list(range(38436)) #full training set
    valid_indices = list(range(4600))

    '''TCN'''
    model = LogisticRegression(input_dim=267, output_dim=10).to(args.device)

    criterion = nn.BCELoss(reduction='none')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, verbose=True)


    # loss tracker
    train_losses = np.zeros(args.epochs)
    val_losses = np.zeros(args.epochs)
    # accuracies tracker
    train_acc = np.zeros((args.epochs, 10))
    val_acc = np.zeros((args.epochs, 10))
    # frequency tracker
    train_freq = np.zeros((args.epochs, 10))
    val_freq = np.zeros((args.epochs, 10))
    # AUC traker
    train_auc = np.zeros((args.epochs, 10))
    val_auc = np.zeros((args.epochs, 10))

    # val data same every epoch
    val_data = Dataset(valid_indices, valid_path)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    start = time.time()
    for epoch in range(args.epochs):
        train_loader = make_train_loader()
        model.train()
        train()
        model.eval()
        valid()
        scheduler.step(-np.mean(val_auc[epoch])) # step on avg. AUC plateau

        if args.prog:
            show_prog(epoch, train_losses[epoch], train_acc[epoch], train_auc[epoch],
                        val_losses[epoch], val_acc[epoch], val_auc[epoch], (time.time() - start))

        best_loss = val_losses[epoch] == min(val_losses[:epoch+1])
        best_t_loss = train_losses[epoch] == min(train_losses[:epoch+1])
        best_auc = np.mean(val_auc[epoch]) == max(np.mean(val_auc[:,:epoch+1],axis=0))

        if save:
            save_prog(model, save_path, train_losses, val_losses, epoch, best_loss, best_t_loss, best_auc)

    # PLOT GRAPHS
    if save:
        auc_plotter(model_name, train_losses,
                val_losses, val_auc, save=save_path, show=False)
    else:
        auc_plotter(model_name, train_losses,
                val_losses, val_auc, save=False, show=True)

    print('Model:', model_name, 'completed ; ', args.epochs, 'args.epochs', 'in %ds' % (time.time()-start))
    print('min vl_loss: %0.3f at epoch %d' % (min(val_losses), val_losses.argmin()+1))
    print('min tr_loss: %0.3f at epoch %d' % (min(train_losses), train_losses.argmin()+1))
    print('max avg AUC: %0.3f at epoch %d' % (max(np.mean(val_auc[:epoch+1],axis=1)), np.mean(val_auc[:epoch+1],axis=1).argmax()+1))
