import matplotlib.pyplot as plt
import numpy as np

import torch
from sklearn import metrics

def auc_plotter(model_name, train_losses, 
            val_losses, val_auc, save=False, show=True, loss_type='Weighted BCE Loss', name=False):
    '''
    Plots loss and accuracy curves
    '''
    fig1 = plt.figure(1)
    plt.clf()
    colours = ['b','g','r','c','m']
    classes = ['Survival', 'Cardiac', 'Graft failure', 'Cancer', 'Infection']

    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(loss_type)
    plt.title('Loss')
    plt.grid()
    #plt.ylim(0,1)

    if bool(save):
        if name:
            plt.savefig(save + model_name + '_loss.png')
        else:
            plt.savefig(save+'loss_curves.png')

    fig2 = plt.figure(2)
    plt.clf()

    for i in range(5):      
        plt.plot(val_auc[:,i], label=classes[i], c=colours[i])

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('5-year AUC')
    plt.grid()
    plt.ylim(-0.01, 1.01)
    plt.suptitle(model_name)
    
    if bool(save):
        if name:
            plt.savefig(save + model_name + '_AUC_5.png')
        else:
            plt.savefig(save+'_AUC_5.png')

    fig3 = plt.figure(3)
    plt.clf()
    
    for i in range(5):      
        plt.plot(val_auc[:,i+5], label=classes[i], c=colours[i])

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('1-year AUC')
    plt.grid()
    plt.ylim(-0.01, 1.01)

    if bool(save):
        if name:
            plt.savefig(save + model_name + '_AUC_1.png')
        else:
            plt.savefig(save+'_AUC_1.png')

    if show:
        plt.show()
    plt.clf()
    return

def show_prog(epoch, train_loss, train_acc, train_auc, val_loss, val_acc, val_auc, time_elapsed):
    '''
    Prints current epoch's losses, accuracy and runtime
    '''
    print('E %03d --- RUNTIME: %ds' % (epoch+1, time_elapsed))
    print('TRAIN  ||  loss: %.3f  ||  acc: %.2f | %.2f | %.2f | %.2f | %.2f || 5auc: %.2f | %.2f | %.2f | %.2f | %.2f || 1auc: %.2f | %.2f | %.2f | %.2f | %.2f' % 
            (train_loss, train_acc[0], train_acc[1], train_acc[2], train_acc[3], train_acc[4],
            train_auc[0], train_auc[1], train_auc[2], train_auc[3], train_auc[4],
            train_auc[5], train_auc[6], train_auc[7], train_auc[8], train_auc[9]))
    print('VALID  ||  loss: %.3f  ||  acc: %.2f | %.2f | %.2f | %.2f | %.2f || 5auc: %.2f | %.2f | %.2f | %.2f | %.2f || 1auc: %.2f | %.2f | %.2f | %.2f | %.2f || avg: %.3f' % 
            (val_loss, val_acc[0], val_acc[1], val_acc[2], val_acc[3], val_acc[4],
            val_auc[0], val_auc[1], val_auc[2], val_auc[3], val_auc[4],
            val_auc[5], val_auc[6], val_auc[7], val_auc[8], val_auc[9], sum(val_auc)/len(val_auc)))
    
def save_prog(model, model_path, train_losses, val_losses, epoch, best_loss, best_t_loss, best_auc):
    '''
    Saves losses and accuracys to model folder
    Saves model state dict every save_rate epochs
    '''
    np.save(model_path +'train_losses', train_losses)
    np.save(model_path +'val_losses', val_losses)

    #save model dict
    if best_loss: 
        torch.save(model.state_dict(), model_path + 'best_loss_model')
    if best_t_loss:
        torch.save(model.state_dict(), model_path + 'best_train_loss_model')
    if best_auc:
        torch.save(model.state_dict(), model_path + 'best_auc_model')

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

