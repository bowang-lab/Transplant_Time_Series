import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from TCN_attn import DynamicTCN
from util import Dataset, collate_fn
from torch.utils.data import DataLoader
from class_saliency import VanillaSaliency
import pickle


def pi_display(probs):
    #TODO Bold title
    #     change colors
    #     just plot single fig

    def polar_labels(theta, labels, offset):

        if type(offset) != type(np.array([])):
            offset = [offset] * 5

        for x, label, offset in zip(theta, labels, offset):
            lab = ax.text(0, 0, label, transform=None, 
                    ha='center', va='center')
            renderer = ax.figure.canvas.get_renderer()
            bbox = lab.get_window_extent(renderer=renderer)
            invb = ax.transData.inverted().transform([[0,0],[bbox.width,0] ])
            lab.set_position((x,offset+(invb[1][0]-invb[0][0])/2.*2.7 ) )
            lab.set_transform(ax.get_xaxis_transform())
    
    groups = ['survival','cardiac', 'graft', 'cancer', 'infection']
    probs5 = [str(int(100*probs[i])) + '%' for i in range(5)]
    probs1 = [str(int(100*probs[i])) + '%' for i in range(5,10)]

    N = 5
    theta = np.linspace(0.0, 2 * np.pi, 5, endpoint=False)
    #rotations = np.rad2deg(theta)
    radii5 = probs[:5]
    radii1 = probs[5:]
    
    width = np.pi * 2 / 5
    colors = plt.cm.Accent((0,1,2,3,4))
    
    fig = plt.figure(figsize=(12,6))
    ax = plt.subplot(121, projection='polar')
    ax.set_title('5 year outlook', fontweight="bold")
    for i in range(5):
        ax.bar(theta[i], radii5[i], width=width, bottom=0.0, edgecolor='k', color=colors[i], alpha=0.5)
        ax.plot((0,theta[i]-np.pi/5), (0,max(radii5)*1.05), color='k', lw=1, alpha=0.5)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    ax.set_theta_offset(np.pi/2 - np.pi/5)
    ax.set_theta_direction(-1)
    y0,y1 = ax.get_ylim()

    p_loc = ((radii5/(y1-y0)) - 0.1)
    p_loc[p_loc < 0.2] = 0.3
    polar_labels(theta, groups, 1.3)
    polar_labels(theta, probs5, p_loc)

    ax = plt.subplot(122, projection='polar')
    ax.set_title('1 year outlook', fontweight="bold")
    for i in range(5):
        ax.bar(theta[i], radii1[i], width=width, bottom=0.0, edgecolor='k', color=colors[i], alpha=0.5)
        ax.plot((0,theta[i]-np.pi/5), (0,max(radii1)*1.05), color='k', lw=1, alpha=0.5)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    ax.set_theta_offset(np.pi/2 - np.pi/5)
    ax.set_theta_direction(-1)
    y0,y1 = ax.get_ylim()

    p_loc = (radii1/(y1-y0)) - 0.1
    p_loc[p_loc < 0.2] = 0.3
    polar_labels(theta, groups, 1.3)
    polar_labels(theta, probs1, p_loc)
    plt.subplots_adjust(wspace=0.35)

    plt.show()


if __name__ == '__main__':


    path = '/home/osvald/Projects/Diagnostics/github/srtr_data/multi_label/backup/n_train_tensors/'
    groups =['survival/','cardiac/', 'graft/', 'cancer/', 'infection/']


    #load model
    indices = list(range(4804))
    data = Dataset(indices, path)
    loader = DataLoader(data, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model_folder = '/home/osvald/Projects/Diagnostics/github/models/TCN/normalized/search/[64, 64]_fcl32_att0/lr0.0007787054002686635_b1_0.7982063652094732_b2_0.6107009808891577_gamma0.8319009616491654_drop0.2357377200377962_l2_0.006814594805294124/'
    model = DynamicTCN(input_size=267, output_size=10, num_channels=[64, 64], fcl=32,
                attention=0, kernel_size=2, dropout=0)
    model.load_state_dict(torch.load(model_folder + '/best_auc_model'))
    model.eval()

    for batch, labels, seq_len in loader:

        # pass to GPU if available
        batch, labels = batch, labels
        batch = batch.permute(0,2,1)

        outputs = model(batch)
        break

    for i in range(outputs.shape[1]):
        pi_display(outputs[0, i, :].detach().numpy())
    
