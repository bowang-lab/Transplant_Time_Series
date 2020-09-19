import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


if __name__ == '__main__':

    location = '/home/osvald/Projects/Diagnostics/github/Multi-Class/Salience/total/importance_absolute.pickle'
    main_dir = '/home/osvald/Projects/Diagnostics/github/Multi-Class/Salience/'
    save_dir = '/home/osvald/Projects/Diagnostics/github/Multi-Class/Salience/figures/'
    groups =['survival/','cardiac/', 'graft/', 'cancer/', 'infection/']
    suffix = ['signed', 'absolute']
    outlook = ['5', '1']
    colors = ['#10498e', '#cd1041']

    with open(location, 'rb') as handle:
        importance = pickle.load(handle)

    fig = plt.figure(0)
    plt.bar(np.arange(0.5,20.5,1),[t[1]/importance[-1][1] for t in importance[-1:-21:-1]], color='#3d4b54')
    plt.xticks(np.arange(0.5,20.5,1), (t[0].strip('"') for t in importance[-1:-21:-1]), rotation='vertical')
    plt.title('Most Salient Features')
    plt.ylabel('Relative Predictive Power')
    plt.xlim(0,20)
    plt.tight_layout()
    plt.savefig(save_dir + 'Most Salient Features.png')

    for c in [0,5, 1,6, 2,7, 3,8, 4,9]:

        #TODO currently always signed
        location = main_dir + groups[c%5] + outlook[int(c>4)] + '_year_signed.pickle'

        with open(location, 'rb') as handle:
            importance = pickle.load(handle)
        
        title = str('%s year outlook for %s' % (outlook[int(c>4)], groups[c%5].strip('/')))

        fig = plt.figure(c+1)
        plt.bar(np.arange(0.5,15.5,1),[abs(t[1]/importance[-1][1]) for t in importance[-1:-16:-1]], color=[colors[int(t[1]<0)] for t in importance[-1:-16:-1]], linewidth=2)
        plt.xticks(np.arange(0.5,15.5,1), (t[0].strip('"') for t in importance[-1:-16:-1]), rotation='vertical')
        plt.title(title)
        plt.ylabel('Relative Predictive Power')
        plt.xlim(0,15)
        plt.tight_layout()
        #plt.savefig(save_dir + title + '.png')

    plt.show()