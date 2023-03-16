import sys
import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.stats import sem
import matplotlib.pyplot as plt
import pickle as pkl
import copy
import torch.nn as nn
import torch.nn.functional as F
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import miscellaneous_sparseauto
from mpl_toolkits.axes_grid1 import make_axes_locatable
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
nan=float('nan')

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines: 
            if loc=='left':
                spine.set_position(('outward', 10))  # outward by 10 points
            if loc=='bottom':
                spine.set_position(('outward', 0))  # outward by 10 points
         #   spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine
    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

# Ramon's Local Path to save figures
path_plots='/home/ramon/Documents/github_repos/AutoEnthorinal/'

# Working params
# sig_neu=0.5 # noise neurons autoencoder
# sig_inp=0.5 # noise input
# sig_init=0.25 #noise weight initialization autoencoder
# n_inp=10
# n_hidden=20 # number hidden units in the autoencoder
# beta=0.999 # between 0 and 1. 0 only reconstruction, 1 only decoding
# beta_sp=10 (20 even better)
# p_norm=2
# n_trials=100
# n_files=5 # number of files (sessions)
# batch_size=10 # batch size when fitting network
# lr=1e-2 # learning rate
# n_epochs=200 #number of max epochs if conv criteria is not reached
# Also works with p_norm=1 and betas=1

#############################
# Parameters Training
#noise during training the autoencoder
sig_neu_vec=[0.1,0.5,1] 
sig_inp_vec=[0.1,0.5,1]
sig_init_vec=[0.1,0.25,0.5,1] 
n_inp_vec=[10,20,100]
n_hidden_vec=[10,20,100]
betar_vec=[1e-4,1e-3]
betac_vec=[1,10]
betas_vec=[1,10]
n_trials_vec=[10,100]
batch_size_vec=[10,50]

a=[]
for i in sig_neu_vec:
    for ii in sig_inp_vec:
        for iii in sig_init_vec:
            for j in n_inp_vec:
                for jj in n_hidden_vec:
                    for jjj in betar_vec:
                        for k in betac_vec:
                            for kk in betas_vec:
                                for kkk in n_trials_vec:
                                    for h in batch_size_vec:
                                        a.append([i,ii,iii,j,jj,jjj,k,kk,kkk,h])

a=np.array(a)

# Define the stimulus
x_pre=np.array([[-1,-1],
                [-1,1],
                [1,-1],
                [1,1]])

lr=1e-2 
n_epochs=200
p_norm=2
n_files=5 

for dd in range(len(a)):
    print ('SIMULATION ',dd)
    params=a[dd]
    sig_neu=params[0]
    sig_inp=params[1]
    sig_init=params[2]
    n_inp=int(params[3])
    n_hidden=int(params[4])
    betar=params[5]
    betac=params[6]
    betas=params[7]
    n_trials=int(params[8])
    batch_size=int(params[9])

    perf_dire=np.zeros((n_files,n_epochs,2))
    perf_speed=np.zeros((n_files,n_epochs,2))
    perf_dire_diff=np.zeros((n_files,n_epochs,2))
    perf_speed_diff=np.zeros((n_files,n_epochs,2))
    perfh_dire=np.zeros((n_files,n_epochs,2))
    perfh_speed=np.zeros((n_files,n_epochs,2))
    loss_epochs=np.zeros((n_files,n_epochs,4))
    for k in range(n_files):
        print (k)
        mat_exp=np.random.normal(0,1/np.sqrt(n_inp),(2,n_inp))
        x_exp=np.dot(x_pre,mat_exp)
        x=np.zeros((len(x_pre)*n_trials,n_inp))
        clase=np.zeros((len(x_pre)*n_trials,2)) # dim0: direction, dim1: speed
        for i in range(len(x_pre)):
            x[i*n_trials:(i+1)*n_trials]=(np.random.normal(x_exp[i],sig_inp,(n_trials,n_inp)))
            clase[i*n_trials:(i+1)*n_trials]=x_pre[i]
        clase[clase==-1]=0
                                
        # Fit the autoencoders
        x_torch=Variable(torch.from_numpy(np.array(x,dtype=np.float32)),requires_grad=False)
        clase_torch=Variable(torch.from_numpy(np.array(clase[:,0],dtype=np.int64)),requires_grad=False) # Only dim0 (direction)
        model=miscellaneous_sparseauto.sparse_autoencoder_1(n_inp=n_inp,n_hidden=n_hidden,sigma_init=sig_init) 
        loss_rec_vec,loss_ce_vec,loss_sp_vec,loss_vec,data_epochs,data_hidden=miscellaneous_sparseauto.fit_autoencoder(model=model,data=x_torch,clase=clase_torch,n_epochs=n_epochs,batch_size=batch_size,lr=lr,sigma_noise=sig_neu,betar=betar,betac=betac,betas=betas,p_norm=p_norm)
        loss_epochs[k,:,0]=loss_rec_vec
        loss_epochs[k,:,1]=loss_ce_vec
        loss_epochs[k,:,2]=loss_sp_vec
        loss_epochs[k,:,3]=loss_vec

        for i in range(n_epochs):
            perf_dire[k,i]=miscellaneous_sparseauto.classifier(data_epochs[i],clase[:,0],1) # Decode direction
            perf_speed[k,i]=miscellaneous_sparseauto.classifier(data_epochs[i],clase[:,1],1) # Decode Speed
            perf_dire_diff[k,i]=miscellaneous_sparseauto.classifier(x-data_epochs[i],clase[:,0],1) # Decode direction
            perf_speed_diff[k,i]=miscellaneous_sparseauto.classifier(x-data_epochs[i],clase[:,1],1) # Decode Speed
            perfh_dire[k,i]=miscellaneous_sparseauto.classifier(data_hidden[i],clase[:,0],1) # Decode direction
            perfh_speed[k,i]=miscellaneous_sparseauto.classifier(data_hidden[i],clase[:,1],1) # Decode Speed
            
        
    # Plot performance
    #perf_m=np.mean(perf_orig,axis=0)
    perf_dire_m=np.mean(perf_dire,axis=0)
    perf_speed_m=np.mean(perf_speed,axis=0)
    perf_dire_dm=np.mean(perf_dire_diff,axis=0)
    perf_speed_dm=np.mean(perf_speed_diff,axis=0)
    perfh_dire_m=np.mean(perfh_dire,axis=0)
    perfh_speed_m=np.mean(perfh_speed,axis=0)

    # Plot Direction
    fig=plt.figure(figsize=(6,4))
    ax=fig.add_subplot(1,1,1)
    adjust_spines(ax,['left','bottom'])
    plt.plot(perf_dire_dm[:,1],color='red')
    plt.plot(perf_speed_dm[:,1],color='blue')
    plt.plot(0.5*np.ones(n_epochs),color='black',linestyle='--')
    ax.set_ylim([0.3,1])
    ax.set_title('Decode Direction')
    ax.set_ylabel('Decoding Performance')
    ax.set_xlabel('Training Stage')
    
    fig.savefig('/home/ramon/Documents/github_repos/AutoPerirhinal/plots_explore/decoding_direction_speed_lr0.01_nepochs200_signeu_%.2f_siginp_%.2f_siginit_%.2f_ninp_%i_nhidden_%i_betar_%.4f_betac_%.1f_betas_%.1f_ntrials_%i_batchsize_%i.png'%(sig_neu,sig_inp,sig_init,n_inp,n_hidden,betar,betac,betas,n_trials,batch_size),dpi=500,bbox_inches='tight')
