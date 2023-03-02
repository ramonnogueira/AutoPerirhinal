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

# Ramon's Local Path to save figures
path_plots='/home/ramon/Documents/github_repos/AutoEnthorinal/'

#############################
# Parameters Training
#noise during training the autoencoder
sig_neu=0.5 # noise neurons autoencoder
sig_inp=0.5 # noise input
sig_init=0.1 #noise weight initialization autoencoder
n_inp=10
n_hidden=20 # number hidden units in the autoencoder
beta=1 # between 0 and 1. 0 only reconstruction, 1 only decoding
beta_sp=1e-10
p_norm=2

n_trials=100
n_files=2 # number of files (sessions)

batch_size=10 # batch size when fitting network
lr=1e-3 # learning rate
n_epochs=500 #number of max epochs if conv criteria is not reached

# Define the stimulus
x_pre=np.array([[-1,-1],
                [-1,1],
                [1,-1],
                [1,1]])

#perf_orig=np.zeros((n_files,2))
perf_dire=np.zeros((n_files,n_epochs,2))
perf_speed=np.zeros((n_files,n_epochs,2))
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
    #perf_orig[k]=miscellaneous_sparseauto.classifier(x,clase[:,1],1)
                                
    # Fit the autoencoders
    x_torch=Variable(torch.from_numpy(np.array(x,dtype=np.float32)),requires_grad=False)
    clase_torch=Variable(torch.from_numpy(np.array(clase[:,0],dtype=np.int64)),requires_grad=False) # Only dim0 (direction)
    model=miscellaneous_sparseauto.sparse_autoencoder_1(n_inp=n_inp,n_hidden=n_hidden,sigma_init=sig_init) 
    loss_rec_vec,loss_ce_vec,loss_sp_vec,loss_vec,data_epochs,data_hidden=miscellaneous_sparseauto.fit_autoencoder(model=model,data=x_torch,clase=clase_torch,n_epochs=n_epochs,batch_size=batch_size,lr=lr,sigma_noise=sig_neu,beta=beta,beta_sp=beta_sp,p_norm=p_norm)
    loss_epochs[k,:,0]=loss_rec_vec
    loss_epochs[k,:,1]=loss_ce_vec
    loss_epochs[k,:,2]=loss_sp_vec
    loss_epochs[k,:,3]=loss_vec

    for i in range(n_epochs):
        #perf_dire[k,i]=miscellaneous_sparseauto.classifier(x-data_epochs[i],clase[:,0],1) # Decode direction
        #perf_speed[k,i]=miscellaneous_sparseauto.classifier(x-data_epochs[i],clase[:,1],1) # Decode Speed
        perf_dire[k,i]=miscellaneous_sparseauto.classifier(data_epochs[i],clase[:,0],1) # Decode direction
        perf_speed[k,i]=miscellaneous_sparseauto.classifier(data_epochs[i],clase[:,1],1) # Decode Speed
        perfh_dire[k,i]=miscellaneous_sparseauto.classifier(data_hidden[i],clase[:,0],1) # Decode direction
        perfh_speed[k,i]=miscellaneous_sparseauto.classifier(data_hidden[i],clase[:,1],1) # Decode Speed

# Plot Loss
loss_m=np.mean(loss_epochs,axis=0)
plt.plot(loss_m[:,0],color='blue',label='Reconstr.')
plt.plot(loss_m[:,1],color='red',label='Class.')
plt.plot(loss_m[:,2],color='green',label='Sparsity')
plt.plot(loss_m[:,3],color='black',label='Total')
plt.ylabel('Training Loss')
plt.xlabel('Epochs')
plt.legend(loc='best')
plt.show()
        
# Plot performance
#perf_m=np.mean(perf_orig,axis=0)
perf_dire_m=np.mean(perf_dire,axis=0)
perf_speed_m=np.mean(perf_speed,axis=0)
perfh_dire_m=np.mean(perfh_dire,axis=0)
perfh_speed_m=np.mean(perfh_speed,axis=0)

plt.plot(perf_dire_m[:,1],color='red',label='Direction')
plt.plot(perf_speed_m[:,1],color='blue',label='Speed')
#plt.plot(perf_m[1]*np.ones(n_epochs),color='grey',label='Input')
plt.plot(perfh_dire_m[:,1],color='red',linestyle='--',label='Dire H')
plt.plot(perfh_speed_m[:,1],color='blue',linestyle='--',label='Speed H')
#plt.plot(perf_m[0]*np.ones(n_epochs),color='grey',linestyle='--')
plt.plot(0.5*np.ones(n_epochs),color='black',linestyle='--')
plt.ylim([0,1.1])
plt.ylabel('Decoding Performance')
plt.xlabel('Epochs')
plt.legend(loc='best')
plt.show()
