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
# sig_neu parece que controla donde empieza la perf de la curva azul y donde acaba. Si es ~0.3, empieza muy arriba, baja mucho y vuelve a subir. Si es ~0.8, empieza bastante abajo pero no sube lo suficiente. 
# Parameters Training
#noise during training the autoencoder
sig_neu=0.5 # noise neurons autoencoder
sig_inp=0.5 # noise input
sig_init=0.25 #noise weight initialization autoencoder
n_inp=10
n_hidden=20 # number hidden units in the autoencoder
#beta=0.999#0.999 # between 0 and 1. 0 only reconstruction, 1 only decoding
#beta_sp=10
betar=1e-4
betac=1
betas=10
p_norm=2

n_trials=100
n_files=2 # number of files (sessions)

delta=1

batch_size=10 # batch size when fitting network
lr=1e-2 # learning rate
n_epochs=200 #number of max epochs if conv criteria is not reached

# Define the stimulus
x0=np.array([[-1,-1],
             [-1,1],
             [1,-1],
             [1,1]])

perf_orig=np.zeros((n_files,2,2))
perf_dire=np.zeros((n_files,n_epochs,2))
perf_speed=np.zeros((n_files,n_epochs,2))
perf_dire_diff=np.zeros((n_files,n_epochs,2))
perf_speed_diff=np.zeros((n_files,n_epochs,2))
perfh_dire=np.zeros((n_files,n_epochs,2))
perfh_speed=np.zeros((n_files,n_epochs,2))
loss_epochs=np.zeros((n_files,n_epochs,4))
for k in range(n_files):
    print (k)
    # Data Model pretraining
    mat_pre=np.random.normal(0,1/np.sqrt(n_inp),(2,n_inp))
    x_pre=np.dot(x0,mat_pre)
    x_pretrain=np.zeros((len(x0)*n_trials,n_inp)) 
    # Data fit CV
    mat_exp=np.random.normal(0,1/np.sqrt(n_inp),(2,n_inp))
    x_exp=np.dot(x0,mat_exp)
    x_auto=np.zeros((len(x0)*n_trials,n_inp)) # Dataset to train the autoencoder and the additional cross-entropy for the additional unit
    x=np.zeros((len(x0)*n_trials,n_inp)) # Dataset to train/test the classifiers
    clase=np.zeros((len(x0)*n_trials,2)) # dim0: direction, dim1: speed
    for i in range(len(x0)):
        x_pretrain[i*n_trials:(i+1)*n_trials]=(np.random.normal(x_pre[i],sig_inp,(n_trials,n_inp)))
        x_auto[i*n_trials:(i+1)*n_trials]=(np.random.normal(x_exp[i],sig_inp,(n_trials,n_inp)))
        x[i*n_trials:(i+1)*n_trials]=(np.random.normal(x_exp[i],sig_inp,(n_trials,n_inp)))
        clase[i*n_trials:(i+1)*n_trials]=x0[i]
    clase[clase==-1]=0
    perf_orig[k,0]=miscellaneous_sparseauto.classifier(x_auto,clase[:,0],1)
    perf_orig[k,1]=miscellaneous_sparseauto.classifier(x,clase[:,0],1)
                                
    # Fit the autoencoders
    x_pretrain_torch=Variable(torch.from_numpy(np.array(x_pretrain,dtype=np.float32)),requires_grad=False)
    x_auto_torch=Variable(torch.from_numpy(np.array(x_auto,dtype=np.float32)),requires_grad=False)
    x_torch=Variable(torch.from_numpy(np.array(x,dtype=np.float32)),requires_grad=False)
    clase_torch=Variable(torch.from_numpy(np.array(clase[:,0],dtype=np.int64)),requires_grad=False) # Only dim0 (direction)

    # Model pretraining
    #print ('Pretraining model...')
    #model=miscellaneous_sparseauto.sparse_autoencoder_1(n_inp=n_inp,n_hidden=n_hidden,sigma_init=sig_init)
    #miscellaneous_sparseauto.fit_autoencoder(model=model,data=x_pretrain_torch,data_cv=x_pretrain_torch,clase=clase_torch,n_epochs=int(0.25*n_epochs),batch_size=batch_size,lr=lr,sigma_noise=sig_neu,betar=1,betac=0,betas=betas,p_norm=p_norm)

    # Model training
    print ('Training model...')
    model=miscellaneous_sparseauto.sparse_autoencoder_1(n_inp=n_inp,n_hidden=n_hidden,sigma_init=sig_init)
    loss_rec_vec,loss_ce_vec,loss_sp_vec,loss_vec,data_epochs,data_hidden=miscellaneous_sparseauto.fit_autoencoder(model=model,data=x_auto_torch,data_cv=x_torch,clase=clase_torch,n_epochs=n_epochs,batch_size=batch_size,lr=lr,sigma_noise=sig_neu,betar=betar,betac=betac,betas=betas,p_norm=p_norm)
    loss_epochs[k,:,0]=loss_rec_vec
    loss_epochs[k,:,1]=loss_ce_vec
    loss_epochs[k,:,2]=loss_sp_vec
    loss_epochs[k,:,3]=loss_vec

    for i in range(n_epochs):
        perf_dire[k,i]=miscellaneous_sparseauto.classifier(data_epochs[i],clase[:,0],1) # Decode direction
        perf_speed[k,i]=miscellaneous_sparseauto.classifier(data_epochs[i],clase[:,1],1) # Decode Speed
        perf_dire_diff[k,i]=miscellaneous_sparseauto.classifier(x_auto-data_epochs[i],clase[:,0],1) # Decode direction
        perf_speed_diff[k,i]=miscellaneous_sparseauto.classifier(x_auto-data_epochs[i],clase[:,1],1) # Decode Speed
        perfh_dire[k,i]=miscellaneous_sparseauto.classifier(data_hidden[i],clase[:,0],1) # Decode direction
        perfh_speed[k,i]=miscellaneous_sparseauto.classifier(data_hidden[i],clase[:,1],1) # Decode Speed

print ('Perf input ',np.mean(perf_orig,axis=0))

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
perf_dire_dm=np.mean(perf_dire_diff,axis=0)
perf_speed_dm=np.mean(perf_speed_diff,axis=0)
perfh_dire_m=np.mean(perfh_dire,axis=0)
perfh_speed_m=np.mean(perfh_speed,axis=0)

plt.plot(perf_dire_dm[:,1],color='red',label='Output Direction')
plt.plot(perf_speed_dm[:,1],color='blue',label='Output Speed')
#plt.plot(perf_dire_dm[:,0],color='red',linestyle='--',label='Hidden Direction')
#plt.plot(perf_speed_dm[:,0],color='blue',linestyle='--',label='Hidden Speed')
#plt.plot(perf_dire_m[:,1],color='red',label='Output Direction')
#plt.plot(perf_speed_m[:,1],color='blue',label='Output Speed')
#plt.plot(perf_m[1]*np.ones(n_epochs),color='grey',label='Input')
plt.plot(perfh_dire_m[:,1],color='red',linestyle='--',label='Hidden Direction')
plt.plot(perfh_speed_m[:,1],color='blue',linestyle='--',label='Hidden Speed')
#plt.plot(perf_m[0]*np.ones(n_epochs),color='grey',linestyle='--')
plt.plot(0.5*np.ones(n_epochs),color='black',linestyle='--')
plt.ylim([0,1.1])
plt.ylabel('Decoding Performance')
plt.xlabel('Epochs')
plt.legend(loc='best')
plt.show()

# Plot Direction
# fig=plt.figure(figsize=(5,2))
# ax=fig.add_subplot(1,2,1)
# adjust_spines(ax,['left','bottom'])
# plt.plot(perf_dire_dm[:,1],color='black')
# plt.plot(0.5*np.ones(n_epochs),color='black',linestyle='--')
# ax.set_ylim([0.4,1])
# ax.set_title('Decode Direction')
# ax.set_ylabel('Decoding Performance')
# ax.set_xlabel('Training Stage')

# # Plot Speed
# ax=fig.add_subplot(1,2,2)
# adjust_spines(ax,['left','bottom'])
# plt.plot(perf_speed_dm[:,1],color='black')
# plt.plot(0.5*np.ones(n_epochs),color='black',linestyle='--')
# ax.set_ylim([0.4,1])
# ax.set_title('Decode Speed')
# ax.set_xlabel('Training Stage')
# fig.savefig('/home/ramon/Documents/github_repos/AutoPerirhinal/plots/decoding_direction_speed.pdf',dpi=500,bbox_inches='tight')
