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
from numpy.random import permutation
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

#Semi working parameters new
# sig_neu=1 # noise neurons autoencoder
# sig_inp=0.35 # noise input
# sig_init=0.3#0.25 #noise weight initialization autoencoder
# n_inp=10
# n_hidden=20 # number hidden units in the autoencoder
# betar=0.2
# betac=5
# betas=1
# p_norm=2
# n_trials=100
# n_files=5 # number of files (sessions)
# batch_size=10 # batch size when fitting network
# lr=5*1e-4 # learning rate
# n_epochs=200 #number of max epochs 

#############################
# Parameters Training
#noise during training the autoencoder
sig_neu=1 #1 noise neurons autoencoder
sig_inp=0.5 # noise input
sig_init=0.5#0.4#0.35 #noise weight initialization autoencoder
n_inp=10
n_hidden=20 # number hidden units in the autoencoder
betar=0.1# Beta 1: 1, Beta 2: 1, Beta 3: 0.1, Beta 4: 0.1
betac=1#1 # Beta 1: 0, Beta 2: 1, Beta 3: 1, Beta 4: 1
betas=20#1 Beta 1: 0, Beta 2: 0, Beta 3: 1, Beta 4: 20
p_norm=1

n_trials=100
n_files=100 # number of files (sessions)

batch_size=10 # batch size when fitting network
lr=2*1e-3 #Beta 1: 1e-2, Beta 2: 1e-2, Beta 3: 2*1e-3, Beta 4: 2*1e-3
n_epochs=50 #number of max epochs 

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
perf_xor=np.zeros((n_files,n_epochs,3))
perf_ccgp=np.zeros((n_files,n_epochs,2,2))
perfh_xor=np.zeros((n_files,n_epochs,3))
perfh_ccgp=np.zeros((n_files,n_epochs,2,2))
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

    ind=permutation(np.arange(len(clase)))
    x_pretrain=x_pretrain[ind]
    x_auto=x_auto[ind]
    x=x[ind]
    clase=clase[ind]
    perf_orig[k,0]=miscellaneous_sparseauto.classifier(x_auto,clase[:,0],1)
    perf_orig[k,1]=miscellaneous_sparseauto.classifier(x,clase[:,0],1)
                                
    # Fit the autoencoders
    x_pretrain_torch=Variable(torch.from_numpy(np.array(x_pretrain,dtype=np.float32)),requires_grad=False)
    x_auto_torch=Variable(torch.from_numpy(np.array(x_auto,dtype=np.float32)),requires_grad=False)
    x_torch=Variable(torch.from_numpy(np.array(x,dtype=np.float32)),requires_grad=False)
    clase_torch=Variable(torch.from_numpy(np.array(clase[:,0],dtype=np.int64)),requires_grad=False) # Only dim0 (direction)
    #clase_torch=Variable(torch.from_numpy(np.array(clase[:,0],dtype=np.float32)),requires_grad=False) # Only dim0 (direction)

    # Model pretraining
    # print ('Pretraining model...')
    # model=miscellaneous_sparseauto.sparse_autoencoder_1(n_inp=n_inp,n_hidden=n_hidden,sigma_init=sig_init)
    # ep_pt=5
    # lr_pt=1e-4
    # miscellaneous_sparseauto.fit_autoencoder(model=model,data=x_pretrain_torch,data_cv=x_pretrain_torch,clase=clase_torch,n_epochs=ep_pt,batch_size=batch_size,lr=lr_pt,sigma_noise=sig_neu,betar=1,betac=0,betas=betas,p_norm=p_norm)

    # Model training
    print ('Training model...')
    model=miscellaneous_sparseauto.sparse_autoencoder_1(n_inp=n_inp,n_hidden=n_hidden,sigma_init=sig_init)
    loss_rec_vec,loss_ce_vec,loss_sp_vec,loss_vec,data_epochs,data_hidden=miscellaneous_sparseauto.fit_autoencoder(model=model,data=x_auto_torch,data_cv=x_torch,clase=clase_torch,n_epochs=n_epochs,batch_size=batch_size,lr=lr,sigma_noise=sig_neu,betar=betar,betac=betac,betas=betas,p_norm=p_norm)
    loss_epochs[k,:,0]=loss_rec_vec
    loss_epochs[k,:,1]=loss_ce_vec
    loss_epochs[k,:,2]=loss_sp_vec
    loss_epochs[k,:,3]=loss_vec

    for i in range(n_epochs):
        #if i%100==0:
        #    print ('Epoch ',i)
        perf_dire[k,i]=miscellaneous_sparseauto.classifier(data_epochs[i],clase[:,0],1) # Decode direction `
        perf_speed[k,i]=miscellaneous_sparseauto.classifier(data_epochs[i],clase[:,1],1) # Decode Speed
        perf_dire_diff[k,i]=miscellaneous_sparseauto.classifier(x_auto-data_epochs[i],clase[:,0],1) # Decode direction
        perf_speed_diff[k,i]=miscellaneous_sparseauto.classifier(x_auto-data_epochs[i],clase[:,1],1) # Decode Speed
        perfh_dire[k,i]=miscellaneous_sparseauto.classifier(data_hidden[i],clase[:,0],1) # Decode direction
        perfh_speed[k,i]=miscellaneous_sparseauto.classifier(data_hidden[i],clase[:,1],1) # Decode Speed

        # geo=miscellaneous_sparseauto.geometry_2D(data_epochs[i],clase,1)
        # perf_xor[k,i]=geo[0][:,1]
        # perf_ccgp[k,i]=geo[1][:,:,1]
        # geoh=miscellaneous_sparseauto.geometry_2D(data_hidden[i],clase,1)
        # perfh_xor[k,i]=geoh[0][:,1]
        # perfh_ccgp[k,i]=geoh[1][:,:,1]
        
print ('Perf input ',np.mean(perf_orig,axis=0))

################################
# Plot Loss
# loss_m=np.mean(loss_epochs,axis=0)
# plt.plot(loss_m[:,0],color='blue',label='Reconstr.')
# plt.plot(loss_m[:,1],color='red',label='Class.')
# plt.plot(loss_m[:,2],color='green',label='Sparsity')
# #plt.plot(loss_m[:,3],color='black',label='Total')
# plt.ylabel('Training Loss')
# plt.xlabel('Epochs')
# plt.legend(loc='best')
# plt.show()

# #################################
# # Plot geometry
# perf_xor_m=np.mean(perf_xor,axis=0)
# perfh_xor_m=np.mean(perfh_xor,axis=0)
# perf_ccgp_m=np.mean(perf_ccgp,axis=(0,3))
# perfh_ccgp_m=np.mean(perfh_ccgp,axis=(0,3))

# plt.plot(perf_xor_m[:,0],color='red')
# plt.plot(perfh_xor_m[:,0],color='red',linestyle='--')
# plt.plot(perf_xor_m[:,1],color='blue')
# plt.plot(perfh_xor_m[:,1],color='blue',linestyle='--')
# plt.plot(perf_xor_m[:,2],color='grey')
# plt.plot(perfh_xor_m[:,2],color='grey',linestyle='--')
# plt.plot(perf_ccgp_m[:,0],color='salmon')
# plt.plot(perfh_ccgp_m[:,0],color='salmon',linestyle='--')
# plt.plot(perf_ccgp_m[:,1],color='royalblue')
# plt.plot(perfh_ccgp_m[:,1],color='royalblue',linestyle='--')
# plt.plot(0.5*np.ones(n_epochs),color='black',linestyle='--')
# plt.ylim([0,1.1])
# plt.ylabel('Decoding Performance')
# plt.xlabel('Epochs')
# #plt.legend(loc='best')
# plt.show()

################################
# Plot final metrics
perf_dire_m=np.mean(perf_dire,axis=0)
perf_speed_m=np.mean(perf_speed,axis=0)
perf_dire_dm=np.mean(perf_dire_diff,axis=0)
perf_speed_dm=np.mean(perf_speed_diff,axis=0)
perfh_dire_m=np.mean(perfh_dire,axis=0)
perfh_speed_m=np.mean(perfh_speed,axis=0)

perf_dire_std=sem(perf_dire,axis=0)
perf_speed_std=sem(perf_speed,axis=0)
perf_dire_dstd=sem(perf_dire_diff,axis=0)
perf_speed_dstd=sem(perf_speed_diff,axis=0)
perfh_dire_std=sem(perfh_dire,axis=0)
perfh_speed_std=sem(perfh_speed,axis=0)

# plt.plot(perf_dire_dm[:,1],color='red',label='Output Direction')
# plt.plot(perf_speed_dm[:,1],color='blue',label='Output Speed')
# #plt.plot(perf_dire_dm[:,0],color='red',linestyle='--',label='Hidden Direction')
# #plt.plot(perf_speed_dm[:,0],color='blue',linestyle='--',label='Hidden Speed')
# #plt.plot(perf_dire_m[:,1],color='red',label='Output Direction')
# #plt.plot(perf_speed_m[:,1],color='blue',label='Output Speed')
# #plt.plot(perf_m[1]*np.ones(n_epochs),color='grey',label='Input')
# plt.plot(perfh_dire_m[:,1],color='red',linestyle='--',label='Hidden Direction')
# plt.plot(perfh_speed_m[:,1],color='blue',linestyle='--',label='Hidden Speed')
# #plt.plot(perf_m[0]*np.ones(n_epochs),color='grey',linestyle='--')
# plt.plot(0.5*np.ones(n_epochs),color='black',linestyle='--')
# plt.ylim([0,1.1])
# plt.ylabel('Decoding Performance')
# plt.xlabel('Epochs')
# plt.legend(loc='best')
# plt.show()

# Plot Direction
fig=plt.figure(figsize=(5,2))
ax=fig.add_subplot(1,2,1)
adjust_spines(ax,['left','bottom'])
plt.errorbar(np.arange(n_epochs),perf_dire_dm[:,1],perf_dire_dstd[:,1],color='black')
#plt.plot(perf_dire_dm[:,1],color='black',linewidth=3)
plt.plot(0.5*np.ones(n_epochs),color='black',linestyle='--')
ax.set_ylim([0.4,1])
ax.set_title('Decode Direction')
ax.set_ylabel('Decoding Performance')
ax.set_xlabel('Training Stage')

# # Plot Speed
ax=fig.add_subplot(1,2,2)
adjust_spines(ax,['left','bottom'])
plt.errorbar(np.arange(n_epochs),perf_speed_dm[:,1],perf_speed_dstd[:,1],color='black')
#plt.plot(perf_speed_dm[:,1],color='black',linewidth=3)
plt.plot(0.5*np.ones(n_epochs),color='black',linestyle='--')
ax.set_ylim([0.4,1])
ax.set_title('Decode Speed')
ax.set_xlabel('Training Stage')
#fig.savefig('/home/ramon/Dropbox/JerryChen/figure/decoding_direction_speed_new.pdf',dpi=500,bbox_inches='tight')

# Supp figures
fig=plt.figure(figsize=(2.5,2))
ax=fig.add_subplot(1,1,1)
adjust_spines(ax,['left','bottom'])
#ax.errorbar(np.arange(n_epochs),perf_dire_m[:,1],perf_dire_std[:,1],color='red',label='Output Direction')
#ax.errorbar(np.arange(n_epochs),perf_speed_m[:,1],perf_speed_std[:,1],color='blue',label='Output Speed')
#ax.errorbar(np.arange(n_epochs),perfh_dire_m[:,1],perfh_dire_std[:,1],color='red',linestyle='--',label='Hidden Direction')
#ax.errorbar(np.arange(n_epochs),perfh_speed_m[:,1],perfh_speed_std[:,1],color='blue',linestyle='--',label='Hidden Speed')
ax.plot(np.arange(n_epochs),perf_dire_m[:,1],color='red',label='Output Direction')
ax.plot(np.arange(n_epochs),perf_speed_m[:,1],color='blue',label='Output Speed')
ax.plot(np.arange(n_epochs),perfh_dire_m[:,1],color='red',linestyle='--',label='Hidden Direction')
ax.plot(np.arange(n_epochs),perfh_speed_m[:,1],color='blue',linestyle='--',label='Hidden Speed')
ax.plot(0.5*np.ones(n_epochs),color='black',linestyle='--')
ax.set_ylim([0.4,1])
ax.set_xlabel('Training Stage')
plt.legend(loc='best')
fig.savefig('/home/ramon/Dropbox/JerryChen/figure/decoding_direction_speed_supp4_new.pdf',dpi=500,bbox_inches='tight')

# Main Figure
# Plot Direction
# fig=plt.figure(figsize=(5,2))
# ax=fig.add_subplot(1,2,1)
# adjust_spines(ax,['left','bottom'])
# plt.errorbar(np.arange(n_epochs),perf_dire_dm[:,1],perf_dire_dstd[:,1],color='black')
# #plt.plot(perf_dire_dm[:,1],color='black',linewidth=3)
# plt.plot(0.5*np.ones(n_epochs),color='black',linestyle='--')
# ax.set_ylim([0.4,1])
# ax.set_title('Decode Direction')
# ax.set_ylabel('Decoding Performance')
# ax.set_xlabel('Training Stage')

# # # Plot Speed
# ax=fig.add_subplot(1,2,2)
# adjust_spines(ax,['left','bottom'])
# plt.errorbar(np.arange(n_epochs),perf_speed_dm[:,1],perf_speed_dstd[:,1],color='black')
# #plt.plot(perf_speed_dm[:,1],color='black',linewidth=3)
# plt.plot(0.5*np.ones(n_epochs),color='black',linestyle='--')
# ax.set_ylim([0.4,1])
# ax.set_title('Decode Speed')
# ax.set_xlabel('Training Stage')
# fig.savefig('/home/ramon/Dropbox/JerryChen/figure/decoding_direction_speed_supp1.pdf',dpi=500,bbox_inches='tight')
