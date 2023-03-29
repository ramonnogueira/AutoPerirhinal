import os
import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
nan=float('nan')


# Standard classifier
def classifier(data,clase,reg):
    n_splits=5
    perf=nan*np.zeros((n_splits,2))
    cv=StratifiedKFold(n_splits=n_splits)
    g=-1
    for train_index, test_index in cv.split(data,clase):
        g=(g+1)
        clf = LogisticRegression(C=reg,class_weight='balanced')
        clf.fit(data[train_index],clase[train_index])
        perf[g,0]=clf.score(data[train_index],clase[train_index])
        perf[g,1]=clf.score(data[test_index],clase[test_index])
    return np.mean(perf,axis=0)

# Fit the autoencoder. The data needs to be in torch format
def fit_autoencoder(model,data,data_cv,clase,n_epochs,batch_size,lr,sigma_noise,betar,betac,betas,p_norm):
    train_loader=DataLoader(torch.utils.data.TensorDataset(data,data,clase),batch_size=batch_size,shuffle=True)
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    loss1=torch.nn.MSELoss()
    loss2=torch.nn.CrossEntropyLoss()
    model.train()
    
    loss_rec_vec=[]
    loss_ce_vec=[]
    loss_sp_vec=[]
    loss_vec=[]
    data_epochs=[]
    data_hidden=[]
    t=0
    while t<n_epochs: 
        #print (t)
        outp=model(data,sigma_noise)
        outp_cv=model(data_cv,sigma_noise)
        data_epochs.append(outp_cv[0].detach().numpy())# Careful here, this is on the CV data not used for training
        data_hidden.append(outp_cv[1].detach().numpy())# Careful here, this is on the CV data not used for training
        loss_rec=loss1(outp[0],data).item()
        loss_ce=loss2(outp[2],clase).item()
        loss_sp=sparsity_loss(outp[2],p_norm).item()
        loss_total=(betar*loss_rec+betac*loss_ce+betas*loss_sp)
        loss_rec_vec.append(loss_rec)
        loss_ce_vec.append(loss_ce)
        loss_sp_vec.append(loss_sp)
        loss_vec.append(loss_total)
        if t==0 or t==(n_epochs-1):
            print (t,'rec ',loss_rec,'ce ',loss_ce,'sp ',loss_sp,'total ',loss_total)
        for batch_idx, (targ1, targ2, cla) in enumerate(train_loader):
            optimizer.zero_grad()
            output=model(targ1,sigma_noise)
            loss_r=loss1(output[0],targ2) # reconstruction error
            loss_cla=loss2(output[2],cla) # cross entropy error
            loss_s=sparsity_loss(output[2],p_norm)
            loss_t=(betar*loss_r+betac*loss_cla+betas*loss_s)
            loss_t.backward() # compute gradient
            optimizer.step() # weight update
        t=(t+1)
    model.eval()
    return np.array(loss_rec_vec),np.array(loss_ce_vec),np.array(loss_sp_vec),np.array(loss_vec),np.array(data_epochs),np.array(data_hidden)

# Autoencoder Architecture
class sparse_autoencoder_1(nn.Module):
    def __init__(self,n_inp,n_hidden,sigma_init):
        super(sparse_autoencoder_1,self).__init__()
        self.n_inp=n_inp
        self.n_hidden=n_hidden
        self.sigma_init=sigma_init
        self.enc=torch.nn.Linear(n_inp,n_hidden)
        self.dec=torch.nn.Linear(n_hidden,n_inp)
        self.dec2=torch.nn.Linear(n_hidden,2)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.sigma_init)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=self.sigma_init)
        
    def forward(self,x,sigma_noise):
        x_hidden = F.relu(self.enc(x))+sigma_noise*torch.randn(x.size(0),self.n_hidden)
        x = self.dec(x_hidden)
        x2 = self.dec2(x_hidden)
        return x,x_hidden,x2

def sparsity_loss(data,p):
    #shap=data.size()
    #nt=shap[0]*shap[1]
    #loss=(1/nt)*torch.norm(data,p)
    #loss=torch.norm(data,p)
    #loss=torch.mean(torch.sigmoid(100*(data-0.1)),axis=(0,1))
    loss=torch.mean(torch.pow(abs(data),p),axis=(0,1))
    return loss

# Evaluate Geometry
# Feat decoding is the features to be decoded (e.g. neural activity). Matrix number of trials x number of features
# Feat binary is the variables to decode. Matrix number of trials x 2. Each trial is a 2D binary word ie [0,1] (two variables to values each variable)
# reg is regularization
def geometry_2D(feat_decod,feat_binary,reg):
    
    # Assigns to each binary word a number from 0 to 3: [0,0] -> 0, [0,1] -> 1, [1,0] -> 2, [1,1] -> 3.
    exp_uq=np.unique(feat_binary,axis=0)
    feat_binary_exp=np.zeros(len(feat_binary))
    for t in range(len(feat_binary)):
        for tt in range((len(exp_uq))):
            gg=(np.sum(feat_binary[t]==exp_uq[tt])==len(feat_binary[0]))
            if gg:
                feat_binary_exp[t]=tt

    ###################################
    # Evaluate decoding perf on variable 1, variable 2 and xor tasks.
    xor=np.sum(feat_binary,axis=1)%2 # Define the XOR function wrt to the two variables
    n_cv=5
    perf_tasks_pre=np.zeros((n_cv,3,2))

    # Variable 1
    skf=StratifiedKFold(n_splits=n_cv)
    g=-1
    for train, test in skf.split(feat_decod,feat_binary[:,0]):
        g=(g+1)
        supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
        mod=supp.fit(feat_decod[train],feat_binary[:,0][train])
        perf_tasks_pre[g,0,0]=supp.score(feat_decod[train],feat_binary[:,0][train])
        perf_tasks_pre[g,0,1]=supp.score(feat_decod[test],feat_binary[:,0][test])

    # Variable 2
    skf=StratifiedKFold(n_splits=n_cv)
    g=-1
    for train, test in skf.split(feat_decod,feat_binary[:,1]):
        g=(g+1)
        supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
        mod=supp.fit(feat_decod[train],feat_binary[:,1][train])
        perf_tasks_pre[g,1,0]=supp.score(feat_decod[train],feat_binary[:,1][train])
        perf_tasks_pre[g,1,1]=supp.score(feat_decod[test],feat_binary[:,1][test])

    # XOR
    skf=StratifiedKFold(n_splits=n_cv)
    g=-1
    for train, test in skf.split(feat_decod,xor):
        g=(g+1)
        supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
        mod=supp.fit(feat_decod[train],xor[train])
        perf_tasks_pre[g,2,0]=supp.score(feat_decod[train],xor[train])
        perf_tasks_pre[g,2,1]=supp.score(feat_decod[test],xor[test])

    perf_tasks=np.mean(perf_tasks_pre,axis=0)

    ###############################################
    # Calculate Abstraction (CCGP)
    
    # Define the dichotomies for the 2D case            
    dichotomies=np.array([[0,0,1,1],[0,1,0,1]])
    train_dich=np.array([[[0,2],[1,3]],[[0,1],[2,3]]])
    test_dich=np.array([[[1,3],[0,2]],[[2,3],[0,1]]])

    # Evaluates CCGP (abstraction)
    perf_ccgp=nan*np.zeros((len(dichotomies),len(train_dich[0]),2))
    for k in range(len(dichotomies)): #Loop on "dichotomies"
      for kk in range(len(train_dich[0])): #Loop on ways to train this particular "dichotomy"
         ind_train=np.where((feat_binary_exp==train_dich[k][kk][0])|(feat_binary_exp==train_dich[k][kk][1]))[0]
         ind_test=np.where((feat_binary_exp==test_dich[k][kk][0])|(feat_binary_exp==test_dich[k][kk][1]))[0]

         task=nan*np.zeros(len(feat_binary_exp))
         for i in range(4):
             ind_task=(feat_binary_exp==i)
             task[ind_task]=dichotomies[k][i]

         supp=LogisticRegression(C=reg,class_weight='balanced',solver='lbfgs')
         #supp=LinearSVC(C=reg,class_weight='balanced')
         mod=supp.fit(feat_decod[ind_train],task[ind_train])
         perf_ccgp[k,kk,0]=supp.score(feat_decod[ind_train],task[ind_train])
         perf_ccgp[k,kk,1]=supp.score(feat_decod[ind_test],task[ind_test])
         
    return perf_tasks,perf_ccgp
