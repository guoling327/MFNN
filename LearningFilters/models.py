import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,ARMAConv,global_mean_pool,GATConv,ChebConv,GCNConv)
from Bern import BernConv
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class GPR_prop(MessagePassing):
    def __init__(self, K, alpha=0.1, Init='Random', Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[-1] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
class GPRNet(torch.nn.Module):
    def __init__(self,K=10):
        super(GPRNet, self).__init__()
        self.lin1 = Linear(1, 32)
        self.lin2 = Linear(32, 64)
        self.prop1 = GPR_prop(K)
        self.fc2 = torch.nn.Linear(64, 1)

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x=data.x_tmp
        edge_index=data.edge_index

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        x = self.prop1(x, edge_index)
        return self.fc2(x)


class ARMANet(nn.Module):
    def __init__(self):
        super(ARMANet, self).__init__()
        self.conv1 = ARMAConv(1,32,1,1,False,dropout=0)
        self.conv2 = ARMAConv(32,64,1,1,False,dropout=0)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, data):

        x=data.x_tmp
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))  
        x = F.relu(self.conv2(x, edge_index))      
        return self.fc2(x)


class GcnNet(nn.Module):
    def __init__(self):
        super(GcnNet, self).__init__()

        self.conv1 = GCNConv(1, 32, cached=False)
        self.conv2 = GCNConv(32, 64, cached=False) 
        
        self.fc2 = torch.nn.Linear(64, 1) 

    def forward(self, data):

        x=data.x_tmp
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))  
        x = F.relu(self.conv2(x, edge_index))      
        return self.fc2(x)

class GatNet(nn.Module):
    def __init__(self):
        super(GatNet, self).__init__()
        self.conv1 = GATConv(1, 8, heads=4,concat=True, dropout=0.0)  
        self.conv2 = GATConv(32, 8, heads=8,concat=True, dropout=0.0) 
        
        self.fc2 = torch.nn.Linear(64, 1) 

    def forward(self, data):
        x=data.x_tmp
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.elu(self.conv2(x, data.edge_index)) 

        return self.fc2(x) 

class ChebNet(nn.Module):
    def __init__(self,K=3):
        super(ChebNet, self).__init__()
        
        self.conv1 = ChebConv(1, 32,K)    
        self.conv2 = ChebConv(32, 32,K) 
        self.fc2 = torch.nn.Linear(32, 1) 

    def forward(self, data):
        x=data.x_tmp
        edge_index=data.edge_index        
        x = F.relu(self.conv1(x, edge_index))   
        x = F.relu(self.conv2(x, edge_index))  
           
        return self.fc2(x) 


class BernNet(nn.Module):
    def __init__(self,K=10):
        super(BernNet, self).__init__()

        self.conv1 = BernConv(1, 32, K)
        self.conv2 = BernConv(32, 64, K)

        self.fc2 = torch.nn.Linear(64, 1)
        self.coe = Parameter(torch.Tensor(K+1))
        self.reset_parameters()

    def reset_parameters(self):
        self.coe.data.fill_(1)

    def forward(self, data):
        x=data.x_tmp
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index,self.coe))  
        x = F.relu(self.conv2(x, edge_index,self.coe))       
        return self.fc2(x)



import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
from torch.nn.parameter import Parameter

import torch_geometric
from numpy.linalg import eig, eigh

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_graph(g):
    g = np.array(g)
    g = g + g.T
    g[g > 0.] = 1.0
    deg = g.sum(axis=1).reshape(-1)
    deg[deg == 0.] = 1.0
    deg = np.diag(deg ** -0.5)
    adj = np.dot(np.dot(deg, g), deg)
    L = np.eye(g.shape[0]) - adj
    return L

def eigen_decompositon(g):
    "The normalized (unit “length”) eigenvectors, "
    "such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]."
    g = normalize_graph(g)
    e, u = eigh(g)
    return e, u



class MFNN0(nn.Module):
    def __init__(self, hidden_dim=128):
        super(MFNN0, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim+1, 1)
        self.device= torch.device('cuda:' + str(2) if torch.cuda.is_available() else 'cpu')


    def forward(self, data):
      #  x = data.x
        e = np.load('eigenvalues.npy')


        e= torch.Tensor(e).to(self.device)
        ee = e.unsqueeze(1)
      #  print(e.device)
        #ee = ee*ee
        #eeig = ee
        eeig = torch.full(ee.shape, torch.tensor(1.0)).to(self.device)

        for i in range(int(self.hidden_dim)):
            # sin = torch.sin((i+1) * math.pi * ee)
            cos = torch.cos((i+1) * math.pi * ee)
            eeig = torch.cat((eeig, cos), dim=1).to(self.device)

       # print(eeig.device)
        out_e = self.eig_w(eeig).to(self.device)


        return out_e


