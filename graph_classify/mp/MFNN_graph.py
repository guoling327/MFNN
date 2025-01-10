import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
from torch.nn.parameter import Parameter
from torch_sparse import matmul
from mp.nn import get_nonlinearity, get_pooling_fn
from torch.nn import Linear, Sequential, BatchNorm1d as BN,ReLU
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear




class FourierEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super( FourierEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim+1, 1)

    def forward(self, e):
        # input:  [N]
        # output: [N, d]
        #ee = e * self.constant
        ee = e.unsqueeze(1)
        #ee = ee*ee
        #eeig = ee
        eeig = torch.full(ee.shape, torch.tensor(1.0)).to(e.device)

        for i in range(int(self.hidden_dim)):
           # sin = torch.sin((i+1) * math.pi * ee).to(e.device)
            cos = torch.cos((i+1) * math.pi * ee).to(e.device)
            eeig = torch.cat((eeig, cos), dim=1)

        out_e = self.eig_w(eeig).to(e.device)

        out_e =torch.sigmoid(out_e)

        return out_e

class MFNNLayer(nn.Module):

    def __init__(self, nbases, ncombines, prop_dropout=0.0):
        super(MFNNLayer, self).__init__()
        self.prop_dropout = nn.Dropout(prop_dropout)


        self.weight = nn.Parameter(torch.ones((1, nbases, ncombines)))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)
        self.norm = nn.LayerNorm(ncombines)


    def forward(self, x):
        x = self.prop_dropout(x) * self.weight      # [N, m, d] * [1, m, d]
        x = torch.sum(x, dim=1)

        if self.norm is not None:
            x = self.norm(x)
            x = F.relu(x)

        return x

class MFNN_prop(MessagePassing):
    def __init__(self, K, alpha, Order=2, bias=True, **kwargs):
        super(MFNN_prop, self).__init__(aggr='add', **kwargs)
        # K=10, alpha=0.1, Init='PPR',
        self.K = K
        self.alpha = alpha
        self.Order = Order
        self.fW = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.fW)
        for k in range(self.K + 1):
            self.fW.data[k] = self.alpha * (1 - self.alpha) ** k
        self.fW.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, h,u,e):
        hidden = h * (self.fW[0])
        ut = u.permute(1, 0)
        utx = ut @ h
        for k in range(self.K):
            x = u @ (e.unsqueeze(1) * utx)
            gamma = self.fW[k + 1]
            hidden = hidden + gamma * x
        return hidden




#没有MFNN_prop
class MFNN_Message(MessagePassing):

    def __init__(self, num_features, order, hidden, num_classess, dropout_rate, nn2, device,**kwargs):
        kwargs.setdefault('aggr', 'add')
        super(MFNN_Message, self).__init__()

        self.nn = nn2
        self.nfeat = num_features
        self.nlayer = 1
        self.hidden_dim = hidden
        self.dropout = dropout_rate
        self.dprate = dropout_rate
        self.Order = order

        self.lin_in = torch.nn.ModuleList()

        self.feat_encoder = Sequential(
            Linear(num_features, hidden),
            ReLU(),
            Linear(hidden, hidden),
        )
        self.device= device  #关于cuda

        # for arxiv & penn
        #self.linear_encoder = nn.Linear(nclass, nclass)
       #self.classify = nn.Linear(nclass, nclass)

        self.eig_encoder = FourierEncoding(hidden)

        self.feat_dp1 = nn.Dropout(self.dropout)
        self.feat_dp2 = nn.Dropout(self.dropout)

        self.layers = nn.ModuleList([MFNNLayer(2, hidden, self.dprate) for i in range(self.nlayer)])

        self.attn_layer = nn.Linear(hidden, 1)

        self.attn_layer = nn.Linear(hidden, 1)
        self.K = 10
        self.alpha = 0.5
        self.fW = Parameter(torch.Tensor(self.K + 1))

        # dim = int(args.Order * dataset.num_classes)
        self.lin3 = nn.Linear(hidden, num_classess)

        self.att_0, self.att_1, self.att_2, self.att_3 = 0, 0, 0, 0
        self.att_vec_0, self.att_vec_1, self.att_vec_2, self.att_vec_3 = (
            Parameter(torch.FloatTensor(1 * hidden, 1).to(self.device)),
            Parameter(torch.FloatTensor(1 * hidden, 1).to(self.device)),
            Parameter(torch.FloatTensor(1 * hidden, 1).to(self.device)),
            Parameter(torch.FloatTensor(1 * hidden, 1).to(self.device)),
        )
        self.att_vec = Parameter(torch.FloatTensor(3, 3).to(self.device))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.fW)
        for k in range(self.K + 1):
            self.fW.data[k] = self.alpha * (1 - self.alpha) ** k
        self.fW.data[-1] = (1 - self.alpha) ** self.K


        std_att = 1.0 / math.sqrt(self.att_vec_2.size(1))
        std_att_vec = 1.0 / math.sqrt(self.att_vec.size(1))

        self.att_vec_1.data.uniform_(-std_att, std_att)
        self.att_vec_0.data.uniform_(-std_att, std_att)
        self.att_vec_2.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)


    def attention3(self, output_0, output_1, output_2):
        logits = (
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm((output_0), self.att_vec_0),
                            torch.mm((output_1), self.att_vec_1),
                            torch.mm((output_2), self.att_vec_2),
                        ],
                        1,
                    )
                ),
                self.att_vec,
            )
        )
        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]


    def forward(self, data,e,u):
        x = data.x
        result = []
        l= len(e)


        h = self.feat_dp1(x)
        h = self.feat_encoder(h)
        h = self.feat_dp2(h)

        result.append(h)

        for i in range(l):

            N = e[i].size(0)
            ut = u[i].permute(1, 0)

            eig = self.eig_encoder(e[i])  # [N, d]
            # (n,a)
            new_e = eig
           # print(new_e.shape) #(183,1)

            for conv in self.layers:
                basic_feats = [h]
                # list
                utx = ut @ h

                hidden = h * (self.fW[0])

                for k in range(self.K):
                    #print(new_e[:, i].unsqueeze(1).shape)
                    r = u[i] @ (new_e * utx)
                    gamma = self.fW[k + 1]
                    hidden = hidden + gamma * r
                basic_feats.append(hidden)
                # (n,d)
                # for i in range(self.nheads):
             #   basic_feats.append(u[i] @ (new_e * utx))  # [N, d]

                basic_feats = torch.stack(basic_feats, axis=1)  # [N, m, d]
                h = conv(basic_feats)
               # print(h.shape)


            result.append(h)

        self.att_0, self.att_1, self.att_2, self.att_3 = self.attention4((result[0]), (result[1]), (result[2]), (result[3])
        h=self.lin3(h)

        return F.log_softmax(h, dim=1)

#PTC MUATG
class MFNN(torch.nn.Module):
    def __init__(self, max_petal_dim, num_features, num_layers, hidden, num_classes, device,readout='sum',
                 dropout_rate=0.5, nonlinearity='relu'):
        super(MFNN, self).__init__()
        self.order = max_petal_dim
        self.device = device
        self.pooling_fn = get_pooling_fn(readout)
        self.nonlinearity = nonlinearity
        self.dropout_rate = dropout_rate
        conv_nonlinearity = get_nonlinearity(nonlinearity, return_module=True)
        self.conv1 = MFNN_Message(num_features, self.order, hidden, hidden, dropout_rate,
            Sequential(
                Linear(hidden, hidden),
                BN(hidden),
                conv_nonlinearity(),
                Linear(hidden, hidden),
                BN(hidden),
                conv_nonlinearity(),
            ), device)
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers - 1):
            self.convs.append(
                MFNN_Message(hidden, self.order, hidden, hidden, dropout_rate,
                    Sequential(
                        Linear(hidden, hidden),
                        BN(hidden),
                        conv_nonlinearity(),
                        Linear(hidden, hidden),
                        BN(hidden),
                        conv_nonlinearity(),
                    ), device)
            )
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        model_nonlinearity = get_nonlinearity(self.nonlinearity, return_module=False)
        x, HL, batch = data.x, data.HL, data.batch

        S_list = []
        U_list = []
        for i in range(self.order):
            HL_sparse_tensor = HL[i+1]
            dense_tensor = HL_sparse_tensor.to_dense()
            L = torch.eye(dense_tensor.shape[0]) - dense_tensor

            # # 使用torch.svd函数进行SVD分解
            # U, S, V = torch.svd(L)
            # # print("U",U.shape)
            # dense_tensor = HL_sparse_tensor.to_dense()
            L = L.float()

          #  L = torch.eye(dense_tensor.shape[0]) - dense_tensor  是否需要单位阵减
            U, S, V = torch.svd(L)

            S_list.append(S.to(self.device))
            U_list.append(U.to(self.device))

        #e, u=data.e,data.u
        x = self.conv1(x,S_list,U_list)
        for conv in self.convs:
            x = conv(x, HL)
        x = self.pooling_fn(x, batch)#readout
        x = model_nonlinearity(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x



