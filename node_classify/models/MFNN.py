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
from torch_geometric.nn import MessagePassing



class FourierEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(FourierEncoding, self).__init__()
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
        #out_e = torch.sigmoid(out_e)

        return out_e

class FourierEncoding0(nn.Module):
    def __init__(self, hidden_dim=128):
        super(FourierEncoding0, self).__init__()
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
        out_e = torch.sigmoid(out_e)

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



class MFNN(nn.Module):

    def __init__(self,dataset, args):
        super(MFNN, self).__init__()

        self.norm = args.norm
        self.nfeat = dataset.num_features
        self.nlayer = args.nlayer
        self.nheads = args.nheads
        self.hidden_dim = args.hidden
        self.feat_encoder = nn.Sequential(
            nn.Linear(dataset.num_features, args.hidden),
            nn.ReLU(),
            nn.Linear(args.hidden, args.hidden),
        )
        self.device=args.cuda
        self.dataset=args.dataset

        self.eig_encoder = FourierEncoding(args.hidden)

        self.feat_dp1 = nn.Dropout(args.dropout)
        self.feat_dp2 = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList([MFNNLayer(2, args.hidden, args.dprate) for i in range(args.nlayer)])

        self.attn_layer = nn.Linear(args.hidden, 1)

        self.attn_layer = nn.Linear(args.hidden, 1)
        self.K = args.K
        self.alpha = args.alpha
        self.fW = Parameter(torch.Tensor(self.K + 1))

        self.lin3 = nn.Linear(args.hidden, dataset.num_classes)

        self.att_0, self.att_1, self.att_2 = 0, 0, 0
        self.att_vec_0, self.att_vec_1, self.att_vec_2 = (
            Parameter(torch.FloatTensor(1 * args.hidden, 1).to(self.device)),
            Parameter(torch.FloatTensor(1 * args.hidden, 1).to(self.device)),
            Parameter(torch.FloatTensor(1 * args.hidden, 1).to(self.device)),
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
        #print(l)
       #  print("S_list0",e[0].shape)
       #  print("S_list1",e[1].shape)

        # for i in range(l):
        #     print("eilistii", e[i].shape)

        h = self.feat_dp1(x)
        h = self.feat_encoder(h)
        h = self.feat_dp2(h)


        result.append(h)
        eigencode=[]

        for i in range(l):

            N = e[i].size(0)
            ut = u[i].permute(1, 0)

            eig = self.eig_encoder(e[i])  # [N, d]

            eigencode.append(eig)

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

       #  # print(result.shape)
       #  h =  self.lin3(result)

        self.att_0, self.att_1, self.att_2 = self.attention3((result[0]), (result[1]), (result[2]))
        # print(self.att_0.size())
        h = self.att_0 * result[0] + self.att_1 * result[1] + self.att_2 * result[2]


        h=self.lin3(h)

        return F.log_softmax(h, dim=1)



class MFNN2(nn.Module):
    def __init__(self,dataset, args):
        super(MFNN2, self).__init__()


        self.nfeat = dataset.num_features
        self.nlayer = args.nlayer
        self.nheads = args.nheads
        self.hidden_dim = args.hidden

        self.feat_encoder = nn.Sequential(
            nn.Linear(dataset.num_features, args.hidden),
            nn.ReLU(),
            nn.Linear(args.hidden, args.hidden),
        )
        self.device=args.cuda
        self.Order=args.Order

        self.eig_encoder = FourierEncoding2(args.hidden)

        self.feat_dp1 = nn.Dropout(args.dropout)
        self.feat_dp2 = nn.Dropout(args.dropout)

        self.layers = nn.ModuleList([MFNNLayer(2, args.hidden, args.dprate) for i in range(args.nlayer)])
        self.K=args.K
        self.alpha = args.alpha
        self.fW = Parameter(torch.Tensor(self.K + 1))
        self.attn_layer = nn.Linear(args.hidden, 1)
        self.lin3 = nn.Linear(args.hidden, dataset.num_classes)

        self.att_0, self.att_1, self.att_2 = 0, 0, 0
        self.att_vec_0, self.att_vec_1, self.att_vec_2= (
            Parameter(torch.FloatTensor(1 * args.hidden, 1).to(self.device)),
            Parameter(torch.FloatTensor(1 * args.hidden, 1).to(self.device)),
            Parameter(torch.FloatTensor(1 * args.hidden, 1).to(self.device))
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
        # print("00", h)
        result.append(h)
       # print("00",result)

        for i in range(l):
            N = e[i].size(0)
            ut = u[i].permute(1, 0)

            eig = self.eig_encoder(e[i])  # [N, d]
            # (n,a)
            new_e = eig

            for conv in self.layers:
                basic_feats = [h]
                # list
                utx = ut @ h

                hidden = h * (self.fW[0])
                for k in range(self.K):
                    # print(new_e[:, i].unsqueeze(1).shape)
                    r = u[i] @ (new_e * utx)
                    gamma = self.fW[k + 1]
                    hidden = hidden + gamma * r
                basic_feats.append(hidden)

                basic_feats = torch.stack(basic_feats, axis=1)  # [N, m, d]
                h = conv(basic_feats)
            result.append(h)

        self.att_0, self.att_1, self.att_2 = self.attention3((result[0]), (result[1]), (result[2]))
        # print(self.att_0.size())
        h = self.att_0 * result[0] + self.att_1 * result[1] + self.att_2 * result[2]

        h=self.lin3(h)

        return F.log_softmax(h, dim=1)



