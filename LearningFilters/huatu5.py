import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import detrend

# 生成x轴特征值
# x = torch.linspace(0,2,200)
x = np.load('eigenvalues.npy')
#print(x)
# ys_c = detrend(x, type='constant')
# ys_l = detrend(x, type='linear')
x= torch.from_numpy(x)
x = x.float()

device= torch.device('cuda:' + str(2) if torch.cuda.is_available() else 'cpu')
x=x.to(device)
#groundtruth = 1-torch.exp(-10 * x**2)
groundtruth  = torch.abs(torch.sin(np.pi * x))
#groundtruth = torch.exp(-10 * (x-1)**2)

groundtruth =groundtruth.to(device)


import torch
from torch_geometric.nn import MessagePassing
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from scipy.special import comb





class BernProp(MessagePassing):
    def __init__(self, K, **kwargs):
        super(BernProp, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self, x):
        TEMP = F.relu(self.temp)

        out = None
        for k in range(self.K + 1):
            coeff = comb(self.K, k) / (2 ** self.K) * TEMP[k]
            if out is None:
                out = coeff * x.pow(k) * (2 - x).pow(self.K - k)
            else:
                out += coeff * x.pow(k) * (2 - x).pow(self.K - k)
      #  print("bern",out.shape)
     #   out = out.view(-1, 1)
        return out



class GPRProp(MessagePassing):
    def __init__(self, K, **kwargs):
        super(GPRProp, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self, x):
        TEMP =self.temp

        total = 0
        for k in range(self.K + 1):
            total += TEMP[k] * torch.pow((1 - x), k)
        return total
        # out = None
        # for k in range(self.K + 1):
        #     coeff = TEMP[k]
        #     if out is None:
        #         out = coeff * x**k
        #     else:
        #         out += coeff * x**k


class SepSineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SepSineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, 1)

    def forward(self, e):
        # input:  [N]
        # output: [N, d]

        ee = e * self.constant
       # print(ee.shape) #(2777)
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000)/self.hidden_dim)).to(e.device)
       # print(div.shape) #(16)
        pe = ee.unsqueeze(1) * div
        #print(pe.shape) #(2777,16)
        eeig = torch.cat((e.unsqueeze(1), torch.sin(pe), torch.cos(pe)), dim=1)
        # print(torch.sin(pe).shape)
        # print(eeig.shape)
       # print("sep",  eeig.shape)  #(100,1)

        ee2=self.eig_w(eeig).squeeze()

        return ee2 #(100,1)



#MFNN:
class OurSineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(OurSineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim+1, 1)

    def forward(self, e):
        ee = e.unsqueeze(1)
        #ee = ee*ee
        #eeig = ee
        eeig = torch.full(ee.shape, torch.tensor(1.0)).to(e.device)

        for i in range(int(self.hidden_dim)):
            #sin = torch.sin((i+1) * math.pi /2 * ee).to(e.device)
            cos = torch.cos((i+1) * math.pi /2 * ee).to(e.device)
            eeig = torch.cat((eeig, cos), dim=1)
           #eeig = torch.cat((eeig, sin, cos), dim=1)

        out_e = self.eig_w(eeig).squeeze().to(e.device)
        #print(out_e.shape) #(100,1)
        return out_e


class GPRProp2(MessagePassing):
    def __init__(self, K, alpha, Init='Random', Gamma=None, bias=True, **kwargs):
        super(GPRProp2, self).__init__(aggr='add', **kwargs)
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

    def forward(self, x):
        TEMP =self.temp
        total = 0
        for k in range(self.K + 1):
            total += TEMP[k] * torch.pow((1 - x), k)
        return total




fourier = OurSineEncoding(32).to(device)
optimizer = torch.optim.Adam(fourier.parameters(), lr=0.01, weight_decay=0.0005)

for epoch in range(1000):
    optimizer.zero_grad()
    h = fourier(x)
    loss = F.mse_loss(h, groundtruth)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')



Bern=BernProp(32).to(device)
optimizer2 = torch.optim.Adam(Bern.parameters(), lr=0.01, weight_decay=0.0005)
# groundtruth = groundtruth.view(-1, 1)
for epoch in range(1000):
    optimizer2.zero_grad()
    h2 = Bern(x)
    loss2 = F.mse_loss(h2, groundtruth)
    loss2.backward()
    optimizer2.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss2.item()}')



GPR= GPRProp2(32,0.5).to(device)
optimizer4 = torch.optim.Adam(GPR.parameters(), lr=0.01, weight_decay=0.0005)
# groundtruth = groundtruth.view(-1, 1)
for epoch in range(1000):
    optimizer4.zero_grad()
    h4 = GPR(x)
    #print(h4)
    loss4 = F.mse_loss(h4, groundtruth)
    loss4.backward()
    optimizer4.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss4.item()}')



Sep = SepSineEncoding(32).to(device)
optimizer3 = torch.optim.Adam(Sep.parameters(), lr=0.01, weight_decay=0.0005)

for epoch in range(1000):
    optimizer3.zero_grad()
    h3 = Sep(x)
    loss3 = F.mse_loss(h3, groundtruth)
    loss3.backward()
    optimizer3.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss3.item()}')
# 绘制图形

markers = ['o', 's', '^', 'p', '.', 'v',
           ',', 'd', 'h', '2', 'x',
           '4', 'd', '+']
Linestyle = ['-', '--', '-.', ':', '-', '--',
             '-.', ':', '-', '--', '-.',
             ':', '-', '--']

plt.figure(figsize=(10, 7))

plt.plot(x.cpu().numpy(), groundtruth.cpu().numpy(), label='GroundTruth', color='red')
plt.plot(x.cpu().numpy(), h4.cpu().detach().numpy(), label='GPR-GNN',  linestyle=Linestyle[1],color='m')
plt.plot(x.cpu().numpy(), h2.cpu().detach().numpy(), label='BernNet',  linestyle=Linestyle[2],color='green')
plt.plot(x.cpu().numpy(), h3.cpu().detach().numpy(), label='Sepecformer', linestyle=Linestyle[3], color='blue')
plt.plot(x.cpu().numpy(), h.cpu().detach().numpy(), label='MFNN', linestyle=Linestyle[1],color='black')
# markers[i]

plt.xlabel('Raw Eigenvalues λ',fontsize=23)
plt.ylabel('New Eigenvalues h(λ)',fontsize=23)
#plt.title('1-exp(-10λ$^2$)',fontsize=23)
#plt.title('exp(-10(λ-1)$^2$)',fontsize=23)
plt.xticks(fontsize=20)
# 把y轴刻度标签字体大小改为14号
plt.yticks(fontsize=20)

plt.title('|sin(πλ)|',fontsize=23)
# plt.xlim(0,2)
# plt.ylim(0,1.5)
plt.legend(fontsize=20)
plt.grid(True)
plt.savefig('../LearningFilters/tu/comb7.jpg', dpi=600)
plt.savefig('../LearningFilters/tu/comb7.eps', dpi=600)
# plt.savefig('/home/luwei/HiGCN-master/node_classify/tu/try5.eps', dpi=600)
plt.show()