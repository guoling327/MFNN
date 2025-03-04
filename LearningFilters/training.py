from utils import filtering,filtering2,visualize,TwoDGrid
import numpy as np
import os 
import argparse
import torch
from models import ChebNet,BernNet,GcnNet,GatNet,ARMANet,GPRNet,MFNN
from sklearn.metrics import r2_score


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--early_stopping', type=int,default=100)
parser.add_argument('--lr',type=float,default=0.01)
parser.add_argument('--filter_type',type=str,choices=['low','high','band','rejection','comb','low_band'],default='band')
parser.add_argument('--net',type=str,choices=['ChebNet','BernNet','GcnNet','GatNet','ARMANet','GPR','MFNN0'],default='MFNN')
parser.add_argument('--img_num',type=int,default=50)
parser.add_argument('--cuda',type=int,default=2)
args = parser.parse_args()
print(args)
print("---------------------------------------------")

if os.path.exists('y_'+args.filter_type+'.npy'):
	y=np.load('y_'+args.filter_type+'.npy')
else:
	y=filtering(args.filter_type)
y=torch.Tensor(y)

if os.path.exists('lamda_'+args.filter_type+'.npy'):
	lamda=np.load('lamda_'+args.filter_type+'.npy')
else:
	lamda=filtering2(args.filter_type)

lamda=torch.Tensor(lamda)

# lamda=np.load('lamda_'+args.filter_type+'.npy')
dataset = TwoDGrid(root='data/2Dgrid', pre_transform=None)
data=dataset[0]

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
# y=y.to(device)
data=data.to(device)
lamda=lamda.unsqueeze(1).to(device)
#
# print("lamda",lamda.shape)


def train(img_idx, model, optimizer):
	model.train()
	optimizer.zero_grad()
	pre = model(data)
	#print("pre", pre.shape)

	loss = torch.square(data.m * (pre - lamda)).sum()
	loss.backward()
	optimizer.step()
	# print("pre", pre)
	# print("lamda", lamda)

	a = pre[data.m == 1]
	# b=lamda[:,img_idx:img_idx+1]
	b = lamda[data.m == 1]
	# print("a", a)
	# print("b", b)

	r2 = r2_score(b.cpu().detach().numpy(), a.cpu().detach().numpy())

	return loss.item(), r2

results=[]
for img_idx in range(args.img_num):
	data.x_tmp=data.x[:,img_idx:img_idx+1]
	data.edge_index_tmp = data.edge_index[:, img_idx:img_idx + 1].to(device)
	#
	# print(data.edge_index.shape) #torch.Size([2, 39600])
	# print(data.edge_index_tmp.shape) #torch.Size([2, 1])

	if args.net=='ChebNet':
		model=ChebNet().to(device)
	elif args.net=='BernNet':
		model=BernNet().to(device)
	elif args.net=='GcnNet':
		model=GcnNet().to(device)
	elif args.net=='GatNet':
		model=GatNet().to(device)
	elif args.net=='ARMANet':
		model=ARMANet().to(device)
	elif args.net=='GPRNet':
		model=GPRNet().to(device)
	elif args.net=='MFNN':
		model=MFNN().to(device)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	best_r2=0
	min_loss=float('inf')
	cnt=0
	re_epoch=args.epochs

	for epoch in range(args.epochs):
		loss,r2=train(img_idx,model,optimizer)
		if(min_loss>loss):
			min_loss=loss
			best_r2=r2
			cnt=0
		else:
			cnt+=1
		if(cnt>args.early_stopping):
			re_epoch=epoch+1
			break

	results.append([min_loss,best_r2])
	print(f'img: {img_idx+1} \t loss= {min_loss:.4f} \t r2= {best_r2:.4f} \t epoch: {re_epoch}')
	if(args.net=="BernNet"):
		TEST = model.coe.clone()
		theta = TEST.detach().cpu()
		theta=torch.relu(theta).numpy()
		print('Theta:', [float('{:.4f}'.format(i)) for i in theta])

loss_mean, r2_mean= np.mean(results, axis=0)
print("---------------------------------------------")
print(f'mean loss= {loss_mean:.4f} \t mean r2 acc = {r2_mean:.4f}')

f = open("../LearningFilters/res/{}_{}.txt".format(args.filter_type,args.net), 'a')
f.write(f'mean loss= {loss_mean:.4f} \t mean r2 acc = {r2_mean:.4f}')
f.write("\n")
f.close()

