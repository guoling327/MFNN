#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
import os
import os.path as osp
import pickle
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from ..utils.dataset_utils import graphLoader

def Loader(name):
    hl_path = osp.join('..', 'data', name + '\\HL_' + name + '.pt')
    HL = torch.load(hl_path)
    A=[]
    for i in range(2):
        HL_sparse_tensor = HL[i + 1]
        dense_tensor = HL_sparse_tensor.to_dense()
        A.append(dense_tensor)
    return A




def encode_onehot(labels):
    # 使用LabelEncoder将标签转换为整数
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    onehot_encoded = np.eye(len(label_encoder.classes_))[integer_encoded]

    return onehot_encoded



def compute_homo(name,Order):
    """Load .mat dataset."""
    if name in ['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'chameleon', 'squirrel', 'film', 'texas', 'cornell',
                'wisconsin']:
        data, dataset = graphLoader(name)
        A=Loader(name)

        #adj = A[Order]
        adj=torch.eye(A[Order].shape[0])#0单纯性
        # features = graph.ndata['feature']
        #label = graph.ndata['label'].unsqueeze(0)
        label=data.y.squeeze(-1)

    all_idx = list(range(adj.shape[0]))
    all_labels = np.squeeze(np.array(label))
    all_anormal_idx = [i for i in all_idx if all_labels[i] == 1]
    all_normal_idx = [i for i in all_idx if all_labels[i] == 0]
    print('anomal nodes number:')
    print(len(all_anormal_idx))
    # d = torch.sum(adj_dense_tensor1,dim=1)
    num_node = int(adj.shape[0])
    labels = np.transpose(label)
    labels = encode_onehot(list(labels))
    # labels_sparse = sp.csr_matrix(labels)  # 转为稀疏矩阵形式
    # same_class_sparse = labels_sparse.dot(labels_sparse.T)  # 稀疏乘法，结果也是稀疏矩阵
    same_class = np.dot(labels, labels.T)
    tensor_same_class = torch.tensor(same_class)

    same_class_node_first = adj * tensor_same_class



    same_class_sum = torch.sum(same_class_node_first, dim=1)
    edge_sum_first = torch.sum(adj, dim=1)

    # # 计算每个节点的同配性
    homo_first = same_class_sum / (edge_sum_first + 1e-6)
    print(homo_first)
    #np.save(f'homo_{name}_{Order + 1}.npy', homo_first)
    np.save(f'homo_{name}_0.npy', homo_first)

#
# compute_homo("wisconsin",0)
# compute_homo("wisconsin",1)
# compute_homo("citeseer",0)
# compute_homo("citeseer",1)
# compute_homo("texas",0)
# compute_homo("texas",1)
# compute_homo("film",0)
# compute_homo("film",1)
# compute_homo("cora",0)
# compute_homo("cora",1)
# compute_homo("squirrel",0)
# compute_homo("squirrel",1)
# compute_homo("cornell",0)
# compute_homo("cornell",1)
compute_homo("chameleon",0)
