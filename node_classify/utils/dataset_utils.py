#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch

import os
import os.path as osp
import pickle
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import Actor
import pandas as pd
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import homophily
from node_classify.utils.gen_HoHLaplacian import creat_L_SparseTensor
import torch
import numpy as np
import networkx as nx
from torch_sparse import SparseTensor, fill_diag, matmul, mul

from numpy.linalg import eig, eigh
import torch_geometric

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class dataset_heterophily(InMemoryDataset):
    def __init__(self, root='data/', name=None,
                 p2raw=None,
                 train_percent=0.01,
                 transform=None, pre_transform=None):

        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(
            root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class WebKB(InMemoryDataset):
    r"""The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           'master/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def graphLoader(name):
    root_path  = osp.join('/home/luwei/die/', 'data')
   # root_path = '/home/luwei/die/'
    if name in ['cora', 'citeseer', 'pubmed']:
        path = osp.join(root_path, name)
        dataset = Planetoid(path, name=name)
        data = dataset[0]
    elif name in ['computers', 'photo']:
        path = osp.join(root_path, name)
        dataset = Amazon(path, name, T.NormalizeFeatures())
        data = dataset[0]
    elif name in ['chameleon', 'squirrel']:
        # use everything from "geom_gcn_preprocess=False" and
        # only the node label y from "geom_gcn_preprocess=True"
        preProcDs = WikipediaNetwork(
            root=root_path, name=name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(
            root=root_path, name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index
        # return dataset, data
    elif name in ['film']:
        dataset = Actor(root=osp.join(root_path, 'film'), transform=T.NormalizeFeatures())
        data = dataset[0]
    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root=root_path, name=name, transform=T.NormalizeFeatures())
        data = dataset[0]
    else:
        raise ValueError(f'dataset {name} not supported in graphLoader')

    return data, dataset


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



def DataLoader(name, args):
    # calculate higher_order adj-matrix and  save
    calculate_ = True if args.net in ['HiGCN', 'MFNN', 'MFNN2'] else False
    #hl_path = osp.join('..\\data\\' + name + '\\HL_' + name + '.pt') # 存储hl的路径
    hl_path = osp.join('..', 'data', name + '\\HL_' + name + '.pt') # 存储hl的路径

    # if name in ['cora', 'citeseer', 'pubmed']:
    #     root_path = '../'
    #     path = osp.join(root_path, 'data', name)
    #     dataset = Planetoid(path, name=name)
    #     data = dataset[0]
    # elif name in ['computers', 'photo']:
    #     root_path = '../'
    #     path = osp.join(root_path, 'data', name)
    #     dataset = Amazon(path, name, T.NormalizeFeatures())
    #     data = dataset[0]
    # elif name in ['chameleon', 'squirrel']:
    #     # use everything from "geom_gcn_preprocess=False" and
    #     # only the node label y from "geom_gcn_preprocess=True"
    #     preProcDs = WikipediaNetwork(
    #         root='../data/', name=name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
    #     dataset = WikipediaNetwork(
    #         root='../data/', name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
    #     data = dataset[0]
    #     data.edge_index = preProcDs[0].edge_index
    #     # return dataset, data
    # elif name in ['film']:
    #     dataset = Actor(
    #         root='../data/film', transform=T.NormalizeFeatures())
    #     data = dataset[0]
    # elif name in ['texas', 'cornell', 'wisconsin']:
    #     dataset = WebKB(root='../data/', name=name, transform=T.NormalizeFeatures())
    #     data = dataset[0]

    if name in ['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'chameleon', 'squirrel', 'film', 'texas', 'cornell', 'wisconsin']:
        data, dataset = graphLoader(name)

    elif name in ['Texas_null']:
        """
        Texas_null is a null model to test different effect of higher-order structures
        """
        name = 'Texas'
        path = '../data/nullModel_Texas/'
        dataset = WebKB(root='../data/',
                        name=name, transform=T.NormalizeFeatures())
        data = dataset[0]
        change = args.rho

        G = nx.Graph()
        graph_edge_list = []
        #dataset_path = '..\\data\\nullModel_' + name + '\\' + name + '_1_generate' + change + '.txt'
        dataset_path = osp.join('..','data','nullModel_'+name, name+'_1_generate' + change + '.txt')
        lines = pd.read_csv(dataset_path)
        G.add_edge(int(lines.keys()[0].split(' ')[0]), int(lines.keys()[0].split(' ')[1]))
        graph_edge_list.append([int(lines.keys()[0].split(' ')[0]), int(lines.keys()[0].split(' ')[1])])
        graph_edge_list.append([int(lines.keys()[0].split(' ')[1]), int(lines.keys()[0].split(' ')[0])])
        for line in lines.values:
            G.add_edge(int(line[0].split(' ')[0]), int(line[0].split(' ')[1]))
            graph_edge_list.append([int(line[0].split(' ')[0]), int(line[0].split(' ')[1])])
            graph_edge_list.append([int(line[0].split(' ')[1]), int(line[0].split(' ')[0])])

        data.edge_index = torch.tensor(graph_edge_list).H
        # data.HL = creat_L_SparseTensor(new_graph, maxCliqueSize=args.Order)
        #calculate_ = False
        hl_path = osp.join('..', 'data', 'nullModel_' + name, name + '_1_generate' + change + '_HL.pt')# 存储hl的路径

    else:
        raise ValueError(f'dataset {name} not supported in dataloader')




    if calculate_ and osp.exists(hl_path):
        data.HL = torch.load(hl_path)
        calculate_ = False
        if len(data.HL) < args.Order:
            calculate_ = True

    if calculate_:
        try:
            # G has been defined in the null model
            print("Runing Null model", G.number_of_nodes())
        except:
            G = nx.Graph()
            G.add_nodes_from(range(data.num_nodes))
            G.add_edges_from(data.edge_index.numpy().transpose())

        print("Calucating higher-order laplacian matix...")
        print(hl_path)
        data.HL = creat_L_SparseTensor(G, maxCliqueSize=args.Order)
        torch.save(data.HL, hl_path)
   # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    S_list = []
    U_list = []

    for i in range(args.Order):
        HL_sparse_tensor = data.HL[i + 1]  # 这里选择HL中的第一个稀疏张量进行SVD分解
        # 将稀疏张量转换为稠密张量
        # dense_tensor = HL_sparse_tensor.to_dense()
        dense_tensor = HL_sparse_tensor.to_dense()
        L = torch.eye(dense_tensor.shape[0]) - dense_tensor

        # 使用torch.svd函数进行SVD分解
        U, S, V = torch.svd(L)
        # print("U",U.shape)
        # print("S",S.shape)

        S_list.append(S.to(device))
        U_list.append(U.to(device))


    homo = homophily(data.edge_index, data.y)
    print("Home:", homo)
    print("Finish load data!")
    return dataset, data,S_list,U_list



def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        # index = (data.y[:, i] == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#
def index_to_mask2(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_planetoid_splits2(data, num_classes, percls_trn=20, val_lb=500, seed=12134):
    index = [i for i in range(0, data.y.shape[0])]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]
    # print(test_idx)

    data.train_mask = index_to_mask2(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask2(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask2(test_idx, size=data.num_nodes)

    return data

class MyTransform(object):
    def __call__(self, data):
        data.y = data.y[:, 8]  # Specify target: 0 = mu
        return data


class MyFilter(object):
    def __call__(self, data):
        return not (data.num_nodes == 7 and data.num_edges == 12) and \
               data.num_nodes < 450
