
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as Var
import inspect
# from gpu_mem_track import MemTracker
# from torch.nn.utils.rnn import pad_sequence
# from torch.nn.utils.rnn import pack_padded_sequence
# from torch.nn.utils.rnn import pad_packed_sequence
# from gensim.models.word2vec import Word2Vec
#
# import torch_geometric.nn as pyg_nn
# import torch_geometric.utils as pyg_utils


import time
from datetime import datetime

import networkx as nx

# from torch_geometric.datasets import TUDataset
# from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, Dataset, download_url
from torch_geometric.nn import GCNConv

# import torch_geometric.transforms as T

# from tensorboardX import SummaryWriter
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import os

import logging
from pathlib import Path
import json
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import math
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


try:
   import cPickle as pickle
except:
   import pickle


class GCN(torch.nn.Module):
    def __init__(self, input_size=128, output_size=2, hidden_size=128):
        super().__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, data):
        # x, edge_index = data.x, data.edge_index
        # print("1 x.shape", x.shape)
        # x = self.conv1(x, edge_index)
        # print("2 x.shape", x.shape)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # print("3 x.shape", x.shape)
        # exit()
        # x = self.conv2(x, edge_index)


        x, edge_index = data.x, data.edge_index
        # print("1 x.shape", x.shape)
        x = self.conv1(x, edge_index)
        # print("2 x.shape", x.shape)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc(x)
        # print("3 x.shape", x.shape)
        # exit()


        return F.log_softmax(x, dim=1)

def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = int((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2)
        w_pad = int((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2)
        # print("h_pad:",h_pad)
        # print("w_pad:",w_pad)
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if (i == 0):
            spp = x.view(num_sample, -1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    # return emb_layer, num_embeddings, embedding_dim
    return emb_layer


class F1_Loss(nn.Module):
    '''
    From https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354

    Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true, ):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, dropout1, device):
        # in_dim is the input dim and mem_dim is the output dim
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        # self.H = []
        self.drop = nn.Dropout(dropout1)
        self.device = device

    def node_forward(self, inputs, child_c, child_h):
        # print("input", inputs.shape)
        inputs = torch.unsqueeze(inputs, 0).to(self.device)
        # print("input unsqueeze",inputs.shape)
        child_h_sum = torch.sum(child_h, dim=0).to(self.device)
        child_h = child_h.to(self.device)
        child_c = child_c.to(self.device)
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(self.fh(child_h) + self.fx(inputs).repeat(len(child_h), 1))
        fc = torch.mul(f, child_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=0)
        h = torch.mul(o, torch.tanh(c))
        # self.H.append(h)
        return c, h

    def forward(self, data):
        tree = data[0]
        inputs = data[1]
        # The inputs here are the tree structure built from class Tree and the input is a list of values with the
        # node ids as the key to store the tree values
        _ = [self.forward([tree.children[idx], inputs]) for idx in range(tree.num_children)]

        if tree.num_children == 0:
            # print("jere",type(inputs[0]))
            # print("before crash",inputs)
            child_c = Var(inputs[tree.id].data.new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[tree.id].data.new(1, self.mem_dim).fill_(0.))
        else:
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
        # print("id",tree.id,"input len",len(inputs))
        tree.state = self.node_forward(inputs[tree.id], child_c, child_h)
        return tree.state


class DLVP(nn.Module):
    """

    """
    def __init__(self, params, lp_weights_matrix, ns_weights_matrix):
        """
        :param input_dim: max(dataset.num_node_features, 1)
        :param hidden_dim: 128
        :param output_dim: dataset.num_classes
        :param task: 'node' or 'graph'
        """
        super(DLVP, self).__init__()
        self.params = params

        # PDT
        self.pdt_M = 15  # num of tokens in a statement
        self.pdt_dim = 85  # [8, 4, 2, 1] 的平方和

        # LP
        self.lp_N = 285  # num of LP in a method
        self.lp_M = 42  # num of tokens in a LP

        # NS
        self.ns_N = 242  # num of NS in a method
        self.ns_M = 16  # num of tokens in a NS

        # Callers & callees
        self.CC_C = 45  # 每个 funtion 最多存储 Caller callee 的数量
        self.CC_N = 242
        self.CC_M = 16

        # num of statement types
        self.st_type_num = 12

        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.output_dim = params['output_dim']
        self.batch_size = params['batch_size']
        self.dropout = params['dropout']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # LP
        self.lp_dim = params['lp_dim']
        self.lp_embedding = create_emb_layer(lp_weights_matrix, True)
        self.lp_gru = nn.GRU(input_size=params['lp_dim'], hidden_size=params['lp_dim'], num_layers=1, batch_first=True)

        # NS
        self.ns_dim = params['ns_dim']
        self.ns_embedding = create_emb_layer(ns_weights_matrix, True)
        self.ns_gru = nn.GRU(input_size=params['ns_dim'], hidden_size=params['ns_dim'], num_layers=1, batch_first=True)

        # PDT (Tree-LSTM)
        # self.tree_lstm = ChildSumTreeLSTM(self.pdt_dim, self.pdt_dim, self.dropout, self.device)

        # Callers and callees
        self.cc_gru = nn.GRU(input_size=params['ns_dim'], hidden_size=params['ns_dim'], num_layers=1, batch_first=True)
        self.fc_cc = nn.Linear(self.CC_C * 85 , params['input_dim'])

        # statement type weights
        # self.st_weights = torch.nn.Parameter(torch.ones(self.st_type_num, dtype=torch.float).requires_grad_())
        self.st_weights = []
        for i in range(self.st_type_num):
            self.st_weights.append( nn.Parameter(torch.tensor(1.)) )

        # EFG (Exception Flow Graph)
        print("init GCNConv...")
        self.efg_conv = GCNConv(params['input_dim'], params['input_dim'])
        print("init GCNConv success")

        self.fc1 = nn.Linear(85 * 2 + self.ns_dim * 3, params['input_dim'])
        self.fc2 = nn.Linear(params['input_dim'], params['hidden_dim'])
        self.fc3 = nn.Linear(params['hidden_dim'], params['output_dim'])
        # self.attention = torch.nn.MultiheadAttention(128 + 84 + 85 + 85, 2)
        # self.fc_lp = nn.Linear(self.lp_N * params['input_dim'], params['input_dim']) # need too much memory
        # self.fc_ns = nn.Linear(self.ns_N * params['input_dim'], params['input_dim']) # need too much memory
        self.softmax = nn.Softmax(dim=1)

        self.loss_layer = nn.CrossEntropyLoss()
        # self.loss_layer = F1_Loss()



    def forward(self, data):
        batch, y = data
        # frame = inspect.currentframe()
        # gpu_tracker = MemTracker(frame)
        # print("new batch start")

        x_batch = []
        n2v_batch = []
        for d in batch:
            # PDT (Tree-LSTM)
            # t1 = time.time()
            x_pdt = d.x_pdt.to(self.device)

            # t2 = time.time()

            # LP
            lp_len_list = torch.tensor(d.lp_len_list)
            x_lp_padded = pad_sequence(d.x_lp, batch_first=True, padding_value=0).to(self.device)
            x_lp = self.lp_embedding(x_lp_padded)
            x_lp_packed = pack_padded_sequence(x_lp, lp_len_list, batch_first=True, enforce_sorted=False)
            _, vec = self.lp_gru(x_lp_packed)
            vec = F.pad(vec, pad=(0, 0, 0, 50 - vec.shape[1])) # vec.shape[1] 太小，后面SPP可能会报错
            x_lp = spatial_pyramid_pool(vec, 1, [vec.shape[1], vec.shape[2]], [8, 4, 2, 1]).view(85, )
            del vec, _, x_lp_padded, x_lp_packed


            # t3 = time.time()

            # NS
            ns_len_list = torch.tensor(d.ns_len_list)
            x_ns_padded = pad_sequence(d.x_ns, batch_first=True, padding_value=0).to(self.device)
            x_ns = self.ns_embedding(x_ns_padded)
            x_ns_packed = pack_padded_sequence(x_ns, ns_len_list, batch_first=True, enforce_sorted=False)
            _, vec = self.ns_gru(x_ns_packed)
            vec = F.pad(vec, pad=(0, 0, 0, 50 - vec.shape[1]))
            x_ns = spatial_pyramid_pool(vec, 1,  [vec.shape[1], vec.shape[2]], [8, 4, 2, 1]).view(85, )
            del vec, _, x_ns_padded, x_ns_packed

            # EFG
            x_efg, efg_edge_index = d.efg_x.to(self.device), d.efg_edge_index
            if efg_edge_index.shape[0] == 0:
                x_efg = torch.zeros(self.ns_dim, dtype=torch.float).to(self.device)
            else:
                efg_edge_index = efg_edge_index.to(self.device)
                # print("1 x.shape", x.shape)
                x_efg = self.efg_conv(x_efg, efg_edge_index)
                x_efg = F.relu(x_efg)
                x_efg = torch.mean(x_efg, 0).view(128, )


            # t4 = time.time()

            if d.has_cc == 1:
                cc_embeddings = []
                # gpu_tracker.track()

                weights = []
                for i in range(len(d.cc_st_types)):
                    weights = weights + d.cc_st_types[i]
                l = len(weights)
                weights = torch.tensor(weights, dtype=torch.float).to(self.device).view(l, 1) # [C * N]


                cc_embeddings = []
                for i in range(len(d.cc_list)):
                    cc_len_list = torch.tensor(d.cc_len_list[i])
                    x_cc_padded = pad_sequence(d.cc_list[i], batch_first=True, padding_value=0).to(self.device)
                    x_cc = self.ns_embedding(x_cc_padded)
                    x_cc_packed = pack_padded_sequence(x_cc, cc_len_list, batch_first=True, enforce_sorted=False)
                    _, vec = self.cc_gru(x_cc_packed) # vec: [1, N, 128]
                    cc_embeddings.append(vec)
                cc_embeddings = torch.cat(cc_embeddings, dim=1).to(self.device) # [1, N*C, 128]
                cc_embeddings = cc_embeddings.squeeze(dim=0) # [N*C, 128]

                x_cc = (cc_embeddings * weights).mean(dim=0)
                n2v_batch.append(d.x_n2v.to(self.device))

                # x_cc = xx_cc * d.x_n2v.to(self.device)
                # t8 = time.time()
                del cc_embeddings, vec, cc_len_list, x_cc_padded, x_cc_packed, _
            else:
                # t5 = time.time()
                # t6 = t5
                # t7 = t5
                # t8 = t5
                x_cc = torch.zeros(self.ns_dim, dtype=torch.int).to(self.device)
                n2v_batch.append(torch.ones(self.ns_dim, dtype=torch.float).to(self.device))

            # t5 = time.time()
            # print("{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format( t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t8-t7 ))
            x_concated = torch.cat([x_pdt, x_lp, x_ns, x_efg, x_cc], 0).to(self.device)
            # print(x_pdt.shape, x_lp.shape,x_ns.shape, x_cc.shape )
            x_batch.append(x_concated)
            del x_pdt, x_lp, x_ns, x_cc, x_efg
        # prof.export_chrome_trace('./dlvp_profile.json')
        # exit()

        x = torch.stack(x_batch).to(self.device)
        n2v = torch.stack(n2v_batch).to(self.device)
        # x = x.view( 1, x.shape[0], x.shape[1] ) # tgt_len, bsz, embed_dim
        # x, _ = self.attention(x, x, x)
        # x = x.view(x.shape[1], x.shape[2])
        # print("x:", x.shape)
        x = F.relu(self.fc1(x))
        x = x * n2v

        # for TSNE
        # torch.cuda.empty_cache()
        # return self.fc2(x)

        x = F.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        del x_batch, n2v, n2v_batch
        torch.cuda.empty_cache()

        return x

    def loss(self, pred, label):
        # CrossEntropyLoss torch.nn.CrossEntropyLoss
        # return F.nll_loss(pred, label)
        # logger.info("pred: {}, label: {}".format(pred.shape, label.shape) )
        # exit()
        return self.loss_layer(pred, label)



if __name__ == '__main__':
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='/tmp/Cora', name='Cora')

    input_feature_size = dataset.num_node_features
    output_size = dataset.num_classes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: {}".format(device))
    model = GCN(input_feature_size, output_size, input_feature_size).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')
