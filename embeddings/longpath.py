import networkx as nx
import os
import json
import numpy as np
import logging
import re
import time
import torch

try:
   import cPickle as pickle
except:
   import pickle


def get_joern_id(line):
    p1 = re.compile(r'joern_id_[(](.*?)[)]', re.S)
    res = re.findall(p1, line)
    if len(res) > 0:
        return res[0]
    return ''


def get_joern_type(line):
    p1 = re.compile(r'joern_type_[(](.*?)[)]', re.S)
    res = re.findall(p1, line)
    if len(res) > 0:
        return res[0]
    return ''


def get_joern_name(line):
    p1 = re.compile(r'joern_name_[(](.*?)[)]', re.S)
    res = re.findall(p1, line)
    if len(res) > 0:
        return res[0]
    return ''


def ast2lp(str_ast):
    # 1. extract nodes & edges
    nodes = {}
    edges = []

    for line in str_ast.splitlines():
        if line.find('" -->> "') > -1:
            a, b = line.split('" -->> "')
            id_a = get_joern_id(a)
            id_b = get_joern_id(b)
            if id_a == '' or id_b == '':
                continue

            type_a = get_joern_type(a)
            name_a = get_joern_name(a)
            if name_a != '':
                v_a = name_a
            else:
                v_a = type_a

            type_b = get_joern_type(b)
            name_b = get_joern_name(b)
            if name_b != '':
                v_b = name_b
            else:
                v_b = type_b

            nodes[id_a] = v_a
            nodes[id_b] = v_b

            edges.append([id_a, id_b])

    # 2. generate LPs
    lp_list = []
    G = nx.DiGraph()
    G.add_edges_from(edges)

    leafnodes = [x for x in G.nodes() if G.out_degree(x) == 0 and G.in_degree(x) == 1]
    rootnode = [x for x in G.nodes() if G.in_degree(x) == 0]
    for node in leafnodes:
        # 找到每一个 从 叶节点 到 根节点 的 path
        try:
            lp_list.append(nx.shortest_path(G, source=rootnode[0], target=node))
        except Exception as e:
            print(str(e))
            pass

    # print(lp_list)

    if len(lp_list) % 2 == 1:
        lp_list.append(lp_list[-1])

    long_path_list = []
    token_list = []
    for i in range(len(lp_list)):
        if i % 2 == 0:
            lp_list[i].reverse()
            lp = lp_list[i] + lp_list[i + 1][1:]
            lp_tokens = [nodes[j] for j in lp]

            long_path_list.append(lp)
            token_list.append(lp_tokens)

            # print( token_list )
    # print(len(token_list))

    return token_list



class Longpath():
    def __init__(self, input_path, output_path, emb_dim=128):
        self.input_path = input_path
        self.output_path = output_path
        self.emb_dim = emb_dim

        self.corpus_file = output_path + "/lp_corpus.pkl"
        self.vector_file = output_path + "/lp_glove_vectors.pkl"  # 即 weights_matrix，用于 pytorch 的 embedding layer
        self.idx_file = output_path + "/lp_glove_idx.pkl"  # word2idx

        self.vectors = None
        self.word2idx = None

        self.retrain = True
        self.load_model()

    def load_model(self):
        if os.path.exists(self.vector_file) and os.path.exists(self.idx_file):
            self.vectors = pickle.load(open(self.vector_file, 'rb'))
            self.word2idx = pickle.load(open(self.idx_file, 'rb'))
        else:
            print("file not existed: {} or {}".format(self.vector_file, self.idx_file))
            exit()

    def get_lp_idx(self, ast_str, N, M):
        path_list = ast2lp(ast_str)
        res = []
        len_list = []
        for i in range(N):
            if i >= len(path_list):
                break
            tokens = path_list[i]
            row = []
            for j in range(M):
                if j >= len(tokens):
                    break
                if tokens[j] in self.word2idx.keys():
                    idx = self.word2idx[tokens[j]]
                    # matrix[i][j] = idx
                    row.append(idx)
            if len(row) > 0:
                res.append( torch.tensor(row) )
                len_list.append( len(row) )
        if len(res) < 8:
            for i in range( 8-len(res) ):
                res.append(torch.tensor([0]))
                len_list.append(1)
        return res, len_list


