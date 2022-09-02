

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torch.nn.utils.rnn import pad_sequence
# from torch.nn.utils.rnn import pack_padded_sequence
# from torch.nn.utils.rnn import pad_packed_sequence
# from gensim.models.word2vec import Word2Vec
#
# import torch_geometric.nn as pyg_nn
# import torch_geometric.utils as pyg_utils

import time
from datetime import datetime

# import networkx as nx

# from torch_geometric.datasets import TUDataset
# from torch_geometric.datasets import Planetoid
# from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, Dataset, download_url

# import torch_geometric.transforms as T

# from tensorboardX import SummaryWriter
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import pandas as pd
import os

import logging
from pathlib import Path
import json
import argparse
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
try:
   import cPickle as pickle
except:
   import pickle
from scenarios import get_function_buggy_code

import random
random.seed(9)

from embeddings.ns import NaturalSeq
from embeddings.longpath import Longpath
from embeddings import pdt, efg
from model import spatial_pyramid_pool
from embeddings.st_types import StType

from config import *

def findAllFile(base, full=True):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if full:
                yield os.path.join(root, f)
            else:
                yield f

def split_trees(tree):
    """
    joern 生成的 AST、PDG 等融合在一块，这个 function 把它们分开。
    """
    res = {}
    tree_type = ""
    tree_body = ""
    for line in tree.splitlines():
        if line.strip() == "":
            continue
        if line[0] == '#':
            if tree_type != "" and tree_body != "":
                res[ tree_type ] = tree_body
                tree_body = ""

            tree_type = line[2:].strip()
            # print(tree_type)
        elif tree_type != "":
            tree_body += line + "\n"
    if tree_body != "":
        res[tree_type] = tree_body
    return res

def read_keys(keys_file):
    keys_index = {}
    ii = 0
    with open(keys_file, "r") as f:
        for line in f.readlines():
            l = line.strip()
            if l != "" and l not in keys_index.keys():
                keys_index[l] = ii
                ii += 1
    return keys_index


class Tree(object):
    # Use this structure to create tree data
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.id = None

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

def build_tree(edge_list, value_list, id, store_value):
    root = Tree()
    root.id = id
    store_value[id] = torch.tensor(value_list.get(id), dtype=torch.float)
    if edge_list.get(id):
        for child_id in edge_list.get(id):
            # print("{} --> {}".format( id, child_id ))
            if child_id not in value_list.keys():
                continue
            new_child, store_value = build_tree(edge_list, value_list, child_id, store_value)
            root.add_child(new_child)
    return root, store_value

def build_tree_for_lstm(pdt_tree, ns_obj, M, dim):
    edge_list = {}
    value_list = {}
    root_id = 10000
    for e in pdt_tree['edges']:
        n1 = e[0]
        n2 = e[1]
        if n2 not in edge_list.keys():
            edge_list[n2] = []
        if n1 != n2:
            edge_list[n2].append(n1)
    for k, statement in pdt_tree['features'].items():
        matrix = ns_obj.statement2vec(statement[0], M)
        matrix = torch.tensor(matrix, dtype=torch.float).view( 1, 1, M, 128 )
        value_list[ int(k) ] = spatial_pyramid_pool(matrix, 1, [M, 128], [8, 4, 2, 1]).view(85,)
        if int(k) < root_id:
            root_id = int(k)

    if root_id == 10000:
        return None, None

    store_value = {}
    tree_t, tree_v = build_tree(edge_list, value_list, root_id, store_value)
    # print(tree_t)
    # print(tree_v)
    return tree_t, tree_v

class MyLargeDataset(Dataset):
    def __init__(self, root="", params=None, rand_key_file_path="", transform=None, pre_transform=None):
        self.save_path = root
        self.params = params
        self.rand_key_file_path = rand_key_file_path
        super(MyLargeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        res = []
        for f in findAllFile(self.processed_dir, full=True):
            if f.find("data_") >= 0 and f.endswith(".pt"):
                pos = f.find("processed") + 10
                res.append(f[pos:])
        return res

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        print("dataset processing...")
        params = self.params

        # Node2vec
        # emb_file_n2v = params.output_path + "/node2vec_embeddings.pkl"
        emb_file_n2v = OUTPUT_PATH + "/node2vec_embeddings.pkl"
        with open(emb_file_n2v, 'rb') as fh:
            n2v_embeddings = pickle.load(fh)

        # Selected func keys
        with open(self.rand_key_file_path) as f:
            selected_lv0_func_keys = f.read().strip().split("\n")

        # PDT
        # pdt_M = 15  # num of tokens in a statement
        # pdt_dim = 85  # [8, 4, 2, 1] 的平方和
        emb_file_pdt = OUTPUT_PATH + "/pdt_graph2vec.pt"
        pdt_emb = torch.load(emb_file_pdt)

        # LP
        lp_N = 285  # num of LP in a method
        lp_M = 42  # num of tokens in a LP

        # NS
        ns_N = 242 # num of NS in a method
        ns_M = 16  # num of tokens in a NS

        # Callers & callees
        CC_C = 45 # 每个 funtion 最多存储 Caller callee 的数量
        CC_N = 242
        CC_M = 16

        ns_obj = NaturalSeq(params.input_path, params.output_path)
        lp_obj = Longpath(params.input_path, params.output_path)


        # files = list(findAllFile(params.input_path, True))
        scenario_1, v1 = 0, 0
        scenario_2, v2 = 0, 0
        scenario_3, v3 = 0, 0
        scenario_4, v4 = 0, 0
        scenario_5, v5 = 0, 0
        scenario_6, v6 = 0, 0
        scenario_6_2, v6_2 = 0, 0
        scenario_7, v7 = 0, 0
        scenario_8, v8 = 0, 0

        cve_list = []

        ii = 0
        for file in findAllFile(params.input_path, True):

            if not file.endswith("entities_1hop.json"):
                continue

            # if ii > 100:
            #     break

            # repoName = file.split("/")[6]
            cve_id = file.split("/")[-1].replace("-entities_1hop.json", "")
            print("cve_id: ", cve_id)

            edges_file = file.replace("entities_1hop", "edges_1hop")

            if not os.path.exists(edges_file):
                logging.info("== no edges file: {}".format(edges_file))
                continue


            if ii % 100 == 0:
                logging.info("now: {}".format(ii) )
                print("now: {}".format(ii) )
            with open(file) as f:
                entities = json.loads(f.read())
            with open(edges_file) as f:
                rel = json.loads(f.read())


            # C -> D -> F -> A -> B
            for lv0_func_key in rel.keys():
                try:
                    if lv0_func_key not in selected_lv0_func_keys:
                        continue
                    if lv0_func_key not in entities.keys():
                        logging.info("== no such a lv0_key in entities: {}".format(lv0_func_key))
                        continue

                    if lv0_func_key not in n2v_embeddings.keys():
                        continue

                    if lv0_func_key not in pdt_emb.keys():
                        continue

                    lv0_func = entities[lv0_func_key]
                    if lv0_func['contents'].strip() == '':
                        logging.info("func contents is empty: {}".format(lv0_func_key))
                        continue

                    if 'tree' not in lv0_func.keys() or lv0_func['tree'] == '':
                        continue

                    # PDT & Tree-LSTM
                    # pdt_tree = pdt.process_joern(lv0_func['tree'], lv0_func['contents'], 'CFG', 'REF', PDT=True)
                    # if len(pdt_tree['edges']) == 0 or len(pdt_tree['features']) == 0:
                    #     continue
                    # tree_t, tree_v = build_tree_for_lstm(pdt_tree, ns_obj, pdt_M, pdt_dim)
                    x_pdt = torch.tensor(pdt_emb[lv0_func_key], dtype=torch.float)

                    # LP
                    tree_dic = split_trees(lv0_func['tree'])
                    lp_matrix, lp_len_list = lp_obj.get_lp_idx(tree_dic['AST'], lp_N, lp_M)
                    # lp_matrix = torch.tensor(lp_matrix, dtype=torch.float)

                    # NS
                    ns_matrix, ns_len_list = ns_obj.get_ns_idx(lv0_func['contents'], ns_N, ns_M)
                    # ns_matrix = torch.tensor(ns_matrix, dtype=torch.float)

                    # EFG
                    efg_statements, efg_edge_index = efg.get_efg(tree_dic['CFG'], lv0_func['contents'])
                    efg_edge_index = torch.tensor(efg_edge_index, dtype=torch.long).t().contiguous()
                    efg_x = []
                    for s in efg_statements:
                        efg_x.append( ns_obj.statement2vec1D(s) )
                    efg_x = torch.tensor(np.array(efg_x), dtype=torch.float)

                    # callers & callees
                    # cc_vector = np.zeros((CC_C, CC_N, CC_M))
                    # cc_st_types = np.zeros((CC_C, CC_N))
                    cc_vector = []
                    cc_len_list = []
                    cc_st_types = []

                    # for i, code_slices in enumerate(all_code_sliced):
                    #     matrix = ns_obj.get_ns_idx(code_slices, CC_N, CC_M)
                    #     matrix = torch.tensor(matrix, dtype=torch.float)
                    #     cc_vector[i] = matrix


                    nodes = rel[lv0_func_key]['nodes']
                    vul = entities[lv0_func_key]['vul']
                    jj = 0
                    has_cc = 0
                    all_code_sliced = []
                    for edge in rel[lv0_func_key]['edge_index']:
                        # F --> A
                        code_sliced = []
                        func_ = None
                        func_key_2 = ''

                        if jj>= CC_C:
                            break

                        if nodes[edge[0]] == lv0_func_key:
                            func_ = entities[nodes[edge[1]]]
                            func_key_2 = nodes[edge[1]]
                            if entities[nodes[edge[1]]]['contents'].strip() == "":
                                continue
                            # print(edge, "A:", nodes[edge[1]])
                            # 1: F call A, no parameters, has return values, use all statements of A.

                            lines_A = entities[nodes[edge[1]]]['contents'].split("\n")
                            if lines_A[0].find("()") > -1 and lines_A[0].find("void") < 0: # 没有参数，有返回值
                                scenario_1 += 1
                                code_sliced = get_function_buggy_code(func_)
                                if vul > 0:
                                    v1 += 1
                            elif lines_A[0].find("()") < 0 and lines_A[0].find("void") < 0: # 有参数，有返回值
                                scenario_2 += 1
                                code_sliced = get_function_buggy_code(func_)
                                if vul > 0:
                                    v2 += 1
                            elif lines_A[0].find("()") < 0 and lines_A[0].find("void") > -1: # 有参数，没有返回值
                                scenario_3 += 1
                                code_sliced = get_function_buggy_code(func_)
                                if vul > 0:
                                    v3 += 1
                            elif lines_A[0].find("()") > -1 and lines_A[0].find("void") > -1: # 没有参数，没有返回值
                                scenario_4 += 1
                                code_sliced = get_function_buggy_code(func_, True)
                                if vul > 0:
                                    v4 += 1
                            else:
                                code_sliced = get_function_buggy_code(func_, True)

                        # D --> F
                        if nodes[edge[1]] == lv0_func_key:
                            if entities[nodes[edge[0]]]['contents'].strip() == "":
                                continue

                            # print(edge, 'D:', nodes[edge[0]])
                            func_ = entities[nodes[edge[0]]]
                            func_key_2 = nodes[edge[0]]
                            lines_F = entities[lv0_func_key]['contents'].split("\n")
                            lines_D = entities[nodes[edge[0]]]['contents'].split("\n")
                            func_name_F = entities[lv0_func_key]['name']
                            func_name_D = entities[nodes[edge[0]]]['name']

                            if lines_F[0].find("()") < 0 and lines_F[0].find("void") > -1: # 有参数，没有返回值
                                scenario_5 += 1
                                code_sliced = get_function_buggy_code(func_)
                                if vul > 0:
                                    v5 += 1
                            elif lines_F[0].find("()") < 0 and lines_F[0].find("void") < 0: # 有参数，有返回值
                                scenario_6 += 1
                                code_sliced = get_function_buggy_code(func_)
                                if vul > 0:
                                    v6 += 1
                                for line in lines_D:
                                    if line.find("return") > -1 and line.find(func_name_D) > -1:
                                        scenario_6_2 += 1
                                        if vul > 0:
                                            v6_2 += 1
                            elif lines_F[0].find("()") > -1 and lines_F[0].find("void") > -1: # 没有参数，没有返回值
                                scenario_7 += 1
                                code_sliced = get_function_buggy_code(func_, True)
                                if vul > 0:
                                    v7 += 1
                            elif lines_F[0].find("()") > -1 and lines_F[0].find("void") < 0: # 没有参数，有返回值
                                scenario_8 += 1
                                code_sliced = get_function_buggy_code(func_)
                                if vul > 0:
                                    v8 += 1
                            else:
                                code_sliced = get_function_buggy_code(func_, True)

                        i = 0
                        statements = [] # 【N，M】
                        statements_len_list = [] # [N]
                        statements_types = [] # [N]
                        for statement, type_id in code_sliced:
                            if i>= CC_N:
                                break
                            idx = ns_obj.get_st_ns_idx(statement, CC_M)
                            if len(idx) > 0:
                                statements.append( torch.tensor(idx) )
                                statements_len_list.append( len(idx) )
                                statements_types.append(type_id)
                            # cc_vector[jj][i] = idx
                            # cc_st_types[jj][i] = type_id

                            i += 1
                            has_cc = 1

                        if len(statements) < 8:
                            for kk in range( 8 - len(statements)):
                                statements.append( torch.tensor([0]))
                                statements_len_list.append(1)
                                statements_types.append(0)

                        cc_vector.append(statements)
                        cc_len_list.append(statements_len_list)
                        cc_st_types.append(statements_types)
                        jj += 1

                    # cc_vector = torch.tensor(cc_vector, dtype=torch.float)
                    # cc_st_types = torch.tensor(cc_st_types, dtype=torch.int)

                    # call graph
                    if lv0_func_key in n2v_embeddings.keys() and n2v_embeddings[lv0_func_key] is not None:
                        x_n2v = torch.tensor(n2v_embeddings[lv0_func_key], dtype=torch.float)
                    else:
                        x_n2v = torch.ones(128, dtype=torch.float)

                    # print("lp_matrix: {}".format(lp_matrix.shape))
                    # print("ns_matrix: {}".format(ns_matrix.shape))
                    # print("tree_v: {}".format(len(tree_v)))
                    # print("x_n2v: {}".format(x_n2v.shape))
                    # print("cc_vector: {}".format(cc_vector.shape))
                    # print("cc_st_types: {}".format(cc_st_types.shape))
                    # print(cc_st_types)
                    # exit()

                    data = Data(num_nodes=1,
                                y=vul,
                                x_lp=lp_matrix,
                                lp_len_list=lp_len_list,
                                x_ns=ns_matrix,
                                ns_len_list=ns_len_list,
                                x_pdt=x_pdt,
                                x_n2v=x_n2v,
                                efg_x=efg_x,
                                efg_edge_index=efg_edge_index,
                                cc_list=cc_vector,
                                cc_len_list=cc_len_list,
                                cc_st_types=cc_st_types,
                                has_cc=has_cc,
                                ii=ii
                                )

                    if 'cve_id' in lv0_func.keys():
                        cve_id = lv0_func['cve_id']
                        if cve_id.find(",") > -1:
                            print(
                                "  one func has multiple cves. func_key: {}, cve_ids: {}".format(lv0_func_key, cve_id))
                            p = cve_id.find(",")
                            cve_id = cve_id[:p]
                        print("  cve in func: {}".format(cve_id))
                    line_counts = len(lv0_func['contents'].strip().split("\n"))
                    cve_list.append([ii, cve_id, lv0_func_key, file, line_counts])
                    save_path = os.path.join(self.processed_dir, str(ii // 1000))
                    Path(save_path).mkdir(parents=True, exist_ok=True)

                    to_file = os.path.join(save_path, 'data_{}.pt'.format(ii))
                    torch.save(data, to_file)

                    ii += 1
                    # logging.info("saved to: {}, vul: {}".format(to_file, vul))
                    print("saved to: {}, vul: {}, has_cc: {}".format(to_file, vul, has_cc))
                except Exception as e:
                    logging.exception(e)
                    logging.info("skipped: {}, file: {}".format(lv0_func_key, file))
                    print("skipped: {}, file: {}".format(lv0_func_key, file))
                    print(str(e))

        cve_list_file = self.save_path + "_cvelist.txt"
        with open(cve_list_file, "w") as fw:
            for row in cve_list:
                fw.write("{},{},{},{},{}\n".format(row[0], row[1], row[2], row[3], row[4]))
        print("saved to {}".format( cve_list_file ) )


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        save_path = os.path.join(self.processed_dir, str(idx // 1000))
        data = torch.load(os.path.join(save_path, 'data_{}.pt'.format(idx)))
        return data


def test():

    # args
    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('--input_path', help='input_path', type=str,
                        default=INPUT_PATH)
    parser.add_argument('--output_path', help='output_path', type=str,
                        default=OUTPUT_PATH)
    args, _ = parser.parse_known_args()
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # 1. data_loader
    i = 1
    rand_key_file_path = RAND_DATA_PATH + "/lv0_func_keys_{}_{}.txt".format(i, 'train')
    dataset_savepath = args.output_path + "/datasets_{}_{}".format(i, 'train')
    Path(dataset_savepath).mkdir(parents=True, exist_ok=True)
    train_dataset = MyLargeDataset(dataset_savepath, args, rand_key_file_path)



if __name__ == '__main__':
    test()