
import networkx as nx
import os
import json
import numpy as np
import logging
import re
import torch

try:
   import cPickle as pickle
except:
   import pickle

#去除code中的comment
def remove_comment(code):
    in_comment = 0
    output = []
    for line in code:
        if in_comment == 0:
            if line.find("/*") != -1:
                if line.find("*/") == -1:
                    in_comment = 1
            else:
                if line.find("//") != -1:
                    if line.find("//") > 0:
                        line = line[:line.find("//")]
                        output.append(line)
                else:
                    output.append(line)
        else:
            if line.find("*/") != -1:
                in_comment = 0
    return output

#多个大写字母相连，可以考虑成一个单词
def merge_code(code):
    output = []
    combine_ = 0
    temp = ""
    for unit in code:
        if combine_ == 0:
            if len(unit) == 1 and unit.istitle():
                temp = temp + unit
                combine_ = 1
            else:
                output.append(unit)
        else:
            if len(unit) == 1 and unit.istitle():
                temp = temp + unit
            else:
                combine_ = 0
                output.append(temp)
                output.append(unit)
                temp = ""
    if temp != "":
        output.append(temp)
    return output

#搜集code用来准备生成glove embedding
def collect_code_data(code_):
    big_code = []

    code_ = code_.splitlines()
    code_ = remove_comment(code_)
    code_ = " ".join(code_)
    code_ = re.sub('[^a-zA-Z0-9]', ' ', code_)
    code_ = re.sub(' +', ' ', code_)
    code_ = re.sub(r"([A-Z])", r" \1", code_).split()
    code_ = merge_code(code_)
    big_code.append(code_)

    return big_code

def statement2tokens(statement):
    code_ = re.sub('[^a-zA-Z0-9]', ' ', statement)
    code_ = re.sub(' +', ' ', code_)
    code_ = re.sub(r"([A-Z])", r" \1", code_).split()
    code_ = merge_code(code_)
    return code_

def code2tokens(code):
    lines = code.splitlines()
    lines = remove_comment(lines)
    ns_list = []
    for line in lines:
        st_tokens = statement2tokens(line)
        if len(st_tokens) > 0:
            ns_list.append( st_tokens )
    return ns_list

class NaturalSeq():
    def __init__(self, input_path, output_path, emb_dim=128):
        self.input_path = input_path
        self.output_path = output_path
        self.emb_dim = emb_dim

        self.corpus_file = output_path + "/ns_corpus.pkl"
        self.vector_file = output_path + "/ns_glove_vectors.pkl"  # 即 weights_matrix，用于 pytorch 的 embedding layer
        self.idx_file = output_path + "/ns_glove_idx.pkl"  # word2idx

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

    def statement2vec(self, statement, M):
        matrix = np.zeros((M, self.emb_dim))
        tokens = statement2tokens(statement)
        for i in range(M):
            if i >= len(tokens):
                break
            token = tokens[i]
            if token in self.word2idx.keys():
                idx = self.word2idx[ token ]
                matrix[i] = self.vectors[idx]

        # normalize
        try:
            norm = np.linalg.norm(matrix, axis=1).reshape(M, 1)
            matrix = np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm != 0)
            # matrix = matrix / np.linalg.norm(matrix, axis=1).reshape(len(doc), 1)
        except RuntimeWarning:
            # print(doc)
            pass
        # matrix = np.array(preprocessing.normalize(matrix, norm='l2'))
        return matrix

    def statement2vec1D(self, statement):
        tokens = statement2tokens(statement)
        matrix = []
        for token in tokens:
            if token in self.word2idx.keys():
                idx = self.word2idx[token]
                matrix.append(self.vectors[idx])
        if len(matrix) == 0:
            return np.zeros(128)
        return np.mean(matrix, axis=0)

    def get_ns_idx(self, code, N, M):
        ns_list = code2tokens(code)
        # matrix = np.zeros((N, M))
        res = []
        len_list = []
        for i in range(N):
            if i >= len(ns_list):
                break
            tokens = ns_list[i]
            row = []
            for j in range(M):
                if j >= len(tokens):
                    break
                if tokens[j] in self.word2idx.keys():
                    idx = self.word2idx[tokens[j]]
                    # matrix[i][j] = idx
                    row.append(idx)
            if len(row) > 0:
                res.append(torch.tensor(row))
                len_list.append(len(row))
        if len(res) < 8: # SPP 至少接受 8 个
            for i in range( 8-len(res) ):
                res.append(torch.tensor([0]))
                len_list.append(1)
        return res, len_list

    def get_st_ns_idx(self, statement, M):
        # vec_idx = np.zeros((M,))
        res = []
        tokens = statement2tokens(statement)
        for i in range(M):
            if i >= len(tokens):
                break
            token = tokens[i]
            if token in self.word2idx.keys():
                idx = self.word2idx[ token ]
                res.append(idx)
        return res


