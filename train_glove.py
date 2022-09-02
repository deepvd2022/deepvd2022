
import networkx as nx
import os
import json
import pandas as pd
from glove import Corpus, Glove
import numpy as np
import logging
import re
import time
from embeddings.ns import code2tokens
try:
   import cPickle as pickle
except:
   import pickle
from config import *
from pathlib import Path

# log file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
Path("logs").mkdir(parents=True, exist_ok=True)
now_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=BASE_DIR + '/logs/' + now_time + '.log')

def findAllFile(base, full=True):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if full:
                yield os.path.join(root, f)
            else:
                yield f


print("INPUT_PATH: {}".format(INPUT_PATH))
print("OUTPUT_PATH: {}".format(OUTPUT_PATH))

Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

#树结构
class Tree(object):
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



class MyGlove():
    def __init__(self, input_path, output_path, name='lp', emb_dim=128):
        # https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

        self.input_path = input_path
        self.output_path = output_path
        self.emb_dim = emb_dim

        self.model_file = output_path + "/{}_glove.bin".format(name)
        self.vector_file = output_path + "/{}_glove_vectors.pkl".format(name)
        self.idx_file = output_path + "/{}_glove_idx.pkl".format(name)

        self.model = None
        self.vectors = None
        self.word2idx = None
        self.retrain = True

    def train(self, train_data):
        if os.path.exists(self.model_file) and self.retrain == False:
            print("train glove done: model existed: {}".format(self.model_file))
            return

        # Creating a corpus object
        corpus = Corpus()
        # Training the corpus to generate the co-occurrence matrix which is used in GloVe
        corpus.fit(train_data, window=10)
        g = Glove(no_components=self.emb_dim, learning_rate=0.05)
        g.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)
        g.add_dictionary(corpus.dictionary)
        g.save(self.model_file)
        print("saved to {}".format(self.model_file))

        maxtrix_len = len(g.dictionary.keys())
        vectors = np.zeros((maxtrix_len, self.emb_dim))
        word2idx = {}

        for k in g.dictionary.keys():
            if len(word2idx.keys()) < 1000:
                print(k)
            idx = g.dictionary[k]
            vectors[idx] = g.word_vectors[idx]
            word2idx[k] = idx
            # dic_pdg[k] = glov_pdg.word_vectors[glov_pdg.dictionary[k]]

        pickle.dump(vectors, open(self.vector_file, "wb"))
        pickle.dump(word2idx, open(self.idx_file, "wb"))
        print("vectors: {}, saved to {}".format(vectors.shape, self.vector_file))
        print("word2idx: {}, saved to {}".format( len(word2idx), self.idx_file))





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

def generate_corpus_for_NS(to_file):
    print("generate_corpus_for_NS")
    corpus = []
    ns_num = [] # 每个 function 有多少 ns
    ns_length = [] # 每个 ns 有多少 token
    for file in findAllFile(INPUT_PATH):
        if file.endswith("entities_1hop.json"):
            with open(file) as f:
                entities = json.loads(f.read())

            for func_key, func in entities.items():
                if func['contents'].strip() != "":
                    ns_list = code2tokens(func['contents'])
                    ns_num.append(len(ns_list))
                    for ns in ns_list:
                        ns_length.append(len(ns))
                    corpus = corpus + ns_list
                    # print("=" * 10)
                    # print(func['contents'].strip())
                    # print(code_)
                    if len(corpus) % 100 == 0:
                        print("collected: {}".format(len(corpus)))
    print("collected: {}".format(len(corpus)))

    print("ns_num:")
    print("mean: {}".format(np.mean(ns_num)))
    print("max: {}".format(np.max(ns_num)))
    print("min: {}".format(np.min(ns_num)))
    print("len: {}".format(len(ns_num)))
    ns_num.sort()
    p = int(len(ns_num) * 0.99)
    print("ns_num[{}]: {}".format(p, ns_num[p]))

    print("=========")
    print("ns_length:")
    print("mean: {}".format(np.mean(ns_length)))
    print("max: {}".format(np.max(ns_length)))
    print("min: {}".format(np.min(ns_length)))
    print("len: {}".format(len(ns_length)))
    ns_length.sort()
    p = int(len(ns_length) * 0.99)
    print("ns_length[{}]: {}".format(p, ns_length[p]))

    """
    ns_num:
    mean: 25.92337298456434
    max: 19721
    min: 0
    len: 197724
    ns_num[195746]: 242
    =========
    ns_length:
    mean: 5.358647342505072
    max: 165
    min: 1
    len: 5125673
    ns_length[5074416]: 16
    """

    return corpus




# ===================================================================================================================


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

def generate_corpus_for_LP(to_file):
    print("generate_corpus_for_LP")
    corpus = []
    ii = 0
    lp_length = [] # 纪录每个 lp 的长度是多少
    lp_num = []    # 纪录每个 function 有多少 lp
    for file in findAllFile(INPUT_PATH):
        if file.endswith("entities_1hop.json"):
            with open(file) as f:
                entities = json.loads(f.read())

            for func_key, func in entities.items():
                if func['contents'].strip() != "" and func['tree'].strip() != "":
                    tree_ = split_trees(func['tree'])
                    # tree --> AST --> tree object -->
                    token_list = ast2lp(tree_['AST'])
                    corpus = corpus + token_list

                    lp_num.append(len(token_list))
                    for lp in token_list:
                        lp_length.append(len(lp))

                    if ii % 100 == 0:
                        print("collected: {}".format(len(corpus)))
                    ii += 1

    pickle.dump(corpus, open(to_file, "wb"))
    logging.info("saved to: {}".format(to_file))

    print("collected: {}".format(len(corpus)))
    logging.info("lp_length: max: {}, min: {}, mean: {}".format( np.max(lp_length), np.min(lp_length), np.mean(lp_length) ))
    logging.info("lp_num: max: {}, min: {}, mean: {}".format( np.max(lp_num), np.min(lp_num), np.mean(lp_num) ))
    lp_statistics = {
        'lp_length': lp_length,
        'lp_num': lp_num
    }
    pickle.dump(lp_statistics, open(OUTPUT_PATH + "/lp_statistics.pkl", "wb"))
    print("saved to : {}".format(OUTPUT_PATH + "/lp_statistics.pkl"))

    lp_length = lp_statistics['lp_length']
    lp_num = lp_statistics['lp_num']

    print("lp_num:")
    print("mean: {}".format(np.mean(lp_num)))
    print("max: {}".format(np.max(lp_num)))
    print("min: {}".format(np.min(lp_num)))
    print("len: {}".format(len(lp_num)))
    lp_num.sort()
    p = int(len(lp_num) * 0.99)
    print("ns_num[{}]: {}".format(p, lp_num[p]))

    print("=========")
    print("lp_length:")
    print("mean: {}".format(np.mean(lp_length)))
    print("max: {}".format(np.max(lp_length)))
    print("min: {}".format(np.min(lp_length)))
    print("len: {}".format(len(lp_length)))
    lp_length.sort()
    p = int(len(lp_length) * 0.99)
    print("ns_length[{}]: {}".format(p, lp_length[p]))

    """
    lp_num:
    mean: 28.01844320343167
    max: 18510
    min: 0
    len: 197688
    ns_num[195711]: 285
    =========
    lp_length:
    mean: 14.001917344748335
    max: 411
    min: 3
    len: 5538910
    ns_length[5483520]: 42
    """

    return corpus

# =====================================================================================================================

def train_glove_for_NS():
    # corpus_file = OUTPUT_PATH + "/ns_corpus.pkl"
    # if not os.path.exists(corpus_file):
    #     generate_corpus_for_NS(corpus_file)
    #     pass
    # exit()
    #
    # corpus = pickle.load(open(corpus_file, 'rb'))
    # print("loaded corpus: {}".format(len(corpus)))
    # print(corpus[:3])

    corpus = generate_corpus_for_NS("")
    my = MyGlove(INPUT_PATH, OUTPUT_PATH, 'ns', 128)
    my.train(corpus)
    print("train_glove_for_NS done")


def train_glove_for_LP():
    corpus_file = OUTPUT_PATH + "/lp_corpus.pkl"
    if not os.path.exists(corpus_file):
        corpus = generate_corpus_for_LP(corpus_file)
    else:
        corpus = pickle.load(open(corpus_file, 'rb'))

    print("loaded corpus: {}".format(len(corpus)))
    print(corpus[:3])

    my = MyGlove(INPUT_PATH, OUTPUT_PATH, 'lp', 128)
    my.train(corpus)
    print("train_glove_for_LP done")


if __name__ == '__main__':
    train_glove_for_NS()
    train_glove_for_LP()

