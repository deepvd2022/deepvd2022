
import time
from datetime import datetime

import networkx as nx
import numpy as np

from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
from gensim.models import Word2Vec

import matplotlib.pyplot as plt
import pandas as pd
import os

import logging
from pathlib import Path
import json
import argparse

try:
    import cPickle as pickle
except:
    import pickle



def findAllFile(base, full=True):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if full:
                yield os.path.join(root, f)
            else:
                yield f

def get_graphs(funcs):
    res = []
    for func_key, attr in funcs.items():
        if len(attr['edge_index']) == 0:
            res.append({
                'func_key': func_key,
                'repo_name': attr['repo_name'],
                'stats': attr['stats'],
                'vul': attr['vul'],
                'unique_funcs': len(attr['nodes']),
                'G': None
            })
        else:
            G = nx.DiGraph()  # directed graph
            nodes = attr['nodes']
            G.add_nodes_from([n for n in nodes])

            edges = []
            for row in attr['edge_index']:
                n1 = nodes[ row[0] ]
                n2 = nodes[ row[1] ]
                edges.append( (n1, n2) )
            G.add_edges_from(edges)
            res.append({
                'func_key': func_key,
                'repo_name': attr['repo_name'],
                'stats': attr['stats'],
                'vul': attr['vul'],
                'unique_funcs': len(nodes),
                'G': G
            })
    return res

def read_graphs(input_path):
    ii = 0
    graphs = []
    for file in findAllFile(input_path, True):

        if not file.endswith("entities_1hop.json"):
            continue

        # repoName = file.split("/")[6]
        # cve_id = file.split("/")[-1].replace("-entities_1hop.json", "")
        edges_file = file.replace("entities_1hop", "edges_1hop")

        if not os.path.exists(edges_file):
            logging.info("== no edges file: {}".format(edges_file))
            continue

        if ii % 100 == 0:
            logging.info("now: {}".format(ii))
        # with open(file) as f:
        #     entities = json.loads(f.read())
        with open(edges_file) as f:
            rel = json.loads(f.read())

        for func_key, attr in rel.items():
            if len(attr['edge_index']) == 0:
                graphs.append({
                    'func_key': func_key,
                    'repo_name': attr['repo_name'],
                    'stats': attr['stats'],
                    # 'vul': attr['vul'],
                    'unique_funcs': len(attr['nodes']),
                    'G': None
                })
            else:
                G = nx.DiGraph()  # directed graph
                nodes = attr['nodes']
                G.add_nodes_from([n for n in nodes])

                edges = []
                for row in attr['edge_index']:
                    n1 = nodes[row[0]]
                    n2 = nodes[row[1]]
                    edges.append((n1, n2))
                G.add_edges_from(edges)
                graphs.append({
                    'func_key': func_key,
                    'repo_name': attr['repo_name'],
                    'stats': attr['stats'],
                    # 'vul': attr['vul'],
                    'unique_funcs': len(nodes),
                    'G': G
                })
            ii += 1
    return graphs


class Node2vec():
    def __init__(self, save_path):
        # self.save_path = save_path + "/"
        self.model_path = save_path + "/node2vec.bin"
        self.model = None
        self.embedding_file = save_path + "/node2vec_embeddings.pkl"

    def train(self, graphs):
        walk_length = 10
        weighted_walks = []
        logging.info("start training node2vec...")
        for g in graphs:
            if g['G'] is not None:
                G = g['G']
                G = G.to_undirected()
                G = StellarGraph.from_networkx(G)
                # print(G.info())
                rw = BiasedRandomWalk(G)
                weighted_walks = weighted_walks + rw.run(
                    nodes=G.nodes(),  # root nodes
                    length=walk_length,  # maximum length of a random walk
                    n=5,    # number of random walks per root node
                    p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
                    q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
                    weighted=True,  # for weighted random walks
                    seed=42,  # random seed fixed for reproducibility
                )
                # print("Number of random walks: {}".format(len(weighted_walks)))

        logging.info("Number of random walks: {}".format(len(weighted_walks)))
        logging.info("training node2vec...")
        print("Number of random walks: {}".format(len(weighted_walks)))
        print("training node2vec...")
        weighted_model = Word2Vec(
            weighted_walks, size=128, window=5, min_count=1, sg=1, workers=1, iter=1
        )

        weighted_model.save(self.model_path)
        self.model = weighted_model
        logging.info("training node2vec done, saved to: {}".format(self.model_path))

    def predict(self):
        pass

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = Word2Vec.load(self.model_path)


    def get_embedding(self, func_key):
        if self.model is None:
            self.load_model()

        if func_key in self.model.wv:
            return self.model.wv[func_key]
        else:
            logging.info("node2vec - no such a key: {}".format(func_key))
        return None

    def save_embeddings(self, graphs):
        embeddings = {}
        for g in graphs:
            func_key = g['func_key']
            embeddings[func_key] = self.get_embedding(func_key)

        pickle.dump(embeddings, open(self.embedding_file, "wb"))
        logging.info("len(n2v_embeddings): {}, saved to {}".format(len(embeddings), self.embedding_file))
        print("len(n2v_embeddings): {}, saved to {}".format(len(embeddings), self.embedding_file))

    def load_embeddings(self):
        if os.path.exists(self.embedding_file):
            return pickle.load( open( self.embedding_file, "rb" ) )



if __name__ == '__main__':
    # Test:
    INPUT_PATH = "data/verstehen"
    OUTPUT_PATH = "data/DeepVD"

    to_file_graphs = OUTPUT_PATH + "/graphs.pkl"

    num_of_nodes = []
    num_of_edges = []
    if os.path.exists(to_file_graphs):
        print("=== loading graphs from: {}".format(to_file_graphs))
        logging.info("loading graphs from: {}".format(to_file_graphs))
        with open(to_file_graphs, 'rb') as fh:
            graphs = pickle.load(fh)
    else:
        print("=== read_graphs")
        graphs = read_graphs(INPUT_PATH)
        pickle.dump(graphs, open(to_file_graphs, "wb"))
        print("saved to {}".format(to_file_graphs))


    for g in graphs:
        if g['G'] is not None:
            G = g['G']
            if G.number_of_nodes() > 1000:
                print("G.number_of_nodes() > 1000, func_key : ", g['func_key'])
                print("G.number_of_nodes() > 1000, repo_name: ", g['repo_name'])
                continue
            num_of_nodes.append(G.number_of_nodes())
            num_of_edges.append(G.number_of_edges())

    print("Nodes:")
    print("mean: {}".format(np.mean(num_of_nodes)))
    print("max: {}".format(np.max(num_of_nodes)))
    print("min: {}".format(np.min(num_of_nodes)))
    print("len: {}".format(len(num_of_nodes)))
    print("=========")
    print("Edges:")
    print("mean: {}".format(np.mean(num_of_edges)))
    print("max: {}".format(np.max(num_of_edges)))
    print("min: {}".format(np.min(num_of_edges)))

    num_of_nodes.sort()
    p = int( len(num_of_nodes) * 0.99 )
    print("num_of_nodes[{}:]".format(p))
    print(num_of_nodes[p:])

    # Node2vec
    n2v = Node2vec(OUTPUT_PATH)
    n2v.train(graphs)
    n2v.save_embeddings(graphs)