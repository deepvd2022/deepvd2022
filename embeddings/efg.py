
import networkx as nx
import os
import json
import numpy as np
import logging
import re
import time
import torch
from nltk.tokenize import word_tokenize

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
    p1 = re.compile(r'joern_type_[(](.*?)[)]_joern_name', re.S)
    res = re.findall(p1, line)
    if len(res) > 0:
        return res[0]
    return ''

def get_joern_code(line):
    p1 = re.compile(r'joern_code_[(](.*?)[)]_joern_type', re.S)
    res = re.findall(p1, line)
    if len(res) > 0:
        return res[0]
    return ''

def get_joern_line(line):
    p1 = re.compile(r'joern_line_[(](.*?)[)]', re.S)
    res = re.findall(p1, line)
    if len(res) > 0:
        return int(res[0])
    return -1


def get_efg(cfg, contents):
    state = 0
    try_catch_pairs = []
    try_line = -1
    cat_line = -1
    for i, line in enumerate(contents.split("\n")):
        line_no = i + 1
        tokens = word_tokenize(line)
        if 'try' in tokens:
            state = 1
            try_line = line_no
        if state == 1 and 'catch' in tokens:
            state = 0
            cat_line = line_no
            try_catch_pairs.append( (try_line, cat_line) )

            try_line = -1
            cat_line = -1

        # print(line_no, line)


    G = nx.Graph()
    for line in cfg.split("\n"):
        if line.find('" -->> "') < 0:
            continue
        a, b = line.split('" -->> "', maxsplit=1)
        line_a = get_joern_line(a)
        line_b = get_joern_line(b)

        for (s1, s2) in try_catch_pairs:
            if line_a >= s1 and line_a < s2:
                G.add_edge(line_a, s2)
                # print("{}--->{}".format(line_a, s2))
            if line_b >= s1 and line_b < s2:
                G.add_edge(line_b, s2)
                # print("{}--->{}".format(line_b, s2))


        G.add_edge(line_a, line_b)

    if G.number_of_nodes() > 10000:
        return None

    all_nodes = []
    for u, v, a in G.edges(data=True):
        if u not in all_nodes:
            all_nodes.append(u)
        if v not in all_nodes:
            all_nodes.append(v)
    all_nodes.sort()


    contents = contents.split("\n")
    statements = []
    edge_index = []
    for v in all_nodes:
        statements.append( contents[v-1] )
        # print(contents[v-1])

    for u, v, a in G.edges(data=True):
        # print(u, v)
        ui = all_nodes.index(u)
        vi = all_nodes.index(v)
        if ui != vi:
            edge_index.append([ui, vi])
        # print(ui, '==>', vi)

    return statements, edge_index


def get_slice(pdg, contents):
    # print(pdg)
    # print("===")
    G = nx.Graph()
    for line in pdg.split("\n"):
        if line.find('" -->> "') < 0:
            continue
        a, b = line.split('" -->> "', maxsplit=1)
        line_a = get_joern_line(a)
        line_b = get_joern_line(b)

        G.add_edge(line_a, line_b)

    if G.number_of_nodes() > 10000:
        return None

    all_nodes = []
    for u, v, a in G.edges(data=True):
        if u not in all_nodes:
            all_nodes.append(u)
        if v not in all_nodes:
            all_nodes.append(v)
    all_nodes.sort()


    contents = contents.split("\n")
    statements = []
    edge_index = []
    for v in all_nodes:
        statements.append( contents[v-1] )

    for u, v, a in G.edges(data=True):
        ui = all_nodes.index(u)
        vi = all_nodes.index(v)
        if ui != vi:
            edge_index.append([ui, vi])
        # print(ui, '==>', vi)

    return statements, edge_index


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


def test():
    for file in findAllFile("/data/verstehen_v8", True):

        if not file.endswith("entities_1hop.json"):
            continue

        edges_file = file.replace("entities_1hop", "edges_1hop")

        if not os.path.exists(edges_file):
            logging.info("== no edges file: {}".format(edges_file))
            continue

        with open(file) as f:
            entities = json.loads(f.read())
        with open(edges_file) as f:
            rel = json.loads(f.read())

        # C -> D -> F -> A -> B
        for lv0_func_key in rel.keys():
            if lv0_func_key not in entities.keys():
                logging.info("== no such a lv0_key in entities: {}".format(lv0_func_key))
                continue

            lv0_func = entities[lv0_func_key]
            if lv0_func['contents'].strip() == '':
                logging.info("func contents is empty: {}".format(lv0_func_key))
                continue

            if 'tree' not in lv0_func.keys() or lv0_func['tree'] == '':
                continue

            trees = split_trees(lv0_func['tree'])
            pdg = trees['PDG']
            sliced_code = get_slice(pdg, lv0_func['contents'])



if __name__ == '__main__':
    test()

