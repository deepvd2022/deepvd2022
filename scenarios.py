
import torch
import time
from datetime import datetime
import re
import networkx as nx
import numpy as np

# from stellargraph.data import BiasedRandomWalk
# from stellargraph import StellarGraph
# from gensim.models import Word2Vec

import pandas as pd
import os

import logging
from pathlib import Path
import json
import argparse
from embeddings.st_types import StType

try:
    import cPickle as pickle
except:
    import pickle

import difflib

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

def findAllFile(base, full=False):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if full:
                yield os.path.join(root, f)
            else:
                yield f



def find_diff(contents1, contents2):
    d = difflib.Differ()
    diffs = d.compare(contents1.split("\n"), contents2.split("\n"))
    line_1 = 0
    line_2 = 0
    diff1 = {}
    diff2 = {}
    for line in diffs:
        # split off the code
        code = line[:2]
        # if the  line is in both files or just b, increment the line number.
        if code == "  ":
            line_1 += 1
            line_2 += 1
            # print(line)
        elif code == '+ ':
            line_2 += 1
            # print(line_2, line)
            diff2[ line_2 ] = line
        elif code == '- ':
            line_1 += 1
            # print(line_1, line)
            diff1[ line_1 ] = line

    return diff1, diff2


def read_tree_file(tree_file):
    tree = {}
    with open(tree_file) as f:
        for line in f:
            l = line.strip()
            if l == "":
                continue
            obj = json.loads(l)
            tree[obj['func_key']] = obj
    return tree


def find_buggy_statements(tree):
    s1 = "digraph g {"
    p = tree.find("# PDG")
    if p > 0:
        dot_str = s1 + tree[p+5 : ]
    else:
        return []

    G = nx.DiGraph()
    for line in dot_str.split("\n"):
        if line.find(" -->> ") > 0:
            try:
                n1, n2 = line.split(" -->> ", 1)
            except Exception as e:
                exit(0)
            n1 = n1.strip().strip('"')
            n2 = n2.strip().strip('"')
            # print(n1, '===' ,n2)
            G.add_edge(n1, n2)

    if len(G.nodes()) == 0:
        return []

    leafnodes = [x for x in G.nodes() if G.out_degree(x) == 0 and G.in_degree(x) == 1]
    start_nodes = [x for x in G.nodes() if str(x).find("joern_type_(METHOD_PARAMETER_IN)") > -1 or str(x).find("joern_type_(RETURN)") > -1]

    p1 = re.compile(r'joern_line_[(](.*?)[)]', re.S)
    buggy_nodes = []

    for n1 in start_nodes:
        for n2 in leafnodes:
            try:
                path = nx.shortest_path(G, source=n1, target=n2)
            except:
                path = []

            for n in path:
                if n not in buggy_nodes:
                    buggy_nodes.append(n)
    return buggy_nodes


def get_function_buggy_code(func, is_full=False):
    if func['tree'].strip() == '':
        return func['contents'].strip()

    trees = split_trees(func['tree'])
    st = StType(trees['AST'])
    stt = st.get_st_types()


    flag = False

    p1 = re.compile(r'joern_line_[(](.*?)[)]', re.S)
    buggy_statements = find_buggy_statements(func['tree'])

    code_lines = func['contents'].split("\n")
    buggy_codes = []
    # print("=-====")
    visited_line = []
    for statement in buggy_statements:
        line_no = int(re.findall(p1, statement)[0])
        if line_no in visited_line:
            continue
        visited_line.append(line_no)

        if line_no in stt.keys():
            st_type_id = stt[line_no]
        else:
            # print("no such line in AST: {}".format(line_no))
            flag = True
            st_type_id = 0
        # print([code_lines[line_no-1], st_type_id])
        buggy_codes.append([code_lines[line_no-1], st_type_id])
    # if flag:
    #     print( trees['AST'] )
    #     for i, line in enumerate( func['contents'].split("\n") ):
    #         print(i, line)
    #     exit()
    return buggy_codes
