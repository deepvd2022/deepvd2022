
import json
import os
import subprocess
# import swifter
import tempfile
import sys
import traceback

# import dask.dataframe as dd
import numpy as np
# import pandas as pd

# from tqdm import tqdm
import networkx as nx
# import logging
# from pathlib import Path



def findAllFile(base, full=True):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if full:
                yield os.path.join(root, f)
            else:
                yield f




def generate_edgelist(ast_root):
    """
    Given a concretised & numbered clang ast, return a list of edges
    in the form:
        [
            [<start_node_id>, <end_node_id>],
            ...
        ]
    """
    edges = []

    def walk_tree_and_add_edges(node):
        for child in node.children:
            edges.append([node.identifier, child.identifier])
            walk_tree_and_add_edges(child)

    walk_tree_and_add_edges(ast_root)

    return edges


def generate_features(ast_root):
    """
    Given a concretised & numbered clang ast, return a dictionary of
    features in the form:
        {
            <node_id>: [<degree>, <type>, <identifier>],
            ...
        }
    """
    features = {}

    def walk_tree_and_set_features(node):
        out_degree = len(node.children)
        in_degree = 1
        degree = out_degree + in_degree

        features[node.identifier] = [str(node.kind)]

        for child in node.children:
            walk_tree_and_set_features(child)

    walk_tree_and_set_features(ast_root)

    return features





def get_info(line, info_start, info_end, separator_index_start = 0,separator_index_end = -1):
    start = line.find(info_start,separator_index_start,separator_index_end)
    end = line.rfind(info_end,separator_index_start,separator_index_end)
    info = line[start+len(info_start):end]
    info_clean_start = info.find("(")
    info_clean_end = info.rfind(")")
    if info_clean_start>-1 and info_clean_end>-1:
        info_clean = info[info_clean_start+1:info_clean_end]
    else:
        info_clean = " "
    return info_clean



def get_edgelist_and_feature(string_graph,string_graph_type,code,string_graph_type_next=" ", PDT = False, Para_link = False, testcase = " "):
    graph = string_graph
    graph_type = string_graph_type
    edgelist = []
    feature_json = {}
    code = code.split("\n")
    # Next, construct an edge list for the graph2vec input:
    start = graph.find("# "+string_graph_type+"\n{")
    if string_graph_type_next == " ":
        end = -1
    else:
        end = graph.find(" }\n# "+string_graph_type_next)
    graph_now = graph[start+5:end]
    graph_now = graph_now.split("\n")
    for line in graph_now:
        line = line.strip()
        separator_index = line.find(' -->> "joern_id_')
        if separator_index>0:
            try:
                source_line = get_info(line,"joern_line_","\"",0,separator_index)
                sink_line = get_info(line,"joern_line_","\"",separator_index_start = separator_index,separator_index_end = len(line))
                if source_line!=sink_line:
                    edgelist.append([int(source_line),int(sink_line)])
                    feature_json[int(source_line)] = [code[int(source_line)-1].strip()]
                    if not (int(sink_line) in feature_json.keys()):
                        feature_json[int(sink_line)] = [code[int(sink_line)-1].strip()]
            except Exception as e:
                traceback.print_exc(file=sys.stdout)
                print("reason", e)
                return None, None

    if PDT:
        # Post-dominance is dominance in reverse CFG obtained by reversing direction of all edges
        # and interchanging roles of START and END.

        try:
            reversed_cfg_edges = []
            for row in edgelist:
                reversed_cfg_edges.append([row[1], row[0]])

            G = nx.DiGraph()
            G.add_edges_from(reversed_cfg_edges)
            start = reversed_cfg_edges[0][0]
            for edge in reversed_cfg_edges:
                for node in edge:
                    if node >= start:
                        start = node
            edgelist = sorted(nx.immediate_dominators(G, start).items())
            edgelist = [list([x[1], x[0]]) for x in edgelist]
        except Exception as e:
            print("reason", e)

    return edgelist,feature_json



def process_joern(tree, contents, graph_type, graph_type_next=" ", PDT=False, Para_link=False, **kwargs):
    """
    Takes in a list of files/datapoints from juliet.csv.zip or
    vdisc_*.csv.gz (as loaded with pandas) matching one particular
    testcase, and preprocesses it ready for the baseline model.
    """

    edgelist, feature_json = get_edgelist_and_feature(tree, graph_type, contents, graph_type_next, PDT, Para_link)
    if feature_json is None:
        return None

    representation = {
        "edges": edgelist,
        "features": feature_json
    }

    return representation


def get_pdt_nodes_edges(tree, contents):
    edgelist, feature_json = get_edgelist_and_feature(tree, 'CFG', contents, 'REF', True, False)
    return feature_json, edgelist