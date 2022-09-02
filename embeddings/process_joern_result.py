import os
import subprocess
from pathlib import Path
import hashlib
import tempfile
import argparse
import logging
import time
import json
import re
from pdt2 import get_pdt_nodes_edges


def cpp_comment_remover(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def generate_prolog(code):
    JOERN_PATH = ""
    if code.strip() == "":
        return ""
    tmp_dir = tempfile.TemporaryDirectory()
    md5_v = hashlib.md5(code.encode()).hexdigest()
    short_filename = "func_" + md5_v + ".cpp"
    with open(tmp_dir.name + "/" + short_filename, 'w') as f:
        f.write(code)
    # print(short_filename)
    # logger.info(short_filename)
    subprocess.check_call([JOERN_PATH + "/joern-parse", tmp_dir.name, "--out", tmp_dir.name + "/cpg.bin.zip"])

    tree = subprocess.check_output(
        "cd {} && ./joern --script cpg_to_dot.sc --params cpgFile=".format(
            JOERN_PATH) + tmp_dir.name + "/cpg.bin.zip",
        shell=True,
        universal_newlines=True,
    )
    pos = tree.find("digraph g {")
    # print(pos)
    if pos > 0:
        tree = tree[pos:]
    tmp_dir.cleanup()
    return tree


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
                res[tree_type] = tree_body
                tree_body = ""

            tree_type = line[2:].strip()
            # print(tree_type)
        elif tree_type != "":
            tree_body += line + "\n"
    if tree_body != "":
        res[tree_type] = tree_body
    return res


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


def get_joern_code(line):
    p1 = re.compile(r'joern_code_[(](.*?)[)]', re.S)
    res = re.findall(p1, line)
    if len(res) > 0:
        return res[0]
    return ''


def get_joern_line_no(line):
    p1 = re.compile(r'joern_line_[(](.*?)[)]', re.S)
    res = re.findall(p1, line)
    if len(res) > 0:
        return res[0]
    return ''


def get_edges(tree):
    visited_edges = []
    cfg_edges = []
    for line in tree.splitlines():
        if line.find('" -->> "') > -1:
            a, b = line.split('" -->> "', 1)

            # id1 = get_joern_id(a)
            # id2 = get_joern_id(b)

            l1 = get_joern_line_no(a)
            l2 = get_joern_line_no(b)

            if l1 == '' or l2 == '':
                continue

            # t1 = get_joern_type(a)
            # t2 = get_joern_type(b)

            k = l1 + "_" + l2
            if k not in visited_edges:
                visited_edges.append(k)
                cfg_edges.append({
                    "node_out": l1,
                    "node_in": l2
                })
    return cfg_edges


def process_pdg_edges(pdg_edges, rdef_edges):
    pdg_edges_fixed = []

    rdef_keys = []
    for e in rdef_edges:
        k = e['node_out'] + "_" + e['node_in']
        # print("rdef_key:", k)
        if k not in rdef_keys:
            rdef_keys.append(k)

    for e in pdg_edges:
        k = e['node_out'] + "_" + e['node_in']

        if k in rdef_keys:
            e['edge_type'] = 'data_dependency'
        else:
            e['edge_type'] = 'control_dependency'
        # print("pdg_key:", k, e['edge_type'])
        pdg_edges_fixed.append(e)
    return pdg_edges_fixed


def get_statement_types(ast_tree):
    lines_nodes = {}
    statement_types = {}
    for line in ast_tree.splitlines():
        if line.find('" -->> "') > -1:
            a, b = line.split('" -->> "', 1)

            # id1 = get_joern_id(a)
            # id2 = get_joern_id(b)

            l1 = get_joern_line_no(a)
            l2 = get_joern_line_no(b)

            if l1 == '' or l2 == '':
                continue
            # if int(l1) > int(l2) :
            #     # print("???", line)
            #     continue

            t1 = get_joern_type(a)
            t2 = get_joern_type(b)

            if int(l1) == int(l2) and l1 not in statement_types:
                # 如果两者行号相等，那么左边的即是 statement type
                statement_types[l1] = t1

            if l1 not in lines_nodes:
                lines_nodes[l1] = []

            lines_nodes[l1].append(line)
    return statement_types


def get_ast_nodes_edges(ast_tree):
    nodes = {}
    edges = []

    for line in ast_tree.splitlines():
        if line.find('" -->> "') > -1:

            a, b = line.split('" -->> "', 1)

            id1 = get_joern_id(a)
            id2 = get_joern_id(b)

            l1 = get_joern_line_no(a)
            l2 = get_joern_line_no(b)

            if l1 == '' or l2 == '':
                continue

            # t1 = get_joern_type(a)
            # t2 = get_joern_type(b)

            c1 = get_joern_code(a)
            c2 = get_joern_code(b)

            if id1 not in nodes:
                nodes[id1] = c1
            if id2 not in nodes:
                nodes[id2] = c2

            edges.append((id1, id2))
    return nodes, edges


def process_nodes(edges, t=1):
    nodes = []
    for e in edges:
        if t == 1:
            a = e['node_out']
            b = e['node_in']
        else:
            a = e[0]
            b = e[1]

        if a not in nodes:
            nodes.append(a)
        if b not in nodes:
            nodes.append(b)
    return nodes


def process_tree(code, tree):
    """
    return:
        nodes: []
        cfg_edges: []
        pdg_edges: []
    """
    trees = split_trees(tree)
    ast_nodes, ast_edges = get_ast_nodes_edges(trees['AST'])

    statement_nodes = {}
    statement_types = get_statement_types(trees['AST'])
    for i, line in enumerate(code.splitlines()):
        s_type = ""
        if str(i + 1) in statement_types:
            s_type = statement_types[str(i + 1)]
        statement_nodes[str(i + 1)] = {
            'code': line.strip(),
            'label': s_type
        }
    # for k, v in nodes.items():
    #     print(k, v)

    cfg_edges = get_edges(trees['CFG'])
    cfg_nodes = process_nodes(cfg_edges)

    pdg_edges = get_edges(trees['PDG'])
    pdg_nodes = process_nodes(pdg_edges)

    # rdef_edges = get_edges(trees['REACHING_DEF'])

    pdt_nodes_, pdt_edges = get_pdt_nodes_edges(tree, code)
    pdt_nodes = process_nodes(pdt_edges, 0)

    # pdg_edges_with_type = process_pdg_edges(pdg_edges, rdef_edges)
    pdg_edges_with_type = None

    return {
        'ast_nodes': ast_nodes,
        'ast_edges': ast_edges,
        'statement_nodes': statement_nodes,

        'cfg_edges': cfg_edges,
        'cfg_nodes': cfg_nodes,

        'pdg_edges': pdg_edges,
        'pdg_nodes': pdg_nodes,

        'pdt_nodes': pdt_nodes,
        'pdt_edges': pdt_edges,
    }

