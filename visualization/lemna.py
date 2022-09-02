"""
1. Select one sample A from the dataset
2. Build a number of (5000 in default) virtual samples using the tokens in A. In each virtual sample, randomly drops some tokens.
3. Use the pre-trained model on these virtual samples to make predictions.
4. Take the virtual samples and predictions as input, and use Lemna to analyze the contribution of each token.

"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import copy
from rpy2 import robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import random
import torch
import argparse
import sys
import json
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
import os


# In[2]:


from embeddings import ns


# In[3]:


lemna_data = "lemna_data"
def findAllFile(base, full=True):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if full:
                yield os.path.join(root, f)
            else:
                yield f


# In[4]:


r = robjects.r
rpy2.robjects.numpy2ri.activate()
importr('genlasso')
importr('gsubfn')
random.seed(178)
torch.manual_seed(178)


# In[5]:


def find_minus_statements(diff=None):
    res = []
    if diff is None: # for test
        diff = {
            '9': '-         int ret = poll(pfds, ts[h].poll_count, 1);'
        }
    for k, v in diff.items():
        if v.startswith("-"):
            v = v[1:].strip()
        # print(k, v)
        res.append(v)
    return res



# In[6]:


def find_minus_tokens(diff=None):
    res = []
    if diff is None: # for test
        diff = {
            '9': '-         int ret = poll(pfds, ts[h].poll_count, 1);'
        }
    for k, v in diff.items():
        if v.startswith("-"):
            vv= v[1:].strip()
            st_tokens = ns.statement2tokens(vv)
            res = res + st_tokens
    return res



# In[7]:


def sort_statements(s=None):
    if s is None:
        s = [
            ('int ret;', 0.2),
            ('int ret = poll(pfds, ts[h].poll_count, -1);', 0.9),
            ('int ret = poll();', 0.7),
        ]
    s.sort(key=lambda x:x[1], reverse=True)
    return s

# In[8]:


def run_lemna(data):
    syth_mat = np.array(data['syth_mat'])
    pred_lst = data['pred_lst']
#     print("syth_mat:", syth_mat)
    X = r.matrix(syth_mat, nrow=syth_mat.shape[0], ncol=syth_mat.shape[1])
    Y = r.matrix(pred_lst, ncol=1)
    
    n = r.nrow(X)
    p = r.ncol(X)
    results = r.fusedlasso1d(y=Y, X=X, gamma=0, approx=True, maxsteps=2000,
                             minlam=0, rtol=1e-07, btol=1e-07, eps=1e-4)
    tmp = r.coef(results, np.sqrt(n * np.log(p)))
#     print(tmp)
    result = np.array(r.coef(results, np.sqrt(n * np.log(p)))[0])[:, -1]
#     print(result)

    token_score = {}
    for token, score in zip(data['tokens'], result):
        token_score[token] = score
        #print(token, score)
    to_file_token_score = file + ".score"
    torch.save(token_score, to_file_token_score)
    print("saved to", to_file_token_score)
    return token_score


# In[49]:


topK = {}
K = 51
for i in range(1, K):
    topK[i] = 0
total = 0
print(topK)


statements_num = [] # per method
tokens_num = []     # per statement
vul_statements_num = []

for file in findAllFile(lemna_data):
    if not file.endswith("14_1.pt"):
        continue
    print(file)
    data = torch.load(file)
    print(data['y_pred'])
    print(data['lv0_func']['vul'])
    
    if data['lv0_func']['vul'] == 0:
        continue
    if data['y_pred'] == 0:
        continue
        
    if 'diff' not in data['lv0_func'].keys():
        continue
    
    file_token_score = file + ".score"
    if os.path.exists(file_token_score):
        token_score = torch.load(file_token_score)
    else:
        token_score = run_lemna(data)
    
    
    statements_num.append(len(data['ns_list']))
    for ns_ in data['ns_list']:
        tokens_num.append(len(ns_['st_tokens']))
        
    vul_statements = find_minus_statements(data['lv0_func']['diff'])
    if len(vul_statements) > 0:
        vul_statements_num.append( len(vul_statements)) 
    else:
        continue
    vul_tokens = find_minus_tokens(data['lv0_func']['diff'])
    
    print(data['cve_id'])
    print(data['func_key'])

    tokens = []
    for k, v in token_score.items():
        tokens.append( (k, v) )
    tokens.sort(key=lambda x:x[1], reverse=True)

    flags = [0] * K
    cover_statements_num = 0
    for i, (token, score) in enumerate(tokens):
        for v_t in vul_tokens:
            if v_t == token:
                print('== ', token)
                cover_statements_num += 1
                for k in range(1, K):
                    if i < k:
                        flags[k] = 1
    for k,v in data['lv0_func']['diff'].items():
        print(k, v)
        
    print("cover_statements_num:", cover_statements_num)
    if cover_statements_num > 0 and cover_statements_num/len(vul_statements) > 0.3:
        print("example")
    for k in range(1, K):
        if flags[k] > 0:
            print("hit top:", k)
            topK[k] += 1
    
    total += 1
    print("=" * 20)


for k in range(1, K):
    print("acc top-{}: {:.4f}".format(k, topK[k] / total) )

