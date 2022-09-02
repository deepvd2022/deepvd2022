#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[2]:




sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 2)


def plot_embedding(X_org, y, perplexity, title=None):
    X, Y = np.asarray(X_org), np.asarray(y)
    # X = X[:10000]
    # Y = Y[:10000]
    # y_v = ['Vulnerable' if yi == 1 else 'Non-Vulnerable' for yi in Y]
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity)
    print('Fitting TSNE!')
    X = tsne.fit_transform(X)
    
    
    
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    
    pmed, nmed = calculate_centroids(X, y)
    dist = calculate_distance(pmed, nmed)
    print(pmed, nmed, dist)
    
    
    file_ = open(str(title) + '-tsne-features.json', 'w')
    if isinstance(X, np.ndarray):
        _x = X.tolist()
        _y = Y.tolist()
    else:
        _x = X
        _y = Y
    json.dump([_x, _y], file_)
    file_.close()
    plt.figure(title)
    # sns.scatterplot(X[:, 0], X[:, 1], hue=y_v, palette=['red', 'green'])
    for i in range(X.shape[0]):
        if Y[i] == 0:
            plt.text(X[i, 0], X[i, 1], 'o',
                     color=plt.cm.Set1(2),
                     fontdict={'size': 9})
        else:
            plt.text(X[i, 0], X[i, 1], '+',
                     color=plt.cm.Set1(0),
                     fontdict={'size': 9})
    # plt.scatter()
    # plt.xticks([]), plt.yticks([])
    if title is None:
        title = "test"
    plt.title(title)
    plt.savefig(title.replace(" ", "_") + '.pdf')
    plt.show()

def calculate_centroids(_features, _labels):
    pos = []
    neg = []
    for f, l  in zip(_features, _labels):
        if l == 1:
            pos.append(f)
        else:
            neg.append(f)
    posx = [x[0] for x in pos]
    posy = [x[1] for x in pos]
    negx = [x[0] for x in neg]
    negy = [x[1] for x in neg]
    _px = np.median(posx)
    _py = np.median(posy)
    _nx = np.median(negx)
    _ny = np.median(negy)
    return (_px, _py), (_nx, _ny)
    pass


def calculate_distance(p1, p2):
    return np.abs(np.sqrt(((p1[0] - p2[0])*(p1[0] - p2[0])) + ((p1[1] - p2[1])*(p1[1] - p2[1]))))
    pass


# In[23]:



features = np.load("tsne_features.npy")
y = np.load("tsne_y.npy")

# if len(features) > 335:
#     _, features, _,  y = train_test_split(features, y, test_size=335)
    
print("features:", features.shape)
print("y:", y.shape)

plot_embedding(features, y, 3, 'DeepVD')



