# DeepVD

## Dataset

The dataset used in DeepVD can be downloaded [here](https://drive.google.com/drive/folders/1VPUGYjrhIEXYOdPjYGdwYrHfvGb4LL7O?usp=sharing). The dataset contains 303 large and popular C/C++ projects covering the CVEs from 2000-2021 with 13,130 vulnerable methods.  

### Input format

The dataset needs to be composed of two types of files (both in .json format): one contains the attribute information of each method (namely xxx-entities.json), and the other is the calling relationship between methods (namely xxx-edges.json).  

#### xxx-entities.json

- `startLine` : The start line number of the method in the file.
- `endLine` : The end line number of the method in the file.
- `name` : The method name.
- `uniquename` : The method name with a unique name in the whole dataset.
- `contents` : The code of the method.
- `type` : The type of the return value of the method.
- `parameters` : The parameters of the method.
- `commit` : The commit of current version of the project.
- `vul` : 0: non-vulnerable, 1: vulnerable
- `tree` : The AST, CFG, CPG, etc. generated using Joern script.

#### xxx-edges.json

- `nodes` : The unique method keys in `xxx-entities.json`.
- `edge_index` : The edges in the call graph. Each number in the edge_index indicates the index of the method in `nodes`. Each pair of numbers indicates an edge starts from one node, ends at another.



## Train DeepVD

1. DeepVD uses Glove as the Embedding Model. Run the following command to train the Glove model:

```
python train_glove.py (The Glove library may need to run in python 3.6)
```

2. DeepVD uses Node2Vec to encode the graph to get the node embedding of each method:

```
python train_node2vec.py 
```

3. DeepVD uses AutoML to adjust the hyperparameters to train the vulnerability detection model automatically. Run the following command to start training, and you can set the search space of hyperparameters in `launch.py`:

```
python launch.py
```
