# DeepVD: Toward Class-Separation Features for Neural Network Vulnerability Detection

The advances of machine learning (ML) including deep learning (DL) have enabled several approaches to implicitly learn vulnerable code patterns to automatically detect software vulnerabilities. Recent study showed that despite successes, the existing ML/DL-based vulnerability detection (VD) models are limited in the ability to distinguish between the two classes of vulnerability and benign code. We propose DeepVD, a graph-based neural network VD model that emphasizes on class-separation features between vulnerability and benign code. DeepVD leverages three types of class-separation features at different levels of abstraction: statement types (similar to Part-of-Speech tagging), Post-Dominator Tree (covering regular flows of execution), and Exception Flow Graph (covering the exception/error-handling flows). We conducted several experiments to evaluate DeepVD in a real-world vulnerability dataset of 303 projects with 13,130 vulnerable methods. Our results show that DeepVD relatively improves over the state-of-the-art ML/DL-based VD approaches 13%–29.6% in precision, 15.6%–28.9% in recall, and 16.4%–25.8% in F-score. Our ablation study confirms that our designed features/components positively contribute to DeepVD’s accuracy.

## Dataset

The dataset used in DeepVD can be downloaded [here](https://drive.google.com/drive/folders/1VPUGYjrhIEXYOdPjYGdwYrHfvGb4LL7O?usp=sharing). It contains 303 large and popular C/C++ projects covering the CVEs from 2000-2021 with 13,130 vulnerable methods.  

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

1. DeepVD uses Glove embeddings trained on the CVE dataset. Run the following command to train the Glove model:

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
