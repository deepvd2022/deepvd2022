# DeepVD

## Dataset

The dataset used in DeepVD can be downloaded [here](https://drive.google.com/drive/folders/1VPUGYjrhIEXYOdPjYGdwYrHfvGb4LL7O?usp=sharing). The dataset contains 303 large and popular C/C++ projects covering the CVEs from 2000-2021 with 13,130 vulnerable methods.  

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
