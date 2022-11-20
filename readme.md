# DeepVD: Toward Class-Separation Features for Neural Network Vulnerability Detection

The advances of machine learning (ML) including deep learning (DL) have enabled several approaches to implicitly learn vulnerable code patterns to automatically detect software vulnerabilities. Recent study showed that despite successes, the existing ML/DL-based vulnerability detection (VD) models are limited in the ability to distinguish between the two classes of vulnerability and benign code. We propose DeepVD, a graph-based neural network VD model that emphasizes on class-separation features between vulnerability and benign code. DeepVD leverages three types of class-separation features at different levels of abstraction: statement types (similar to Part-of-Speech tagging), Post-Dominator Tree (covering regular flows of execution), and Exception Flow Graph (covering the exception/error-handling flows). We conducted several experiments to evaluate DeepVD in a real-world vulnerability dataset of 303 projects with 13,130 vulnerable methods. Our results show that DeepVD relatively improves over the state-of-the-art ML/DL-based VD approaches 13%–29.6% in precision, 15.6%–28.9% in recall, and 16.4%–25.8% in F-score. Our ablation study confirms that our designed features/components positively contribute to DeepVD’s accuracy.

## Dataset

The dataset used in DeepVD can be downloaded [here](https://drive.google.com/drive/folders/1VPUGYjrhIEXYOdPjYGdwYrHfvGb4LL7O?usp=sharing). It contains 303 large and popular C/C++ projects covering the CVEs from 2000-2021 with 13,130 vulnerable methods.  

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
