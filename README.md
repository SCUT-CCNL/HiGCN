# HiGCN

HiGCN: a hierarchical graph convolution network for representation learning of gene expression data

## 1. Requirements

- NumPy (1.17.4) http://www.numpy.org/
- Pandas (0.25.3) https://pandas.pydata.org/
- Scikit-learn (0.22) http://scikit-learn.org/stable/install.html
- pytorch (1.3.1) https://pytorch.org/
- snfpy (0.2.1) https://github.com/rmarkello/snfpy

## 2. Code Base Structure

**train.py**: main script for classification

**model.py**: contains PyTorch model definitions for HiGCN

**sparsegcn_layer.py**: contains PyTorch definitions for Sparse GCN

**featureweightedgcn_layer.py**: contains PyTorch definitions for Feature-weighted GCN

**until.py**: contains definitions for Sample Similarity Graph construction,  C-index, data loader, etc

**cox/train.py**: main script for Cox.

**cox/utils.py**: contains definitions for Sample Similarity Graph construction,  figure plotting, etc

**simulation/crimmix**: **SimData1**. It's generated from Crimmxi.

**simulation/gedfn**: **SimData2**. It's generated based on GEDFN

## 3. Training and Evaluation

By default, you can directly run `python train.py` to get the result of HiGCN on **SimData1**.

If you want to classify other dataset (i.e., **SimData2**). You can directly comment out the loading code of **SimData1** and add code to load your own dataset. (The code to load **SimData2** can be finded in **train.py**)

## Acknowledgments

This code is inspired by [GEDFN](https://github.com/yunchuankong/GEDFN) and [AffinityNet](https://github.com/BeautyOfWeb/AffinityNet), code base structure was inspired by [GCN](https://github.com/tkipf/pygcn/tree/master/pygcn)

