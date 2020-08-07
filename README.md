# HiGCN

HiGCN: a hierarchical graph convolution network for representation learning of gene expression data

**CONTACT**: For questions or comments about the code please contact: kwtan0909@qq.com / cskwtan93@mail.scut.edu.cn / sbdong@scut.edu.cn

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

**simulation/crimmix**: *SimData1*. It's generated from Crimmix (ten datasets in total)

**simulation/crimmix/omic1**: sample-feature matrix of the first dataset

**simulation/crimmix/omic1_gene_A**: gene interaction graph of the first dataset

**simulation/crimmix/omic1_positive**: true signals of the first dataset

**simulation/gedfn**: *SimData2*. It's generated based on GEDFN (ten datasets in total)

**simulation/gedfn/gedfn1_x**: sample-feature matrix of the first dataset

**simulation/gedfn/gedfn1_y**: labels of the first dataset

**simulation/gedfn/gedfn1_gene_A**: gene interaction graph of the first dataset

**simulation/gedfn/gedfn1_position**: true signals of the first dataset

## 3. Training and Evaluation

### Classification

By default, you can directly run `python ./train.py` to get the result of HiGCN on **SimData1**. The `train_portions` in **train.py** is used to set *Training Percentage* (Default: 1%).

If you want to classify other dataset (i.e., **SimData2**). You can directly comment out the loading code of **SimData1** and add code to load your own dataset. (The code to load **SimData2** can be finded in **train.py**)

### Survival analysis

Run `python ./cox/train.py`. The survival analysis data we used is too large to upload. If you need them, please contact us by email.

## Acknowledgments

This code is inspired by [GEDFN](https://github.com/yunchuankong/GEDFN) and [AffinityNet](https://github.com/BeautyOfWeb/AffinityNet), code base structure was inspired by [GCN](https://github.com/tkipf/pygcn/tree/master/pygcn)

