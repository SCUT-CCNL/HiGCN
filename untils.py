import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pingouin as pg

import torch
import snf

if torch.cuda.is_available():
    dtype = {'float': torch.cuda.FloatTensor, 'long': torch.cuda.LongTensor, 'byte': torch.cuda.ByteTensor}
else:
    dtype = {'float': torch.FloatTensor, 'long': torch.LongTensor, 'byte': torch.ByteTensor}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_feature_weight(feature_weight_all, colors, title, figsize=10):
    plt.figure(figsize=(figsize, figsize))
    plt.scatter(range(len(feature_weight_all)), feature_weight_all, c=colors, s=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('index', fontsize=20)
    plt.ylabel('weight', fontsize=20)
    plt.title(title, fontsize=25)
    plt.show()
    plt.close()

def cal_A(x):
    ## SNF
    gene_A = snf.make_affinity(x, metric='cosine', K=20) #N_i
    gene_A = torch.FloatTensor(gene_A)
    ## End of SNF
    ### replace snf with pd.corr()
    # gene_A = pd.DataFrame(x)
    # gene_A = gene_A.T.corr()
    # gene_A = torch.FloatTensor(np.array(gene_A))
    # ### End of the replace
    # ### replace snf with pg.pcorr()
    # gene_A = pd.DataFrame(x)
    # gene_A = gene_A.T.pcorr()
    # gene_A = torch.FloatTensor(np.array(gene_A))
    # ###
    value, index = gene_A.sort(descending=True)
    index_0 = index[:, :10] #k
    gene_A_kNN = torch.zeros_like(gene_A)
    for i in range(len(gene_A)):
        temp = gene_A[i, index_0[i]]
        temp = torch.softmax(temp, dim=0)
        k = 0
        for j in index_0[i]:
            gene_A_kNN[i, j] = temp[k]
            k = k + 1
    return gene_A_kNN

def randperm(idx, random_examples=False, seed=None):
    """Randomly permute indices. From https://github.com/BeautyOfWeb/AffinityNet """

    n = len(idx)
    if isinstance(seed, int):
        torch.manual_seed(seed)
        return idx[torch.randperm(n)]
    if random_examples:
        return idx[torch.randperm(n)]
    else:
        return idx

def split_train_test(x_var, y_var, train_indices, y_true=None, seed=None):
    """Split data into training and test (validation) set. From https://github.com/BeautyOfWeb/AffinityNet"""

    test_indices = dtype['long'](sorted(set(range(x_var.size(0))).difference(train_indices.cpu().numpy())))
    if seed is not None:
        train_indices = randperm(train_indices, random_examples=True, seed=seed)
        test_indices = randperm(test_indices, random_examples=True, seed=seed)
    x_train = x_var[train_indices]
    y_train = y_var[train_indices]
    x_test = x_var[test_indices]
    if y_true is None:
        y_test = y_var[test_indices]
    else:
        y_test = y_true[test_indices]
    return x_train, y_train, x_test, y_test, train_indices, test_indices

def split_data(x_var, y_var, num_examples=1, proportions=None, seed=None, random_examples=False):
    """From https://github.com/BeautyOfWeb/AffinityNet"""

    num_clusters = y_var.max().item() + 1 # assume y_var is LongTensor starting from 0 to num_cls-1
    if proportions is not None:
        if isinstance(proportions, float):
            assert proportions > 0 and proportions < 1
            proportions = [proportions]*num_clusters
        num_examples = [max(1,round(torch.nonzero(y_var==i).size(0) * proportions[i])) for i in range(num_clusters)]
    if isinstance(num_examples, int):
        num_examples_per_class = num_examples
        num_examples = [num_examples_per_class]*num_clusters
    assert num_clusters == len(num_examples)
    train_indices = [randperm(torch.nonzero(y_var==i), random_examples, seed)[:num_examples[i],0]
                     for i in range(num_clusters)]
    train_indices = torch.cat(train_indices, dim=0).data
    return split_train_test(x_var, y_var, train_indices, seed=seed)