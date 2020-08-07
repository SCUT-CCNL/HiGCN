import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence

import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale

from untils import cal_A



def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm

def normalize_adj_con(adj, symmetric=True): # normalize continue adj
    sums = adj.sum(1)
    a_norm = adj.div(sums, axis="rows")
    return a_norm

def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj_con(adj, symmetric)
    laplacian = np.eye(adj.shape[0]) - adj_normalized
    return laplacian

def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian

def load_data(DATASET):
    print('Loading {} dataset...'.format(DATASET))
    X = pd.read_table("/data1/home/tankaiwen/CCNL/PycharmProjects/DATASETS/Pancancer/expression.txt", index_col=0)
    gene_A = pd.read_table("/data1/home/tankaiwen/CCNL/PycharmProjects/DATASETS/Pancancer/exp_gene_A.txt", index_col=0)
    y = pd.read_table("/data1/home/tankaiwen/CCNL/PycharmProjects/DATASETS/Pancancer/clin.txt", index_col=0)
    X = scale(X)

    A = cal_A(X)

    # A = normalize_adj_con(A)
    # A = sp.csr_matrix(A)  # 转成稀疏矩阵
    # A = rescale_laplacian(A)
    ystatus = y[DATASET]
    ytime = y[DATASET+".time"]

    gene_A = torch.FloatTensor(np.asarray(gene_A))
    features = torch.FloatTensor(np.array(X))
    ystatus = torch.LongTensor(np.array(ystatus))
    ytime = torch.LongTensor(np.array(ytime))
    adj = A

    return adj, gene_A, features, ystatus, ytime


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    #preds = output.max(1)[1].type_as(labels)
    #correct = preds.eq(labels).double()
    rocauc = roc_auc_score(labels.cpu(), output.cpu().detach().numpy())
    #correct = correct.sum()
    return rocauc

def CIndex(result, ytime_test, ystatus_test):
    concord = 0.
    total = 0.
    N_test = ystatus_test.shape[0]
    ystatus_test = np.asarray(ystatus_test, dtype=bool)
    theta = result
    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                if ytime_test[j] > ytime_test[i]:
                    total = total + 1
                    if theta[j] < theta[i]: concord = concord + 1
                    elif theta[j] < theta[i]: concord = concord + 0.5
    return(concord/total)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def cal_R(ytime, train_index):
    ytime_train = ytime[train_index]
    N_train = train_index.shape[0]
    R_matrix_train = np.zeros([N_train, N_train], dtype=int)
    for i in range(N_train):
        for j in range(N_train):
            R_matrix_train[i,j] = ytime_train[j] >= ytime_train[i]
    # return torch.LongTensor(R_matrix_train).cuda()
    return torch.LongTensor(R_matrix_train)
