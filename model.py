import torch
import torch.nn as nn
from sparsegnc_layer import sparsegcn
from featureweightedgcn_layer import fwgcn
# from anf_layer import GraphAttentionLayer

class higcn(nn.Module):
    def __init__(self, nfeat, nclass):
        super(higcn, self).__init__()
        self.sgcn = sparsegcn(nfeat, nfeat)
        self.fgcn = fwgcn(nfeat, nfeat)
        # self.anf1 = GraphAttentionLayer(in_dim=nfeat, out_dim=nfeat, k=10, kernel1='cosine', kernel2='gaussian')
        self.linear1 = nn.Linear(nfeat, 64)
        self.linear2 = nn.Linear(64, 16)
        self.linear3 = nn.Linear(16, nclass)

    def forward(self, x, adj, gene_adj):
        ##HiGCN
        x = self.sgcn(x, gene_adj)
        x = torch.tanh(x)
        x = self.fgcn(x, adj)
        x = torch.tanh(x)
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        x = self.linear3(x)
        return x