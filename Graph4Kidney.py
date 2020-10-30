import pandas as pd
import numpy as np
import networkx as nx

#读HINT的binary数据, 并处理为领接矩阵
binary = pd.read_table('./HomoSapiens_binary_hq.txt')
graph = binary[['Gene_A', 'Gene_B']]
l=[]
for i in range(graph.shape[0]):
    l.append(tuple(graph.iloc[i]))
G=nx.Graph(l)
A=nx.adjacency_matrix(G).todense()
A=pd.DataFrame(A)
A.index = G.nodes
A.columns = G.nodes
###
#读所要处理的组学数据,methylation的和（expression，mutation）的略有不同
exp = pd.read_table('/Users/tankaiwen/CCNL/DATASETS/Kidney/KIRP/expression.txt')
exp.index = exp['sample']
del exp['sample']
# mut = pd.read_table('/Users/tankaiwen/CCNL/DATASETS/Kidney/KIRP/mutation.txt')
# mut.index = mut['sample']
# del mut['sample']
# met = pd.read_table('/Users/tankaiwen/CCNL/DATASETS/Kidney/methylation__TSS200-TSS1500__Both.txt')
# met.index = met['UniGene']
# del met['UniGene']

###
#获取共有数据
###
common_exp = set(A.index) & set(exp.index)
exp_gene_adj = A[list(common_exp)].loc[list(common_exp)]
# common_mut = set(A.index) & set(mut.index)
# mut_gene_adj = A[list(common_mut)].loc[list(common_mut)]
# common_met = set(A.index) & set(met.index)
# met_gene_adj = A[list(common_met)].loc[list(common_met)]


#获得对角为1的领结矩阵
np.fill_diagonal(exp_gene_adj.values, 1)
exp_gene_adj.to_csv('./exp_gene_A.txt',sep='\t')
# np.fill_diagonal(mut_gene_adj.values, 1)
# mut_gene_adj.to_csv('./mut_gene_A.txt',sep='\t')
# np.fill_diagonal(met_gene_adj.values, 1)
# met_gene_adj.to_csv('./met_gene_A.txt',sep='\t。/
###
###