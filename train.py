import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn import metrics
import untils
import time
import torch
import random
from model import higcn
from utils import count_parameters
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_weight(importance, title):
    #from AffinityNet
    feature_weight_all = importance.detach().cpu().data.numpy()
    positive_list = []
    noisy_list = [i for i in range(1000)]
    for i in positive:
        # for gedfn
        positive_list.append(int(i))
        # for crimmix
        # positive_list.append(int(i[4:])-1)
    for j in positive_list:
        if j in noisy_list:
            noisy_list.remove(j)
    feature_weight_all = np.concatenate([feature_weight_all[positive_list], feature_weight_all[noisy_list]])
    colors = ['r'] * len(positive_list) + ['b'] * len(noisy_list)
    untils.plot_feature_weight(feature_weight_all, colors, title)

def plot_train_test_loss(loss_train, train, loss_test, test):
    x = range(1,101)

    plt.subplot(2,1,1)
    plt.plot(x, loss_train, label='Train', color='r')
    plt.plot(x, loss_test, label='Test', color='b')
    plt.title('a. GEDFN Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(x, train, label='Train', color='r')
    plt.plot(x, test, label='Test', color='b')
    plt.title('b. GEDFN Balanced ACC')
    plt.xlabel('epoch')
    plt.ylabel('balanced ACC')
    plt.legend()

    plt.subplots_adjust(hspace=2)

    plt.savefig('GEDFN training loss and bacc.eps', ppi=600, format='eps')
    plt.show()
    # plt.close()

def plot_weight_Graph(weight):
    weight = model.sgcn.weight.detach().numpy()
    gene_A = pd.read_table('../DATASETS/Kidney/exp_gene_A.txt', index_col=0)
    gene_A = gene_A * weight

    name = gene_A.index
    gene_A = np.array(gene_A)
    np.fill_diagonal(gene_A, 0)
    gene_A = pd.DataFrame(gene_A)
    gene_A.index = name
    gene_A.columns = name

    gene_A = gene_A.abs()


    gene_A = gene_A[gene_A > 0.09]
    g1 = gene_A.dropna(axis=0, how='all').index
    g2 = gene_A.dropna(axis=1, how='all').columns
    g_name = set(g1).union(set(g2))
    print(len(g_name))
    gene_A = gene_A[g_name].loc[g_name]
    gene_A = gene_A.fillna(0)
    for i in range(len(g_name)):
        for j in range(i, len(g_name)):
            # print(i,j)
            gene_A.iloc[i, j] = gene_A.iloc[i, j] + gene_A.iloc[j, i]
            gene_A.iloc[j, i] = 0

    G = nx.from_pandas_adjacency(gene_A)
    fig = plt.subplots()
    nx.draw_networkx(G, font_size=5, node_size=100, width=0.5, edge_color='r', node_color='#F0F8FF')
    plt.savefig('./genes1.eps', ppi=600, format='eps')
    plt.show()

avg_10_ans = []
for it in range(1,11):
    print("##############################")
    print('it:', it)
    print("##############################")

    # ## Kidney real data
    # x1 = pd.read_table('../DATASETS/Kidney/KICH/expression.txt', index_col=0)
    # x2 = pd.read_table('../DATASETS/Kidney/KIRC/expression.txt', index_col=0)
    # x3 = pd.read_table('../DATASETS/Kidney/KIRP/expression.txt', index_col=0)
    # gene_A = pd.read_table('../DATASETS/Kidney/exp_gene_A.txt', index_col=0)
    # x = pd.concat([x1, x2, x3], axis=1)
    # x = x.loc[gene_A.index].T
    # x = np.asarray(x)
    # gene_A = np.asarray(gene_A)
    # y = np.asarray([0] * x1.shape[1] + [1] * x2.shape[1] + [2] * x3.shape[1])
    # ###

    ### crimmix simulation data
    positive = np.asarray(pd.read_csv('./simulation/crimmix/omic%d_positive.txt'%it, sep='\t', header=None)).reshape(-1)
    gene_A = np.loadtxt('./simulation/crimmix/omic%d_gene_A.txt'%it)
    x = np.asarray(pd.read_csv('./simulation/crimmix/omic%d.txt'%it, sep='\t'))
    y = np.asarray([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100)
    ###

    # ## gednf simulation data
    # positive = np.loadtxt('./simulation/gedfn/gedfn%d_position.txt' % it)
    # gene_A = np.loadtxt('./simulation/gedfn/gedfn%d_gene_A.txt' % it)
    # np.fill_diagonal(gene_A, 1)
    # x = np.loadtxt('./simulation/gedfn/gedfn%d_x.txt' % it)
    # y = np.loadtxt('./simulation/gedfn/gedfn%d_y.txt' % it)
    # ##

    yy = []
    for i in range(len(y)):
        if y[i] == 0:
            yy.append([1, 0, 0, 0])
        elif y[i] == 1:
            yy.append([0, 1, 0, 0])
        elif y[i] == 2:
            yy.append([0, 0, 1, 0])
        elif y[i] == 3:
            yy.append([0, 0, 0, 1])

    x = scale(x)
    start = time.time()
    A = untils.cal_A(x)

    ### For GEDFN
    gamma_c = 50
    gamma_numerator = np.sum(gene_A, axis=0)
    gamma_denominator = np.sum(gene_A, axis=0)
    gamma_numerator[np.where(gamma_numerator > gamma_c)] = gamma_c

    x = torch.FloatTensor(x)
    A = torch.FloatTensor(A)
    gene_A = torch.FloatTensor(gene_A)
    y = torch.LongTensor(y)
    yy = torch.LongTensor(yy)

    ## hyper-parameters and settings
    learning_rate = 0.01
    training_epochs = 100
    weight_decay = 1e-4
    train_portions = [0.01]
    num_cls = y.data.max().item() + 1
    in_dim = x.shape[1]
    var_importance_mean_all = torch.zeros([in_dim])
    loss_train_all=[]
    loss_test_all=[]
    train_bacc_all=[]
    test_bacc_all=[]
    for train_portion in train_portions:
        ans = []
        for re in range(5):
            model = higcn(in_dim, num_cls)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            proportions = [train_portion] * num_cls
            x_train, y_train, x_test, y_test, train_idx, test_idx = untils.split_data(
                x, y, proportions=proportions, seed=random.randint(0,10))
            print('train size: {0}, test size: {1}'.format(y_train.size(0), y_test.size(0)))
            for i in range(training_epochs):
                model.train()
                optimizer.zero_grad()
                output = model(x, A, gene_A)
                loss = torch.nn.CrossEntropyLoss()
                loss_train = loss(output[train_idx], y[train_idx])
                loss_train.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    output = model(x, A, gene_A)
                    loss_test = loss(output[test_idx], y[test_idx])
                    print('Epoch: {:04d}'.format(i),
                          'train_loss: {:.4f}'.format(loss_train.item()),
                          'train_roc_auc: {:.4f}'.format(metrics.roc_auc_score(yy[train_idx].cpu(), output[train_idx].cpu())),
                          'train_balance_acc_train: {:.4f}'.format(
                              metrics.balanced_accuracy_score(y[train_idx].cpu(), output[train_idx].argmax(dim=1).cpu())),

                          'test_roc_auc: {:.4f}'.format(metrics.roc_auc_score(yy[test_idx].cpu(), output[test_idx].cpu())),
                          'test_balance_acc_test: {:.4f}'.format(
                              metrics.balanced_accuracy_score(y[test_idx].cpu(), output[test_idx].argmax(dim=1).cpu())))

            #         loss_train_all.append(loss_train.item())
            #         train_bacc_all.append(metrics.balanced_accuracy_score(y[train_idx].cpu(), output[train_idx].argmax(dim=1).cpu()))
            #         loss_test_all.append(loss_test.item())
            #         test_bacc_all.append(metrics.balanced_accuracy_score(y[test_idx].cpu(), output[test_idx].argmax(dim=1).cpu()))
            # plot_train_test_loss(loss_train_all, train_bacc_all, loss_test_all, test_bacc_all)
            # print('confusion_matrix: ', confusion_matrix(y[test_idx].cpu(), output[test_idx].argmax(dim=1).cpu()))

            ans.append(metrics.balanced_accuracy_score(y[test_idx].cpu(), output[test_idx].argmax(dim=1).cpu()))

            var_left = torch.sum(torch.abs(model.sgcn.weight * gene_A), 0)
            var_left_mean = var_left / var_left.sum()
            var_right = torch.sum(torch.abs(model.linear1.weight), 0)
            var_right_mean = var_right / var_right.sum()
            # var_importance = (var_left * torch.FloatTensor(gamma_numerator)) * (1.0 / torch.FloatTensor(gamma_denominator)) + var_right
            var_importance_mean = var_left_mean + var_right_mean
            var_importance_mean_all += var_importance_mean

            # plot_weight(var_left, 'a. GEDFN_left')
            # plot_weight(var_right, 'b. GEDFN_right')
            # plot_weight(var_importance, 'c. GEDFN_importance')
            # plot_weight(var_left_mean, 'd. HiGCN_left')
            # plot_weight(var_right_mean, 'e. HiGCN_right')
            # plot_weight(var_importance_mean, 'f. HiGCN_importance')

        avg_10_ans.append(np.mean(ans))
        print("bacc: ", ans)
        print("mean: {:.4f}".format(np.mean(ans)))
        print("std: {:.4f}".format(np.std(ans)))

    end = time.time()
    print('time: ', (end - start)/5)

    # plot_weight(var_importance_mean_all, 'title')
    # plot_weight_Graph(model.sgcn.weight.detach().numpy())
print("avg_10_ans: ", avg_10_ans)
print("avg_10_ans_mean: {:.4f}".format(np.mean(avg_10_ans)))
print("avg_10_ans_std: {:.4f}".format(np.std(avg_10_ans)))