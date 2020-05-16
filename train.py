import numpy as np
from sklearn.preprocessing import scale
from sklearn import metrics
import untils
import time
import torch
from model import higcn

def plot_weight(importance, title):
    #from AffinityNet
    feature_weight_all = importance.detach().cpu().data.numpy()
    positive_list = []
    noisy_list = [i for i in range(1000)]
    for i in positive:
        positive_list.append(int(i))
    for j in positive_list:
        if j in noisy_list:
            noisy_list.remove(j)
    feature_weight_all = np.concatenate([feature_weight_all[positive_list], feature_weight_all[noisy_list]])
    colors = ['r'] * len(positive_list) + ['b'] * len(noisy_list)
    untils.plot_feature_weight(feature_weight_all, colors, title)

# ### crimmix simulation data
# positive = np.asarray(pd.read_csv('../../DATASETS/simulation/omic1_positive.txt', sep='\t', header=None)).reshape(-1)
# gene_A = np.loadtxt('../../DATASETS/simulation/omic1_gene_A.txt')
# x = np.asarray(pd.read_csv('../../DATASETS/simulation/omic1.txt', sep='\t'))
# y = np.asarray([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100)
# ###

# ### Kidney real data
# x1 = pd.read_table('../../DATASETS/Kidney/KICH/expression.txt', index_col=0)
# x2 = pd.read_table('../../DATASETS/Kidney/KIRC/expression.txt', index_col=0)
# x3 = pd.read_table('../../DATASETS/Kidney/KIRP/expression.txt', index_col=0)
# gene_A = pd.read_table('../../DATASETS/Kidney/exp_gene_A.txt', index_col=0)
# x = pd.concat([x1, x2, x3], axis=1)
# x = x.loc[gene_A.index].T
# x = np.asarray(x)
# gene_A = np.asarray(gene_A)
# y = np.asarray([0] * x1.shape[1] + [1] * x2.shape[1] + [2] * x3.shape[1])
# ###

## gednf simulation data
positive = np.loadtxt('./simulation/gedfn_position.txt')
gene_A = np.loadtxt('./simulation/gedfn_gene_A.txt')
# np.fill_diagonal(gene_A,1)
x = np.loadtxt('./simulation/gedfn_x.txt')
y = np.loadtxt('./simulation/gedfn_y.txt')
##

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

# ### From gednf
# gamma_c = 50
# gamma_numerator = np.sum(gene_A, axis=0)
# gamma_denominator = np.sum(gene_A, axis=0)
# gamma_numerator[np.where(gamma_numerator > gamma_c)] = gamma_c

x = torch.FloatTensor(x)
A = torch.FloatTensor(A)
gene_A = torch.FloatTensor(gene_A)
y = torch.LongTensor(y)
yy = torch.LongTensor(yy)

## hyper-parameters and settings
learning_rate = 0.01
training_epochs = 100
weight_decay = 1e-4
train_portions = [0.1]
num_cls = y.data.max().item() + 1
in_dim = x.shape[1]
var_importance_mean_all = torch.zeros([in_dim])
for train_portion in train_portions:
    ans = []
    for re in range(5):
        model = higcn(in_dim, num_cls)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        proportions = [train_portion] * num_cls
        x_train, y_train, x_test, y_test, train_idx, test_idx = untils.split_data(
            x, y, proportions=proportions, seed=0)
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
                print('Epoch: {:04d}'.format(i),
                      'train_loss: {:.4f}'.format(loss_train.item()),
                      'train_roc_auc: {:.4f}'.format(metrics.roc_auc_score(yy[train_idx].cpu(), output[train_idx].cpu())),
                      'train_balance_acc_train: {:.4f}'.format(
                          metrics.balanced_accuracy_score(y[train_idx].cpu(), output[train_idx].argmax(dim=1).cpu())),

                      'test_roc_auc: {:.4f}'.format(metrics.roc_auc_score(yy[test_idx].cpu(), output[test_idx].cpu())),
                      'test_balance_acc_test: {:.4f}'.format(
                          metrics.balanced_accuracy_score(y[test_idx].cpu(), output[test_idx].argmax(dim=1).cpu())))

        ans.append(metrics.balanced_accuracy_score(y[test_idx].cpu(), output[test_idx].argmax(dim=1).cpu()))

        var_left = torch.sum(torch.abs(model.sgcn.weight * gene_A), 0)
        var_left_mean = var_left / var_left.sum()
        var_right = torch.sum(torch.abs(model.linear1.weight), 0)
        var_right_mean = var_right / var_right.sum()
        # var_importance_mean = (var_left_mean * torch.FloatTensor(gamma_numerator)) * (
                # 1. / torch.LongTensor(gamma_denominator)) + var_right_mean
        var_importance_mean = var_left_mean + var_right_mean
        var_importance_mean_all += var_importance_mean

    print("bacc: ", ans)
    print("mean: {:.4f}".format(np.mean(ans)))
    print("std: {:.4f}".format(np.std(ans)))

end = time.time()
print('time: ', (end - start)/5)

plot_weight(var_importance_mean_all, 'title')