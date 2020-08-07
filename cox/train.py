from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index

import torch
import torch.optim as optim
import sys

sys.path.append("../../")

from cox.utils import load_data, cal_R
from model import higcn

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=1000,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=1,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def negative_log_likelihood(theta, R_batch, ystatus_batch):
    exp_theta = torch.exp(theta.sum(1))
    return (-torch.mean((theta.sum(1) - torch.log(torch.sum(exp_theta * R_batch,axis=1))) * ystatus_batch))

def train(epoch, idx_train, idx_test, train_R):
    t = time.time()
    model.train()  # tell pytorch you are training a model, so dropout will work
    optimizer.zero_grad()
    output = model(features, adj, gene_A)
    loss_train = negative_log_likelihood(theta=output[idx_train], R_batch=train_R, ystatus_batch=ystatus[idx_train].float())
    loss_train.backward()
    optimizer.step()

    model.eval()
    # if epoch%50==0:
    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(concordance_index(ytime[idx_train].cpu().detach().numpy(), -output[idx_train].cpu().detach().numpy(), ystatus[idx_train].cpu().detach().numpy())),
          'acc_val: {:.4f}'.format(concordance_index(ytime[idx_test].cpu().detach().numpy(), -output[idx_test].cpu().detach().numpy(), ystatus[idx_test].cpu().detach().numpy())),
          'time: {:.4f}s'.format(time.time() - t))


def test(idx_test):
    with torch.no_grad():
        model.eval()
        output = model(features, adj, gene_A)
        acc_test = concordance_index(ytime[idx_test].cpu().detach().numpy(), -output[idx_test].cpu().detach().numpy(), ystatus[idx_test].cpu().detach().numpy())
        print("Test set results:",
              "**** accuracy= {:.4f} ****".format(acc_test))
        return acc_test


# Train model
# datatypes = ['OS', 'DSS', 'PFI', 'DFI']
DATASET = "OS"
print(DATASET)
result = []
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, gene_A, features, ystatus, ytime = load_data(DATASET)

## Graph permutation experiments (Reviewer2)
adj = torch.eye(adj.shape[0],adj.shape[1])
gene_A = torch.eye(gene_A.shape[0], gene_A.shape[1])

for train_index, test_index in StratifiedKFold(n_splits=5, shuffle=True).split(features.cpu(), ystatus.cpu()):
    temp = train_index
    train_index = test_index
    test_index=temp
    print(train_index.shape)
    print(test_index.shape)
    #calculate the R matrix
    print("calculating train_R matrix")
    train_R = cal_R(ytime, train_index)
    # Model and optimizer
    model = higcn(nfeat=features.shape[1],
               nclass=1)
    # vis_graph = make_dot(model(features, adj), params=dict(model.named_parameters()))
    optimizer = optim.Adam(model.parameters(),
                          lr=args.lr, weight_decay=args.weight_decay)
    if args.cuda:
        pass
        # model.cuda()
        # features = features.cuda()
        # adj = adj.cuda()
        # ystatus = ystatus.cuda()
        # ytime = ytime.cuda()
        # train_index = torch.LongTensor(train_index).cuda()
        # test_index = torch.LongTensor(test_index).cuda()
    t_total = time.time()


    for epoch in range(args.epochs):
        train(epoch, train_index, test_index, train_R)
    test_roc = test(test_index)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    result.append(test_roc)
print('{}({})'.format(round(np.mean(result), 4), round(np.std(result), 4)))
