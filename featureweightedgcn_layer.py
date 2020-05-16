import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class fwgcn(Module):
    """
    feature-weighted gcn layer
    """

    def __init__(self, in_features, out_features, bias=False):
        super(fwgcn, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.a = Parameter(torch.Tensor(in_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # kaiming_uniform
        stdv = 1. / math.sqrt(self.a.size(0))
        self.a.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # a = torch.softmax(self.a, dim=0) #softmax/attention
        output = input * self.a
        # output = torch.mm(input, self.weight) #orignal GCN
        output = torch.mm(adj, output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
