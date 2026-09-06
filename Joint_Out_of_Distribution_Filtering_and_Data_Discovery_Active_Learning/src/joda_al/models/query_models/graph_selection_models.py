import math

import torch
from torch import nn as nn
from torch.nn import Module, Parameter, functional as F


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/3664f2dc90cbf971564c0bf186dc794f12446d0c/models.py
    """
    def __init__(self, in_features, out_features, dropout, alpha = 1e-2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        attendence, attentions = [], []
        for att_layer in self.attentions:
            a, attention = att_layer(x, adj)
            attendence.append(a)
            attentions.append(attention)
        x = torch.cat(attendence)
        feat = x
        attention= torch.cat(attentions)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return torch.sigmoid(x), attention, torch.cat((feat,x),1)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.linear = nn.Linear(nclass, 1)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        feat = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(feat, adj)
        #x = self.linear(x)
        # x = F.softmax(x, dim=1)
        return torch.sigmoid(x), feat, torch.cat((feat,x),1)


class GCNMC(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCNMC, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.linear = nn.Linear(nclass, 1)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        feat = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(feat, adj)
        #x = self.linear(x)
        # x = F.softmax(x, dim=1)
        return x, feat, torch.cat((feat,x),1)


class GCNAttention(nn.Module):
    """https://github.com/Diego999/pyGAT/blob/3664f2dc90cbf971564c0bf186dc794f12446d0c/models.py"""
    def __init__(self, nfeat, nhid, nclass, dropout,alpha,nheads=1):
        super(GCNAttention, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.linear = nn.Linear(nclass, 1)
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        feat = F.dropout(x, self.dropout, training=self.training)
        #x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        a, attention = self.out_att(x, adj)
        x = F.elu(a)
        x = self.gc3(x, adj)
        # x = F.softmax(x, dim=1)
        return torch.sigmoid(x), attention, torch.cat((feat,x),1)


class GATNet(nn.Module):
    """GATNet with three GAT layers and variable edge dimension. Output is for binary classification."""
    def __init__(self, in_channels, out_channels, edge_dim, heads):
        super(GATNet, self).__init__()
        self.GAT1 = GATConv(in_channels, out_channels, heads=heads, edge_dim=edge_dim)
        self.GAT2 = GATConv(out_channels * heads, out_channels, heads=heads, edge_dim=edge_dim)
        self.GAT3 = GATConv(out_channels * heads, out_channels, heads=heads, edge_dim=edge_dim)
        self.linear = nn.Linear(out_channels, 1)

    def forward(self, x, adj, edge_attr):
        x = F.elu(self.GAT1(x=x, edge_attr=edge_attr, edge_index=adj))
        feat = F.elu(self.GAT2(x=x, edge_attr=edge_attr, edge_index=adj))
        x = F.elu(self.GAT3(x=x, edge_attr=edge_attr, edge_index=adj))
        x = self.linear(x)
        return torch.sigmoid(x), feat, torch.cat((feat,x),1)


class GeneralConvNet(nn.Module):
    """GeneralNet with three GeneralConv layers and variable edge dimension. Output is for binary classification."""
    def __init__(self, in_channels, out_channels, edge_dim):
        super(GeneralConvNet, self).__init__()
        self.General1 = GeneralConv(in_channels, out_channels, edge_dim=edge_dim)
        self.General2 = GeneralConv(out_channels, out_channels, edge_dim=edge_dim)
        self.General3 = GeneralConv(out_channels, out_channels, edge_dim=edge_dim)
        self.linear = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.General1(x, edge_index, edge_attr))
        feat = F.elu(self.General2(x, edge_index, edge_attr))
        x = F.elu(self.General3(x, edge_index, edge_attr))
        x = self.linear(x)
        return torch.sigmoid(x), feat, torch.cat((feat, x), 1)


class ResGatedGraphNet(nn.Module):
    """ResGatedGraphConv with three ResGatedGraphConv layers and variable edge dimension. Output is for binary classification."""
    def __init__(self, in_channels, out_channels, edge_dim):
        super(ResGatedGraphNet, self).__init__()
        self.ResGated1 = ResGatedGraphConv(in_channels, out_channels, edge_dim=edge_dim)
        self.ResGated2 = ResGatedGraphConv(out_channels, out_channels, edge_dim=edge_dim)
        self.ResGated3 = ResGatedGraphConv(out_channels, out_channels, edge_dim=edge_dim)
        self.linear = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.ResGated1(x, edge_index, edge_attr))
        feat = F.elu(self.ResGated2(x, edge_index, edge_attr))
        x = F.elu(self.ResGated3(x, edge_index, edge_attr))
        x = self.linear(x)
        return torch.sigmoid(x), feat, torch.cat((feat, x), 1)
