import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import euclidean_dist, output_conv_size
import numpy as np
from GAT import GraphAttentionLayer
from torch_geometric.nn import GCNConv, SAGPooling
from torch_geometric.nn import global_mean_pool as gep, global_max_pool as gmp
from DC_Conv import DCConv


class RTNet(nn.Module):

    def __init__(self, nfeat, len_ts, nclass, hidden_dim, pooling_ratio, threshold, dropout, filters, kernels, dilation,
                 layers, att_dim, use_att=True, gru_dim=128):
        super(RTNet, self).__init__()
        self.nclass = nclass
        self.dropout = dropout

        # parameters for random projection
        self.att_dim = att_dim
        self.threshold = threshold

        if True:
            # gcn
            self.num_hidden = hidden_dim
            self.pooling_ratio = pooling_ratio

            self.gat = GraphAttentionLayer(nfeat, len_ts)
            self.graph_conv1 = GCNConv(len_ts, self.num_hidden)
            self.graph_conv2 = GCNConv(self.num_hidden, self.num_hidden)
            self.graph_conv3 = GCNConv(self.num_hidden, self.num_hidden)

            self.graph_pool1 = SAGPooling(self.num_hidden, ratio=self.pooling_ratio)
            self.graph_pool2 = SAGPooling(self.num_hidden, ratio=self.pooling_ratio)
            self.graph_pool3 = SAGPooling(self.num_hidden, ratio=self.pooling_ratio)

            # LSTM
            self.channel = nfeat
            self.ts_length = len_ts

            self.gru_dim = gru_dim
            self.gru = nn.GRU(self.ts_length, self.gru_dim)

            paddings = np.array(dilation) * (np.array(kernels) - 1)

            # if self.use_rp:
            #     self.conv_1_models = nn.ModuleList()
            #     self.idx = []
            #     for i in range(self.rp_group):
            #         self.conv_1_models.append(DCConv(self.rp_dim, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1, padding=paddings[0]))
            #         self.idx.append(np.random.permutation(nfeat)[0: self.rp_dim])
            # else:
            self.conv_1 = DCConv(self.channel, filters[0], kernel_size=kernels[0], stride=1, padding=paddings[0], dilation=dilation[0])

            # self.SE1 = SqueezeExcitation1d(filters[0])

            # self.conv_bn_1 = nn.BatchNorm1d(filters[0])

            self.conv_2 = DCConv(filters[0], filters[1], kernel_size=kernels[1], stride=1, dilation=dilation[1], padding=paddings[1])

            # self.SE2 = SqueezeExcitation1d(filters[1])

            self.conv_3 = DCConv(filters[1], filters[2], kernel_size=kernels[2], stride=1, dilation=dilation[2], padding=paddings[2])

            # self.conv_bn_3 = nn.BatchNorm1d(filters[2])

            # compute the size of input for fully connected layers
            fc_input = 0

            conv_size = len_ts
            for i in range(len(filters)):
                conv_size = output_conv_size(conv_size, kernels[i], 1, paddings[i])
            fc_input += conv_size

            fc_input += conv_size * self.gru_dim

            fc_input = filters[2] + self.gru_dim + 2 * self.num_hidden  # 加了graph

        # Representation mapping function

        layers = [fc_input] + layers
        print("Layers", layers)
        self.mapping = nn.Sequential()
        for i in range(len(layers) - 2):
            self.mapping.add_module("fc_" + str(i), nn.Linear(layers[i], layers[i + 1]))
            self.mapping.add_module("bn_" + str(i), nn.BatchNorm1d(layers[i + 1]))
            self.mapping.add_module("relu_" + str(i), nn.LeakyReLU())

        # add last layer
        self.mapping.add_module("fc_" + str(len(layers) - 2), nn.Linear(layers[-2], layers[-1]))
        if len(layers) == 2:  # if only one layer, add batch normalization
            self.mapping.add_module("bn_" + str(len(layers) - 2), nn.BatchNorm1d(layers[-1]))

        # Attention
        att_dim = 64
        self.use_att = use_att
        if self.use_att:
            self.att_models = nn.ModuleList()
            for _ in range(nclass):
                att_model = nn.Sequential(
                    nn.Linear(layers[-1], att_dim),
                    nn.Tanh(),
                    nn.Linear(att_dim, 1)
                )
                self.att_models.append(att_model)
        
    def forward(self, input_ts):
        x, labels, idx_train = input_ts  # x is N * L, where L is the time-series feature dimension

        if True:
            N = x.size(0)

            A = self.gat(x)

            # y = torch.diag_embed(torch.ones(x.shape[0], x.shape[1]))
            # A = y.clone().detach().to('cuda:0')

            threshold = self.threshold
            batch_size, n_feature, len_feature = x.shape[0], x.shape[1], x.shape[2]
            offset, row, col = (A > threshold).nonzero().t()
            # edge_weight = adj[offset, row, col]
            row += offset * n_feature
            col += offset * n_feature
            edge_index = torch.stack([row, col], dim=0)
            x_gnn = x.contiguous().view(batch_size * n_feature, len_feature)  # 这里改动了长度
            batch = torch.arange(0, batch_size).view(-1, 1).repeat(1, n_feature).view(-1).to('cuda:0')

            x_gnn = F.relu(self.graph_conv1(x_gnn, edge_index))
            x_gnn, edge_index, _, batch, _, _ = self.graph_pool1(x_gnn, edge_index, None, batch)
            x_gnn_1 = torch.cat([gmp(x_gnn, batch), gep(x_gnn, batch)], dim=1)

            x_gnn = F.relu(self.graph_conv2(x_gnn, edge_index))
            x_gnn, edge_index, _, batch, _, _ = self.graph_pool2(x_gnn, edge_index, None, batch)
            x_gnn_2 = torch.cat([gmp(x_gnn, batch), gep(x_gnn, batch)], dim=1)

            x_gnn = F.relu(self.graph_conv3(x_gnn, edge_index))
            x_gnn, edge_index, _, batch, _, _ = self.graph_pool3(x_gnn, edge_index, None, batch)
            x_gnn_3 = torch.cat([gmp(x_gnn, batch), gep(x_gnn, batch)], dim=1)

            # (Batch, 2 * hidden)
            x_gnn = x_gnn_1 + x_gnn_2 + x_gnn_3

            ### 新增

            # LSTM

            x_gru = self.gru(x)[0]
            x_gru = x_gru.mean(1)
            x_gru = x_gru.view(N, -1)

            # Covolutional Network
            # input ts: # N * C * L

            # for i in range(len(self.conv_1_models)):
            x_conv = x
            # x_conv = self.conv_1_models[i](x[:, self.idx[i], :])
            x_conv = self.conv_1(x_conv)
            # x_conv = self.SE1(x_conv)
            # x_conv = self.conv_bn_1(x_conv)
            # x_conv = F.leaky_relu(x_conv)

            x_conv = self.conv_2(x_conv)
            # x_conv = self.SE2(x_conv)
            # x_conv = self.conv_bn_2(x_conv)
            # x_conv = F.leaky_relu(x_conv)

            x_conv = self.conv_3(x_conv)
            # x_conv = self.conv_bn_3(x_conv)
            # x_conv = F.leaky_relu(x_conv)

            x_conv = torch.mean(x_conv, 2)

            #     if i == 0:
            #         x_conv_sum = x_conv
            #     else:
            #         x_conv_sum = torch.cat([x_conv_sum, x_conv], dim=1)
            #
            # x_conv = x_conv_sum

            x = torch.cat([x_conv, x_gru], dim=1)

        # linear mapping to low-dimensional space
        x = torch.cat((x, x_gnn), 1)
        x = self.mapping(x)

        # generate the class protocol with dimension C * D (nclass * dim)
        proto_list = []
        for i in range(self.nclass):
            idx = (labels[idx_train].squeeze() == i).nonzero().squeeze(1)
            if self.use_att:
                A = self.att_models[i](x[idx_train][idx])  # N_k * 1
                A = torch.transpose(A, 1, 0)  # 1 * N_k
                A = F.softmax(A, dim=1)  # softmax over N_k

                class_repr = torch.mm(A, x[idx_train][idx])  # 1 * L
                class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
            else:  # if do not use attention, simply use the mean of training samples with the same labels.
                class_repr = x[idx_train][idx].mean(0)  # L * 1
            proto_list.append(class_repr.view(1, -1))
        x_proto = torch.cat(proto_list, dim=0)

        # prototype distance
        proto_dists = euclidean_dist(x_proto, x_proto)
        proto_dists = torch.exp(-0.5*proto_dists)
        num_proto_pairs = int(self.nclass * (self.nclass - 1) / 2)
        proto_dist = torch.sum(proto_dists) / num_proto_pairs

        dists = euclidean_dist(x, x_proto)

        # dump_embedding(x_proto, x, labels)
        return torch.exp(-0.5*dists), proto_dist
