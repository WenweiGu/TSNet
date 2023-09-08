from __future__ import division
from __future__ import print_function

import math
import sys
import time
import argparse

import torch.optim as optim
from utils import *
import torch.nn.functional as F
from models import RTNet

parser = argparse.ArgumentParser()

# dataset settings
parser.add_argument('--data_path', type=str, default="./data/",
                    help='the path of data.')
parser.add_argument('--dataset', type=str, default="NATOPS",
                    help='time series dataset. Options: See the datasets list')

# cuda settings
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

# Training parameter settings
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Initial learning rate. default:[0.00001]')
parser.add_argument('--wd', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters). default: 5e-3')
parser.add_argument('--stop_thres', type=float, default=1e-9,
                    help='The stop threshold for the training error. If the difference between training losses '
                         'between epoches are less than the threshold, the training will be stopped. Default:1e-9')

# Model parameters

parser.add_argument('--filters', type=str, default=[256, 256, 256],
                    help='filters used for convolutional network. Default:256,256,128')
parser.add_argument('--kernels', type=str, default=[8, 5, 3],
                    help='kernels used for convolutional network. Default:8,5,3')
parser.add_argument('--dilation', type=list, default=[1, 2, 4],
                    help='the dilation used for the first convolutional layer. '
                         'If set to -1, use the automatic number. Default:-1')
parser.add_argument('--layers', type=str, default=[200, 100],
                    help='layer settings of mapping function.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability). Default:0.5')
parser.add_argument('--gru_dim', type=int, default=128,
                    help='Dimension of GRU Embedding.')
parser.add_argument('--hidden_dim', type=int, default=128,
                    help='Dimension of GNN Embedding.')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='Pooling ratio of SAGPool.')
parser.add_argument('--threshold', type=float, default=0.01,
                    help='threshold of graph.')
parser.add_argument('--att_dim', type=int, default=128,
                    help='Dimension of attention.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
args.sparse = True

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

print("Loading dataset", args.dataset, "...")
# Model and optimizer
model_type = "TapNet"

if model_type == "TapNet":

    features, labels, idx_train, idx_val, idx_test, nclass = load_raw_ts(args.data_path, dataset=args.dataset)
    print("Data shape:", features.size())

    # update dilation parameter
    if args.dilation == -1:
        args.dilation = math.floor(features.shape[2] / 64)

    model = RTNet(nfeat=features.shape[1],
                   len_ts=features.shape[2],
                   layers=args.layers,
                   nclass=nclass,
                   hidden_dim=args.hidden_dim,
                   pooling_ratio=args.pooling_ratio,
                   threshold=args.threshold,
                   dropout=args.dropout,
                   filters=args.filters,
                   dilation=args.dilation,
                   kernels=args.kernels,
                   gru_dim=args.gru_dim,
                   att_dim=args.att_dim
                   )

    # cuda
    if args.cuda:
        # model = nn.DataParallel(model) Used when you have more than one GPU. Sometimes work but not stable
        model.cuda()
        features, labels, idx_train = features.cuda(), labels.cuda(), idx_train.cuda()
    input_feature = (features, labels, idx_train)

# init the optimizer
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.wd)


# training function
def train():
    loss_list = [sys.maxsize]
    test_best_possible, best_so_far = 0.0, sys.maxsize
    for epoch in range(args.epochs):

        t = time.time()
        model.train()
        optimizer.zero_grad()

        output, proto_dist = model(input_feature)

        # loss 部分可以加入别的部分, 有一定作用
        loss_train = F.cross_entropy(output[idx_train], torch.squeeze(labels[idx_train])) + 0.1 * 1 / proto_dist ** 0.5

        # 暂时注释掉，因为会立马停止训练
        # if abs(loss_train.item() - loss_list[-1]) < args.stop_thres \
        #         or loss_train.item() > loss_list[-1]:
        #     break

        loss_list.append(loss_train.item())

        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        loss_val = F.cross_entropy(output[idx_val], torch.squeeze(labels[idx_val]))
        acc_val = accuracy(output[idx_val], labels[idx_val])

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.8f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        if acc_val.item() > test_best_possible:
            test_best_possible = acc_val.item()
        if best_so_far > loss_train.item():
            best_so_far = loss_train.item()
            test_acc = acc_val.item()
    print("test_acc: " + str(test_acc))
    print("best possible: " + str(test_best_possible))


# test function
def test():
    output, proto_dist = model(input_feature)
    loss_test = F.cross_entropy(output[idx_test], torch.squeeze(labels[idx_test]))
    if args.use_metric:
        loss_test = loss_test - args.metric_param * proto_dist

    acc_test = accuracy(output[idx_test], labels[idx_test])
    print(args.dataset, "Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
train()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
