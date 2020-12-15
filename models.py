import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
from torch_geometric.nn import SAGEConv
import torch_geometric.utils as pyg_utils

def init_weight_(weight):
    """
    Initialize a weighting tensor
    """
    nn.init.xavier_uniform_(weight)

class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, task='node'):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim))

        self.task = task
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = args.dropout
        self.num_layers = args.num_layers

    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GraphSage':
            return GraphSage
        elif model_type == 'GAT':
            return GAT

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        ############################################################################
        # Each layer in GNN consist of a convolution (specified in model_type)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = nn.Dropout(self.dropout)(x)
        if self.task == 'graph':
            x = pyg_nn.global_max_pool(x, batch)
        
        ############################################################################

        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GraphSage(pyg_nn.MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels, reducer='mean', 
                 normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='mean')

        ############################################################################
        # Define the layers needed for the forward function. 
        self.lin = nn.Linear(in_channels, out_channels)
        self.agg_lin = nn.Linear(in_channels, out_channels)

        ############################################################################
        init_weight_(self.lin.weight)
        init_weight_(self.agg_lin.weight)
        if normalize_embedding:
            self.normalize_emb = True

    def forward(self, x, edge_index, size=None):
        num_nodes = x.size(0)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        out = F.relu(self.agg_lin(x))
        ############################################################################

        return self.propagate(edge_index, size_i = num_nodes, size_j = num_nodes, size=(num_nodes, num_nodes), x=out, ori_x = x)

    def message(self, x_j, edge_index, size_i, size_j):
        # x_j has shape [E, out_channels]
        return x_j
        row, col = edge_index
        deg = pyg_utils.degree(row, size_i, dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out, ori_x):
        ############################################################################
        # if self.normalize_emb:
        #     aggr_out = F.normalize(aggr_out, dim=-1)
        ############################################################################
        ori_x = self.lin(ori_x)
        x = F.relu(ori_x + aggr_out)
        if self.normalize_emb:
            x = F.normalize(x, 2, dim=-1)
        return x


class GAT(pyg_nn.MessagePassing):

    def __init__(self, in_channels, out_channels, num_heads=1, concat=False,
                 dropout=0, bias=True, **kwargs):
        super(GAT, self).__init__(aggr='add', **kwargs)
        print('head--', num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = num_heads
        self.concat = concat 
        self.dropout = dropout

        ############################################################################
        # Define the layers needed for the forward function. 
        self.lin = nn.Linear(self.in_channels, self.out_channels)

        ############################################################################

        ############################################################################
        # The attention mechanism is a single feed-forward neural network parametrized
        # by weight vector self.att.

        self.att = nn.Parameter(torch.Tensor(self.heads, 2 * self.out_channels))

        ############################################################################

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

        ############################################################################

    def forward(self, x, edge_index, size=None):
        ############################################################################
        # linear transformation to the node feature matrix before starting
        # to propagate messages.
        
        x = self.lin(x) # TODO
        ############################################################################

        # Start propagating messages.
        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        #  Constructs messages to node i for each edge (j, i).

        ############################################################################
        # Compute the attention coefficients alpha
        alpha = torch.mm(torch.cat([x_i, x_j], dim=-1), self.att.T)
        alpha = nn.LeakyReLU(0.2)(alpha)
        alpha = pyg_utils.softmax(alpha, edge_index_i, num_nodes=size_i)
        ############################################################################
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        res = (x_j.view(-1,self.out_channels,1) * alpha.view(-1, 1, self.heads)).sum(-1)
        return res

    def update(self, aggr_out):
        # Updates node embedings.
        # if self.concat is True:
        #     aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        # else:
        #     aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
